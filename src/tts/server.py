"""
TTS gRPC Server — Text-to-Speech como servico independente.

Carrega o modelo TTS uma vez e expoe via gRPC para o AI Agent.
Suporta modo batch (Synthesize) e streaming (SynthesizeStream).
"""

import asyncio
import logging
import os
import signal
import time
from typing import AsyncIterator, Optional

import grpc
from grpc_health.v1 import health, health_pb2, health_pb2_grpc
from grpc_reflection.v1alpha import reflection

from shared.grpc_gen import tts_service_pb2 as tts_pb
from shared.grpc_gen import tts_service_pb2_grpc

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("tts-server")


def _configure_logging():
    level = os.getenv("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


# =============================================================================
# TTS Provider loading (reusa providers existentes do ai-agent)
# =============================================================================

async def _create_provider():
    """Cria e conecta o TTS provider configurado via env vars."""
    from tts.providers.base import create_tts_provider
    provider = await create_tts_provider()
    logger.info(f"TTS provider carregado: {provider.provider_name}")
    return provider


# =============================================================================
# gRPC Servicer
# =============================================================================

class TTSServicer(tts_service_pb2_grpc.TTSServiceServicer):
    """Implementacao do TTSService gRPC."""

    def __init__(self, provider):
        self._provider = provider

    async def Synthesize(
        self,
        request: tts_pb.SynthesizeRequest,
        context: grpc.aio.ServicerContext,
    ) -> tts_pb.SynthesizeResponse:
        """Batch: texto completo -> audio completo."""
        start = time.perf_counter()

        if not request.text or not request.text.strip():
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details("text vazio")
            return tts_pb.SynthesizeResponse()

        try:
            audio = await self._provider.synthesize(request.text)
        except Exception as e:
            logger.error(f"Synthesize error: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return tts_pb.SynthesizeResponse()

        processing_ms = (time.perf_counter() - start) * 1000

        # Calcula duracao do audio (PCM 8kHz 16-bit mono)
        sample_rate = 8000
        if request.output_config and request.output_config.sample_rate > 0:
            sample_rate = request.output_config.sample_rate
        audio_duration_ms = (len(audio) / (sample_rate * 2) * 1000) if audio else 0.0

        logger.info(
            f"Synthesize: \"{request.text[:40]}...\" -> "
            f"{len(audio) if audio else 0} bytes "
            f"({audio_duration_ms:.0f}ms audio, {processing_ms:.0f}ms proc)"
        )

        return tts_pb.SynthesizeResponse(
            audio_data=audio or b"",
            duration_ms=audio_duration_ms,
            processing_ms=processing_ms,
        )

    async def SynthesizeStream(
        self,
        request: tts_pb.SynthesizeRequest,
        context: grpc.aio.ServicerContext,
    ) -> AsyncIterator[tts_pb.AudioChunk]:
        """Streaming: texto -> stream de chunks de audio."""
        if not request.text or not request.text.strip():
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details("text vazio")
            return

        try:
            sequence = 0
            async for chunk in self._provider.synthesize_stream(request.text):
                if chunk:
                    yield tts_pb.AudioChunk(
                        audio_payload=chunk,
                        is_last=False,
                        sequence=sequence,
                    )
                    sequence += 1

            # Envia chunk final vazio para sinalizar fim
            yield tts_pb.AudioChunk(
                audio_payload=b"",
                is_last=True,
                sequence=sequence,
            )

            logger.info(
                f"SynthesizeStream: \"{request.text[:40]}...\" -> "
                f"{sequence} chunks"
            )

        except asyncio.CancelledError:
            logger.info("SynthesizeStream cancelado")
        except Exception as e:
            logger.error(f"SynthesizeStream error: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))


# =============================================================================
# Server lifecycle
# =============================================================================

class TTSServer:
    """Wrapper do servidor gRPC para TTS."""

    def __init__(self):
        self._server: Optional[grpc.aio.Server] = None
        self._provider = None

    async def start(self, host: str = None, port: int = None):
        host = host or os.getenv("GRPC_HOST", "0.0.0.0")
        port = port or int(os.getenv("GRPC_PORT", "50070"))

        # Carrega provider
        self._provider = await _create_provider()

        max_msg = int(os.getenv("GRPC_MAX_MESSAGE_SIZE", str(10 * 1024 * 1024)))
        self._server = grpc.aio.server(
            options=[
                ("grpc.max_send_message_length", max_msg),
                ("grpc.max_receive_message_length", max_msg),
                ("grpc.keepalive_time_ms", 30000),
                ("grpc.keepalive_timeout_ms", 10000),
                ("grpc.keepalive_permit_without_calls", True),
                ("grpc.http2.min_recv_ping_interval_without_data_ms", 10000),
                ("grpc.http2.max_ping_strikes", 0),
            ],
        )

        servicer = TTSServicer(self._provider)
        tts_service_pb2_grpc.add_TTSServiceServicer_to_server(servicer, self._server)

        # Health check
        health_servicer = health.HealthServicer()
        health_pb2_grpc.add_HealthServicer_to_server(health_servicer, self._server)
        health_servicer.set(
            "theo.tts.TTSService",
            health_pb2.HealthCheckResponse.SERVING,
        )

        # Reflection
        service_names = (
            tts_pb.DESCRIPTOR.services_by_name["TTSService"].full_name,
            health_pb2.DESCRIPTOR.services_by_name["Health"].full_name,
            reflection.SERVICE_NAME,
        )
        reflection.enable_server_reflection(service_names, self._server)

        self._server.add_insecure_port(f"{host}:{port}")
        await self._server.start()
        logger.info(f"TTS Server started on {host}:{port}")

    async def stop(self):
        if self._server:
            await self._server.stop(grace=5)
        if self._provider:
            await self._provider.disconnect()
        logger.info("TTS Server stopped")

    async def wait(self):
        if self._server:
            await self._server.wait_for_termination()


async def main():
    _configure_logging()
    server = TTSServer()
    await server.start()

    loop = asyncio.get_running_loop()
    stop_event = asyncio.Event()

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, stop_event.set)

    await stop_event.wait()
    logger.info("Shutting down...")
    await server.stop()


if __name__ == "__main__":
    asyncio.run(main())
