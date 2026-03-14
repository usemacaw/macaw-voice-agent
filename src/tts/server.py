"""
TTS gRPC Server — Text-to-Speech como servico independente.

Carrega o modelo TTS uma vez e expoe via gRPC para o AI Agent.
Suporta modo batch (Synthesize) e streaming (SynthesizeStream).
"""

import asyncio
import logging
import time
from typing import AsyncIterator, Optional

import grpc

from shared.grpc_gen import tts_service_pb2 as tts_pb
from shared.grpc_gen import tts_service_pb2_grpc

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("tts-server")


def _configure_logging():
    import os
    level = os.getenv("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


# =============================================================================
# TTS Provider loading
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

    def __init__(self, provider, health_servicer=None):
        self._provider = provider
        self._health = health_servicer
        self._consecutive_errors = 0
        self._MAX_ERRORS_BEFORE_UNHEALTHY = 5

    def _record_success(self):
        if self._consecutive_errors > 0:
            self._consecutive_errors = 0
            if self._health:
                from grpc_health.v1 import health_pb2
                self._health.set(
                    "theo.tts.TTSService",
                    health_pb2.HealthCheckResponse.SERVING,
                )

    def _record_error(self):
        self._consecutive_errors += 1
        if self._consecutive_errors >= self._MAX_ERRORS_BEFORE_UNHEALTHY and self._health:
            from grpc_health.v1 import health_pb2
            self._health.set(
                "theo.tts.TTSService",
                health_pb2.HealthCheckResponse.NOT_SERVING,
            )
            logger.error(
                f"Provider degraded after {self._consecutive_errors} consecutive errors"
            )

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
            self._record_success()
        except Exception as e:
            logger.error(f"Synthesize error: {e}")
            self._record_error()
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
            self._record_success()

            logger.info(
                f"SynthesizeStream: \"{request.text[:40]}...\" -> "
                f"{sequence} chunks"
            )

        except asyncio.CancelledError:
            logger.info("SynthesizeStream cancelado")
        except Exception as e:
            logger.error(f"SynthesizeStream error: {e}")
            self._record_error()
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))


# =============================================================================
# Server lifecycle
# =============================================================================

class TTSServer:
    """Wrapper do servidor gRPC para TTS."""

    def __init__(self):
        from common.grpc_server import GrpcMicroservice
        self._micro = GrpcMicroservice("theo.tts.TTSService", port=None)
        self._provider = None

    async def start(self, host: str = None, port: int = None):
        self._provider = await _create_provider()

        if host:
            self._micro._host = host
        if port:
            self._micro._port = port

        def add_servicers(server, health_servicer):
            servicer = TTSServicer(self._provider, health_servicer)
            tts_service_pb2_grpc.add_TTSServiceServicer_to_server(servicer, server)

        await self._micro.start(
            add_servicers=add_servicers,
            service_names=(
                tts_pb.DESCRIPTOR.services_by_name["TTSService"].full_name,
            ),
            provider=self._provider,
        )

    async def stop(self):
        await self._micro.stop()

    async def wait(self):
        await self._micro.wait()


async def main():
    _configure_logging()
    server = TTSServer()
    await server.start()
    await server._micro.run_until_signal()


if __name__ == "__main__":
    asyncio.run(main())
