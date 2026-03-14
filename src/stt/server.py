"""
STT gRPC Server — Speech-to-Text como servico independente.

Carrega o modelo STT uma vez e expoe via gRPC para o AI Agent.
Suporta modo batch (Transcribe) e streaming (TranscribeStream).
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

from shared.grpc_gen import stt_service_pb2 as stt_pb
from shared.grpc_gen import stt_service_pb2_grpc

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("stt-server")


def _configure_logging():
    level = os.getenv("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


# =============================================================================
# STT Provider loading (reusa providers existentes do ai-agent)
# =============================================================================

async def _create_provider():
    """Cria e conecta o STT provider configurado via env vars."""
    # Importa o sistema de providers do ai-agent
    from stt.providers.base import create_stt_provider
    provider = await create_stt_provider()
    logger.info(f"STT provider carregado: {provider.provider_name}")
    return provider


# =============================================================================
# gRPC Servicer
# =============================================================================

class STTServicer(stt_service_pb2_grpc.STTServiceServicer):
    """Implementacao do STTService gRPC."""

    def __init__(self, provider):
        self._provider = provider

    async def Transcribe(
        self,
        request: stt_pb.TranscribeRequest,
        context: grpc.aio.ServicerContext,
    ) -> stt_pb.TranscribeResponse:
        """Batch: audio completo -> transcricao."""
        start = time.perf_counter()

        if not request.audio_data:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details("audio_data vazio")
            return stt_pb.TranscribeResponse()

        try:
            text = await self._provider.transcribe(request.audio_data)
        except Exception as e:
            logger.error(f"Transcribe error: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return stt_pb.TranscribeResponse()

        processing_ms = (time.perf_counter() - start) * 1000
        result_text = text or ""
        logger.info(
            f"Transcribe: \"{result_text[:60]}\" ({processing_ms:.0f}ms, "
            f"{len(request.audio_data)} bytes)"
        )

        return stt_pb.TranscribeResponse(
            text=result_text,
            processing_ms=processing_ms,
        )

    async def TranscribeStream(
        self,
        request_iterator: AsyncIterator[stt_pb.AudioChunk],
        context: grpc.aio.ServicerContext,
    ) -> AsyncIterator[stt_pb.TranscribeResult]:
        """Streaming: chunks de audio -> transcricoes parciais + final."""
        if not self._provider.supports_streaming:
            context.set_code(grpc.StatusCode.UNIMPLEMENTED)
            context.set_details(
                f"Provider '{self._provider.provider_name}' nao suporta streaming"
            )
            return

        stream_id: Optional[str] = None

        try:
            async for chunk in request_iterator:
                # Primeiro chunk: inicializa streaming
                if stream_id is None:
                    stream_id = chunk.stream_id or "default"
                    await self._provider.start_streaming(stream_id=stream_id)
                    logger.info(f"[{stream_id[:8]}] Streaming STT iniciado")

                # Fim do stream: finaliza e envia resultado final
                if chunk.end_of_stream:
                    final_text = await self._provider.finish_streaming(
                        stream_id=stream_id,
                    )
                    logger.info(
                        f"[{stream_id[:8]}] Streaming STT finalizado: "
                        f"\"{final_text[:60] if final_text else ''}\""
                    )
                    yield stt_pb.TranscribeResult(
                        stream_id=stream_id,
                        text=final_text or "",
                        is_final=True,
                    )
                    return

                # Chunk normal: processa e retorna parcial
                if chunk.audio_payload:
                    partial = await self._provider.process_chunk(
                        chunk.audio_payload,
                        stream_id=stream_id,
                    )
                    if partial:
                        yield stt_pb.TranscribeResult(
                            stream_id=stream_id,
                            text=partial,
                            is_final=False,
                        )

        except asyncio.CancelledError:
            logger.info(f"[{stream_id and stream_id[:8] or '?'}] Stream cancelado")
        except Exception as e:
            logger.error(f"[{stream_id and stream_id[:8] or '?'}] Streaming error: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))


# =============================================================================
# Server lifecycle
# =============================================================================

class STTServer:
    """Wrapper do servidor gRPC para STT."""

    def __init__(self):
        self._server: Optional[grpc.aio.Server] = None
        self._provider = None

    async def start(self, host: str = None, port: int = None):
        host = host or os.getenv("GRPC_HOST", "0.0.0.0")
        port = port or int(os.getenv("GRPC_PORT", "50060"))

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

        servicer = STTServicer(self._provider)
        stt_service_pb2_grpc.add_STTServiceServicer_to_server(servicer, self._server)

        # Health check
        health_servicer = health.HealthServicer()
        health_pb2_grpc.add_HealthServicer_to_server(health_servicer, self._server)
        health_servicer.set(
            "theo.stt.STTService",
            health_pb2.HealthCheckResponse.SERVING,
        )

        # Reflection
        service_names = (
            stt_pb.DESCRIPTOR.services_by_name["STTService"].full_name,
            health_pb2.DESCRIPTOR.services_by_name["Health"].full_name,
            reflection.SERVICE_NAME,
        )
        reflection.enable_server_reflection(service_names, self._server)

        self._server.add_insecure_port(f"{host}:{port}")
        await self._server.start()
        logger.info(f"STT Server started on {host}:{port}")

    async def stop(self):
        if self._server:
            await self._server.stop(grace=5)
        if self._provider:
            await self._provider.disconnect()
        logger.info("STT Server stopped")

    async def wait(self):
        if self._server:
            await self._server.wait_for_termination()


async def main():
    _configure_logging()
    server = STTServer()
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
