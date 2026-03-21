"""
STT gRPC Server — Speech-to-Text como servico independente.

Carrega o modelo STT uma vez e expoe via gRPC para o AI Agent.
Suporta modo batch (Transcribe) e streaming (TranscribeStream).
"""

import asyncio
import logging
import time
from typing import AsyncIterator, Optional

import grpc

from shared.grpc_gen import stt_service_pb2 as stt_pb
from shared.grpc_gen import stt_service_pb2_grpc

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("stt-server")


# =============================================================================
# STT Provider loading
# =============================================================================

async def _create_provider():
    """Cria e conecta o STT provider configurado via env vars."""
    from stt.providers.base import create_stt_provider
    provider = await create_stt_provider()
    logger.info(f"STT provider carregado: {provider.provider_name}")
    return provider


# =============================================================================
# gRPC Servicer
# =============================================================================

class STTServicer(stt_service_pb2_grpc.STTServiceServicer):
    """Implementacao do STTService gRPC."""

    def __init__(self, provider, health_servicer=None):
        self._provider = provider
        from common.grpc_server import HealthTracker
        self._health_tracker = HealthTracker(
            "theo.stt.STTService", health_servicer,
        )

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
            self._health_tracker.record_success()
        except Exception as e:
            logger.error(f"Transcribe error: {e}")
            self._health_tracker.record_error()
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
                    self._health_tracker.record_success()
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
            self._health_tracker.record_error()
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))


# =============================================================================
# Server lifecycle
# =============================================================================

class STTServer:
    """Wrapper do servidor gRPC para STT."""

    def __init__(self):
        from common.grpc_server import GrpcMicroservice
        self._micro = GrpcMicroservice("theo.stt.STTService", port=None)
        self._provider = None

    async def start(self, host: str = None, port: int = None):
        self._provider = await _create_provider()

        if host:
            self._micro._host = host
        if port:
            self._micro._port = port

        def add_servicers(server, health_servicer):
            servicer = STTServicer(self._provider, health_servicer)
            stt_service_pb2_grpc.add_STTServiceServicer_to_server(servicer, server)

        await self._micro.start(
            add_servicers=add_servicers,
            service_names=(
                stt_pb.DESCRIPTOR.services_by_name["STTService"].full_name,
            ),
            provider=self._provider,
        )

    async def stop(self):
        await self._micro.stop()

    async def wait(self):
        await self._micro.wait()


async def main():
    from common.grpc_server import configure_logging
    configure_logging()
    server = STTServer()
    await server.start()
    await server._micro.run_until_signal()


if __name__ == "__main__":
    asyncio.run(main())
