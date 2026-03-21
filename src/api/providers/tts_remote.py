"""
Remote TTS Provider — delegates to TTS Service via gRPC.

Reuses the existing tts-server (proto/tts_service.proto).

Config:
    TTS_PROVIDER=remote
    TTS_REMOTE_TARGET=localhost:50070
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import AsyncGenerator

import grpc

from grpc_gen import tts_service_pb2 as tts_pb
from grpc_gen import tts_service_pb2_grpc

from config import TTS
from providers.tts import TTSProvider, register_tts_provider

logger = logging.getLogger("open-voice-api.tts.remote")


class RemoteTTS(TTSProvider):
    """TTS provider that delegates to external TTS Service via gRPC."""

    provider_name = "remote"

    def __init__(self):
        self._target = TTS.remote_target
        self._language = TTS.language
        self._timeout = TTS.remote_timeout
        self._channel: grpc.aio.Channel | None = None
        self._stub: tts_service_pb2_grpc.TTSServiceStub | None = None

    async def connect(self) -> None:
        max_msg = 10 * 1024 * 1024
        self._channel = grpc.aio.insecure_channel(
            self._target,
            options=[
                ("grpc.max_receive_message_length", max_msg),
                ("grpc.max_send_message_length", max_msg),
                ("grpc.keepalive_time_ms", 30000),
                ("grpc.keepalive_timeout_ms", 10000),
            ],
        )
        self._stub = tts_service_pb2_grpc.TTSServiceStub(self._channel)
        logger.info(f"Remote TTS channel created: {self._target} (lazy connect)")

    async def warmup(self) -> None:
        """Pre-warm gRPC channel by forcing TCP + HTTP/2 handshake."""
        if not self._channel:
            return
        try:
            await asyncio.wait_for(
                self._channel.channel_ready(), timeout=5.0
            )
            logger.info(f"TTS gRPC channel warmed up: {self._target}")
        except Exception as e:
            logger.warning(f"TTS warmup failed (will retry on first call): {e}")

    async def disconnect(self) -> None:
        if self._channel:
            await self._channel.close()
            self._channel = None
            self._stub = None

    async def health_check(self) -> bool:
        """Check gRPC channel connectivity."""
        return self._channel is not None and self._stub is not None

    async def synthesize(self, text: str) -> bytes:
        if not self._stub:
            raise RuntimeError("Remote TTS not connected")
        request = tts_pb.SynthesizeRequest(text=text, language=self._language)
        t0 = time.perf_counter()
        try:
            response = await asyncio.wait_for(
                self._stub.Synthesize(request), timeout=self._timeout
            )
            elapsed_ms = (time.perf_counter() - t0) * 1000
            logger.info(f"TTS batch: {elapsed_ms:.0f}ms → {len(response.audio_data)} bytes: \"{text[:40]}\"")
            return response.audio_data
        except (grpc.aio.AioRpcError, asyncio.TimeoutError) as e:
            elapsed_ms = (time.perf_counter() - t0) * 1000
            logger.error(f"TTS error after {elapsed_ms:.0f}ms: {e}")
            raise RuntimeError(f"TTS service unavailable: {e}") from e

    async def synthesize_stream(self, text: str) -> AsyncGenerator[bytes, None]:
        if not self._stub:
            raise RuntimeError("Remote TTS not connected")
        request = tts_pb.SynthesizeRequest(text=text, language=self._language)
        t0 = time.perf_counter()
        try:
            async for chunk in self._stub.SynthesizeStream(request):
                if chunk.is_last:
                    break
                if chunk.audio_payload:
                    yield chunk.audio_payload
        except (grpc.aio.AioRpcError, asyncio.TimeoutError) as e:
            elapsed_ms = (time.perf_counter() - t0) * 1000
            logger.error(f"TTS stream error after {elapsed_ms:.0f}ms: {e}")
            raise RuntimeError(f"TTS stream failed: {e}") from e

    @property
    def supports_streaming(self) -> bool:
        return True


register_tts_provider("remote", RemoteTTS)
