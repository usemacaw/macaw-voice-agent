"""
Remote ASR Provider — delegates to STT Service via gRPC.

Reuses the existing stt-server (proto/stt_service.proto).

Config:
    ASR_PROVIDER=remote
    ASR_REMOTE_TARGET=localhost:50060
"""

from __future__ import annotations

import asyncio
import logging
import time

import grpc

from grpc_gen import stt_service_pb2 as stt_pb
from grpc_gen import stt_service_pb2_grpc

from config import ASR_CONFIG
from providers.asr import ASRProvider, register_asr_provider

logger = logging.getLogger("open-voice-api.asr.remote")


class RemoteASR(ASRProvider):
    """ASR provider that delegates to external STT Service via gRPC."""

    provider_name = "remote"

    def __init__(self):
        self._target = ASR_CONFIG["remote_target"]
        self._timeout = ASR_CONFIG["remote_timeout"]
        self._language = ASR_CONFIG.get("language", "pt")
        self._channel: grpc.aio.Channel | None = None
        self._stub: stt_service_pb2_grpc.STTServiceStub | None = None
        self._streams: dict[str, _StreamingSession] = {}
        self._pre_buffers: dict[str, list[bytes]] = {}

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
        self._stub = stt_service_pb2_grpc.STTServiceStub(self._channel)
        logger.info(f"Remote ASR channel created: {self._target} (lazy connect)")

    async def warmup(self) -> None:
        """Pre-warm gRPC channel by forcing TCP + HTTP/2 handshake."""
        if not self._channel:
            return
        try:
            await asyncio.wait_for(
                self._channel.channel_ready(), timeout=5.0
            )
            logger.info(f"ASR gRPC channel warmed up: {self._target}")
        except Exception as e:
            logger.warning(f"ASR warmup failed (will retry on first call): {e}")

    async def disconnect(self) -> None:
        for session in list(self._streams.values()):
            session.cancel()
        self._streams.clear()
        self._pre_buffers.clear()
        if self._channel:
            await self._channel.close()
            self._channel = None
            self._stub = None

    async def transcribe(self, audio: bytes) -> str:
        if not self._stub:
            raise RuntimeError("Remote ASR not connected")
        request = stt_pb.TranscribeRequest(audio_data=audio, language=self._language)
        t0 = time.perf_counter()
        try:
            response = await asyncio.wait_for(
                self._stub.Transcribe(request), timeout=self._timeout
            )
            elapsed_ms = (time.perf_counter() - t0) * 1000
            logger.info(f"ASR batch: {elapsed_ms:.0f}ms → \"{response.text[:60]}\"")
            return response.text
        except (grpc.aio.AioRpcError, asyncio.TimeoutError) as e:
            elapsed_ms = (time.perf_counter() - t0) * 1000
            logger.error(f"ASR error after {elapsed_ms:.0f}ms: {e}")
            raise RuntimeError(f"ASR service unavailable: {e}") from e

    @property
    def supports_streaming(self) -> bool:
        return ASR_CONFIG["remote_streaming"]

    async def start_stream(self, stream_id: str) -> None:
        if not self._stub:
            raise RuntimeError("Remote ASR not connected")
        if stream_id in self._streams:
            self._streams[stream_id].cancel()
        session = _StreamingSession(self._stub, stream_id, self._timeout)
        self._streams[stream_id] = session
        await session.start()
        # Flush any buffered chunks that arrived before start
        buffered = self._pre_buffers.pop(stream_id, [])
        if buffered:
            logger.info(f"[{stream_id[:8]}] Flushing {len(buffered)} pre-buffered chunks")
            for chunk in buffered:
                await session.send_chunk(chunk)

    # Max pre-buffer chunks before start_stream (prevents unbounded memory growth)
    _MAX_PRE_BUFFER_CHUNKS = 500  # ~16s at 32ms chunks

    async def feed_chunk(self, audio: bytes, stream_id: str) -> str:
        session = self._streams.get(stream_id)
        if not session:
            # Buffer chunks that arrive before start_stream completes
            if stream_id not in self._pre_buffers:
                self._pre_buffers[stream_id] = []
            buf = self._pre_buffers[stream_id]
            if len(buf) < self._MAX_PRE_BUFFER_CHUNKS:
                buf.append(audio)
            else:
                logger.warning(
                    f"[{stream_id[:8]}] Pre-buffer full ({self._MAX_PRE_BUFFER_CHUNKS} chunks), "
                    f"dropping chunk"
                )
            return ""
        return await session.send_chunk(audio)

    async def finish_stream(self, stream_id: str) -> str:
        session = self._streams.pop(stream_id, None)
        self._pre_buffers.pop(stream_id, None)
        if not session:
            return ""
        return await session.finish()


class _StreamingSession:
    def __init__(self, stub: stt_service_pb2_grpc.STTServiceStub, stream_id: str, timeout: float):
        self._stub = stub
        self._stream_id = stream_id
        self._timeout = timeout
        self._request_queue: asyncio.Queue[stt_pb.AudioChunk | None] = asyncio.Queue()
        self._response_stream = None
        self._read_task: asyncio.Task | None = None
        self._last_partial = ""
        self._final_text = ""
        self._done = asyncio.Event()

    async def start(self):
        self._response_stream = self._stub.TranscribeStream(self._request_iter())
        self._read_task = asyncio.create_task(self._read_responses())

    async def _request_iter(self):
        while True:
            chunk = await self._request_queue.get()
            if chunk is None:
                return
            yield chunk

    async def _read_responses(self):
        try:
            async for result in self._response_stream:
                if result.is_final:
                    self._final_text = result.text
                    self._done.set()
                    return
                self._last_partial = result.text
        except asyncio.CancelledError:
            pass
        except grpc.aio.AioRpcError as e:
            logger.error(f"[{self._stream_id[:8]}] Stream error: {e.code()}")
        finally:
            self._done.set()

    async def send_chunk(self, audio_payload: bytes) -> str:
        await self._request_queue.put(
            stt_pb.AudioChunk(stream_id=self._stream_id, audio_payload=audio_payload)
        )
        return self._last_partial

    async def finish(self) -> str:
        await self._request_queue.put(
            stt_pb.AudioChunk(stream_id=self._stream_id, end_of_stream=True)
        )
        await self._request_queue.put(None)
        try:
            await asyncio.wait_for(self._done.wait(), timeout=self._timeout)
        except asyncio.TimeoutError:
            logger.error(f"[{self._stream_id[:8]}] finish timeout")
            return self._last_partial
        if self._read_task and not self._read_task.done():
            self._read_task.cancel()
            try:
                await self._read_task
            except (asyncio.CancelledError, Exception):
                pass
        return self._final_text

    def cancel(self):
        try:
            self._request_queue.put_nowait(None)
        except asyncio.QueueFull:
            pass
        if self._read_task and not self._read_task.done():
            self._read_task.cancel()
        self._done.set()


register_asr_provider("remote", RemoteASR)
