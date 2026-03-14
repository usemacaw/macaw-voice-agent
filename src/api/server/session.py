"""
RealtimeSession — per-connection state machine.

Manages session config, conversation items, audio buffer, VAD,
and delegates response execution to ResponseRunner.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import time
import uuid
from typing import TYPE_CHECKING

import numpy as np
import websockets.exceptions

from audio.codec import API_SAMPLE_RATE, INTERNAL_SAMPLE_RATE, SAMPLE_WIDTH, decode_audio_from_client
from audio.vad import VADProcessor
from config import LLM_CONFIG, VAD_CONFIG
from protocol import events
from protocol.event_emitter import EventEmitter, SlowClientError
from protocol.models import (
    ConversationItem,
    ConversationItemValidationError,
    ContentPart,
    SessionConfig,
    SessionConfigValidationError,
)
from server.response_runner import ResponseContext, ResponseRunner

from tools.recall_memory import ConversationMemory

if TYPE_CHECKING:
    from providers.asr import ASRProvider
    from providers.llm import LLMProvider
    from providers.tts import TTSProvider
    from tools.registry import ToolRegistry
    from websockets.asyncio.server import ServerConnection

logger = logging.getLogger("open-voice-api.session")

# Maximum audio buffer size in manual commit mode (10 minutes at 8kHz 16-bit)
_MAX_AUDIO_BUFFER_BYTES = 10 * 60 * INTERNAL_SAMPLE_RATE * SAMPLE_WIDTH

# Maximum conversation items kept in memory (FIFO eviction of oldest)
_MAX_CONVERSATION_ITEMS = 200

# Rate limiting: max events per second from a single client
_MAX_EVENTS_PER_SECOND = 200

# Session idle timeout: disconnect if no message received within this period (seconds)
_SESSION_IDLE_TIMEOUT_S = float(600)  # 10 minutes


class RealtimeSession:
    """Per-connection state machine implementing the OpenAI Realtime protocol."""

    def __init__(
        self,
        ws: ServerConnection,
        asr: ASRProvider,
        llm: LLMProvider,
        tts: TTSProvider,
        tool_registry: ToolRegistry | None = None,
    ):
        self._ws = ws
        self._asr = asr
        self._llm = llm
        self._tts = tts
        self._tool_registry = tool_registry

        self.session_id = f"sess_{uuid.uuid4().hex[:24]}"
        self.conversation_id = f"conv_{uuid.uuid4().hex[:24]}"

        self._config = SessionConfig(
            instructions=LLM_CONFIG["system_prompt"],
            temperature=LLM_CONFIG["temperature"],
        )
        self._items: list[ConversationItem] = []
        self._audio_buffer = bytearray()
        self._emitter = EventEmitter(ws, self.session_id)

        # VAD
        self._vad: VADProcessor | None = None
        self._pending_speech_item_id: str | None = None
        self._setup_vad()

        # ASR streaming
        self._asr_stream_id: str | None = None

        # Conversation memory for history search tool
        # Fork registry so each session has its own recall_memory handler
        self._memory = ConversationMemory()
        if self._tool_registry is not None:
            self._tool_registry = self._tool_registry.fork()
            from tools.recall_memory import register_recall_handler
            register_recall_handler(self._tool_registry, self._memory)

        # Active response task
        self._response_task: asyncio.Task | None = None

        # Background tasks (fire-and-forget from VAD callbacks)
        self._background_tasks: set[asyncio.Task] = set()

        # Lock to serialize state mutations (items, response_task, asr_stream_id)
        self._state_lock = asyncio.Lock()

        # Metrics
        self._audio_append_count = 0
        self._audio_bytes_received = 0
        self._last_metrics_log = time.monotonic()
        self._speech_stopped_at: float | None = None  # perf_counter timestamp

        # Per-response observability metrics (reset at each response start)
        self._response_metrics: dict[str, object] = {}

        # Session-level counters for observability
        self._session_start = time.perf_counter()
        self._turn_count = 0
        self._barge_in_count = 0

        # Rate limiting
        self._event_timestamps: list[float] = []

    def _track_task(self, task: asyncio.Task) -> None:
        """Register a fire-and-forget task so exceptions are logged and the task is cleaned up."""
        self._background_tasks.add(task)

        def _done(t: asyncio.Task) -> None:
            self._background_tasks.discard(t)
            if t.cancelled():
                return
            exc = t.exception()
            if exc is not None:
                logger.error(
                    f"[{self.session_id[:8]}] Background task failed: {exc}",
                    exc_info=exc,
                )

        task.add_done_callback(_done)

    def _setup_vad(self) -> None:
        td = self._config.turn_detection
        if td and td.type == "server_vad":
            self._vad = VADProcessor(
                config=td,
                on_speech_started=self._on_speech_started,
                on_speech_stopped=self._on_speech_stopped,
                aggressiveness=VAD_CONFIG["aggressiveness"],
            )
        else:
            self._vad = None

    def _on_speech_started(self, audio_start_ms: int) -> None:
        """Called synchronously by VAD when speech begins.

        NOTE: _pending_speech_item_id and _asr_stream_id are set here
        synchronously (same event loop tick as VAD.feed()). This is safe
        because the event loop is single-threaded — no concurrent coroutine
        can interleave between these assignments and the subsequent
        _handle_speech_started task. The lock is acquired in the async task.
        """
        item_id = f"item_{uuid.uuid4().hex[:24]}"
        self._pending_speech_item_id = item_id
        # Set stream_id synchronously so feed_chunk works immediately
        self._asr_stream_id = item_id if self._asr.supports_streaming else None
        logger.info(
            f"[{self.session_id[:8]}] VAD speech_started at {audio_start_ms}ms, item={item_id[:12]}"
        )
        task = asyncio.create_task(self._handle_speech_started(audio_start_ms, item_id))
        self._track_task(task)

    async def _handle_speech_started(self, audio_start_ms: int, item_id: str) -> None:
        await self._emitter.emit(
            events.input_audio_buffer_speech_started("", audio_start_ms, item_id)
        )

        # Start ASR streaming (stream_id already set synchronously in _on_speech_started)
        if self._asr.supports_streaming:
            await self._asr.start_stream(item_id)

    @staticmethod
    def _compute_rms(speech_audio: bytes) -> float:
        """Compute RMS amplitude of PCM 16-bit audio using numpy."""
        if not speech_audio:
            return 0.0
        n_bytes = len(speech_audio)
        usable = n_bytes - (n_bytes % SAMPLE_WIDTH)
        if usable == 0:
            return 0.0
        samples = np.frombuffer(speech_audio[:usable], dtype=np.int16).astype(np.float64)
        return float(np.sqrt(np.mean(samples * samples)))

    def _on_speech_stopped(self, audio_end_ms: int, speech_audio: bytes) -> None:
        """Called synchronously by VAD when speech ends."""
        item_id = self._pending_speech_item_id or f"item_{uuid.uuid4().hex[:24]}"
        self._pending_speech_item_id = None
        task = asyncio.create_task(
            self._handle_speech_stopped_with_rms(audio_end_ms, speech_audio, item_id)
        )
        self._track_task(task)

    async def _handle_speech_stopped_with_rms(
        self, audio_end_ms: int, speech_audio: bytes, item_id: str
    ) -> None:
        """Validate RMS then dispatch to _handle_speech_stopped."""
        speech_duration_ms = len(speech_audio) / (INTERNAL_SAMPLE_RATE * SAMPLE_WIDTH) * 1000

        rms = self._compute_rms(speech_audio)

        min_rms = VAD_CONFIG.get("min_speech_rms", 150)
        if rms < min_rms:
            logger.debug(
                f"[{self.session_id[:8]}] VAD speech_stopped DISCARDED: rms={rms:.0f} < {min_rms} "
                f"(likely noise/echo), {speech_duration_ms:.0f}ms, item={item_id[:12]}"
            )
            async with self._state_lock:
                if self._asr_stream_id == item_id:
                    self._asr_stream_id = None
            return

        logger.info(
            f"[{self.session_id[:8]}] VAD speech_stopped at {audio_end_ms}ms, "
            f"speech_audio={len(speech_audio)} bytes ({speech_duration_ms:.0f}ms), "
            f"rms={rms:.0f}, item={item_id[:12]}"
        )

        # NOW interrupt active response (barge-in) — we know this is real speech
        async with self._state_lock:
            td = self._config.turn_detection
            if td and td.interrupt_response and self._response_task and not self._response_task.done():
                self._barge_in_count += 1
                logger.info(f"[{self.session_id[:8]}] Barge-in #{self._barge_in_count}: cancelling active response (rms={rms:.0f})")
                self._response_task.cancel()

        self._speech_stopped_at = time.perf_counter()
        # Store VAD data for metrics
        self._response_metrics["speech_rms"] = round(rms, 1)
        await self._handle_speech_stopped(audio_end_ms, speech_audio, item_id)

    async def _handle_speech_stopped(
        self, audio_end_ms: int, speech_audio: bytes, item_id: str
    ) -> None:
        await self._emitter.emit(
            events.input_audio_buffer_speech_stopped("", audio_end_ms, item_id)
        )

        # Transcribe
        transcript = await self._transcribe_audio(speech_audio, item_id)

        if not transcript:
            audio_ms = len(speech_audio) / (INTERNAL_SAMPLE_RATE * SAMPLE_WIDTH) * 1000
            logger.warning(
                f"[{self.session_id[:8]}] ASR returned empty — skipping response. "
                f"speech_audio={len(speech_audio)} bytes ({audio_ms:.0f}ms)"
            )
            return

        # Create user item with transcription (state mutation under lock)
        item = ConversationItem(
            id=item_id,
            type="message",
            role="user",
            content=[ContentPart(type="input_audio", transcript=transcript)],
            status="completed",
        )
        async with self._state_lock:
            prev_id = self._items[-1].id if self._items else ""
            self._append_item(item)

        await self._emitter.emit(events.input_audio_buffer_committed("", prev_id, item_id))
        await self._emitter.emit(events.conversation_item_created("", prev_id, item))
        await self._emitter.emit(
            events.input_audio_transcription_completed("", item_id, 0, transcript)
        )

        # Auto-create response if configured
        td = self._config.turn_detection
        if td and td.create_response:
            await self._start_response()

    async def _transcribe_audio(self, speech_audio: bytes, item_id: str) -> str:
        """Transcribe audio using streaming or batch ASR."""
        t0 = time.perf_counter()
        transcript = ""
        asr_mode = "batch"

        if self._asr_stream_id == item_id and self._asr.supports_streaming:
            asr_mode = "streaming"
            transcript = await self._asr.finish_stream(item_id)
            async with self._state_lock:
                if self._asr_stream_id == item_id:
                    self._asr_stream_id = None
            asr_ms = (time.perf_counter() - t0) * 1000
            logger.info(
                f"[{self.session_id[:8]}] ASR (streaming): {asr_ms:.0f}ms → "
                f"\"{transcript[:80]}\" (empty={not transcript})"
            )

            # Fallback to batch if streaming returned empty
            if not transcript and speech_audio:
                asr_mode = "batch-fallback"
                logger.info(
                    f"[{self.session_id[:8]}] ASR streaming empty, falling back to batch "
                    f"({len(speech_audio)} bytes)"
                )
                t0 = time.perf_counter()
                transcript = await self._asr.transcribe(speech_audio)
                asr_ms = (time.perf_counter() - t0) * 1000
                logger.info(
                    f"[{self.session_id[:8]}] ASR (batch-fallback): {asr_ms:.0f}ms → "
                    f"\"{transcript[:80]}\" (empty={not transcript})"
                )
        else:
            if self._asr_stream_id == item_id:
                try:
                    await self._asr.finish_stream(item_id)
                except Exception as e:
                    logger.warning(f"[{self.session_id[:8]}] Error finishing ASR stream {item_id[:12]}: {e}")
                async with self._state_lock:
                    if self._asr_stream_id == item_id:
                        self._asr_stream_id = None
            transcript = await self._asr.transcribe(speech_audio)
            asr_ms = (time.perf_counter() - t0) * 1000
            logger.info(
                f"[{self.session_id[:8]}] ASR (batch): {asr_ms:.0f}ms → "
                f"\"{transcript[:80]}\" (empty={not transcript})"
            )

        # Store for per-response observability
        total_asr_ms = (time.perf_counter() - t0) * 1000
        speech_ms = len(speech_audio) / (INTERNAL_SAMPLE_RATE * SAMPLE_WIDTH) * 1000
        self._response_metrics["asr_ms"] = round(total_asr_ms, 1)
        self._response_metrics["speech_ms"] = round(speech_ms, 1)
        self._response_metrics["asr_mode"] = asr_mode
        self._response_metrics["input_chars"] = len(transcript)

        return transcript

    def _append_item(self, item: ConversationItem) -> None:
        """Append item to conversation and feed conversation memory.

        Must be called under _state_lock.
        """
        self._items.append(item)
        self._enforce_items_limit()

        # Feed conversation memory for recall_memory tool
        if item.type == "message" and item.role in ("user", "assistant"):
            text = ""
            for part in item.content:
                if part.type in ("input_text", "text") and part.text:
                    text += part.text + " "
                elif part.type in ("input_audio", "audio") and part.transcript:
                    text += part.transcript + " "
            if text.strip():
                self._memory.add(item.role, text.strip())

    def _enforce_items_limit(self) -> None:
        """Evict oldest items when conversation exceeds max size.

        Must be called under _state_lock.
        """
        while len(self._items) > _MAX_CONVERSATION_ITEMS:
            evicted = self._items.pop(0)
            logger.debug(
                f"[{self.session_id[:8]}] Evicted oldest item {evicted.id[:12]} "
                f"(items={len(self._items)})"
            )

    def _check_rate_limit(self) -> bool:
        """Check if client is sending events too fast. Returns True if allowed."""
        now = time.monotonic()
        # Remove timestamps older than 1 second
        cutoff = now - 1.0
        self._event_timestamps = [t for t in self._event_timestamps if t > cutoff]
        if len(self._event_timestamps) >= _MAX_EVENTS_PER_SECOND:
            return False
        self._event_timestamps.append(now)
        return True

    # ---- Session Loop ----

    async def run(self) -> None:
        """Main session loop: receive and dispatch client events."""
        # Send initial events
        await self._emitter.emit(
            events.session_created("", self.session_id, self._config)
        )
        await self._emitter.emit(
            events.conversation_created("", self.conversation_id)
        )

        try:
            while True:
                try:
                    message = await asyncio.wait_for(
                        self._ws.recv(), timeout=_SESSION_IDLE_TIMEOUT_S
                    )
                except asyncio.TimeoutError:
                    logger.warning(
                        f"[{self.session_id[:8]}] Session idle timeout "
                        f"({_SESSION_IDLE_TIMEOUT_S:.0f}s), disconnecting"
                    )
                    break

                try:
                    # Rate limiting
                    if not self._check_rate_limit():
                        await self._emitter.emit(
                            events.error_event(
                                "", "Rate limit exceeded. Slow down.",
                                code="rate_limit_exceeded",
                            )
                        )
                        continue

                    data = json.loads(message)
                    await self._dispatch(data)
                except json.JSONDecodeError:
                    await self._emitter.emit(
                        events.error_event("", "Invalid JSON", code="invalid_json")
                    )
                except SlowClientError:
                    logger.warning(f"[{self.session_id[:8]}] Session terminated: slow client")
                    break
                except Exception as e:
                    logger.error(f"[{self.session_id[:8]}] Handler error: {e}", exc_info=True)
                    await self._emitter.emit(
                        events.error_event("", str(e), error_type="server_error")
                    )
        except websockets.exceptions.ConnectionClosed:
            logger.info(f"[{self.session_id[:8]}] WebSocket connection closed")
        except SlowClientError:
            logger.warning(f"[{self.session_id[:8]}] Session terminated: slow client")
        except asyncio.CancelledError:
            logger.info(f"[{self.session_id[:8]}] Session cancelled")
        except Exception as e:
            logger.error(f"[{self.session_id[:8]}] Session loop error: {e}", exc_info=True)
        finally:
            await self._cleanup()

    async def _cleanup(self) -> None:
        if self._response_task and not self._response_task.done():
            self._response_task.cancel()
            try:
                await self._response_task
            except asyncio.CancelledError:
                pass
            except Exception as e:
                logger.warning(f"[{self.session_id[:8]}] Error during response task cleanup: {e}")
        # Cancel and await all tracked background tasks
        for task in list(self._background_tasks):
            if not task.done():
                task.cancel()
        if self._background_tasks:
            await asyncio.gather(*self._background_tasks, return_exceptions=True)
            self._background_tasks.clear()

        if self._asr_stream_id:
            try:
                await self._asr.finish_stream(self._asr_stream_id)
            except Exception as e:
                logger.warning(f"[{self.session_id[:8]}] Error finishing ASR stream on cleanup: {e}")
        logger.info(f"[{self.session_id[:8]}] Session cleaned up")

    async def _dispatch(self, data: dict) -> None:
        event_type = data.get("type", "")
        handler_name = self._HANDLER_MAP.get(event_type)
        if handler_name is None:
            await self._emitter.emit(
                events.error_event(
                    "", f"Unknown event type: {event_type}",
                    code="unknown_event",
                )
            )
            return
        handler = getattr(self, handler_name)
        await handler(data)

    # ---- Client Event Handlers ----

    async def _handle_session_update(self, data: dict) -> None:
        session_data = data.get("session", {})
        try:
            self._config.update(session_data)
        except SessionConfigValidationError as e:
            await self._emitter.emit(
                events.error_event("", str(e), code="invalid_session_config")
            )
            return
        self._setup_vad()
        logger.info(
            f"[{self.session_id[:8]}] session.updated: "
            f"modalities={self._config.modalities}, "
            f"input_format={self._config.input_audio_format}, "
            f"output_format={self._config.output_audio_format}, "
            f"vad={'server_vad' if self._vad else 'none'}"
        )
        await self._emitter.emit(
            events.session_updated("", self.session_id, self._config)
        )

    async def _handle_input_audio_buffer_append(self, data: dict) -> None:
        audio_b64 = data.get("audio", "")
        if not audio_b64:
            return

        pcm = decode_audio_from_client(audio_b64, self._config.input_audio_format)
        self._audio_append_count += 1
        self._audio_bytes_received += len(pcm)

        # In VAD mode, don't accumulate in _audio_buffer (VAD tracks its own speech_audio).
        # Only accumulate for manual commit mode.
        if not self._vad:
            if len(self._audio_buffer) + len(pcm) > _MAX_AUDIO_BUFFER_BYTES:
                logger.warning(
                    f"[{self.session_id[:8]}] Audio buffer full "
                    f"({len(self._audio_buffer)} bytes), rejecting append"
                )
                await self._emitter.emit(
                    events.error_event(
                        "", "Audio buffer full. Commit or clear before appending more.",
                        code="buffer_full",
                    )
                )
                return
            self._audio_buffer.extend(pcm)

        # Periodic metrics (every 5s)
        now = time.monotonic()
        if now - self._last_metrics_log >= 5.0:
            audio_duration_ms = self._audio_bytes_received / (INTERNAL_SAMPLE_RATE * SAMPLE_WIDTH) * 1000
            vad_status = "N/A"
            if self._vad:
                vad_status = f"speaking={self._vad.is_speaking}, total_audio={self._vad.total_audio_ms}ms"
            logger.info(
                f"[{self.session_id[:8]}] METRICS: "
                f"appends={self._audio_append_count}, "
                f"pcm_bytes={self._audio_bytes_received} ({audio_duration_ms:.0f}ms), "
                f"vad=[{vad_status}], "
                f"asr_stream={self._asr_stream_id is not None}, "
                f"items={len(self._items)}"
            )
            self._last_metrics_log = now

        # Feed to VAD
        if self._vad:
            self._vad.feed(pcm)

        # Feed to ASR streaming
        if self._asr_stream_id and self._asr.supports_streaming:
            await self._asr.feed_chunk(pcm, self._asr_stream_id)

    async def _handle_input_audio_buffer_commit(self, data: dict) -> None:
        if not self._audio_buffer:
            await self._emitter.emit(
                events.error_event("", "Audio buffer is empty", code="empty_buffer")
            )
            return

        item_id = f"item_{uuid.uuid4().hex[:24]}"
        audio_data = bytes(self._audio_buffer)
        self._audio_buffer.clear()

        if self._vad:
            self._vad.reset()

        # Transcribe
        transcript = await self._asr.transcribe(audio_data)

        item = ConversationItem(
            id=item_id,
            type="message",
            role="user",
            content=[ContentPart(type="input_audio", transcript=transcript)],
            status="completed",
        )
        async with self._state_lock:
            prev_id = self._items[-1].id if self._items else ""
            self._append_item(item)

        await self._emitter.emit(events.input_audio_buffer_committed("", prev_id, item_id))
        await self._emitter.emit(events.conversation_item_created("", prev_id, item))

        if transcript:
            await self._emitter.emit(
                events.input_audio_transcription_completed("", item_id, 0, transcript)
            )

    async def _handle_input_audio_buffer_clear(self, data: dict) -> None:
        self._audio_buffer.clear()
        if self._vad:
            self._vad.reset()
        await self._emitter.emit(events.input_audio_buffer_cleared(""))

    async def _handle_conversation_item_create(self, data: dict) -> None:
        item_data = data.get("item", {})
        if not item_data.get("id"):
            item_data["id"] = f"item_{uuid.uuid4().hex[:24]}"
        item = ConversationItem.from_dict(item_data)

        # Validate item from client input
        try:
            item.validate()
        except ConversationItemValidationError as e:
            await self._emitter.emit(
                events.error_event("", str(e), code="invalid_item")
            )
            return

        async with self._state_lock:
            prev_id = self._items[-1].id if self._items else ""
            self._append_item(item)
        await self._emitter.emit(events.conversation_item_created("", prev_id, item))

    async def _handle_conversation_item_delete(self, data: dict) -> None:
        item_id = data.get("item_id", "")
        async with self._state_lock:
            original_len = len(self._items)
            self._items = [i for i in self._items if i.id != item_id]
            found = len(self._items) < original_len
        if not found:
            await self._emitter.emit(
                events.error_event("", f"Item not found: {item_id}", code="item_not_found")
            )
            return
        await self._emitter.emit(events.conversation_item_deleted("", item_id))

    async def _handle_conversation_item_retrieve(self, data: dict) -> None:
        item_id = data.get("item_id", "")
        async with self._state_lock:
            found_item = None
            for item in self._items:
                if item.id == item_id:
                    found_item = item
                    break
        if found_item is not None:
            await self._emitter.emit(events.conversation_item_retrieved("", found_item))
        else:
            await self._emitter.emit(
                events.error_event("", f"Item not found: {item_id}", code="item_not_found")
            )

    async def _handle_conversation_item_truncate(self, data: dict) -> None:
        item_id = data.get("item_id", "")
        content_index = data.get("content_index", 0)
        audio_end_ms = data.get("audio_end_ms", 0)

        async with self._state_lock:
            found = False
            for item in self._items:
                if item.id == item_id:
                    found = True
                    # Truncate audio content
                    if content_index < len(item.content):
                        part = item.content[content_index]
                        if part.audio:
                            # Calculate bytes to keep (API_SAMPLE_RATE, 16-bit)
                            bytes_to_keep = int(audio_end_ms / 1000 * API_SAMPLE_RATE * SAMPLE_WIDTH)
                            audio_bytes = base64.b64decode(part.audio)
                            part.audio = base64.b64encode(audio_bytes[:bytes_to_keep]).decode("ascii")
                    break

        if found:
            await self._emitter.emit(
                events.conversation_item_truncated("", item_id, content_index, audio_end_ms)
            )
        else:
            await self._emitter.emit(
                events.error_event("", f"Item not found: {item_id}", code="item_not_found")
            )

    async def _handle_response_create(self, data: dict) -> None:
        await self._start_response()

    async def _handle_response_cancel(self, data: dict) -> None:
        task = None
        async with self._state_lock:
            if self._response_task and not self._response_task.done():
                self._response_task.cancel()
                task = self._response_task
        # Await outside lock to avoid holding lock during cancellation
        if task is not None:
            try:
                await task
            except asyncio.CancelledError:
                pass
            except Exception as e:
                logger.warning(f"[{self.session_id[:8]}] Error during response cancel: {e}")

    async def _handle_output_audio_buffer_clear(self, data: dict) -> None:
        # Client wants to clear output audio — cancel active response
        async with self._state_lock:
            if self._response_task and not self._response_task.done():
                self._response_task.cancel()

    # ---- Response Pipeline ----

    async def _start_response(self) -> None:
        # Cancel any existing response OUTSIDE the lock to avoid deadlock
        # (the response task itself acquires _state_lock internally)
        prev_task = None
        async with self._state_lock:
            if self._response_task and not self._response_task.done():
                self._response_task.cancel()
                prev_task = self._response_task

        if prev_task is not None:
            try:
                await prev_task
            except asyncio.CancelledError:
                pass
            except Exception as e:
                logger.warning(f"[{self.session_id[:8]}] Error cancelling previous response: {e}")

        async with self._state_lock:
            self._response_task = asyncio.create_task(self._run_response())

    async def _run_response(self) -> None:
        """Create a ResponseRunner and delegate the full response cycle."""
        response_id = f"resp_{uuid.uuid4().hex[:24]}"

        # Prepare prior metrics from speech/ASR phase
        self._turn_count += 1
        prior_metrics = dict(self._response_metrics)

        runner = ResponseRunner(
            session_id=self.session_id,
            emitter=self._emitter,
            llm=self._llm,
            tts=self._tts,
            config=self._config,
            tool_registry=self._tool_registry,
        )

        ctx = ResponseContext(
            items=self._items,
            state_lock=self._state_lock,
            append_item=self._append_item,
            speech_stopped_at=self._speech_stopped_at,
            turn_count=self._turn_count,
            session_start=self._session_start,
            barge_in_count=self._barge_in_count,
        )

        await runner.run(response_id, ctx, prior_metrics)

        # Store runner metrics for the next cycle
        self._response_metrics = runner.metrics

    # Handler dispatch map: event type → method name (avoids unbound method references)
    _HANDLER_MAP: dict[str, str] = {
        "session.update": "_handle_session_update",
        "input_audio_buffer.append": "_handle_input_audio_buffer_append",
        "input_audio_buffer.commit": "_handle_input_audio_buffer_commit",
        "input_audio_buffer.clear": "_handle_input_audio_buffer_clear",
        "conversation.item.create": "_handle_conversation_item_create",
        "conversation.item.delete": "_handle_conversation_item_delete",
        "conversation.item.retrieve": "_handle_conversation_item_retrieve",
        "conversation.item.truncate": "_handle_conversation_item_truncate",
        "response.create": "_handle_response_create",
        "response.cancel": "_handle_response_cancel",
        "output_audio_buffer.clear": "_handle_output_audio_buffer_clear",
    }
