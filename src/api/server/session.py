"""
RealtimeSession — per-connection protocol facade.

Thin layer that:
- Dispatches client events to handlers
- Manages connection lifecycle (auth, rate limit, idle timeout)
- Delegates conversation state to ConversationStore
- Delegates response execution to ResponseRunner

Does NOT contain: conversation logic, response generation, audio processing,
tool execution, or context building.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import time
import uuid
from typing import TYPE_CHECKING

import websockets.exceptions

from audio.codec import API_SAMPLE_RATE, INTERNAL_SAMPLE_RATE, SAMPLE_WIDTH, decode_audio_from_client
from config import LLM, STREAMING
from protocol import events
from protocol.contract import MAX_EVENTS_PER_SECOND, SESSION_IDLE_TIMEOUT_S
from protocol.event_emitter import EventEmitter, SlowClientError
from protocol.models import (
    ConversationItem,
    ConversationItemValidationError,
    ContentPart,
    SessionConfig,
    SessionConfigValidationError,
)
from server.audio_input import AudioInputCallbacks, AudioInputHandler
from server.conversation_store import ConversationStore
from server.response_runner import ResponseContext, ResponseRunner
from server.system_metrics import SYSTEM_METRICS

if TYPE_CHECKING:
    from providers.asr import ASRProvider
    from providers.llm import LLMProvider
    from providers.tts import TTSProvider
    from tools.registry import ToolRegistry
    from websockets.asyncio.server import ServerConnection

logger = logging.getLogger("open-voice-api.session")

# Maximum audio buffer size in manual commit mode (10 minutes at 8kHz 16-bit)
_MAX_AUDIO_BUFFER_BYTES = 10 * 60 * INTERNAL_SAMPLE_RATE * SAMPLE_WIDTH


class RealtimeSession:
    """Per-connection protocol facade implementing the OpenAI Realtime protocol.

    This class is intentionally thin — a facade that delegates to:
    - ConversationStore: items, memory, locking
    - AudioInputHandler: VAD, ASR, speech detection
    - ResponseRunner: LLM, tools, TTS, audio streaming
    """

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
            instructions=LLM.system_prompt,
            temperature=LLM.temperature,
        )
        self._store = ConversationStore()
        self._audio_buffer = bytearray()
        self._emitter = EventEmitter(ws, self.session_id)

        # Fork tool registry per session for recall_memory
        if self._tool_registry is not None:
            self._tool_registry = self._setup_session_tools(self._tool_registry)

        # Active response task
        self._response_task: asyncio.Task | None = None
        self._active_response_id: str | None = None

        # Early LLM trigger state (E2E streaming)
        self._early_trigger_active = False
        self._previous_partial: str = ""
        self._early_trigger_item_id: str | None = None

        # Audio input handler (VAD + ASR)
        self._audio_input = AudioInputHandler(
            asr=asr,
            config=self._config,
            callbacks=AudioInputCallbacks(
                cancel_active_response=self._cancel_active_response,
                append_user_item_and_respond=self._on_user_speech_complete,
                emit=self._emitter.emit,
                on_partial_transcript=(
                    self._on_partial_transcript
                    if STREAMING.enable_early_llm_trigger
                    else None
                ),
            ),
            emitter=self._emitter,
            session_id=self.session_id,
        )

        # Metrics
        self._audio_append_count = 0
        self._audio_bytes_received = 0
        self._last_metrics_log = time.monotonic()

        # Per-response observability metrics (reset at each response start)
        self._response_metrics: dict[str, object] = {}

        # Session-level counters for observability
        self._session_start = time.perf_counter()
        self._turn_count = 0

        # Rate limiting
        self._event_timestamps: list[float] = []

        SYSTEM_METRICS.total_sessions += 1

    def _setup_session_tools(self, registry: ToolRegistry) -> ToolRegistry:
        """Fork the global tool registry and bind per-session tools."""
        from tools.recall_memory import register_recall_handler
        forked = registry.fork()
        register_recall_handler(forked, self._store.memory)
        return forked

    # ---- Audio Input Callbacks ----

    async def _cancel_active_response(self) -> bool:
        """Barge-in: cancel active response if running. Returns True if cancelled."""
        async with self._store.lock:
            if self._response_task and not self._response_task.done():
                self._response_task.cancel()
                # Fence: invalidate response so late events are dropped
                if self._active_response_id:
                    self._emitter.invalidate_response(self._active_response_id)
                    self._active_response_id = None
                SYSTEM_METRICS.record_cancel()
                logger.info(
                    f"[{self.session_id[:8]}] Barge-in: cancelling active response"
                )
                return True
        return False

    async def _on_user_speech_complete(
        self, item: ConversationItem, transcript: str
    ) -> None:
        """Called when ASR transcription is ready: add item, emit events, auto-respond."""
        async with self._store.lock:
            prev_id = self._store.last_id()
            self._store.append(item)

        await self._emitter.emit(
            events.input_audio_buffer_committed("", prev_id, item.id)
        )
        await self._emitter.emit(
            events.conversation_item_created("", prev_id, item)
        )
        await self._emitter.emit(
            events.input_audio_transcription_completed("", item.id, 0, transcript)
        )

        # Auto-create response if configured
        td = self._config.turn_detection
        if td and td.create_response:
            # If early trigger already started a response, don't start another
            if not self._early_trigger_active:
                await self._start_response()
            else:
                # Update the in-progress item with final transcript
                async with self._store.lock:
                    existing = self._store.find(item.id)
                    if existing:
                        existing.content = item.content
                        existing.status = "completed"
                logger.info(
                    f"[{self.session_id[:8]}] Final transcript confirms early trigger: "
                    f'"{transcript[:60]}"'
                )
                self._early_trigger_active = False

    async def _on_partial_transcript(self, item_id: str, partial: str) -> None:
        """Handle partial ASR transcript — may trigger early LLM response.

        Called during speech when ASR emits a partial transcript.
        Triggers LLM when enough stable words are detected.
        """
        if self._early_trigger_active:
            return  # Already triggered

        if not partial.strip():
            return

        # Count stable words (present in both current and previous partial)
        stable_count = self._count_stable_words(partial, self._previous_partial)
        self._previous_partial = partial

        if stable_count < STREAMING.min_stable_words:
            return

        # Trigger early LLM response
        logger.info(
            f"[{self.session_id[:8]}] Early LLM trigger: {stable_count} stable words: "
            f'"{partial[:60]}"'
        )
        self._early_trigger_active = True
        self._early_trigger_item_id = item_id

        # Create in-progress conversation item
        item = ConversationItem(
            id=item_id,
            type="message",
            role="user",
            content=[ContentPart(type="input_audio", transcript=partial)],
            status="in_progress",
        )

        async with self._store.lock:
            prev_id = self._store.last_id()
            self._store.append(item)

        await self._emitter.emit(
            events.input_audio_buffer_committed("", prev_id, item.id)
        )
        await self._emitter.emit(
            events.conversation_item_created("", prev_id, item)
        )

        self._response_metrics["early_trigger_words"] = stable_count
        self._response_metrics["early_trigger_partial"] = partial[:80]
        await self._start_response()

    @staticmethod
    def _count_stable_words(current: str, previous: str) -> int:
        """Count words that are stable between two consecutive partials.

        A word is stable if it appears at the same position in both partials.
        Returns the count of matching prefix words.
        """
        if not previous:
            return 0

        current_words = current.strip().lower().split()
        previous_words = previous.strip().lower().split()

        stable = 0
        for cw, pw in zip(current_words, previous_words):
            if cw == pw:
                stable += 1
            else:
                break

        return stable

    def _check_rate_limit(self) -> bool:
        """Check if client is sending events too fast. Returns True if allowed."""
        now = time.monotonic()
        cutoff = now - 1.0
        self._event_timestamps = [t for t in self._event_timestamps if t > cutoff]
        if len(self._event_timestamps) >= MAX_EVENTS_PER_SECOND:
            return False
        self._event_timestamps.append(now)
        return True

    # ---- Session Loop ----

    async def run(self) -> None:
        """Main session loop: receive and dispatch client events."""
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
                        self._ws.recv(), timeout=SESSION_IDLE_TIMEOUT_S
                    )
                except asyncio.TimeoutError:
                    logger.warning(
                        f"[{self.session_id[:8]}] Session idle timeout "
                        f"({SESSION_IDLE_TIMEOUT_S:.0f}s), disconnecting"
                    )
                    break

                try:
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

        await self._audio_input.cleanup()
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
        self._audio_input.setup_vad()
        logger.info(
            f"[{self.session_id[:8]}] session.updated: "
            f"modalities={self._config.modalities}, "
            f"input_format={self._config.input_audio_format}, "
            f"output_format={self._config.output_audio_format}, "
            f"vad={'server_vad' if self._audio_input.vad else 'none'}"
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

        if not self._audio_input.vad:
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
            if self._audio_input.vad:
                vad_status = (
                    f"speaking={self._audio_input.vad.is_speaking}, "
                    f"total_audio={self._audio_input.vad.total_audio_ms}ms"
                )
            logger.info(
                f"[{self.session_id[:8]}] METRICS: "
                f"appends={self._audio_append_count}, "
                f"pcm_bytes={self._audio_bytes_received} ({audio_duration_ms:.0f}ms), "
                f"vad=[{vad_status}], "
                f"asr_stream={self._audio_input._asr_stream_id is not None}, "
                f"items={len(self._store.items)}"
            )
            self._last_metrics_log = now

        self._audio_input.feed_audio(pcm)
        await self._audio_input.feed_asr_chunk(pcm)

    async def _handle_input_audio_buffer_commit(self, data: dict) -> None:
        if not self._audio_buffer:
            await self._emitter.emit(
                events.error_event("", "Audio buffer is empty", code="empty_buffer")
            )
            return

        item_id = f"item_{uuid.uuid4().hex[:24]}"
        audio_data = bytes(self._audio_buffer)
        self._audio_buffer.clear()

        if self._audio_input.vad:
            self._audio_input.vad.reset()

        transcript = await self._asr.transcribe(audio_data)

        item = ConversationItem(
            id=item_id,
            type="message",
            role="user",
            content=[ContentPart(type="input_audio", transcript=transcript)],
            status="completed",
        )
        async with self._store.lock:
            prev_id = self._store.last_id()
            self._store.append(item)

        await self._emitter.emit(events.input_audio_buffer_committed("", prev_id, item_id))
        await self._emitter.emit(events.conversation_item_created("", prev_id, item))

        if transcript:
            await self._emitter.emit(
                events.input_audio_transcription_completed("", item_id, 0, transcript)
            )

    async def _handle_input_audio_buffer_clear(self, data: dict) -> None:
        self._audio_buffer.clear()
        if self._audio_input.vad:
            self._audio_input.vad.reset()
        await self._emitter.emit(events.input_audio_buffer_cleared(""))

    async def _handle_conversation_item_create(self, data: dict) -> None:
        item_data = data.get("item", {})
        if not item_data.get("id"):
            item_data["id"] = f"item_{uuid.uuid4().hex[:24]}"
        item = ConversationItem.from_dict(item_data)

        try:
            item.validate()
        except ConversationItemValidationError as e:
            await self._emitter.emit(
                events.error_event("", str(e), code="invalid_item")
            )
            return

        async with self._store.lock:
            prev_id = self._store.last_id()
            self._store.append(item)
        await self._emitter.emit(events.conversation_item_created("", prev_id, item))

    async def _handle_conversation_item_delete(self, data: dict) -> None:
        item_id = data.get("item_id", "")
        async with self._store.lock:
            found = self._store.delete(item_id)
        if not found:
            await self._emitter.emit(
                events.error_event("", f"Item not found: {item_id}", code="item_not_found")
            )
            return
        await self._emitter.emit(events.conversation_item_deleted("", item_id))

    async def _handle_conversation_item_retrieve(self, data: dict) -> None:
        item_id = data.get("item_id", "")
        async with self._store.lock:
            found_item = self._store.find(item_id)
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

        async with self._store.lock:
            item = self._store.find(item_id)
            if item and content_index < len(item.content):
                part = item.content[content_index]
                if part.audio:
                    bytes_to_keep = int(audio_end_ms / 1000 * API_SAMPLE_RATE * SAMPLE_WIDTH)
                    audio_bytes = base64.b64decode(part.audio)
                    part.audio = base64.b64encode(audio_bytes[:bytes_to_keep]).decode("ascii")

        if item:
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
        async with self._store.lock:
            if self._response_task and not self._response_task.done():
                self._response_task.cancel()
                if self._active_response_id:
                    self._emitter.invalidate_response(self._active_response_id)
                    self._active_response_id = None
                task = self._response_task
        if task is not None:
            try:
                await task
            except asyncio.CancelledError:
                pass
            except Exception as e:
                logger.warning(f"[{self.session_id[:8]}] Error during response cancel: {e}")

    async def _handle_output_audio_buffer_clear(self, data: dict) -> None:
        async with self._store.lock:
            if self._response_task and not self._response_task.done():
                self._response_task.cancel()
                if self._active_response_id:
                    self._emitter.invalidate_response(self._active_response_id)
                    self._active_response_id = None

    # ---- Response Pipeline ----

    async def _start_response(self) -> None:
        prev_task = None
        async with self._store.lock:
            if self._response_task and not self._response_task.done():
                self._response_task.cancel()
                # Fence: invalidate previous response so late events are dropped
                if self._active_response_id:
                    self._emitter.invalidate_response(self._active_response_id)
                    self._active_response_id = None
                prev_task = self._response_task

        if prev_task is not None:
            try:
                await prev_task
            except asyncio.CancelledError:
                pass
            except Exception as e:
                logger.warning(f"[{self.session_id[:8]}] Error cancelling previous response: {e}")

        async with self._store.lock:
            self._response_task = asyncio.create_task(self._run_response())

    async def _run_response(self) -> None:
        """Create a ResponseRunner and delegate the full response cycle."""
        response_id = f"resp_{uuid.uuid4().hex[:24]}"

        # Register fence so late events from previous responses are dropped
        self._active_response_id = response_id
        self._emitter.set_active_response(response_id)

        self._turn_count += 1
        SYSTEM_METRICS.record_response()
        prior_metrics = dict(self._audio_input.response_metrics)

        runner = ResponseRunner(
            session_id=self.session_id,
            emitter=self._emitter,
            llm=self._llm,
            tts=self._tts,
            config=self._config,
            tool_registry=self._tool_registry,
        )

        ctx = ResponseContext(
            items=self._store.items,
            state_lock=self._store.lock,
            append_item=self._store.append,
            speech_stopped_at=self._audio_input.speech_stopped_at,
            turn_count=self._turn_count,
            session_start=self._session_start,
            barge_in_count=self._audio_input.barge_in_count,
        )

        await runner.run(response_id, ctx, prior_metrics)

        self._audio_input.response_metrics = runner.metrics

    # Handler dispatch map
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
