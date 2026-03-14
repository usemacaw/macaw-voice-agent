"""
RealtimeSession — per-connection state machine.

Manages session config, conversation items, audio buffer, VAD,
and the response pipeline for a single WebSocket connection.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import time
import uuid
from typing import TYPE_CHECKING

import re

import numpy as np
import websockets.exceptions

from audio.codec import API_SAMPLE_RATE, INTERNAL_SAMPLE_RATE, SAMPLE_WIDTH, decode_audio_from_client, encode_audio_for_client
from audio.vad import VADProcessor
from config import LLM_CONFIG, VAD_CONFIG, PIPELINE_CONFIG
from pipeline.conversation import items_to_messages, items_to_windowed_messages
from pipeline.sentence_pipeline import SentencePipeline
from protocol import events
from protocol.event_emitter import EventEmitter, SlowClientError
from protocol.models import (
    ConversationItem,
    ConversationItemValidationError,
    ContentPart,
    SessionConfig,
    SessionConfigValidationError,
    TurnDetection,
)

from tools.recall_memory import ConversationMemory

if TYPE_CHECKING:
    from providers.asr import ASRProvider
    from providers.llm import LLMProvider
    from providers.tts import TTSProvider
    from tools.registry import ToolRegistry
    from websockets.asyncio.server import ServerConnection

logger = logging.getLogger("open-voice-api.session")

# Strip <think>...</think> blocks from "thinking" models (e.g. Qwen3)
_THINK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)


def _strip_think(text: str) -> str:
    """Remove <think>...</think> reasoning blocks from LLM output."""
    return _THINK_RE.sub("", text).strip()


import random

# Filler templates: (prefix_with_query, prefix_without_query)
# {q} is replaced by the search query when available.
_SEARCH_FILLERS = [
    ("Vou pesquisar sobre {q}, aguarde.", "Vou pesquisar, aguarde."),
    ("Deixa eu buscar sobre {q}.", "Deixa eu buscar isso pra você."),
    ("Vou verificar sobre {q}, um momento.", "Vou verificar, um momento."),
    ("Vou buscar informações sobre {q}.", "Vou buscar informações pra você."),
    ("Um momento, vou procurar sobre {q}.", "Um momento, vou procurar."),
    ("Espere um pouco, vou pesquisar sobre {q}.", "Espere um pouco, vou pesquisar."),
    ("Aguarde, vou buscar sobre {q}.", "Aguarde, vou buscar."),
]

_MEMORY_FILLERS = [
    "Deixa eu verificar, um momento.",
    "Vou checar, aguarde.",
    "Um momento, vou lembrar.",
]

_GENERIC_FILLERS = [
    "Um momento, por favor.",
    "Aguarde um instante.",
    "Só um momento.",
]


def _build_dynamic_filler(tool_name: str, arguments_json: str) -> str:
    """Build a contextual filler phrase based on tool name and arguments.

    Uses randomized templates so the voice assistant doesn't repeat
    the same phrase every time, sounding more natural.
    """
    try:
        args = json.loads(arguments_json) if arguments_json else {}
    except (json.JSONDecodeError, TypeError):
        args = {}

    if tool_name == "web_search":
        query = args.get("query", "")
        with_q, without_q = random.choice(_SEARCH_FILLERS)
        if query:
            short = query[:60].rstrip()
            return with_q.format(q=short)
        return without_q

    if tool_name == "recall_memory":
        return random.choice(_MEMORY_FILLERS)

    return random.choice(_GENERIC_FILLERS)

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
                logger.info(f"[{self.session_id[:8]}] Barge-in: cancelling active response (rms={rms:.0f})")
                self._response_task.cancel()

        self._speech_stopped_at = time.perf_counter()
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

        if self._asr_stream_id == item_id and self._asr.supports_streaming:
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
        response_id = f"resp_{uuid.uuid4().hex[:24]}"
        item_id = f"item_{uuid.uuid4().hex[:24]}"
        output_index = 0
        content_index = 0
        response_start = time.perf_counter()
        has_audio = "audio" in self._config.modalities
        has_tools = bool(self._config.tools) or (
            self._tool_registry is not None and self._tool_registry.has_server_tools
        )
        assistant_item = None

        logger.info(
            f"[{self.session_id[:8]}] RESPONSE START: "
            f"items={len(self._items)}, has_audio={has_audio}, has_tools={has_tools}"
        )

        try:
            # Build messages from conversation history (windowed to reduce context)
            async with self._state_lock:
                if has_tools:
                    messages = items_to_windowed_messages(list(self._items), window=8)
                else:
                    messages = items_to_messages(list(self._items))
            system = self._config.instructions
            temperature = self._config.temperature
            max_tokens = (
                self._config.max_response_output_tokens
                if isinstance(self._config.max_response_output_tokens, int)
                else LLM_CONFIG["max_tokens"]
            )

            if has_tools:
                await self._run_response_with_tools(
                    response_id, messages, system, temperature, max_tokens, has_audio,
                )
            else:
                assistant_item = await self._setup_response(
                    response_id, item_id, output_index, content_index, has_audio
                )

                if has_audio:
                    full_transcript = await self._run_audio_response(
                        response_id, item_id, output_index, content_index,
                        assistant_item, messages, system, temperature, max_tokens,
                    )
                    full_text = ""
                else:
                    full_text = await self._run_text_response(
                        response_id, item_id, output_index, content_index,
                        assistant_item, messages, system, temperature, max_tokens,
                    )
                    full_transcript = ""

                await self._finalize_response(
                    response_id, output_index, content_index,
                    assistant_item, response_start, full_transcript, full_text,
                )

        except asyncio.CancelledError:
            if assistant_item is not None:
                assistant_item.status = "incomplete"
            try:
                await self._emitter.emit(
                    events.response_done("", response_id, status="cancelled")
                )
            except Exception:
                pass
            raise

        except SlowClientError:
            raise

        except Exception as e:
            logger.error(f"[{self.session_id[:8]}] Response error: {e}", exc_info=True)
            await self._emitter.emit(
                events.response_done("", response_id, status="failed")
            )

    async def _run_response_with_tools(
        self,
        response_id: str,
        messages: list[dict],
        system: str,
        temperature: float,
        max_tokens: int,
        has_audio: bool,
    ) -> None:
        """Run response with function calling support.

        Two modes:
        - Server-side execution (tool_registry has handlers): executes tools
          automatically, sends filler TTS, re-calls LLM with results.
        - Client-side execution (no tool_registry): emits function_call events
          for the client to execute (OpenAI Realtime API compatible).
        """
        from providers.llm import LLMStreamEvent

        server_side = (
            self._tool_registry is not None
            and self._tool_registry.has_server_tools
        )

        await self._emitter.emit(events.response_created("", response_id))

        response_start = time.perf_counter()
        output_index = 0
        max_rounds = (
            self._tool_registry.max_rounds if self._tool_registry else 5
        )

        # Merge tool schemas: session config tools + server-side tool schemas
        tools = list(self._config.tools) if self._config.tools else []
        if server_side:
            server_schemas = self._tool_registry.get_schemas()
            existing_names = {
                t.get("function", {}).get("name") for t in tools if isinstance(t, dict)
            }
            for schema in server_schemas:
                name = schema.get("function", {}).get("name", "")
                if name not in existing_names:
                    tools.append(schema)

        # max_rounds = max tool executions. After tools run, one final LLM
        # call without tools generates the text/audio response.
        # Total LLM calls = max_rounds (with tools) + 1 (final, no tools).
        tools_used = 0
        tool_round = 0
        for tool_round in range(max_rounds + 1):
            # After max tool executions, force text-only response with fewer tokens
            allow_tools = tools if tools_used < max_rounds else None
            round_max_tokens = max_tokens if allow_tools else min(max_tokens, 40)

            # --- Final round (no tools) + audio: use pipelined LLM→TTS ---
            if not allow_tools and has_audio:
                logger.info(
                    f"[{self.session_id[:8]}] LLM call round {tool_round}: "
                    f"msgs={len(messages)}, tools=no (pipelined), "
                    f"messages={[m.get('role','?')+':'+str(m.get('content') or '')[:40] for m in messages[-4:]]}"
                )
                await self._emit_tool_response_audio_streamed(
                    response_id, output_index, response_start,
                    messages, system, temperature, round_max_tokens,
                )
                break

            logger.info(
                f"[{self.session_id[:8]}] LLM call round {tool_round}: "
                f"msgs={len(messages)}, tools={'yes' if allow_tools else 'no'}, "
                f"messages={[m.get('role','?')+':'+str(m.get('content') or '')[:40] for m in messages[-4:]]}"
            )

            # Collect full LLM stream: text + tool calls
            collected_text = ""
            collected_tool_calls: list[dict] = []  # {id, name, arguments}

            current_tool_call_id = ""
            current_tool_name = ""
            tool_arguments_buffer = ""
            in_tool_call = False

            async for event in self._llm.generate_stream_with_tools(
                messages, system=system,
                tools=allow_tools or None,
                temperature=temperature,
                max_tokens=round_max_tokens,
            ):
                if event.type == "text_delta":
                    collected_text += event.text

                elif event.type == "tool_call_start":
                    current_tool_call_id = event.tool_call_id
                    current_tool_name = event.tool_name
                    tool_arguments_buffer = ""
                    in_tool_call = True

                elif event.type == "tool_call_delta" and in_tool_call:
                    tool_arguments_buffer += event.tool_arguments_delta

                elif event.type == "tool_call_end" and in_tool_call:
                    collected_tool_calls.append({
                        "id": current_tool_call_id,
                        "name": current_tool_name,
                        "arguments": tool_arguments_buffer,
                    })
                    in_tool_call = False

            # --- No tool calls: emit final response (text or audio) ---
            collected_text = _strip_think(collected_text)
            if not collected_tool_calls:
                if collected_text:
                    logger.info(
                        f"[{self.session_id[:8]}] LLM text (round {tool_round}): "
                        f"{collected_text[:200]!r}"
                    )
                    if has_audio:
                        await self._emit_tool_response_audio(
                            response_id, output_index, response_start,
                            collected_text,
                        )
                    else:
                        await self._emit_tool_response_text(
                            response_id, output_index, collected_text,
                        )
                else:
                    # No text and no tool calls — empty response
                    logger.warning(
                        f"[{self.session_id[:8]}] Tool round {tool_round}: "
                        "no text and no tool calls"
                    )
                break

            # --- Has tool calls ---
            logger.info(
                f"[{self.session_id[:8]}] Tool round {tool_round}: "
                f"{len(collected_tool_calls)} tool call(s): "
                f"{[tc['name'] for tc in collected_tool_calls]}"
            )

            if server_side:
                # Server-side execution: filler + execute + re-call LLM
                all_ok = await self._execute_tools_server_side(
                    response_id, output_index, collected_tool_calls, has_audio,
                )
                output_index += len(collected_tool_calls)
                tools_used += 1

                # Rebuild messages with tool results for next round (windowed)
                async with self._state_lock:
                    messages = items_to_windowed_messages(list(self._items), window=8)

                # If any tool returned an error, exhaust rounds so next
                # LLM call is forced to respond in text
                if not all_ok:
                    logger.info(
                        f"[{self.session_id[:8]}] Tool error detected, "
                        "forcing text response on next call"
                    )
                    tools_used = max_rounds
            else:
                # Client-side execution: emit events and stop
                await self._emit_tool_calls_for_client(
                    response_id, output_index, collected_tool_calls,
                )
                break
        else:
            logger.warning(
                f"[{self.session_id[:8]}] Tool execution exceeded max rounds ({max_rounds})"
            )

        response_ms = (time.perf_counter() - response_start) * 1000
        logger.info(
            f"[{self.session_id[:8]}] RESPONSE DONE (tools): {response_ms:.0f}ms, "
            f"rounds={tool_round + 1}"
        )
        await self._emitter.emit(
            events.response_done("", response_id, status="completed")
        )

    async def _execute_tools_server_side(
        self,
        response_id: str,
        output_index: int,
        tool_calls: list[dict],
        has_audio: bool,
    ) -> bool:
        """Execute tool calls server-side with filler TTS.

        For each tool call:
        1. Emit function_call events (for observability)
        2. Send filler audio if in audio mode
        3. Execute tool via registry
        4. Add function_call + function_call_output items to conversation

        Returns:
            True if all tools succeeded, False if any returned an error.
        """
        any_error = False
        # Send filler audio for the first tool call
        if has_audio and tool_calls:
            first_tool = tool_calls[0]
            filler = _build_dynamic_filler(first_tool["name"], first_tool["arguments"])
            await self._send_filler_audio(response_id, output_index, filler)

        for tc in tool_calls:
            tc_id = tc["id"] or f"call_{uuid.uuid4().hex[:12]}"
            tc_name = tc["name"]
            tc_args = tc["arguments"]

            # Create function_call item
            fc_item_id = f"item_{uuid.uuid4().hex[:24]}"
            fc_item = ConversationItem(
                id=fc_item_id,
                type="function_call",
                status="completed",
                call_id=tc_id,
                name=tc_name,
                arguments=tc_args,
            )
            await self._emitter.emit(
                events.response_output_item_added(
                    "", response_id, output_index, fc_item
                )
            )
            async with self._state_lock:
                self._append_item(fc_item)

            await self._emitter.emit(
                events.response_function_call_arguments_done(
                    "", response_id, fc_item_id, output_index, tc_id, tc_args,
                )
            )
            await self._emitter.emit(
                events.response_output_item_done(
                    "", response_id, output_index, fc_item
                )
            )

            # Execute tool
            result_json = await self._tool_registry.execute(tc_name, tc_args)
            logger.info(
                f"[{self.session_id[:8]}] Tool '{tc_name}' result: "
                f"{result_json[:200]}"
            )

            # Check if tool returned an error
            try:
                result_data = json.loads(result_json)
                if isinstance(result_data, dict) and "error" in result_data:
                    any_error = True
            except (json.JSONDecodeError, TypeError):
                pass

            # Create function_call_output item
            fco_item_id = f"item_{uuid.uuid4().hex[:24]}"
            fco_item = ConversationItem(
                id=fco_item_id,
                type="function_call_output",
                status="completed",
                call_id=tc_id,
                output=result_json,
            )
            async with self._state_lock:
                self._append_item(fco_item)
            await self._emitter.emit(
                events.conversation_item_created("", fc_item_id, fco_item)
            )

            output_index += 1

        return not any_error

    async def _send_filler_audio(
        self, response_id: str, output_index: int, filler_text: str
    ) -> None:
        """Synthesize and send a filler phrase via TTS while tools execute."""
        try:
            filler_item_id = f"item_{uuid.uuid4().hex[:24]}"
            filler_item = ConversationItem(
                id=filler_item_id,
                type="message",
                role="assistant",
                status="in_progress",
                content=[ContentPart(type="audio", audio="", transcript="")],
            )
            await self._emitter.emit(
                events.response_output_item_added(
                    "", response_id, output_index, filler_item
                )
            )
            # NOTE: filler is NOT added to self._items on purpose.
            # Storing it would pollute the LLM context and teach the
            # model to mimic filler text instead of calling tools.

            # Synthesize filler
            if self._tts.supports_streaming:
                audio_chunks = []
                async for chunk in self._tts.synthesize_stream(filler_text):
                    if chunk:
                        audio_chunks.append(chunk)
                        audio_b64 = encode_audio_for_client(
                            chunk, self._config.output_audio_format
                        )
                        await self._emitter.emit(
                            events.response_audio_delta(
                                "", response_id, filler_item_id, output_index, 0,
                                audio_b64,
                            )
                        )
            else:
                audio = await self._tts.synthesize(filler_text)
                if audio:
                    audio_b64 = encode_audio_for_client(
                        audio, self._config.output_audio_format
                    )
                    await self._emitter.emit(
                        events.response_audio_delta(
                            "", response_id, filler_item_id, output_index, 0,
                            audio_b64,
                        )
                    )

            # Emit filler transcript
            await self._emitter.emit(
                events.response_audio_transcript_delta(
                    "", response_id, filler_item_id, output_index, 0, filler_text,
                )
            )

            # Finalize filler item
            filler_item.content[0] = ContentPart(
                type="audio", transcript=filler_text
            )
            filler_item.status = "completed"
            await self._emitter.emit(
                events.response_audio_done(
                    "", response_id, filler_item_id, output_index, 0
                )
            )
            await self._emitter.emit(
                events.response_output_item_done(
                    "", response_id, output_index, filler_item
                )
            )
            logger.info(
                f"[{self.session_id[:8]}] Filler sent: \"{filler_text}\""
            )
        except Exception as e:
            logger.warning(
                f"[{self.session_id[:8]}] Filler TTS failed (non-critical): {e}"
            )

    async def _emit_tool_response_audio(
        self,
        response_id: str,
        output_index: int,
        response_start: float,
        collected_text: str,
    ) -> None:
        """Synthesize already-collected text through TTS (no extra LLM call)."""
        item_id = f"item_{uuid.uuid4().hex[:24]}"
        content_index = 0

        assistant_item = ConversationItem(
            id=item_id,
            type="message",
            role="assistant",
            status="in_progress",
            content=[ContentPart(type="audio", audio="", transcript="")],
        )
        await self._emitter.emit(
            events.response_output_item_added(
                "", response_id, output_index, assistant_item
            )
        )
        async with self._state_lock:
            self._append_item(assistant_item)

        full_transcript = collected_text.strip()
        first_audio_sent = False

        # Synthesize text directly through TTS — no LLM call needed
        logger.info(
            f"[{self.session_id[:8]}] TTS direct synth: {full_transcript[:100]!r}"
        )
        if self._tts.supports_streaming:
            async for audio_chunk in self._tts.synthesize_stream(full_transcript):
                if not audio_chunk:
                    continue
                audio_b64 = encode_audio_for_client(
                    audio_chunk, self._config.output_audio_format
                )
                await self._emitter.emit(
                    events.response_audio_delta(
                        "", response_id, item_id, output_index, content_index,
                        audio_b64,
                    )
                )
                if not first_audio_sent:
                    first_audio_sent = True
                    if self._speech_stopped_at is not None:
                        e2e_ms = (
                            time.perf_counter() - self._speech_stopped_at
                        ) * 1000
                        logger.info(
                            f"[{self.session_id[:8]}] E2E LATENCY: {e2e_ms:.0f}ms "
                            f"(speech_stopped → first_audio_sent)"
                        )
        else:
            audio = await self._tts.synthesize(full_transcript)
            if audio:
                audio_b64 = encode_audio_for_client(
                    audio, self._config.output_audio_format
                )
                await self._emitter.emit(
                    events.response_audio_delta(
                        "", response_id, item_id, output_index, content_index,
                        audio_b64,
                    )
                )
                if self._speech_stopped_at is not None:
                    e2e_ms = (
                        time.perf_counter() - self._speech_stopped_at
                    ) * 1000
                    logger.info(
                        f"[{self.session_id[:8]}] E2E LATENCY: {e2e_ms:.0f}ms "
                        f"(speech_stopped → first_audio_sent)"
                    )

        # Emit transcript delta
        if full_transcript:
            await self._emitter.emit(
                events.response_audio_transcript_delta(
                    "", response_id, item_id, output_index, content_index,
                    full_transcript,
                )
            )

        # Update assistant item
        assistant_item.content[content_index] = ContentPart(
            type="audio", transcript=full_transcript
        )

        # Audio done + transcript done
        await self._emitter.emit(
            events.response_audio_done(
                "", response_id, item_id, output_index, content_index
            )
        )
        await self._emitter.emit(
            events.response_audio_transcript_done(
                "", response_id, item_id, output_index, content_index,
                full_transcript,
            )
        )

        await self._emitter.emit(
            events.response_content_part_done(
                "", response_id, item_id, output_index, content_index,
                assistant_item.content[content_index],
            )
        )
        assistant_item.status = "completed"
        await self._emitter.emit(
            events.response_output_item_done(
                "", response_id, output_index, assistant_item
            )
        )

    async def _emit_tool_response_audio_streamed(
        self,
        response_id: str,
        output_index: int,
        response_start: float,
        messages: list[dict],
        system: str,
        temperature: float,
        max_tokens: int,
    ) -> None:
        """Final tool round: pipelined LLM→TTS via SentencePipeline.

        Instead of collecting all LLM text first and then synthesizing,
        this streams LLM output sentence-by-sentence into TTS, so the
        first audio chunk is sent as soon as the first sentence completes.
        """
        item_id = f"item_{uuid.uuid4().hex[:24]}"
        content_index = 0

        assistant_item = ConversationItem(
            id=item_id,
            type="message",
            role="assistant",
            status="in_progress",
            content=[ContentPart(type="audio", audio="", transcript="")],
        )
        await self._emitter.emit(
            events.response_output_item_added(
                "", response_id, output_index, assistant_item
            )
        )
        async with self._state_lock:
            self._append_item(assistant_item)

        # Run pipelined LLM→TTS (no tools for the final round)
        full_transcript = await self._run_audio_response(
            response_id, item_id, output_index, content_index,
            assistant_item, messages, system, temperature, max_tokens,
            tools=None,
        )

        # Emit completion events
        await self._emitter.emit(
            events.response_content_part_done(
                "", response_id, item_id, output_index, content_index,
                assistant_item.content[content_index],
            )
        )
        assistant_item.status = "completed"
        await self._emitter.emit(
            events.response_output_item_done(
                "", response_id, output_index, assistant_item
            )
        )

    async def _emit_tool_response_text(
        self,
        response_id: str,
        output_index: int,
        text: str,
    ) -> None:
        """Emit collected text as a text response item."""
        item_id = f"item_{uuid.uuid4().hex[:24]}"
        text_item = ConversationItem(
            id=item_id,
            type="message",
            role="assistant",
            status="completed",
            content=[ContentPart(type="text", text=text)],
        )
        await self._emitter.emit(
            events.response_output_item_added(
                "", response_id, output_index, text_item
            )
        )
        async with self._state_lock:
            self._append_item(text_item)

        await self._emitter.emit(
            events.response_text_done(
                "", response_id, item_id, output_index, 0, text
            )
        )
        await self._emitter.emit(
            events.response_output_item_done(
                "", response_id, output_index, text_item
            )
        )

    async def _emit_tool_calls_for_client(
        self,
        response_id: str,
        output_index: int,
        tool_calls: list[dict],
    ) -> None:
        """Emit tool call events for client-side execution (OpenAI Realtime API compat)."""
        for tc in tool_calls:
            tc_id = tc["id"] or f"call_{uuid.uuid4().hex[:12]}"
            fc_item_id = f"item_{uuid.uuid4().hex[:24]}"
            fc_item = ConversationItem(
                id=fc_item_id,
                type="function_call",
                status="completed",
                call_id=tc_id,
                name=tc["name"],
                arguments=tc["arguments"],
            )
            await self._emitter.emit(
                events.response_output_item_added(
                    "", response_id, output_index, fc_item
                )
            )
            async with self._state_lock:
                self._append_item(fc_item)

            await self._emitter.emit(
                events.response_function_call_arguments_done(
                    "", response_id, fc_item_id, output_index,
                    tc_id, tc["arguments"],
                )
            )
            await self._emitter.emit(
                events.response_output_item_done(
                    "", response_id, output_index, fc_item
                )
            )
            output_index += 1

    async def _setup_response(
        self,
        response_id: str,
        item_id: str,
        output_index: int,
        content_index: int,
        has_audio: bool,
    ) -> ConversationItem:
        """Emit initial response events and create the assistant item."""
        await self._emitter.emit(events.response_created("", response_id))

        assistant_item = ConversationItem(
            id=item_id,
            type="message",
            role="assistant",
            status="in_progress",
        )

        if has_audio:
            assistant_item.content = [
                ContentPart(type="audio", audio="", transcript=""),
            ]
        else:
            assistant_item.content = [ContentPart(type="text", text="")]

        await self._emitter.emit(
            events.response_output_item_added("", response_id, output_index, assistant_item)
        )
        async with self._state_lock:
            prev_id = self._items[-1].id if self._items else ""
            self._append_item(assistant_item)
        await self._emitter.emit(
            events.conversation_item_created("", prev_id, assistant_item)
        )

        part = assistant_item.content[content_index]
        await self._emitter.emit(
            events.response_content_part_added(
                "", response_id, item_id, output_index, content_index, part
            )
        )

        return assistant_item

    async def _run_audio_response(
        self,
        response_id: str,
        item_id: str,
        output_index: int,
        content_index: int,
        assistant_item: ConversationItem,
        messages: list[dict],
        system: str,
        temperature: float,
        max_tokens: int,
        tools: list[dict] | None = ...,
    ) -> str:
        """Run LLM→TTS pipeline, stream audio events. Returns full transcript.

        Args:
            tools: Tool schemas to pass to LLM. Use ``...`` (default) to use
                   session config tools; pass ``None`` explicitly to disable.
        """
        if tools is ...:
            tools = self._config.tools or None

        full_transcript = ""
        first_audio_sent = False

        pipeline = SentencePipeline(self._llm, self._tts)
        async for sentence, audio_chunk, is_new_sentence in pipeline.process_streaming(
            messages,
            system=system,
            tools=tools,
            temperature=temperature,
            max_tokens=max_tokens,
        ):
            # Audio delta
            audio_b64 = encode_audio_for_client(
                audio_chunk, self._config.output_audio_format
            )
            await self._emitter.emit(
                events.response_audio_delta(
                    "", response_id, item_id, output_index, content_index, audio_b64
                )
            )

            # Log end-to-end latency on first audio chunk
            if not first_audio_sent:
                first_audio_sent = True
                if self._speech_stopped_at is not None:
                    e2e_ms = (time.perf_counter() - self._speech_stopped_at) * 1000
                    logger.info(
                        f"[{self.session_id[:8]}] E2E LATENCY: {e2e_ms:.0f}ms "
                        f"(speech_stopped → first_audio_sent)"
                    )

            # Transcript delta (only on the first chunk of each new sentence)
            if is_new_sentence and sentence:
                delta = sentence
                if full_transcript:
                    delta = " " + sentence
                await self._emitter.emit(
                    events.response_audio_transcript_delta(
                        "", response_id, item_id, output_index, content_index, delta
                    )
                )
                full_transcript += delta

        # Update assistant item — don't store accumulated audio base64
        # (already streamed via deltas, storing would bloat memory/messages)
        assistant_item.content[content_index] = ContentPart(
            type="audio", transcript=full_transcript
        )

        # Audio done + transcript done
        await self._emitter.emit(
            events.response_audio_done("", response_id, item_id, output_index, content_index)
        )
        await self._emitter.emit(
            events.response_audio_transcript_done(
                "", response_id, item_id, output_index, content_index, full_transcript
            )
        )

        return full_transcript

    async def _run_text_response(
        self,
        response_id: str,
        item_id: str,
        output_index: int,
        content_index: int,
        assistant_item: ConversationItem,
        messages: list[dict],
        system: str,
        temperature: float,
        max_tokens: int,
    ) -> str:
        """Run LLM text-only streaming. Returns full text."""
        full_text = ""

        async for chunk in self._llm.generate_stream(
            messages,
            system=system,
            tools=self._config.tools or None,
            temperature=temperature,
            max_tokens=max_tokens,
        ):
            await self._emitter.emit(
                events.response_text_delta(
                    "", response_id, item_id, output_index, content_index, chunk
                )
            )
            full_text += chunk

        assistant_item.content[content_index] = ContentPart(
            type="text", text=full_text
        )
        await self._emitter.emit(
            events.response_text_done(
                "", response_id, item_id, output_index, content_index, full_text
            )
        )

        return full_text

    async def _finalize_response(
        self,
        response_id: str,
        output_index: int,
        content_index: int,
        assistant_item: ConversationItem,
        response_start: float,
        full_transcript: str,
        full_text: str,
    ) -> None:
        """Emit final response lifecycle events."""
        await self._emitter.emit(
            events.response_content_part_done(
                "", response_id, assistant_item.id, output_index, content_index,
                assistant_item.content[content_index],
            )
        )

        assistant_item.status = "completed"
        await self._emitter.emit(
            events.response_output_item_done("", response_id, output_index, assistant_item)
        )

        response_ms = (time.perf_counter() - response_start) * 1000
        logger.info(
            f"[{self.session_id[:8]}] RESPONSE DONE: {response_ms:.0f}ms, "
            f"transcript=\"{full_transcript[:60] or full_text[:60]}\""
        )
        await self._emitter.emit(
            events.response_done(
                "", response_id, status="completed",
                output=[assistant_item.to_dict()],
            )
        )

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
