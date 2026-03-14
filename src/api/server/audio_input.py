"""
AudioInputHandler — VAD, ASR, and speech detection logic.

Extracted from RealtimeSession to isolate audio input processing
from session lifecycle management.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Awaitable, Callable

import numpy as np

from audio.codec import INTERNAL_SAMPLE_RATE, SAMPLE_WIDTH
from audio.vad import VADProcessor
from config import VAD
from protocol import events
from protocol.models import ContentPart, ConversationItem
from providers.admission import ADMISSION

if TYPE_CHECKING:
    from protocol.event_emitter import EventEmitter
    from protocol.models import SessionConfig
    from providers.asr import ASRProvider

logger = logging.getLogger("open-voice-api.audio-input")


@dataclass
class AudioInputCallbacks:
    """Callbacks from AudioInputHandler back to RealtimeSession."""

    cancel_active_response: Callable[[], Awaitable[bool]]  # returns True if cancelled
    append_user_item_and_respond: Callable[[ConversationItem, str], Awaitable[None]]
    emit: Callable[[dict], Awaitable[None]]


class AudioInputHandler:
    """Handles VAD, ASR, and speech detection for a session.

    Owns:
      - VAD processor and its callbacks
      - ASR streaming state (stream_id)
      - Speech metrics (RMS, barge-in count, speech_stopped_at)
      - Background tasks spawned by VAD callbacks

    Does NOT own:
      - Conversation items (managed by session)
      - Response task (managed by session)
      - WebSocket / emitter (uses callback)
    """

    def __init__(
        self,
        asr: ASRProvider,
        config: SessionConfig,
        callbacks: AudioInputCallbacks,
        emitter: EventEmitter,
        session_id: str,
    ):
        self._asr = asr
        self._config = config
        self._callbacks = callbacks
        self._emitter = emitter
        self._session_id = session_id

        # VAD
        self._vad: VADProcessor | None = None
        self._pending_speech_item_id: str | None = None
        self.setup_vad()

        # ASR streaming
        self._asr_stream_id: str | None = None
        self._asr_lock = asyncio.Lock()

        # Metrics
        self._speech_stopped_at: float | None = None
        self._barge_in_count = 0
        self._response_metrics: dict[str, object] = {}

        # Background tasks (fire-and-forget from VAD callbacks)
        self._background_tasks: set[asyncio.Task] = set()

    # ---- Public API ----

    @property
    def vad(self) -> VADProcessor | None:
        return self._vad

    @property
    def speech_stopped_at(self) -> float | None:
        return self._speech_stopped_at

    @property
    def barge_in_count(self) -> int:
        return self._barge_in_count

    @property
    def response_metrics(self) -> dict[str, object]:
        return self._response_metrics

    @response_metrics.setter
    def response_metrics(self, value: dict[str, object]) -> None:
        self._response_metrics = value

    def setup_vad(self) -> None:
        """Create or reset the VAD processor from current config."""
        td = self._config.turn_detection
        if td and td.type == "server_vad":
            self._vad = VADProcessor(
                config=td,
                on_speech_started=self._on_speech_started,
                on_speech_stopped=self._on_speech_stopped,
                aggressiveness=VAD.aggressiveness,
            )
        else:
            self._vad = None

    def feed_audio(self, pcm: bytes) -> None:
        """Feed PCM audio to VAD (if active)."""
        if self._vad:
            self._vad.feed(pcm)

    async def feed_asr_chunk(self, pcm: bytes) -> None:
        """Feed PCM audio to ASR streaming (if active)."""
        if self._asr_stream_id and self._asr.supports_streaming:
            await self._asr.feed_chunk(pcm, self._asr_stream_id)

    async def cleanup(self) -> None:
        """Cancel background tasks and finish any active ASR stream."""
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
                logger.warning(
                    f"[{self._session_id[:8]}] Error finishing ASR stream on cleanup: {e}"
                )
            self._asr_stream_id = None

    # ---- VAD Callbacks (called synchronously by VAD.feed) ----

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
            f"[{self._session_id[:8]}] VAD speech_started at {audio_start_ms}ms, "
            f"item={item_id[:12]}"
        )
        task = asyncio.create_task(self._handle_speech_started(audio_start_ms, item_id))
        self._track_task(task)

    async def _handle_speech_started(self, audio_start_ms: int, item_id: str) -> None:
        await self._emitter.emit(
            events.input_audio_buffer_speech_started("", audio_start_ms, item_id)
        )

        # Start ASR streaming (stream_id already set synchronously)
        if self._asr.supports_streaming:
            await self._asr.start_stream(item_id)

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

        min_rms = VAD.min_speech_rms
        if rms < min_rms:
            logger.debug(
                f"[{self._session_id[:8]}] VAD speech_stopped DISCARDED: rms={rms:.0f} < {min_rms} "
                f"(likely noise/echo), {speech_duration_ms:.0f}ms, item={item_id[:12]}"
            )
            async with self._asr_lock:
                if self._asr_stream_id == item_id:
                    self._asr_stream_id = None
            return

        logger.info(
            f"[{self._session_id[:8]}] VAD speech_stopped at {audio_end_ms}ms, "
            f"speech_audio={len(speech_audio)} bytes ({speech_duration_ms:.0f}ms), "
            f"rms={rms:.0f}, item={item_id[:12]}"
        )

        # Barge-in: interrupt active response if this is real speech
        td = self._config.turn_detection
        if td and td.interrupt_response:
            cancelled = await self._callbacks.cancel_active_response()
            if cancelled:
                self._barge_in_count += 1

        self._speech_stopped_at = time.perf_counter()
        self._response_metrics["speech_rms"] = round(rms, 1)

        # Capture turn detection metrics from VAD (silence wait, Smart Turn timing)
        if self._vad and self._vad.last_turn_metrics:
            self._response_metrics.update(self._vad.last_turn_metrics)

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
                f"[{self._session_id[:8]}] ASR returned empty — skipping response. "
                f"speech_audio={len(speech_audio)} bytes ({audio_ms:.0f}ms)"
            )
            return

        # Create user item with transcription
        item = ConversationItem(
            id=item_id,
            type="message",
            role="user",
            content=[ContentPart(type="input_audio", transcript=transcript)],
            status="completed",
        )

        await self._callbacks.append_user_item_and_respond(item, transcript)

    async def _transcribe_audio(self, speech_audio: bytes, item_id: str) -> str:
        """Transcribe audio using streaming or batch ASR."""
        t0 = time.perf_counter()
        transcript = ""
        asr_mode = "batch"

        if self._asr_stream_id == item_id and self._asr.supports_streaming:
            asr_mode = "streaming"
            async with ADMISSION.asr.acquire():
                transcript = await self._asr.finish_stream(item_id)
            async with self._asr_lock:
                if self._asr_stream_id == item_id:
                    self._asr_stream_id = None
            asr_ms = (time.perf_counter() - t0) * 1000
            logger.info(
                f"[{self._session_id[:8]}] ASR (streaming): {asr_ms:.0f}ms → "
                f'"{transcript[:80]}" (empty={not transcript})'
            )

            # Fallback to batch if streaming returned empty
            if not transcript and speech_audio:
                asr_mode = "batch-fallback"
                logger.info(
                    f"[{self._session_id[:8]}] ASR streaming empty, falling back to batch "
                    f"({len(speech_audio)} bytes)"
                )
                t0 = time.perf_counter()
                async with ADMISSION.asr.acquire():
                    transcript = await self._asr.transcribe(speech_audio)
                asr_ms = (time.perf_counter() - t0) * 1000
                logger.info(
                    f"[{self._session_id[:8]}] ASR (batch-fallback): {asr_ms:.0f}ms → "
                    f'"{transcript[:80]}" (empty={not transcript})'
                )
        else:
            if self._asr_stream_id == item_id:
                try:
                    await self._asr.finish_stream(item_id)
                except Exception as e:
                    logger.warning(
                        f"[{self._session_id[:8]}] Error finishing ASR stream "
                        f"{item_id[:12]}: {e}"
                    )
                async with self._asr_lock:
                    if self._asr_stream_id == item_id:
                        self._asr_stream_id = None
            async with ADMISSION.asr.acquire():
                transcript = await self._asr.transcribe(speech_audio)
            asr_ms = (time.perf_counter() - t0) * 1000
            logger.info(
                f"[{self._session_id[:8]}] ASR (batch): {asr_ms:.0f}ms → "
                f'"{transcript[:80]}" (empty={not transcript})'
            )

        # Store for per-response observability
        total_asr_ms = (time.perf_counter() - t0) * 1000
        speech_ms = len(speech_audio) / (INTERNAL_SAMPLE_RATE * SAMPLE_WIDTH) * 1000
        self._response_metrics["asr_ms"] = round(total_asr_ms, 1)
        self._response_metrics["speech_ms"] = round(speech_ms, 1)
        self._response_metrics["asr_mode"] = asr_mode
        self._response_metrics["input_chars"] = len(transcript)

        return transcript

    # ---- Helpers ----

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

    def _track_task(self, task: asyncio.Task) -> None:
        """Register a fire-and-forget task for cleanup."""
        self._background_tasks.add(task)

        def _done(t: asyncio.Task) -> None:
            self._background_tasks.discard(t)
            if t.cancelled():
                return
            exc = t.exception()
            if exc is not None:
                logger.error(
                    f"[{self._session_id[:8]}] Audio input task failed: {exc}",
                    exc_info=exc,
                )

        task.add_done_callback(_done)
