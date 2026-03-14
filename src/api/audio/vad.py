"""
Server-side Voice Activity Detection using Silero VAD + Smart Turn.

Two-stage pipeline:
  1. Silero VAD (acoustic): detects speech vs silence in 32ms chunks
  2. Smart Turn (semantic): analyzes prosody/intonation to decide if
     the speaker has finished their turn or is just pausing

This eliminates premature responses when the user pauses mid-thought.
"""

from __future__ import annotations

import logging
import time
from collections import deque
from typing import Callable

import numpy as np
import torch

from protocol.models import TurnDetection

logger = logging.getLogger("open-voice-api.vad")

# Silero VAD at 8kHz expects 256-sample chunks (32ms)
SAMPLE_RATE = 8000
CHUNK_SAMPLES = 256
CHUNK_BYTES = CHUNK_SAMPLES * 2  # 16-bit PCM
CHUNK_DURATION_MS = CHUNK_SAMPLES * 1000 // SAMPLE_RATE  # 32ms

# Speech probability threshold — above this = speech detected
_DEFAULT_THRESHOLD = 0.5


class VADProcessor:
    """Server-side VAD using Silero VAD model.

    Much more accurate than webrtcvad for distinguishing real speech
    from noise, keyboard typing, background sounds, etc.

    Accumulates audio frames and calls back on speech_started/speech_stopped
    based on configurable silence duration and minimum speech length.
    """

    def __init__(
        self,
        config: TurnDetection,
        on_speech_started: Callable[[int], None] | None = None,
        on_speech_stopped: Callable[[int, bytes], None] | None = None,
        aggressiveness: int = 2,
    ):
        from silero_vad import load_silero_vad

        self._model = load_silero_vad(onnx=True)
        self._threshold = config.threshold if config.threshold > 0 else _DEFAULT_THRESHOLD
        self._config = config

        self._on_speech_started = on_speech_started
        self._on_speech_stopped = on_speech_stopped

        # Frame tracking
        self._frame_buffer = bytearray()
        self._total_audio_ms = 0
        self._vad_error_count = 0

        # Speech state
        self._is_speaking = False
        self._speech_start_ms = 0
        self._silence_chunks = 0
        self._speech_chunks = 0

        # Prefix padding buffer (ring buffer of recent chunks)
        max_prefix_chunks = max(1, config.prefix_padding_ms // CHUNK_DURATION_MS)
        self._prefix_buffer: deque[bytes] = deque(maxlen=max_prefix_chunks)

        # Accumulated speech audio
        self._speech_audio = bytearray()

        # Thresholds in chunks
        self._silence_threshold_chunks = max(1, config.silence_duration_ms // CHUNK_DURATION_MS)
        self._min_speech_chunks = max(1, 250 // CHUNK_DURATION_MS)  # 250ms min

        # Smart Turn semantic detector (second stage)
        self._smart_turn = None
        self._smart_turn_wait_count = 0
        # Max times Smart Turn can say "incomplete" before we force the turn
        self._smart_turn_max_waits = 4

        # Turn detection metrics (populated per speech_stopped, read by AudioInputHandler)
        self._silence_start_time: float | None = None
        self._smart_turn_accum_ms: float = 0.0
        self.last_turn_metrics: dict[str, object] = {}
        try:
            from audio.smart_turn import SmartTurnDetector
            self._smart_turn = SmartTurnDetector()
        except Exception as e:
            logger.warning(f"Smart Turn not available, using acoustic-only VAD: {e}")

        logger.info(
            f"Silero VAD initialized: threshold={self._threshold}, "
            f"silence={config.silence_duration_ms}ms, "
            f"prefix={config.prefix_padding_ms}ms, "
            f"smart_turn={'enabled' if self._smart_turn else 'disabled'}"
        )

    def feed(self, audio: bytes) -> None:
        """Feed PCM 8kHz 16-bit audio. Processes in 32ms chunks."""
        self._frame_buffer.extend(audio)

        while len(self._frame_buffer) >= CHUNK_BYTES:
            chunk = bytes(self._frame_buffer[:CHUNK_BYTES])
            del self._frame_buffer[:CHUNK_BYTES]
            self._process_chunk(chunk)

    def _process_chunk(self, chunk: bytes) -> None:
        self._total_audio_ms += CHUNK_DURATION_MS

        # Convert PCM 16-bit to float32 tensor for Silero
        samples = np.frombuffer(chunk, dtype=np.int16).astype(np.float32) / 32768.0
        audio_tensor = torch.from_numpy(samples)

        try:
            prob = self._model(audio_tensor, SAMPLE_RATE).item()
            self._vad_error_count = 0
        except Exception as e:
            self._vad_error_count += 1
            logger.error(f"VAD model inference failed ({self._vad_error_count} consecutive): {e}")
            if self._vad_error_count >= 10:
                raise RuntimeError(
                    f"VAD model failed {self._vad_error_count} consecutive times, last error: {e}"
                ) from e
            prob = 0.0

        is_speech = prob >= self._threshold

        if not self._is_speaking:
            self._prefix_buffer.append(chunk)

            if is_speech:
                self._speech_chunks += 1
                # Need a few consecutive speech chunks to trigger
                if self._speech_chunks >= 3:
                    self._is_speaking = True
                    self._speech_start_ms = self._total_audio_ms - (self._speech_chunks * CHUNK_DURATION_MS)
                    self._silence_chunks = 0

                    # Include prefix padding
                    self._speech_audio.clear()
                    for pf in self._prefix_buffer:
                        self._speech_audio.extend(pf)

                    if self._on_speech_started:
                        self._on_speech_started(self._speech_start_ms)
            else:
                self._speech_chunks = 0

        else:
            self._speech_audio.extend(chunk)

            if is_speech:
                self._silence_chunks = 0
                self._smart_turn_wait_count = 0
                self._silence_start_time = None
                self._smart_turn_accum_ms = 0.0
            else:
                self._silence_chunks += 1
                # Mark when silence begins (first non-speech chunk)
                if self._silence_start_time is None:
                    self._silence_start_time = time.perf_counter()

                if self._silence_chunks >= self._silence_threshold_chunks:
                    speech_duration_ms = self._total_audio_ms - self._speech_start_ms
                    if speech_duration_ms >= self._min_speech_chunks * CHUNK_DURATION_MS:
                        # Smart Turn: check if speaker actually finished
                        if self._smart_turn and self._smart_turn_wait_count < self._smart_turn_max_waits:
                            t0 = time.perf_counter()
                            is_complete, prob = self._smart_turn.predict(
                                bytes(self._speech_audio), source_sample_rate=SAMPLE_RATE
                            )
                            st_ms = (time.perf_counter() - t0) * 1000
                            self._smart_turn_accum_ms += st_ms
                            logger.info(
                                f"Smart Turn: complete={is_complete}, prob={prob:.2f}, "
                                f"inference={st_ms:.0f}ms, wait_count={self._smart_turn_wait_count}"
                            )
                            if not is_complete:
                                # Speaker not done — reset silence counter and keep listening
                                self._smart_turn_wait_count += 1
                                self._silence_chunks = 0
                                return
                        # Turn is complete (or max waits exceeded) — populate metrics
                        now = time.perf_counter()
                        silence_wait_ms = (
                            (now - self._silence_start_time) * 1000
                            if self._silence_start_time else 0.0
                        )
                        self.last_turn_metrics = {
                            "vad_silence_wait_ms": round(silence_wait_ms, 1),
                            "smart_turn_inference_ms": round(self._smart_turn_accum_ms, 1),
                            "smart_turn_waits": self._smart_turn_wait_count,
                        }
                        # Fire callback
                        if self._on_speech_stopped:
                            self._on_speech_stopped(
                                self._total_audio_ms,
                                bytes(self._speech_audio),
                            )
                    else:
                        logger.debug(
                            f"Speech too short ({speech_duration_ms}ms), discarding"
                        )

                    # Reset
                    self._is_speaking = False
                    self._speech_chunks = 0
                    self._silence_chunks = 0
                    self._smart_turn_wait_count = 0
                    self._silence_start_time = None
                    self._smart_turn_accum_ms = 0.0
                    self._speech_audio.clear()
                    self._prefix_buffer.clear()

    def reset(self) -> None:
        """Reset VAD state."""
        self._frame_buffer.clear()
        self._total_audio_ms = 0
        self._is_speaking = False
        self._speech_start_ms = 0
        self._silence_chunks = 0
        self._speech_chunks = 0
        self._smart_turn_wait_count = 0
        self._silence_start_time = None
        self._smart_turn_accum_ms = 0.0
        self._prefix_buffer.clear()
        self._speech_audio.clear()
        # Reset Silero model state
        self._model.reset_states()

    @property
    def is_speaking(self) -> bool:
        return self._is_speaking

    @property
    def total_audio_ms(self) -> int:
        return self._total_audio_ms
