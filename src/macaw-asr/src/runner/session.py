"""Streaming session — implements ISession contract.

Manages per-connection state: audio accumulation, background precomputation.
SRP: Only handles streaming pipeline. Does NOT load models or manage lifecycle.
"""

from __future__ import annotations

import asyncio
import logging
import time as _time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

import numpy as np

from macaw_asr._executor import run_in_executor
from macaw_asr.audio.preprocessing import pcm_to_float32, resample
from macaw_asr.config import EngineConfig
from macaw_asr.decode.postprocess import clean_asr_text
from macaw_asr.decode.strategies import DecodeStrategy
from macaw_asr.models.contracts import IASRModel
from macaw_asr.audio.preprocessing import AudioPreprocessor
from macaw_asr.runner.contracts import ISession

logger = logging.getLogger("macaw-asr.runner.session")


@dataclass
class SessionMetrics:
    total_resample_ms: float = 0.0
    total_processor_ms: float = 0.0
    total_generate_ms: float = 0.0
    bg_decode_count: int = 0


class StreamingSession(ISession):
    """Per-connection streaming ASR session.

    Accumulates raw PCM at input rate, resamples as one block on finish.
    Background precompute runs during speech if model supports streaming.
    """

    def __init__(
        self, session_id: str, model: IASRModel, preprocessor: AudioPreprocessor,
        strategy: DecodeStrategy | None, config: EngineConfig, executor: ThreadPoolExecutor,
    ) -> None:
        self._session_id = session_id
        self._model = model
        self._strategy = strategy
        self._config = config
        self._executor = executor

        self._input_rate = config.audio.input_sample_rate
        self._model_rate = config.audio.model_sample_rate
        self._raw_buffer: list[np.ndarray] = []
        self._raw_samples: int = 0
        self._max_samples = int(300 * self._input_rate)  # 5 min max session
        self._trigger_samples = int(config.streaming.chunk_trigger_sec * self._input_rate)
        self._samples_since_trigger: int = 0
        self._metrics = SessionMetrics()

        self._cached_inputs: Any = None
        self._cached_audio_len: int = 0
        self._bg_task: asyncio.Task | None = None
        self._bg_lock = asyncio.Lock()
        self._partial_text: str = ""
        self._partial_raw: str = ""
        self._text: str = ""

    @property
    def session_id(self) -> str:
        return self._session_id

    @property
    def text(self) -> str:
        return self._text

    @property
    def metrics(self) -> SessionMetrics:
        return self._metrics

    async def push_audio(self, pcm_chunk: bytes) -> str:
        if not pcm_chunk:
            return self._text

        t0 = _time.perf_counter()
        try:
            float_chunk = pcm_to_float32(pcm_chunk)
        except ValueError:
            logger.warning("Malformed PCM chunk (%d bytes), skipping", len(pcm_chunk))
            return self._text
        self._metrics.total_resample_ms += (_time.perf_counter() - t0) * 1000

        self._raw_buffer.append(float_chunk)
        self._raw_samples += len(float_chunk)
        self._samples_since_trigger += len(float_chunk)

        # Cap buffer: discard oldest chunks until within limit
        if self._raw_samples > self._max_samples:
            while self._raw_samples > self._max_samples and len(self._raw_buffer) > 1:
                removed = self._raw_buffer.pop(0)
                self._raw_samples -= len(removed)
            logger.warning("Session %s exceeded max duration, trimmed to %ds",
                           self._session_id[:8], self._raw_samples // self._input_rate)

        should_trigger = self._samples_since_trigger >= self._trigger_samples
        if should_trigger:
            self._samples_since_trigger = 0

        if should_trigger and self._config.streaming.enable_background_compute:
            if self._bg_task is None or self._bg_task.done():
                snapshot = self._get_resampled_audio()
                self._bg_task = asyncio.create_task(self._bg_precompute(snapshot))

        return self._text

    async def finish(self) -> str:
        t_finish = _time.perf_counter()
        await self._cancel_bg()

        if self._raw_samples == 0:
            return ""

        audio = self._get_resampled_audio()
        audio_len = len(audio)

        t0 = _time.perf_counter()
        inputs, mode = await self._prepare_finish(audio, audio_len)
        self._metrics.total_processor_ms += (_time.perf_counter() - t0) * 1000

        t0 = _time.perf_counter()
        output = await run_in_executor(self._executor, self._model.generate, inputs, self._strategy)
        self._metrics.total_generate_ms += (_time.perf_counter() - t0) * 1000

        text = clean_asr_text(output.text)
        logger.info("FINISH: '%s' | recompute=%s finish=%.0fms bg=%d",
                     text[:40], mode, (_time.perf_counter() - t_finish) * 1000, self._metrics.bg_decode_count)
        return text

    async def cancel(self) -> None:
        await self._cancel_bg()

    # ==================== Internal ====================

    def _get_resampled_audio(self) -> np.ndarray:
        if not self._raw_buffer:
            return np.zeros(0, dtype=np.float32)
        raw = np.concatenate(self._raw_buffer)
        return resample(raw, self._input_rate, self._model_rate)

    async def _prepare_finish(self, audio: np.ndarray, audio_len: int) -> tuple[Any, str]:
        if self._cached_inputs is not None and self._cached_audio_len > 0:
            if self._cached_audio_len == audio_len:
                return self._cached_inputs, "cached"
            fast = await run_in_executor(
                self._executor, self._model.fast_finish_inputs,
                audio, self._cached_inputs, self._cached_audio_len, self._partial_raw or "",
            )
            if fast is not None:
                return fast, "fast"
        inputs = await run_in_executor(
            self._executor, self._model.prepare_inputs, audio, self._partial_raw or "",
        )
        return inputs, "full"

    async def _bg_precompute(self, audio: np.ndarray) -> None:
        async with self._bg_lock:
            try:
                inputs = await run_in_executor(self._executor, self._model.prepare_inputs, audio, "")
                self._cached_inputs = inputs
                self._cached_audio_len = len(audio)

                output = await run_in_executor(self._executor, self._model.generate, inputs, self._strategy)
                self._partial_text = clean_asr_text(output.text)
                self._partial_raw = output.raw_text
                self._text = self._partial_text
                self._metrics.bg_decode_count += 1
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.warning("BG decode error", exc_info=True)  # Fix #6: log stack trace

    async def _cancel_bg(self) -> None:
        if self._bg_task and not self._bg_task.done():
            self._bg_task.cancel()
            try:
                await self._bg_task
            except (asyncio.CancelledError, Exception):
                pass
