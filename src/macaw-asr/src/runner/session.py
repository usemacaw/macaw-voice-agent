"""Streaming session for ASR inference.

Manages per-connection state: audio accumulation, background precomputation,
and partial results. Equivalent to Ollama's runner state per request.

The session coordinates the streaming pipeline:
    push_audio → accumulate → [background] prepare + generate → partial text
    finish     → fast_finish or recompute → final generate → clean text
"""

from __future__ import annotations

import asyncio
import logging
import time as _time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from macaw_asr._executor import run_in_executor
from macaw_asr.audio.accumulator import ChunkAccumulator
from macaw_asr.audio.preprocessing import AudioPreprocessor
from macaw_asr.config import EngineConfig
from macaw_asr.decode.postprocess import clean_asr_text
from macaw_asr.decode.strategies import DecodeStrategy
from macaw_asr.models.base import ASRModel

logger = logging.getLogger("macaw-asr.runner.session")


@dataclass
class SessionMetrics:
    """Timing metrics accumulated during a streaming session."""

    total_resample_ms: float = 0.0
    total_processor_ms: float = 0.0
    total_generate_ms: float = 0.0
    bg_decode_count: int = 0


class StreamingSession:
    """Per-connection streaming ASR session.

    Owns audio accumulation, background precomputation, and partial results.
    Delegates actual inference to the ASRModel (DIP).
    """

    def __init__(
        self,
        session_id: str,
        model: ASRModel,
        preprocessor: AudioPreprocessor,
        strategy: DecodeStrategy,
        config: EngineConfig,
        executor: ThreadPoolExecutor,
    ) -> None:
        self._session_id = session_id
        self._model = model
        self._preprocessor = preprocessor
        self._strategy = strategy
        self._config = config
        self._executor = executor

        trigger_samples = int(
            config.streaming.chunk_trigger_sec * config.audio.model_sample_rate
        )
        self._accumulator = ChunkAccumulator(trigger_samples)
        self._metrics = SessionMetrics()

        # Background precomputation cache
        self._cached_inputs: Any = None
        self._cached_audio_len: int = 0

        # Background task
        self._bg_task: asyncio.Task | None = None
        self._bg_lock = asyncio.Lock()

        # Partial results
        self._partial_text: str = ""
        self._partial_raw: str = ""
        self._text: str = ""

    @property
    def session_id(self) -> str:
        return self._session_id

    @property
    def text(self) -> str:
        """Current best transcription."""
        return self._text

    @property
    def metrics(self) -> SessionMetrics:
        return self._metrics

    async def push_audio(self, pcm_chunk: bytes) -> str:
        """Push an audio chunk. Returns current best transcription."""
        if not pcm_chunk:
            return self._text

        t0 = _time.perf_counter()
        float_chunk = self._preprocessor.process(pcm_chunk)
        self._metrics.total_resample_ms += (_time.perf_counter() - t0) * 1000

        should_trigger = self._accumulator.push(float_chunk)

        if should_trigger and self._config.streaming.enable_background_compute:
            if self._bg_task is None or self._bg_task.done():
                snapshot = self._accumulator.snapshot()
                self._bg_task = asyncio.create_task(
                    self._background_precompute_and_decode(snapshot)
                )

        return self._text

    async def finish(self) -> str:
        """Finish session and get final transcription."""
        t_finish = _time.perf_counter()

        # Cancel background task (prevents GPU contention)
        bg_wait_ms = await self._cancel_bg_task()

        if self._accumulator.is_empty:
            logger.debug("Session finish: no audio (%s)", self._session_id[:8])
            return ""

        audio = self._accumulator.get_all()
        audio_len = len(audio)

        # Try fast finish, fall back to full recompute
        t0 = _time.perf_counter()
        inputs, recompute_mode = await self._prepare_finish_inputs(audio, audio_len)
        proc_ms = (_time.perf_counter() - t0) * 1000
        self._metrics.total_processor_ms += proc_ms

        # Generate
        t0 = _time.perf_counter()
        output = await run_in_executor(
            self._executor, self._model.generate, inputs, self._strategy
        )
        gen_ms = (_time.perf_counter() - t0) * 1000
        self._metrics.total_generate_ms += gen_ms

        text = clean_asr_text(output.text)
        finish_ms = (_time.perf_counter() - t_finish) * 1000

        logger.info(
            "FINISH: '%s' | bg_wait=%.0fms recompute=%s proc=%.0fms "
            "gen=%.0fms(pf=%.0fms %dtok=%.0fms) finish=%.0fms | "
            "session: resample=%.0fms proc=%.0fms gen=%.0fms bg_decodes=%d",
            text, bg_wait_ms, recompute_mode, proc_ms,
            gen_ms, output.timings.get("prefill_ms", 0),
            output.n_tokens, output.timings.get("decode_ms", 0),
            finish_ms,
            self._metrics.total_resample_ms, self._metrics.total_processor_ms,
            self._metrics.total_generate_ms, self._metrics.bg_decode_count,
        )
        return text

    async def cancel(self) -> None:
        """Cancel session and cleanup resources."""
        await self._cancel_bg_task()

    # ==================== Internal ====================

    async def _prepare_finish_inputs(
        self, audio: np.ndarray, audio_len: int
    ) -> tuple[Any, str]:
        """Prepare inputs for final generate, using fast path when possible."""
        if self._cached_inputs is not None and self._cached_audio_len > 0:
            if self._cached_audio_len == audio_len:
                return self._cached_inputs, "cached"

            fast_inputs = await run_in_executor(
                self._executor,
                self._model.fast_finish_inputs,
                audio,
                self._cached_inputs,
                self._cached_audio_len,
                self._partial_raw or "",
            )
            if fast_inputs is not None:
                return fast_inputs, "fast"

        prefix = self._partial_raw or ""
        inputs = await run_in_executor(
            self._executor, self._model.prepare_inputs, audio, prefix
        )
        return inputs, "full"

    async def _background_precompute_and_decode(self, audio: np.ndarray) -> None:
        """Background: prepare_inputs + generate during speech."""
        async with self._bg_lock:
            try:
                t_total = _time.perf_counter()

                t0 = _time.perf_counter()
                inputs = await run_in_executor(
                    self._executor, self._model.prepare_inputs, audio, ""
                )
                proc_ms = (_time.perf_counter() - t0) * 1000
                self._metrics.total_processor_ms += proc_ms

                self._cached_inputs = inputs
                self._cached_audio_len = len(audio)

                t0 = _time.perf_counter()
                output = await run_in_executor(
                    self._executor, self._model.generate, inputs, self._strategy
                )
                gen_ms = (_time.perf_counter() - t0) * 1000
                self._metrics.total_generate_ms += gen_ms

                self._partial_text = clean_asr_text(output.text)
                self._partial_raw = output.raw_text
                self._text = self._partial_text
                self._metrics.bg_decode_count += 1

                total_ms = (_time.perf_counter() - t_total) * 1000
                logger.info(
                    "BG[%d]: '%s' | proc=%.0fms gen=%.0fms total=%.0fms",
                    self._metrics.bg_decode_count, self._partial_text[:40],
                    proc_ms, gen_ms, total_ms,
                )
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.warning("BG decode error: %s", e)

    async def _cancel_bg_task(self) -> float:
        """Cancel background task. Returns wait time in ms."""
        if self._bg_task and not self._bg_task.done():
            t0 = _time.perf_counter()
            self._bg_task.cancel()
            try:
                await self._bg_task
            except (asyncio.CancelledError, Exception):
                pass
            return (_time.perf_counter() - t0) * 1000
        return 0.0
