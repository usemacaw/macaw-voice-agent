"""ASR Engine — inference orchestrator. Implements IEngine.

Fixes from audit:
- #1/#2: transcribe_stream() and create_strategy() are public — routes don't access internals
- #4: apply_compiled_module() delegated to model, no feature envy
- #6: _create_strategy no longer swallows exceptions silently
- executor.shutdown(wait=True) for clean stop
"""

from __future__ import annotations

import logging
import time as _time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Generator

import numpy as np

from macaw_asr._executor import create_executor, run_in_executor
from macaw_asr.audio.preprocessing import AudioPreprocessor
from macaw_asr.config import EngineConfig
from macaw_asr.decode.postprocess import clean_asr_text
from macaw_asr.decode.strategies import DecodeStrategy, GreedyWithEarlyStopping
from macaw_asr.models import create_model, IASRModel
from macaw_asr.models.types import ModelOutput
from macaw_asr.runner.contracts import IEngine
from macaw_asr.runner.session import StreamingSession

logger = logging.getLogger("macaw-asr.runner.engine")


class ASREngine(IEngine):
    """Main inference orchestrator."""

    def __init__(self, config: EngineConfig) -> None:
        self._config = config
        self._model: IASRModel | None = None
        self._preprocessor = AudioPreprocessor(config.audio)
        self._executor: ThreadPoolExecutor | None = None
        self._sessions: dict[str, StreamingSession] = {}
        self._started = False
        self._startup_timings: dict[str, float] = {}

    @property
    def config(self) -> EngineConfig:
        return self._config

    @property
    def is_started(self) -> bool:
        return self._started

    @property
    def startup_timings(self) -> dict[str, float]:
        return dict(self._startup_timings)

    async def start(self) -> None:
        if self._started:
            return

        t_total = _time.perf_counter()
        self._executor = create_executor(self._config.max_inference_workers)
        self._model = create_model(self._config.model_name)

        # Load
        t0 = _time.perf_counter()
        await run_in_executor(self._executor, self._model.load, self._config)
        load_ms = (_time.perf_counter() - t0) * 1000

        # Compile (fix #4: delegate to model via apply_compiled_module)
        compile_ms = 0.0
        if self._config.enable_compile:
            compile_ms = await self._apply_compile()

        # Warmup
        t0 = _time.perf_counter()
        await run_in_executor(self._executor, self._model.warmup, self._config)
        warmup_ms = (_time.perf_counter() - t0) * 1000

        total_ms = (_time.perf_counter() - t_total) * 1000
        self._startup_timings = {
            "load_ms": load_ms, "compile_ms": compile_ms,
            "warmup_ms": warmup_ms, "total_ms": total_ms,
        }

        self._started = True
        logger.info(
            "ASREngine started: model=%s load=%.0fms compile=%.0fms warmup=%.0fms total=%.0fms",
            self._config.model_name, load_ms, compile_ms, warmup_ms, total_ms,
        )

    async def stop(self) -> None:
        if not self._started:
            return
        try:
            for sid in list(self._sessions):
                await self._cancel_session(sid)
            if self._model:
                await run_in_executor(self._executor, self._model.unload)
                self._model = None
        finally:
            if self._executor:
                self._executor.shutdown(wait=True)
                self._executor = None
            self._started = False
            logger.info("ASREngine stopped")

    # ==================== Batch ====================

    async def transcribe(self, pcm_data: bytes) -> str:
        result = await self.transcribe_with_metrics(pcm_data)
        return result.text

    async def transcribe_with_metrics(self, pcm_data: bytes) -> TranscribeResult:
        """Transcribe with full timing breakdown."""
        from macaw_asr.models.types import TranscribeResult

        self._ensure_started()
        if not pcm_data:
            return TranscribeResult(text="")

        t_total = _time.perf_counter()

        t0 = _time.perf_counter()
        float_audio = self._preprocessor.process(pcm_data)
        preprocess_ms = (_time.perf_counter() - t0) * 1000

        t0 = _time.perf_counter()
        inputs = await run_in_executor(self._executor, self._model.prepare_inputs, float_audio, "")
        prepare_ms = (_time.perf_counter() - t0) * 1000

        strategy = self.create_strategy()

        t0 = _time.perf_counter()
        output = await run_in_executor(self._executor, self._model.generate, inputs, strategy)
        inference_ms = (_time.perf_counter() - t0) * 1000

        text = clean_asr_text(output.text)
        e2e_ms = (_time.perf_counter() - t_total) * 1000

        timings = {
            "preprocess_ms": round(preprocess_ms, 2),
            "inference_ms": round(inference_ms, 2),
            "e2e_ms": round(e2e_ms, 2),
            **{k: round(v, 2) for k, v in output.timings.items()},
        }

        return TranscribeResult(text=text, timings=timings)

    # ==================== Streaming SSE (fix #1/#2: public method, uses executor) ====================

    def transcribe_stream(
        self, audio_model_rate: np.ndarray, strategy: DecodeStrategy | None = None,
    ) -> Generator:
        """Prepare inputs + generate_stream. Runs in calling thread (use via executor).

        Returns generator of (delta, is_done, output). Routes call this
        through run_in_executor — no raw thread creation needed.
        """
        self._ensure_started()
        inputs = self._model.prepare_inputs(audio_model_rate, "")
        strat = strategy or self.create_strategy()
        return self._model.generate_stream(inputs, strat)

    # ==================== Public Strategy (fix #1: no more _create_strategy) ====================

    def create_strategy(self) -> DecodeStrategy | None:
        """Create decode strategy with model's real EOS token.

        Fix #6: no try/except — if model is loaded, eos_token_id must work.
        """
        self._ensure_started()
        return GreedyWithEarlyStopping(
            eos_token_id=self._model.eos_token_id,
            repetition_window=self._config.streaming.repetition_window,
        )

    # ==================== Sessions ====================

    async def create_session(self, session_id: str = "") -> None:
        self._ensure_started()
        if not session_id:
            import uuid
            session_id = str(uuid.uuid4())
        self._sessions[session_id] = StreamingSession(
            session_id=session_id, model=self._model,
            preprocessor=self._preprocessor, strategy=self.create_strategy(),
            config=self._config, executor=self._executor,
        )

    async def push_audio(self, session_id: str, pcm_chunk: bytes) -> str:
        return await self._get_session(session_id).push_audio(pcm_chunk)

    async def finish_session(self, session_id: str) -> str:
        session = self._sessions.pop(session_id, None)
        if session is None:
            raise RuntimeError(f"Session not found: {session_id}")
        return await session.finish()

    async def has_session(self, session_id: str) -> bool:
        return session_id in self._sessions

    def session_exists(self, session_id: str) -> bool:
        """Sync version for scheduler lookup."""
        return session_id in self._sessions

    # ==================== Internal ====================

    async def _apply_compile(self) -> float:
        """Fix #4: delegate compilation to model — no feature envy."""
        module = self._model.compilable_module()
        if module is None:
            return 0.0
        try:
            import torch
            t0 = _time.perf_counter()
            compiled = torch.compile(module, mode="default")
            # Ask model to replace its internal module
            if hasattr(self._model, 'apply_compiled_module'):
                self._model.apply_compiled_module(compiled)
            ms = (_time.perf_counter() - t0) * 1000
            logger.info("torch.compile applied: %.0fms", ms)
            return ms
        except Exception as e:
            logger.warning("torch.compile failed: %s", e)
            return 0.0

    async def _cancel_session(self, sid: str) -> None:
        session = self._sessions.pop(sid, None)
        if session:
            await session.cancel()

    def _ensure_started(self) -> None:
        if not self._started:
            raise RuntimeError("ASREngine not started.")

    def _get_session(self, sid: str) -> StreamingSession:
        session = self._sessions.get(sid)
        if session is None:
            raise RuntimeError(f"Session '{sid}' not found.")
        return session
