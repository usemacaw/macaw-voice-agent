"""ASR Engine — inference orchestrator.

Equivalent to Ollama's runner dispatcher: owns model lifecycle
and delegates per-request work to StreamingSession instances.

M1: Warmup with config passed to model
M2: Strategy creation aware of model type (strategy=None for non-autoregressive)
M3: torch.compile applied generically via compilable_module()
M5: Structured startup metrics (load_ms, compile_ms, warmup_ms)
"""

from __future__ import annotations

import logging
import time as _time
from concurrent.futures import ThreadPoolExecutor

from macaw_asr._executor import create_executor, run_in_executor
from macaw_asr.audio.preprocessing import AudioPreprocessor
from macaw_asr.config import EngineConfig
from macaw_asr.decode.postprocess import clean_asr_text
from macaw_asr.decode.strategies import DecodeStrategy, GreedyWithEarlyStopping
from macaw_asr.models.base import ASRModel, create_model
from macaw_asr.runner.session import StreamingSession

logger = logging.getLogger("macaw-asr.runner.engine")


class ASREngine:
    """Main inference orchestrator.

    Responsibilities (SRP):
    - Model lifecycle (load/compile/warmup/unload)
    - Session management (create/finish/cancel)
    - Batch transcription
    """

    def __init__(self, config: EngineConfig) -> None:
        self._config = config
        self._model: ASRModel | None = None
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
        """Load model, optionally compile, warmup, prepare for inference."""
        if self._started:
            return

        t_total = _time.perf_counter()
        self._executor = create_executor(self._config.max_inference_workers)
        self._model = create_model(self._config.model_name)

        # Load
        t0 = _time.perf_counter()
        await run_in_executor(self._executor, self._model.load, self._config)
        load_ms = (_time.perf_counter() - t0) * 1000

        # Compile (M3 — generic torch.compile via compilable_module)
        compile_ms = 0.0
        if self._config.enable_compile:
            compile_ms = await self._apply_compile()

        # Warmup (M1 — passes config to model for multi-shape warmup)
        t0 = _time.perf_counter()
        await run_in_executor(self._executor, self._model.warmup, self._config)
        warmup_ms = (_time.perf_counter() - t0) * 1000

        total_ms = (_time.perf_counter() - t_total) * 1000

        self._startup_timings = {
            "load_ms": load_ms,
            "compile_ms": compile_ms,
            "warmup_ms": warmup_ms,
            "total_ms": total_ms,
        }

        self._started = True
        logger.info(
            "ASREngine started: model=%s load=%.0fms compile=%.0fms "
            "warmup=%.0fms total=%.0fms",
            self._config.model_name, load_ms, compile_ms, warmup_ms, total_ms,
        )

    async def stop(self) -> None:
        """Unload model and free resources."""
        if not self._started:
            return

        for session_id in list(self._sessions.keys()):
            try:
                await self._cancel_session(session_id)
            except Exception as e:
                logger.warning("Error cancelling session %s: %s", session_id, e)

        if self._model is not None:
            await run_in_executor(self._executor, self._model.unload)
            self._model = None

        if self._executor is not None:
            self._executor.shutdown(wait=False)
            self._executor = None

        self._started = False
        logger.info("ASREngine stopped")

    # ==================== Batch ====================

    async def transcribe(self, pcm_data: bytes) -> str:
        """Transcribe a complete audio segment."""
        self._ensure_started()
        if not pcm_data:
            return ""

        t_total = _time.perf_counter()

        t0 = _time.perf_counter()
        float_audio = self._preprocessor.process(pcm_data)
        resample_ms = (_time.perf_counter() - t0) * 1000

        t0 = _time.perf_counter()
        inputs = await run_in_executor(
            self._executor, self._model.prepare_inputs, float_audio, ""
        )
        proc_ms = (_time.perf_counter() - t0) * 1000

        strategy = self._create_strategy()
        t0 = _time.perf_counter()
        output = await run_in_executor(
            self._executor, self._model.generate, inputs, strategy
        )
        gen_ms = (_time.perf_counter() - t0) * 1000

        text = clean_asr_text(output.text)
        total_ms = (_time.perf_counter() - t_total) * 1000

        logger.info(
            "Batch: '%s' | resample=%.0fms proc=%.0fms gen=%.0fms "
            "(pf=%.0fms %dtok=%.0fms) total=%.0fms",
            text, resample_ms, proc_ms, gen_ms,
            output.timings.get("prefill_ms", 0),
            output.n_tokens,
            output.timings.get("decode_ms", 0),
            total_ms,
        )
        return text

    # ==================== Streaming ====================

    async def create_session(self, session_id: str = "") -> None:
        """Start a streaming session."""
        self._ensure_started()
        self._sessions[session_id] = StreamingSession(
            session_id=session_id,
            model=self._model,
            preprocessor=self._preprocessor,
            strategy=self._create_strategy(),
            config=self._config,
            executor=self._executor,
        )
        logger.debug("Session created: %s", session_id[:8] or "default")

    async def push_audio(self, session_id: str, pcm_chunk: bytes) -> str:
        session = self._get_session(session_id)
        return await session.push_audio(pcm_chunk)

    async def finish_session(self, session_id: str) -> str:
        session = self._sessions.pop(session_id, None)
        if session is None:
            raise RuntimeError(f"Session not found: {session_id}")
        return await session.finish()

    async def has_session(self, session_id: str) -> bool:
        return session_id in self._sessions

    # ==================== Internal ====================

    async def _apply_compile(self) -> float:
        """Apply torch.compile to the model's compilable module (M3)."""
        module = self._model.compilable_module()
        if module is None:
            logger.info("Model does not support torch.compile — skipping")
            return 0.0

        try:
            import torch
            t0 = _time.perf_counter()
            # 'default' mode: kernel fusion without CUDA graph capture.
            # 'reduce-overhead' uses CUDA graphs which fail with dynamic
            # KV cache shapes in autoregressive models (PyTorch 2.5.x bug).
            compiled = torch.compile(module, mode="default")
            # Replace the module reference in the model
            # This is model-specific — for Qwen it's _thinker
            if hasattr(self._model, "_thinker") and self._model._thinker is module:
                self._model._thinker = compiled
            compile_ms = (_time.perf_counter() - t0) * 1000
            logger.info("torch.compile applied: %.0fms", compile_ms)
            return compile_ms
        except Exception as e:
            logger.warning("torch.compile failed: %s — continuing without", e)
            return 0.0

    async def _cancel_session(self, session_id: str) -> None:
        session = self._sessions.pop(session_id, None)
        if session is not None:
            await session.cancel()

    def _create_strategy(self) -> DecodeStrategy | None:
        """Create decode strategy based on model capabilities (M2)."""
        try:
            eos = self._model.eos_token_id
        except (RuntimeError, AttributeError):
            return None

        return GreedyWithEarlyStopping(
            eos_token_id=eos,
            repetition_window=self._config.streaming.repetition_window,
        )

    def _ensure_started(self) -> None:
        if not self._started:
            raise RuntimeError(
                "ASREngine not started. Call await engine.start() first."
            )

    def _get_session(self, session_id: str) -> StreamingSession:
        session = self._sessions.get(session_id)
        if session is None:
            raise RuntimeError(
                f"Session '{session_id}' not found. "
                "Call await engine.create_session() first."
            )
        return session
