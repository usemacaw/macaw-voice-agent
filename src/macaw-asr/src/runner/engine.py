"""ASR Engine — inference orchestrator.

Equivalent to Ollama's runner dispatcher: owns model lifecycle
and delegates per-request work to StreamingSession instances.

Usage (batch):
    engine = ASREngine(config)
    await engine.start()
    text = await engine.transcribe(pcm_bytes)
    await engine.stop()

Usage (streaming):
    await engine.create_session("s1")
    for chunk in audio_chunks:
        partial = await engine.push_audio("s1", chunk)
    final = await engine.finish_session("s1")
"""

from __future__ import annotations

import logging
import time as _time
from concurrent.futures import ThreadPoolExecutor

from macaw_asr._executor import create_executor, run_in_executor
from macaw_asr.audio.preprocessing import AudioPreprocessor
from macaw_asr.config import EngineConfig
from macaw_asr.decode.postprocess import clean_asr_text
from macaw_asr.decode.strategies import GreedyWithEarlyStopping
from macaw_asr.models.base import ASRModel, create_model
from macaw_asr.runner.session import StreamingSession

logger = logging.getLogger("macaw-asr.runner.engine")


class ASREngine:
    """Main inference orchestrator.

    Responsibilities (SRP):
    - Model lifecycle (load/unload)
    - Session management (create/finish/cancel)
    - Batch transcription

    Does NOT contain streaming logic — that's in StreamingSession.
    """

    def __init__(self, config: EngineConfig) -> None:
        self._config = config
        self._model: ASRModel | None = None
        self._preprocessor = AudioPreprocessor(config.audio)
        self._executor: ThreadPoolExecutor | None = None
        self._sessions: dict[str, StreamingSession] = {}
        self._started = False

    @property
    def config(self) -> EngineConfig:
        return self._config

    @property
    def is_started(self) -> bool:
        return self._started

    async def start(self) -> None:
        """Load model and prepare for inference."""
        if self._started:
            return

        self._executor = create_executor(self._config.max_inference_workers)
        self._model = create_model(self._config.model_name)

        logger.info(
            "Loading model: name=%s, id=%s, device=%s",
            self._config.model_name,
            self._config.model_id,
            self._config.device,
        )

        await run_in_executor(self._executor, self._model.load, self._config)
        await run_in_executor(self._executor, self._model.warmup)

        self._started = True
        logger.info("ASREngine started: %s", self._config.model_name)

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
        """Push an audio chunk to a streaming session."""
        session = self._get_session(session_id)
        return await session.push_audio(pcm_chunk)

    async def finish_session(self, session_id: str) -> str:
        """Finish a streaming session and get final transcription."""
        session = self._sessions.pop(session_id, None)
        if session is None:
            raise RuntimeError(f"Session not found: {session_id}")
        return await session.finish()

    async def has_session(self, session_id: str) -> bool:
        return session_id in self._sessions

    # ==================== Internal ====================

    async def _cancel_session(self, session_id: str) -> None:
        session = self._sessions.pop(session_id, None)
        if session is not None:
            await session.cancel()

    def _create_strategy(self) -> GreedyWithEarlyStopping:
        return GreedyWithEarlyStopping(
            eos_token_id=self._model.eos_token_id,
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
