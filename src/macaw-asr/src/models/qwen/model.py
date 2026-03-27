"""Qwen3-ASR model — Facade composing loader, preprocessor, decoder.

Facade Pattern: single entry point for engine, delegates to specialized components.
Builder Pattern: __init__ is lightweight, load() builds all internal components.
Composition over Inheritance: uses IModelLoader, IPreprocessor, IDecoder, IStreamDecoder.
"""

from __future__ import annotations

import logging
import time as _time
from typing import Any, Generator

import numpy as np

from macaw_asr.config import EngineConfig
from macaw_asr.decode.strategies import DecodeStrategy, GreedyWithEarlyStopping
from macaw_asr.models.contracts import IASRModel
from macaw_asr.models.types import ModelOutput

from macaw_asr.models.qwen.decoder import QwenDecoder
from macaw_asr.models.qwen.loader import QwenModelLoader
from macaw_asr.models.qwen.preprocessor import QwenPreprocessor
from macaw_asr.models.qwen.prompt import QwenPromptBuilder

logger = logging.getLogger("macaw-asr.models.qwen")


class QwenASRModel(IASRModel):
    """Qwen3-ASR: autoregressive model with streaming, fast_finish, torch.compile.

    Composed of:
        QwenModelLoader   — weight lifecycle (SRP)
        QwenPreprocessor  — audio → tensors (SRP)
        QwenDecoder       — tensors → text, batch + stream (SRP, DRY)
        QwenPromptBuilder — chat template construction (SRP)
    """

    def __init__(self) -> None:
        self._loader = QwenModelLoader()
        self._preprocessor: QwenPreprocessor | None = None
        self._decoder: QwenDecoder | None = None
        self._config: EngineConfig | None = None

    # ==================== Properties (Contract) ====================

    @property
    def eos_token_id(self) -> int:
        if self._loader.eos_id is None:
            raise RuntimeError("Model not loaded")
        return self._loader.eos_id

    @property
    def supports_streaming(self) -> bool:
        return True

    @property
    def supports_cuda_graphs(self) -> bool:
        return True

    def compilable_module(self) -> Any:
        return self._loader.thinker

    def apply_compiled_module(self, compiled_module: Any) -> None:
        """Replace thinker with compiled version. Fix #4: no external feature envy."""
        self._loader.thinker = compiled_module
        # Rebuild decoder with compiled thinker
        if self._decoder and self._config:
            self._decoder = QwenDecoder(
                compiled_module, self._loader.processor,
                self._loader.eos_id, self._config.streaming.max_new_tokens,
            )

    # ==================== Lifecycle (Contract) ====================

    def load(self, config: EngineConfig) -> None:
        """Builder Pattern: load() builds all internal components."""
        self._config = config
        self._loader.load(config)

        prompt_builder = QwenPromptBuilder(self._loader.processor, config.language_name)
        self._preprocessor = QwenPreprocessor(self._loader.thinker, self._loader.processor, prompt_builder)
        self._decoder = QwenDecoder(
            self._loader.thinker, self._loader.processor,
            self._loader.eos_id, config.streaming.max_new_tokens,
        )

    def unload(self) -> None:
        self._decoder = None
        self._preprocessor = None
        self._loader.unload()

    # ==================== Warmup (Contract) ====================

    def warmup(self, config: EngineConfig | None = None) -> None:
        if not self._loader.is_loaded():
            return

        t0 = _time.perf_counter()
        shapes_ok = 0

        for dur in [0.5, 1.0, 3.0]:
            try:
                dummy = np.zeros(int(dur * 16000), dtype=np.float32)
                inputs = self.prepare_inputs(dummy)
                self.generate(inputs)
                shapes_ok += 1
            except Exception as e:
                logger.warning("Warmup failed for %.1fs shape: %s", dur, e)

        # Warmup fast_finish path
        try:
            dummy_full = np.zeros(32000, dtype=np.float32)
            dummy_partial = np.zeros(16000, dtype=np.float32)
            cached = self.prepare_inputs(dummy_partial)
            fast = self.fast_finish_inputs(dummy_full, cached, len(dummy_partial))
            if fast is not None:
                self.generate(fast)
        except Exception as e:
            logger.warning("Warmup fast_finish failed: %s", e)

        logger.info("Qwen3-ASR warmup: %.0fms (%d/3 shapes)", (_time.perf_counter() - t0) * 1000, shapes_ok)

    # ==================== Inference (Contract) ====================

    def prepare_inputs(self, audio: np.ndarray, prefix: str = "") -> Any:
        return self._preprocessor.prepare_inputs(audio, prefix)

    def generate(self, inputs: Any, strategy: DecodeStrategy | None = None) -> ModelOutput:
        return self._decoder.decode(inputs, strategy)

    def generate_stream(
        self, inputs: Any, strategy: DecodeStrategy | None = None,
    ) -> Generator[tuple[str, bool, ModelOutput | None], None, None]:
        return self._decoder.decode_stream(inputs, strategy)

    def fast_finish_inputs(
        self, audio: np.ndarray, cached_inputs: Any,
        cached_audio_len: int, prefix: str = "",
    ) -> Any | None:
        return self._preprocessor.fast_finish_inputs(audio, cached_inputs, cached_audio_len, prefix)
