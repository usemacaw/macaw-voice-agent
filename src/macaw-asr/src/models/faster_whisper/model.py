"""Faster-Whisper ASR model — Facade composing loader, preprocessor, decoder.

Implements IASRModel contract using CTranslate2 runtime.
No PyTorch dependency. 4x faster than HF Whisper.
"""

from __future__ import annotations

import logging
import time as _time
from typing import Any, Generator

import numpy as np

from macaw_asr.config import EngineConfig
from macaw_asr.decode.strategies import DecodeStrategy
from macaw_asr.models.contracts import IASRModel
from macaw_asr.models.types import ModelOutput
from macaw_asr.models.faster_whisper.decoder import FasterWhisperDecoder
from macaw_asr.models.faster_whisper.loader import FasterWhisperModelLoader
from macaw_asr.models.faster_whisper.preprocessor import FasterWhisperPreprocessor

logger = logging.getLogger("macaw-asr.models.faster_whisper")

# Whisper EOS token (constant across all model sizes).
# Required by IASRModel contract but never used — faster-whisper
# handles stopping internally and DecodeStrategy is ignored.
_WHISPER_EOS_TOKEN_ID = 50257


class FasterWhisperASRModel(IASRModel):
    """Whisper via CTranslate2: fast, lightweight, no PyTorch.

    Variants: tiny (39M), small (244M), medium (769M), large-v3 (1.5B).
    Composed of FasterWhisperModelLoader + FasterWhisperPreprocessor + FasterWhisperDecoder.
    """

    def __init__(self) -> None:
        self._loader = FasterWhisperModelLoader()
        self._preprocessor: FasterWhisperPreprocessor | None = None
        self._decoder: FasterWhisperDecoder | None = None

    # ==================== Properties ====================

    @property
    def eos_token_id(self) -> int:
        return _WHISPER_EOS_TOKEN_ID

    @property
    def supports_streaming(self) -> bool:
        return False

    @property
    def supports_cuda_graphs(self) -> bool:
        return False

    # ==================== Lifecycle ====================

    def load(self, config: EngineConfig) -> None:
        self._loader.load(config)
        self._preprocessor = FasterWhisperPreprocessor(config.language)
        self._decoder = FasterWhisperDecoder(self._loader.model)

    def unload(self) -> None:
        self._decoder = None
        self._preprocessor = None
        self._loader.unload()

    # ==================== Inference ====================

    def prepare_inputs(self, audio: np.ndarray, prefix: str = "") -> Any:
        return self._preprocessor.prepare_inputs(audio, prefix)

    def generate(self, inputs: Any, strategy: DecodeStrategy | None = None) -> ModelOutput:
        return self._decoder.decode(inputs, strategy)

    def generate_stream(
        self, inputs: Any, strategy: DecodeStrategy | None = None,
    ) -> Generator[tuple[str, bool, ModelOutput | None], None, None]:
        return self._decoder.decode_stream(inputs, strategy)

    def warmup(self, config: EngineConfig | None = None) -> None:
        if not self._loader.is_loaded():
            return
        t0 = _time.perf_counter()
        for dur in [1.0, 3.0]:
            dummy = np.zeros(int(dur * 16000), dtype=np.float32)
            inputs = self.prepare_inputs(dummy)
            self.generate(inputs)
        logger.info("Faster-Whisper warmup: %.0fms", (_time.perf_counter() - t0) * 1000)
