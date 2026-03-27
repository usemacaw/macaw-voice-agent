"""Whisper ASR model — Facade composing loader, preprocessor, decoder.

Implements IASRModel contract. Encoder-decoder model:
- No manual decode loop (uses HF generate)
- No token-by-token streaming (batch only)
- No fast_finish_inputs (no incremental mel)
- No CUDA graphs (not autoregressive)
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
from macaw_asr.models.whisper.decoder import WhisperDecoder
from macaw_asr.models.whisper.loader import WhisperModelLoader
from macaw_asr.models.whisper.preprocessor import WhisperPreprocessor

logger = logging.getLogger("macaw-asr.models.whisper")


class WhisperASRModel(IASRModel):
    """OpenAI Whisper: encoder-decoder ASR model.

    Variants: tiny (39M), small (244M), medium (769M), large-v3 (1.5B).
    Composed of WhisperModelLoader + WhisperPreprocessor + WhisperDecoder.
    """

    def __init__(self) -> None:
        self._loader = WhisperModelLoader()
        self._preprocessor: WhisperPreprocessor | None = None
        self._decoder: WhisperDecoder | None = None
        self._config: EngineConfig | None = None

    # ==================== Properties ====================

    @property
    def eos_token_id(self) -> int:
        if self._loader.eos_id is None:
            raise RuntimeError("Model not loaded")
        return self._loader.eos_id

    @property
    def supports_streaming(self) -> bool:
        return False  # Whisper is batch-only

    @property
    def supports_cuda_graphs(self) -> bool:
        return False  # Not autoregressive (uses HF generate)

    def compilable_module(self) -> Any:
        return self._loader.model  # Can torch.compile the encoder

    # ==================== Lifecycle ====================

    def load(self, config: EngineConfig) -> None:
        self._config = config
        self._loader.load(config)
        self._preprocessor = WhisperPreprocessor(
            self._loader.model, self._loader.processor,
            config.device, config.language, config.dtype,
        )
        self._decoder = WhisperDecoder(
            self._loader.model, self._loader.processor, config.device,
        )

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

    def fast_finish_inputs(self, audio, cached_inputs, cached_audio_len, prefix=""):
        return None  # Whisper doesn't support fast finish

    def warmup(self, config: EngineConfig | None = None) -> None:
        if not self._loader.is_loaded():
            return
        t0 = _time.perf_counter()
        for dur in [1.0, 3.0]:
            dummy = np.zeros(int(dur * 16000), dtype=np.float32)
            inputs = self.prepare_inputs(dummy)
            self.generate(inputs)
        logger.info("Whisper warmup: %.0fms", (_time.perf_counter() - t0) * 1000)
