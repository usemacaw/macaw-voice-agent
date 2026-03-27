"""Parakeet ASR model — Facade composing loader, preprocessor, decoder.

NVIDIA Parakeet TDT v3: 600M params, 25 languages including PT-BR.
Top 1 on Open ASR Leaderboard (2025).

Architecture: FastConformer encoder + TDT decoder.
No manual decode loop, no DecodeStrategy, no fast_finish.
NeMo handles everything internally via model.transcribe().
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
from macaw_asr.models.parakeet.decoder import ParakeetDecoder
from macaw_asr.models.parakeet.loader import ParakeetModelLoader
from macaw_asr.models.parakeet.preprocessor import ParakeetPreprocessor

logger = logging.getLogger("macaw-asr.models.parakeet")


class ParakeetASRModel(IASRModel):
    """NVIDIA Parakeet TDT/CTC/RNNT ASR model.

    Batch-only (no token-by-token streaming).
    No DecodeStrategy (NeMo handles decoding).
    No fast_finish (no incremental mel).
    """

    def __init__(self) -> None:
        self._loader = ParakeetModelLoader()
        self._preprocessor: ParakeetPreprocessor | None = None
        self._decoder: ParakeetDecoder | None = None
        self._config: EngineConfig | None = None

    # ==================== Properties ====================

    @property
    def eos_token_id(self) -> int:
        return self._loader.eos_id

    @property
    def supports_streaming(self) -> bool:
        return False  # Batch-only

    @property
    def supports_cuda_graphs(self) -> bool:
        return False

    def compilable_module(self) -> Any:
        return None  # NeMo models not compatible with torch.compile

    # ==================== Lifecycle ====================

    def load(self, config: EngineConfig) -> None:
        self._config = config
        self._loader.load(config)
        self._preprocessor = ParakeetPreprocessor(self._loader.model, config.language)
        self._decoder = ParakeetDecoder(self._loader.model)

    def unload(self) -> None:
        self._decoder = None
        self._preprocessor = None
        self._loader.unload()

    def warmup(self, config: EngineConfig | None = None) -> None:
        if not self._loader.is_loaded():
            return
        t0 = _time.perf_counter()
        shapes_ok = 0
        for dur in [1.0, 3.0]:
            try:
                dummy = np.zeros(int(dur * 16000), dtype=np.float32)
                inputs = self.prepare_inputs(dummy)
                self.generate(inputs)
                shapes_ok += 1
            except Exception as e:
                logger.warning("Warmup failed for %.1fs shape: %s", dur, e)
        logger.info("Parakeet warmup: %.0fms (%d/2 shapes)", (_time.perf_counter() - t0) * 1000, shapes_ok)

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
        return None
