"""Mock ASR model — implements full IASRModel contract without GPU."""

from __future__ import annotations

from typing import Any, Generator

import numpy as np

from macaw_asr.config import EngineConfig
from macaw_asr.decode.strategies import DecodeStrategy
from macaw_asr.models.contracts import IASRModel
from macaw_asr.models.types import (
    TIMING_DECODE_MS, TIMING_DECODE_PER_TOKEN_MS,
    TIMING_PREFILL_MS, TIMING_PREPARE_MS, TIMING_TOTAL_MS,
    ModelOutput,
)


class MockASRModel(IASRModel):
    """Returns fixed text. Implements full contract for testing."""

    def __init__(self, fixed_text: str = "Texto de teste do mock ASR.") -> None:
        self._fixed_text = fixed_text
        self._loaded = False

    @property
    def eos_token_id(self) -> int:
        return 0

    @property
    def supports_streaming(self) -> bool:
        return True

    @property
    def supports_cuda_graphs(self) -> bool:
        return False

    def load(self, config: EngineConfig) -> None:
        self._loaded = True

    def unload(self) -> None:
        self._loaded = False

    def prepare_inputs(self, audio: np.ndarray, prefix: str = "") -> Any:
        return {"audio_len": len(audio), "prefix": prefix}

    def generate(self, inputs: Any, strategy: DecodeStrategy | None = None) -> ModelOutput:
        if not self._loaded:
            raise RuntimeError("Model not loaded")
        n = len(self._fixed_text.split())
        return ModelOutput(
            text=self._fixed_text, raw_text=self._fixed_text, n_tokens=n,
            timings={TIMING_PREPARE_MS: 0, TIMING_PREFILL_MS: 0, TIMING_DECODE_MS: 0,
                     TIMING_DECODE_PER_TOKEN_MS: 0, TIMING_TOTAL_MS: 0},
        )

    def generate_stream(
        self, inputs: Any, strategy: DecodeStrategy | None = None,
    ) -> Generator[tuple[str, bool, ModelOutput | None], None, None]:
        output = self.generate(inputs, strategy)
        yield output.text, True, output

    def fast_finish_inputs(self, audio, cached_inputs, cached_audio_len, prefix=""):
        return {"audio_len": len(audio), "prefix": prefix, "fast": True}
