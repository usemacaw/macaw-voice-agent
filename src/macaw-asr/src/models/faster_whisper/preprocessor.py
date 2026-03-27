"""Faster-Whisper preprocessor — passthrough.

SRP: Packages raw audio for the decoder.
faster-whisper handles mel extraction internally in transcribe(),
so this is a thin wrapper that satisfies the IPreprocessor contract.
"""

from __future__ import annotations

import time as _time
from typing import Any

import numpy as np

from macaw_asr.models.contracts import IPreprocessor
from macaw_asr.models.types import InputsWrapper


class FasterWhisperPreprocessor(IPreprocessor):
    """Passthrough preprocessor — faster-whisper does its own mel."""

    def __init__(self, language: str) -> None:
        self._language = language

    def prepare_inputs(self, audio: np.ndarray, prefix: str = "") -> Any:
        t0 = _time.perf_counter()
        result = InputsWrapper({
            "audio": audio,
            "language": self._language or None,
        })
        result._prepare_ms = (_time.perf_counter() - t0) * 1000
        return result
