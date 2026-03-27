"""Parakeet preprocessor — audio to NeMo-ready format.

SRP: Converts audio to format expected by NeMo ASRModel.transcribe().
Parakeet expects 16kHz float32 mono audio — NeMo handles mel internally.
"""

from __future__ import annotations

import logging
import time as _time
from typing import Any

import numpy as np

from macaw_asr.models.contracts import IPreprocessor
from macaw_asr.models.types import InputsWrapper

logger = logging.getLogger("macaw-asr.models.parakeet.preprocessor")

PARAKEET_SAMPLE_RATE = 16000


class ParakeetPreprocessor(IPreprocessor):
    """Converts audio to Parakeet input format.

    NeMo's transcribe() accepts raw float32 arrays at 16kHz.
    No mel spectrogram needed — NeMo handles feature extraction internally.
    """

    def __init__(self, model, language: str = "") -> None:
        self._model = model
        self._language = language

    def prepare_inputs(self, audio: np.ndarray, prefix: str = "") -> Any:
        """Wrap audio for NeMo transcribe(). Minimal preprocessing."""
        t0 = _time.perf_counter()
        result = InputsWrapper({"audio": audio, "language": self._language or None})
        result._prepare_ms = (_time.perf_counter() - t0) * 1000
        return result

    def fast_finish_inputs(self, audio, cached_inputs, cached_audio_len, prefix=""):
        """Parakeet doesn't support fast_finish."""
        return None
