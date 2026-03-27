"""Whisper preprocessor — audio to model-ready tensors.

SRP: Converts audio to Whisper's expected input format (log-mel spectrogram).
Whisper processor handles mel computation internally via WhisperFeatureExtractor.
"""

from __future__ import annotations

import logging
import time as _time
from typing import Any

import numpy as np

from macaw_asr.models.contracts import IPreprocessor
from macaw_asr.models.types import InputsWrapper

logger = logging.getLogger("macaw-asr.models.whisper.preprocessor")

WHISPER_SAMPLE_RATE = 16000


class WhisperPreprocessor(IPreprocessor):
    """Converts audio to Whisper input tensors via WhisperProcessor.

    Whisper expects 16kHz audio. The processor computes log-mel spectrogram.
    """

    def __init__(self, model, processor, device: str, language: str = "", dtype: str = "float16") -> None:
        self._model = model
        self._processor = processor
        self._device = device
        self._language = language
        self._dtype = dtype

    def prepare_inputs(self, audio: np.ndarray, prefix: str = "") -> Any:
        """Convert float32 audio (16kHz) to Whisper input features."""
        import torch

        t0 = _time.perf_counter()

        import torch
        dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
        torch_dtype = dtype_map.get(self._dtype, torch.float16)

        input_features = self._processor(
            audio, sampling_rate=WHISPER_SAMPLE_RATE, return_tensors="pt"
        ).input_features.to(device=self._device, dtype=torch_dtype)

        # Build forced decoder IDs for language
        forced_decoder_ids = None
        if self._language:
            forced_decoder_ids = self._processor.get_decoder_prompt_ids(
                language=self._language, task="transcribe"
            )

        result = InputsWrapper({
            "input_features": input_features,
            "forced_decoder_ids": forced_decoder_ids,
        })
        result._prepare_ms = (_time.perf_counter() - t0) * 1000
        return result

    def fast_finish_inputs(self, audio, cached_inputs, cached_audio_len, prefix=""):
        """Whisper doesn't support fast_finish (no incremental mel)."""
        return None
