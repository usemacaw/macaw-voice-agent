"""
Semantic end-of-turn detection using Pipecat Smart Turn v3.2.

Analyzes raw audio waveform (prosody, intonation) to determine if
the speaker has finished their conversational turn. Runs on CPU
via ONNX Runtime in ~12ms.

Used as a second-stage filter after Silero VAD detects silence:
  Silero VAD (acoustic) → Smart Turn (semantic) → respond or wait
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import numpy as np
import onnxruntime as ort

from common.audio_utils import resample as _resample_audio

logger = logging.getLogger("open-voice-api.smart-turn")

_MODEL_DIR = Path(__file__).parent / "models"
_DEFAULT_MODEL = "smart-turn-v3.2-cpu.onnx"
_SMART_TURN_SAMPLE_RATE = 16000
_MAX_AUDIO_SECONDS = 8
_DEFAULT_THRESHOLD = 0.5


class SmartTurnDetector:
    """Semantic end-of-turn detector using Pipecat Smart Turn v3.2 ONNX model."""

    def __init__(self, threshold: float = _DEFAULT_THRESHOLD):
        model_path = _MODEL_DIR / _DEFAULT_MODEL
        if not model_path.exists():
            raise FileNotFoundError(
                f"Smart Turn model not found at {model_path}. "
                f"Download from https://huggingface.co/pipecat-ai/smart-turn-v3"
            )

        self._threshold = threshold

        # Load ONNX session (CPU optimized)
        so = ort.SessionOptions()
        so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        so.inter_op_num_threads = 1
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self._session = ort.InferenceSession(str(model_path), sess_options=so)

        # Load Whisper feature extractor (lazy import to avoid startup cost)
        from transformers import WhisperFeatureExtractor
        self._feature_extractor = WhisperFeatureExtractor(chunk_length=_MAX_AUDIO_SECONDS)

        logger.info(
            f"Smart Turn detector initialized: model={_DEFAULT_MODEL}, "
            f"threshold={self._threshold}"
        )

    def predict(self, speech_audio: bytes, source_sample_rate: int = 8000) -> tuple[bool, float]:
        """Predict whether the speaker has finished their turn.

        Args:
            speech_audio: PCM 16-bit mono audio bytes of the speech segment.
            source_sample_rate: Sample rate of the input audio (default 8kHz).

        Returns:
            (is_complete, probability) where is_complete is True if the
            speaker likely finished their turn.
        """
        # Convert PCM16 bytes to float32 [-1, 1]
        samples = np.frombuffer(speech_audio, dtype=np.int16).astype(np.float32) / 32768.0

        # Resample to 16kHz if needed
        if source_sample_rate != _SMART_TURN_SAMPLE_RATE:
            samples = _resample_audio(samples, source_sample_rate, _SMART_TURN_SAMPLE_RATE)
            if len(samples) == 0:
                return True, 1.0  # Empty audio = turn complete

        # Truncate/pad to last N seconds
        samples = self._truncate_to_last_n_seconds(samples)

        # Extract Whisper features
        inputs = self._feature_extractor(
            samples,
            sampling_rate=_SMART_TURN_SAMPLE_RATE,
            return_tensors="np",
            padding="max_length",
            max_length=_MAX_AUDIO_SECONDS * _SMART_TURN_SAMPLE_RATE,
            truncation=True,
            do_normalize=True,
        )
        input_features = inputs.input_features.squeeze(0).astype(np.float32)
        input_features = np.expand_dims(input_features, axis=0)

        # Run inference
        outputs = self._session.run(None, {"input_features": input_features})
        probability = float(outputs[0][0].item())
        is_complete = probability > self._threshold

        return is_complete, probability

    @staticmethod
    def _truncate_to_last_n_seconds(
        samples: np.ndarray,
        n_seconds: int = _MAX_AUDIO_SECONDS,
    ) -> np.ndarray:
        max_samples = n_seconds * _SMART_TURN_SAMPLE_RATE
        if len(samples) > max_samples:
            return samples[-max_samples:]
        elif len(samples) < max_samples:
            padding = max_samples - len(samples)
            return np.pad(samples, (padding, 0), mode="constant", constant_values=0)
        return samples
