"""Audio preprocessing: PCM conversion and resampling.

Pure functions for audio format conversion. No model dependencies.
Uses scipy for quality resampling when available, numpy fallback otherwise.
"""

from __future__ import annotations

import numpy as np

from macaw_asr.config import AudioConfig

try:
    from scipy.signal import resample_poly as _scipy_resample_poly

    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False


def pcm_to_float32(pcm_data: bytes) -> np.ndarray:
    """PCM 16-bit signed little-endian -> numpy float32 normalized [-1, 1].

    Raises:
        ValueError: If pcm_data has odd byte count (invalid for 16-bit).
    """
    if len(pcm_data) % 2 != 0:
        raise ValueError(
            f"PCM 16-bit requires even byte count, got {len(pcm_data)}"
        )
    samples = np.frombuffer(pcm_data, dtype=np.int16)
    return samples.astype(np.float32) / 32768.0


def float32_to_pcm(samples: np.ndarray) -> bytes:
    """numpy float32 [-1, 1] -> PCM 16-bit signed little-endian bytes.

    Values outside [-1, 1] are clipped to prevent overflow.
    """
    clipped = np.clip(samples, -1.0, 1.0)
    int_samples = (clipped * 32767.0).astype(np.int16)
    return int_samples.tobytes()


def resample(samples: np.ndarray, from_rate: int, to_rate: int) -> np.ndarray:
    """Resample audio with anti-aliasing when scipy is available.

    Uses scipy.signal.resample_poly (polyphase filter) when available,
    falls back to linear interpolation via numpy.
    """
    if from_rate == to_rate:
        return samples
    if len(samples) == 0:
        return np.array([], dtype=samples.dtype)

    if _HAS_SCIPY:
        from math import gcd

        g = gcd(from_rate, to_rate)
        up = to_rate // g
        down = from_rate // g
        return _scipy_resample_poly(samples, up, down).astype(samples.dtype)

    # Fallback: linear interpolation (no anti-aliasing)
    duration = len(samples) / from_rate
    target_length = int(duration * to_rate)
    if target_length == 0:
        return np.array([], dtype=samples.dtype)

    indices = np.linspace(0, len(samples) - 1, target_length)
    return np.interp(indices, np.arange(len(samples)), samples).astype(
        samples.dtype
    )


class AudioPreprocessor:
    """Stateless audio preprocessor. Converts raw PCM to model-ready float32.

    Encapsulates the PCM→float32→resample pipeline that every ASR model needs.
    Eliminates duplication across model implementations.
    """

    def __init__(self, config: AudioConfig) -> None:
        self._input_rate = config.input_sample_rate
        self._model_rate = config.model_sample_rate

    def process(self, pcm_data: bytes) -> np.ndarray:
        """Convert PCM bytes to resampled float32 array at model sample rate.

        Args:
            pcm_data: Raw PCM 16-bit audio bytes at input_sample_rate.

        Returns:
            Float32 numpy array at model_sample_rate.
        """
        float_audio = pcm_to_float32(pcm_data)
        return resample(float_audio, self._input_rate, self._model_rate)

    def process_float(self, float_audio: np.ndarray) -> np.ndarray:
        """Resample float32 audio from input rate to model rate."""
        return resample(float_audio, self._input_rate, self._model_rate)

    @property
    def input_rate(self) -> int:
        return self._input_rate

    @property
    def model_rate(self) -> int:
        return self._model_rate
