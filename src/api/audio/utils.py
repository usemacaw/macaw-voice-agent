"""
Audio utilities: PCM/float32 conversion, resampling.

Adapted from ai-agent/providers/audio_utils.py.
"""

import numpy as np

try:
    from scipy.signal import resample_poly as _scipy_resample_poly
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False


def pcm_to_float32(pcm_data: bytes) -> np.ndarray:
    """PCM 16-bit signed little-endian -> numpy float32 normalized [-1, 1]."""
    if len(pcm_data) % 2 != 0:
        raise ValueError(f"PCM 16-bit requires even byte count, got {len(pcm_data)}")
    samples = np.frombuffer(pcm_data, dtype=np.int16)
    return samples.astype(np.float32) / 32768.0


def float32_to_pcm(samples: np.ndarray) -> bytes:
    """numpy float32 [-1, 1] -> PCM 16-bit signed little-endian bytes."""
    clipped = np.clip(samples, -1.0, 1.0)
    int_samples = (clipped * 32767.0).astype(np.int16)
    return int_samples.tobytes()


def resample(samples: np.ndarray, from_rate: int, to_rate: int) -> np.ndarray:
    """Resample audio with proper anti-aliasing.

    Uses scipy.signal.resample_poly when available (polyphase filter with
    anti-aliasing), falls back to numpy linear interpolation otherwise.
    """
    if from_rate == to_rate:
        return samples
    if len(samples) == 0:
        return np.array([], dtype=samples.dtype)

    if _HAS_SCIPY:
        # Find GCD for rational resampling
        from math import gcd
        g = gcd(from_rate, to_rate)
        up = to_rate // g
        down = from_rate // g
        resampled = _scipy_resample_poly(samples, up, down).astype(samples.dtype)
        return resampled

    # Fallback: linear interpolation (no anti-aliasing — lower quality)
    duration = len(samples) / from_rate
    target_length = int(duration * to_rate)
    if target_length == 0:
        return np.array([], dtype=samples.dtype)
    indices = np.linspace(0, len(samples) - 1, target_length)
    return np.interp(indices, np.arange(len(samples)), samples).astype(samples.dtype)
