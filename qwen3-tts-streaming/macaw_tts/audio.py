"""Audio processing utilities for streaming TTS.

Handles resampling (24kHz → 8kHz) and format conversion (float32 ↔ PCM16).
"""

from __future__ import annotations

import numpy as np


# Qwen3-TTS 12Hz codec output rate
CODEC_SAMPLE_RATE = 24000

# Macaw Voice Agent internal rate
INTERNAL_SAMPLE_RATE = 8000

# Samples per codec frame: validated from model.speech_tokenizer.get_decode_upsample_rate()
# 12Hz codec at 24kHz output = 1920 samples/frame (NOT 2000)
SAMPLES_PER_FRAME = 1920


def resample(audio: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    """Resample audio from src_rate to dst_rate.

    Uses scipy.signal.resample_poly for efficient polyphase resampling.
    For 24kHz → 8kHz: ratio = 1/3, very fast.

    Args:
        audio: Float32 audio samples.
        src_rate: Source sample rate.
        dst_rate: Destination sample rate.

    Returns:
        Resampled float32 audio.
    """
    if src_rate == dst_rate:
        return audio
    if len(audio) == 0:
        return audio

    from math import gcd

    g = gcd(src_rate, dst_rate)
    up = dst_rate // g
    down = src_rate // g

    from scipy.signal import resample_poly

    return resample_poly(audio, up, down).astype(np.float32)


def float32_to_pcm16(audio: np.ndarray) -> bytes:
    """Convert float32 [-1, 1] audio to PCM 16-bit little-endian bytes.

    Args:
        audio: Float32 audio array.

    Returns:
        PCM16 LE bytes.
    """
    clipped = np.clip(audio, -1.0, 1.0)
    pcm = (clipped * 32767).astype(np.int16)
    return pcm.tobytes()


def pcm16_to_float32(pcm: bytes) -> np.ndarray:
    """Convert PCM 16-bit LE bytes to float32 [-1, 1] array.

    Args:
        pcm: PCM16 LE bytes.

    Returns:
        Float32 audio array.
    """
    samples = np.frombuffer(pcm, dtype=np.int16)
    return samples.astype(np.float32) / 32768.0


def resample_to_internal(audio: np.ndarray, src_rate: int = CODEC_SAMPLE_RATE) -> np.ndarray:
    """Resample from codec rate (24kHz) to Macaw internal rate (8kHz)."""
    return resample(audio, src_rate, INTERNAL_SAMPLE_RATE)
