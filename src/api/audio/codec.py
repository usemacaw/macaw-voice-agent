"""
Audio codec for OpenAI Realtime API.

Handles base64 encoding/decoding, PCM format conversion,
resampling between 24kHz (API) and 8kHz (internal/gRPC),
and G.711 mu-law/A-law encoding/decoding.

G.711 NOTE: Both mu-law and A-law codecs expect and produce 8kHz sample rate
audio (ITU-T G.711 standard). No resampling is performed — clients MUST send
G.711 at 8kHz, which matches the OpenAI Realtime API specification.

Uses numpy-based G.711 implementation to avoid the deprecated
audioop module (removed in Python 3.13).
"""

from __future__ import annotations

import base64

import numpy as np

from audio.utils import pcm_to_float32, float32_to_pcm, resample

# OpenAI Realtime API uses 24kHz 16-bit mono PCM
API_SAMPLE_RATE = 24000
# Internal pipeline uses 8kHz (matching gRPC STT/TTS servers)
INTERNAL_SAMPLE_RATE = 8000
# 16-bit PCM = 2 bytes per sample
SAMPLE_WIDTH = 2

# G.711 mu-law constants
_ULAW_BIAS = 0x84
_ULAW_CLIP = 32635


def _build_ulaw_decode_table() -> np.ndarray:
    """Pre-compute mu-law decode table (256 entries) using vectorized ops."""
    indices = np.arange(256, dtype=np.int32)
    val = ~indices & 0xFF
    sign = val & 0x80
    exponent = (val >> 4) & 0x07
    mantissa = val & 0x0F
    sample = ((mantissa << 3) + _ULAW_BIAS) << exponent
    sample = sample - _ULAW_BIAS
    sample = np.where(sign, -sample, sample)
    return sample.astype(np.int16)


def _build_alaw_decode_table() -> np.ndarray:
    """Pre-compute A-law decode table (256 entries) using vectorized ops."""
    indices = np.arange(256, dtype=np.int32)
    val = indices ^ 0x55
    sign = val & 0x80
    exponent = (val >> 4) & 0x07
    mantissa = val & 0x0F
    sample = np.where(
        exponent == 0,
        (mantissa << 4) + 8,
        ((mantissa << 4) + 0x108) << (exponent - 1),
    )
    sample = np.where(sign, -sample, sample)
    return sample.astype(np.int16)


def _build_ulaw_encode_table() -> np.ndarray:
    """Pre-compute mu-law encode table (65536 entries) using vectorized ops."""
    # Interpret uint16 indices as signed int16 samples
    samples = np.arange(65536, dtype=np.uint16).view(np.int16).astype(np.int32)
    sign_byte = np.where(samples < 0, 0x80, 0x00).astype(np.uint8)
    mag = np.minimum(np.abs(samples), _ULAW_CLIP) + _ULAW_BIAS

    # Find exponent: highest bit position
    exp = np.zeros(65536, dtype=np.int32)
    for e in range(7, -1, -1):
        mask = (mag >= (1 << (e + 7))) & (exp == 0)
        exp = np.where(mask, e, exp)

    mantissa_bits = (mag >> (exp + 3)) & 0x0F
    result = (~(sign_byte.astype(np.int32) | (exp << 4) | mantissa_bits)) & 0xFF
    return result.astype(np.uint8)


def _build_alaw_encode_table() -> np.ndarray:
    """Pre-compute A-law encode table (65536 entries) using vectorized ops."""
    samples = np.arange(65536, dtype=np.uint16).view(np.int16).astype(np.int32)
    sign_byte = np.where(samples < 0, 0x80, 0x00).astype(np.int32)
    mag = np.abs(samples)

    exp = np.zeros(65536, dtype=np.int32)
    for e in range(7, 0, -1):
        mask = (mag >= (1 << (e + 7))) & (exp == 0)
        exp = np.where(mask, e, exp)

    mantissa_bits = np.where(
        exp == 0,
        (mag >> 4) & 0x0F,
        (mag >> (exp + 3)) & 0x0F,
    )
    result = ((sign_byte | (exp << 4) | mantissa_bits) ^ 0x55) & 0xFF
    return result.astype(np.uint8)


# Build lookup tables once at import time (vectorized — ~5ms vs ~100ms with Python loops)
_ULAW_DECODE_TABLE = _build_ulaw_decode_table()
_ALAW_DECODE_TABLE = _build_alaw_decode_table()
_ULAW_ENCODE_TABLE = _build_ulaw_encode_table()
_ALAW_ENCODE_TABLE = _build_alaw_encode_table()


def _ulaw_decode(data: bytes) -> bytes:
    """Decode G.711 mu-law to PCM 16-bit."""
    indices = np.frombuffer(data, dtype=np.uint8)
    return _ULAW_DECODE_TABLE[indices].tobytes()


def _ulaw_encode(pcm: bytes) -> bytes:
    """Encode PCM 16-bit to G.711 mu-law using pre-computed lookup table."""
    samples = np.frombuffer(pcm, dtype=np.int16).view(np.uint16)
    return _ULAW_ENCODE_TABLE[samples].tobytes()


def _alaw_decode(data: bytes) -> bytes:
    """Decode G.711 A-law to PCM 16-bit."""
    indices = np.frombuffer(data, dtype=np.uint8)
    return _ALAW_DECODE_TABLE[indices].tobytes()


def _alaw_encode(pcm: bytes) -> bytes:
    """Encode PCM 16-bit to G.711 A-law using pre-computed lookup table."""
    samples = np.frombuffer(pcm, dtype=np.int16).view(np.uint16)
    return _ALAW_ENCODE_TABLE[samples].tobytes()


def decode_audio_from_client(audio_b64: str, input_format: str = "pcm16") -> bytes:
    """Decode base64 client audio to internal PCM 8kHz 16-bit.

    Args:
        audio_b64: Base64-encoded audio from client.
        input_format: "pcm16" (24kHz), "g711_ulaw" (8kHz), or "g711_alaw" (8kHz).

    Returns:
        PCM 16-bit 8kHz mono bytes.
    """
    raw = base64.b64decode(audio_b64)
    if not raw:
        return b""

    if input_format == "pcm16":
        # Client sends 24kHz PCM16 — resample to 8kHz
        samples = pcm_to_float32(raw)
        resampled = resample(samples, API_SAMPLE_RATE, INTERNAL_SAMPLE_RATE)
        return float32_to_pcm(resampled)

    elif input_format == "g711_ulaw":
        # G.711 mu-law is 8kHz by spec (ITU-T G.711) — no resample needed
        return _ulaw_decode(raw)

    elif input_format == "g711_alaw":
        # G.711 A-law is 8kHz by spec (ITU-T G.711) — no resample needed
        return _alaw_decode(raw)

    else:
        raise ValueError(f"Unsupported input_audio_format: {input_format}")


def encode_audio_for_client(pcm_8k: bytes, output_format: str = "pcm16") -> str:
    """Encode internal PCM 8kHz to base64 for client.

    Args:
        pcm_8k: PCM 16-bit 8kHz mono bytes.
        output_format: "pcm16" (24kHz), "g711_ulaw" (8kHz), or "g711_alaw" (8kHz).

    Returns:
        Base64-encoded audio string.
    """
    if not pcm_8k:
        return ""

    if output_format == "pcm16":
        # Resample 8kHz → 24kHz for API
        samples = pcm_to_float32(pcm_8k)
        resampled = resample(samples, INTERNAL_SAMPLE_RATE, API_SAMPLE_RATE)
        raw = float32_to_pcm(resampled)

    elif output_format == "g711_ulaw":
        raw = _ulaw_encode(pcm_8k)

    elif output_format == "g711_alaw":
        raw = _alaw_encode(pcm_8k)

    else:
        raise ValueError(f"Unsupported output_audio_format: {output_format}")

    return base64.b64encode(raw).decode("ascii")
