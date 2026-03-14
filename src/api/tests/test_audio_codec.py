"""Tests for audio codec: base64, PCM, G.711, resampling roundtrips."""

import base64
import struct

import numpy as np
import pytest

from audio.codec import (
    decode_audio_from_client,
    encode_audio_for_client,
    API_SAMPLE_RATE,
    INTERNAL_SAMPLE_RATE,
)
from common.audio_utils import pcm_to_float32, float32_to_pcm, resample


class TestPCMConversion:
    def test_pcm_to_float32_roundtrip(self):
        # Generate 100ms of 440Hz sine wave at 8kHz
        t = np.linspace(0, 0.1, 800, endpoint=False)
        samples = (np.sin(2 * np.pi * 440 * t) * 0.5).astype(np.float32)
        pcm = float32_to_pcm(samples)
        recovered = pcm_to_float32(pcm)

        # Should be close (16-bit quantization introduces small error)
        # Max error is 1/32768 per sample, but float32→int16→float32 uses /32768 then /32768.0
        np.testing.assert_allclose(samples, recovered, atol=5e-05)

    def test_pcm_to_float32_odd_bytes_raises(self):
        with pytest.raises(ValueError, match="numero par de bytes|even byte count"):
            pcm_to_float32(b"\x00\x01\x02")

    def test_float32_to_pcm_clips(self):
        samples = np.array([1.5, -1.5], dtype=np.float32)
        pcm = float32_to_pcm(samples)
        values = np.frombuffer(pcm, dtype=np.int16)
        assert values[0] == 32767
        assert values[1] == -32767

    def test_empty_input(self):
        assert pcm_to_float32(b"").shape == (0,)
        assert float32_to_pcm(np.array([], dtype=np.float32)) == b""


class TestResampling:
    def test_same_rate_no_change(self):
        samples = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        result = resample(samples, 8000, 8000)
        np.testing.assert_array_equal(samples, result)

    def test_upsample_8k_to_24k(self):
        # 10ms at 8kHz = 80 samples
        samples = np.ones(80, dtype=np.float32)
        result = resample(samples, 8000, 24000)
        # Should be ~240 samples (30ms at 24kHz would be 720, but 10ms at 24kHz = 240)
        assert len(result) == 240
        np.testing.assert_allclose(result, 1.0, atol=0.01)

    def test_downsample_24k_to_8k(self):
        samples = np.ones(240, dtype=np.float32)
        result = resample(samples, 24000, 8000)
        assert len(result) == 80
        np.testing.assert_allclose(result, 1.0, atol=0.01)

    def test_roundtrip_preserves_signal(self):
        # Create a simple low-frequency signal
        t = np.linspace(0, 0.1, 800, endpoint=False)
        original = np.sin(2 * np.pi * 100 * t).astype(np.float32)  # 100Hz

        upsampled = resample(original, 8000, 24000)
        downsampled = resample(upsampled, 24000, 8000)

        # Signal should be preserved (within interpolation error)
        np.testing.assert_allclose(original, downsampled, atol=0.02)

    def test_empty_input(self):
        result = resample(np.array([], dtype=np.float32), 8000, 24000)
        assert len(result) == 0


class TestCodecPCM16:
    def test_encode_decode_roundtrip(self):
        """PCM 8kHz → encode (resample to 24kHz, base64) → decode (resample back) → PCM 8kHz."""
        # Generate 100ms of silence at 8kHz
        pcm_8k = b"\x00\x00" * 800

        encoded = encode_audio_for_client(pcm_8k, "pcm16")
        assert isinstance(encoded, str)
        assert len(encoded) > 0

        # Decode: 24kHz PCM16 → 8kHz
        decoded = decode_audio_from_client(encoded, "pcm16")
        assert isinstance(decoded, bytes)
        assert len(decoded) > 0

        # Duration should be preserved (~100ms)
        decoded_duration_ms = len(decoded) / (INTERNAL_SAMPLE_RATE * 2) * 1000
        assert abs(decoded_duration_ms - 100) < 5  # within 5ms

    def test_signal_preserved_through_roundtrip(self):
        """A low-frequency sine wave survives encode/decode roundtrip."""
        t = np.linspace(0, 0.1, 800, endpoint=False)
        sine = (np.sin(2 * np.pi * 200 * t) * 0.5).astype(np.float32)
        pcm_8k = float32_to_pcm(sine)

        encoded = encode_audio_for_client(pcm_8k, "pcm16")
        decoded = decode_audio_from_client(encoded, "pcm16")

        original_samples = pcm_to_float32(pcm_8k)
        decoded_samples = pcm_to_float32(decoded)

        # Align lengths (resampling may introduce small differences)
        min_len = min(len(original_samples), len(decoded_samples))
        np.testing.assert_allclose(
            original_samples[:min_len],
            decoded_samples[:min_len],
            atol=0.05,
        )


class TestCodecG711:
    def test_g711_ulaw_encode_decode(self):
        pcm_8k = b"\x00\x00" * 800  # 100ms silence

        encoded = encode_audio_for_client(pcm_8k, "g711_ulaw")
        assert isinstance(encoded, str)

        decoded = decode_audio_from_client(encoded, "g711_ulaw")
        assert isinstance(decoded, bytes)
        # G.711 at 8kHz: 1 byte per sample → 800 bytes → decoded to 16-bit = 1600 bytes
        assert len(decoded) == 1600

    def test_g711_alaw_encode_decode(self):
        pcm_8k = b"\x00\x00" * 800

        encoded = encode_audio_for_client(pcm_8k, "g711_alaw")
        decoded = decode_audio_from_client(encoded, "g711_alaw")
        assert len(decoded) == 1600

    def test_unsupported_format_raises(self):
        with pytest.raises(ValueError, match="Unsupported"):
            decode_audio_from_client("AAAA", "opus")

        with pytest.raises(ValueError, match="Unsupported"):
            encode_audio_for_client(b"\x00\x00", "opus")


class TestEdgeCases:
    def test_empty_audio_encode(self):
        assert encode_audio_for_client(b"", "pcm16") == ""

    def test_empty_audio_decode(self):
        empty_b64 = base64.b64encode(b"").decode("ascii")
        assert decode_audio_from_client(empty_b64, "pcm16") == b""
