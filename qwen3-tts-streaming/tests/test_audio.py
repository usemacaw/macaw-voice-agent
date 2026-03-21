"""Tests for audio utilities — 100% CPU, no GPU needed."""

import numpy as np
import pytest

from macaw_tts.audio import (
    resample,
    float32_to_pcm16,
    pcm16_to_float32,
    resample_to_internal,
    CODEC_SAMPLE_RATE,
    INTERNAL_SAMPLE_RATE,
    SAMPLES_PER_FRAME,
)


class TestConstants:

    def test_codec_sample_rate(self):
        assert CODEC_SAMPLE_RATE == 24000

    def test_internal_sample_rate(self):
        assert INTERNAL_SAMPLE_RATE == 8000

    def test_samples_per_frame(self):
        # Validated from Qwen3-TTS model: speech_tokenizer.get_decode_upsample_rate() = 1920
        assert SAMPLES_PER_FRAME == 1920


class TestResample:

    def test_same_rate_returns_input(self):
        audio = np.ones(100, dtype=np.float32)
        result = resample(audio, 24000, 24000)
        np.testing.assert_array_equal(result, audio)

    def test_empty_returns_empty(self):
        result = resample(np.array([], dtype=np.float32), 24000, 8000)
        assert len(result) == 0

    def test_downsample_24k_to_8k(self):
        # 24kHz → 8kHz = 1/3 ratio
        audio = np.ones(2400, dtype=np.float32)
        result = resample(audio, 24000, 8000)
        # Should have ~1/3 the samples
        assert abs(len(result) - 800) <= 10

    def test_upsample_8k_to_24k(self):
        audio = np.ones(800, dtype=np.float32)
        result = resample(audio, 8000, 24000)
        assert abs(len(result) - 2400) <= 10

    def test_preserves_dc_component(self):
        """Resampling a constant signal should preserve amplitude."""
        audio = np.ones(2400, dtype=np.float32) * 0.5
        result = resample(audio, 24000, 8000)
        # DC component should be preserved
        np.testing.assert_allclose(result[10:-10], 0.5, atol=0.05)


class TestResampleToInternal:

    def test_resamples_to_8k(self):
        audio = np.ones(2400, dtype=np.float32)
        result = resample_to_internal(audio)
        assert abs(len(result) - 800) <= 10


class TestFloat32ToPcm16:

    def test_silence_produces_zeros(self):
        audio = np.zeros(100, dtype=np.float32)
        pcm = float32_to_pcm16(audio)
        samples = np.frombuffer(pcm, dtype=np.int16)
        assert np.all(samples == 0)

    def test_max_amplitude(self):
        audio = np.array([1.0], dtype=np.float32)
        pcm = float32_to_pcm16(audio)
        samples = np.frombuffer(pcm, dtype=np.int16)
        assert samples[0] == 32767

    def test_min_amplitude(self):
        audio = np.array([-1.0], dtype=np.float32)
        pcm = float32_to_pcm16(audio)
        samples = np.frombuffer(pcm, dtype=np.int16)
        assert samples[0] == -32767

    def test_clips_above_one(self):
        audio = np.array([2.0], dtype=np.float32)
        pcm = float32_to_pcm16(audio)
        samples = np.frombuffer(pcm, dtype=np.int16)
        assert samples[0] == 32767

    def test_byte_length(self):
        audio = np.ones(100, dtype=np.float32)
        pcm = float32_to_pcm16(audio)
        assert len(pcm) == 200  # 100 samples × 2 bytes


class TestPcm16ToFloat32:

    def test_roundtrip(self):
        original = np.array([0.0, 0.5, -0.5, 1.0, -1.0], dtype=np.float32)
        pcm = float32_to_pcm16(original)
        result = pcm16_to_float32(pcm)
        np.testing.assert_allclose(result, original, atol=0.001)

    def test_silence(self):
        pcm = np.zeros(10, dtype=np.int16).tobytes()
        result = pcm16_to_float32(pcm)
        np.testing.assert_array_equal(result, np.zeros(10, dtype=np.float32))

    def test_most_negative_value_within_range(self):
        """Most negative PCM16 (-32768) should map to [-1, 1] range."""
        pcm = np.array([-32768], dtype=np.int16).tobytes()
        result = pcm16_to_float32(pcm)
        assert result[0] >= -1.0
        assert result[0] <= 1.0


class TestValidateAudioPath:
    """Tests for voice clone ref_audio path validation."""

    def test_rejects_non_audio_extension(self):
        from macaw_tts.model import _validate_audio_path
        with pytest.raises(ValueError, match="must be an audio file"):
            _validate_audio_path("/etc/passwd")

    def test_rejects_python_file(self):
        from macaw_tts.model import _validate_audio_path
        with pytest.raises(ValueError, match="must be an audio file"):
            _validate_audio_path("/tmp/evil.py")

    def test_rejects_nonexistent_audio_file(self):
        from macaw_tts.model import _validate_audio_path
        with pytest.raises(FileNotFoundError):
            _validate_audio_path("/tmp/nonexistent_file_12345.wav")

    def test_accepts_valid_wav_extension(self, tmp_path):
        from macaw_tts.model import _validate_audio_path
        wav_file = tmp_path / "test.wav"
        wav_file.write_bytes(b"\x00" * 100)
        _validate_audio_path(str(wav_file))  # should not raise

    def test_accepts_valid_flac_extension(self, tmp_path):
        from macaw_tts.model import _validate_audio_path
        flac_file = tmp_path / "test.flac"
        flac_file.write_bytes(b"\x00" * 100)
        _validate_audio_path(str(flac_file))  # should not raise
