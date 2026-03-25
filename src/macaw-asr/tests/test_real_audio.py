"""RED/GREEN tests: Real audio preprocessing on GPU.

Validates the full pipeline from PCM to model-ready tensors.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from macaw_asr.audio.preprocessing import AudioPreprocessor, pcm_to_float32, resample
from macaw_asr.config import AudioConfig


class TestResampleCorrectness:

    def test_8k_to_16k_doubles_samples(self):
        audio = np.ones(8000, dtype=np.float32)
        result = resample(audio, 8000, 16000)
        assert len(result) == 16000

    def test_energy_preserved(self):
        np.random.seed(42)
        signal = np.random.randn(8000).astype(np.float32) * 0.5
        energy_in = np.sum(signal ** 2) / len(signal)
        resampled = resample(signal, 8000, 16000)
        energy_out = np.sum(resampled ** 2) / len(resampled)
        ratio = energy_out / energy_in
        assert 0.9 < ratio < 1.1, f"Energy ratio {ratio:.3f} outside 10% tolerance"

    def test_resample_performance(self):
        """Resampling 1s of audio should take < 10ms."""
        audio = np.random.randn(8000).astype(np.float32)
        t0 = time.perf_counter()
        for _ in range(100):
            resample(audio, 8000, 16000)
        avg_ms = (time.perf_counter() - t0) * 1000 / 100
        assert avg_ms < 10, f"Resample took {avg_ms:.1f}ms avg"


class TestPreprocessorPipeline:

    def test_pcm_to_model_rate(self):
        config = AudioConfig(input_sample_rate=8000, model_sample_rate=16000)
        proc = AudioPreprocessor(config)
        pcm = np.zeros(8000, dtype=np.int16).tobytes()  # 1s at 8kHz
        result = proc.process(pcm)
        assert len(result) == 16000
        assert result.dtype == np.float32

    def test_pcm_roundtrip_fidelity(self):
        """PCM → float32 → PCM must preserve signal within 1 LSB."""
        from macaw_asr.audio.preprocessing import float32_to_pcm
        original = np.array([0, 1000, -1000, 16000, -16000, 32767, -32768], dtype=np.int16)
        pcm = original.tobytes()
        float_audio = pcm_to_float32(pcm)
        recovered = np.frombuffer(float32_to_pcm(float_audio), dtype=np.int16)
        np.testing.assert_array_almost_equal(recovered, original, decimal=0)
