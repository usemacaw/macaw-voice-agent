"""Performance benchmarks: Audio preprocessing pipeline.

Extracted from:
- Ollama x/tokenizer/tokenizer_benchmark_test.go: encode/decode at multiple sizes
- Ollama sample/samplers_benchmark_test.go: variable-size, b.ResetTimer pattern
- NeMo: mel spectrogram extraction timing
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from macaw_asr.audio.preprocessing import (
    AudioPreprocessor,
    float32_to_pcm,
    pcm_to_float32,
    resample,
)
from macaw_asr.config import AudioConfig


def _bench(fn, n_warmup=3, n_runs=100):
    """Ollama pattern: warmup + N runs, return times in ms."""
    for _ in range(n_warmup):
        fn()
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000)
    return times


# ==================== PCM Conversion (Ollama tokenizer encode/decode pattern) ====================


class TestPcmConversionPerformance:
    """From Ollama BenchmarkTokenizerEncodeBPE/DecodeBPE: multiple sizes."""

    @pytest.mark.parametrize("n_samples", [800, 8000, 16000, 48000, 160000])
    def test_pcm_to_float32_throughput(self, n_samples):
        pcm = np.random.randint(-32768, 32767, n_samples, dtype=np.int16).tobytes()
        times = _bench(lambda: pcm_to_float32(pcm), n_warmup=5, n_runs=100)
        p50 = np.percentile(times, 50)
        mb_per_sec = (len(pcm) / 1e6) / (p50 / 1000) if p50 > 0 else float("inf")

        print(f"\n  pcm_to_float32 {n_samples} samples: p50={p50:.3f}ms ({mb_per_sec:.0f} MB/s)")

        # Must process at least 100 MB/s
        assert mb_per_sec > 100, f"Throughput {mb_per_sec:.0f} MB/s below 100 MB/s"

    @pytest.mark.parametrize("n_samples", [800, 8000, 16000, 48000, 160000])
    def test_float32_to_pcm_throughput(self, n_samples):
        samples = np.random.randn(n_samples).astype(np.float32) * 0.5
        times = _bench(lambda: float32_to_pcm(samples), n_warmup=5, n_runs=100)
        p50 = np.percentile(times, 50)
        mb_per_sec = (n_samples * 4 / 1e6) / (p50 / 1000) if p50 > 0 else float("inf")

        print(f"\n  float32_to_pcm {n_samples} samples: p50={p50:.3f}ms ({mb_per_sec:.0f} MB/s)")

        assert mb_per_sec > 100, f"Throughput {mb_per_sec:.0f} MB/s below 100 MB/s"


# ==================== Resampling (Ollama variable-size benchmark pattern) ====================


class TestResamplingPerformance:
    """From Ollama BenchmarkWeightedSampler: test across multiple sizes."""

    @pytest.mark.parametrize(
        "duration_sec,from_rate,to_rate",
        [
            (0.032, 8000, 16000),   # 32ms chunk (real-time streaming)
            (0.5, 8000, 16000),     # 500ms
            (1.0, 8000, 16000),     # 1s
            (5.0, 8000, 16000),     # 5s
            (10.0, 8000, 16000),    # 10s
            (1.0, 44100, 16000),    # CD quality → 16k
        ],
        ids=["32ms", "500ms", "1s", "5s", "10s", "44.1k→16k"],
    )
    def test_resample_latency(self, duration_sec, from_rate, to_rate):
        n_samples = int(duration_sec * from_rate)
        audio = np.random.randn(n_samples).astype(np.float32) * 0.5
        times = _bench(lambda: resample(audio, from_rate, to_rate), n_warmup=3, n_runs=50)
        p50 = np.percentile(times, 50)
        p99 = np.percentile(times, 99)

        # RTF for resample alone: must be << 1.0
        rtf = p50 / (duration_sec * 1000)
        print(f"\n  Resample {duration_sec}s ({from_rate}→{to_rate}): p50={p50:.2f}ms p99={p99:.2f}ms RTF={rtf:.4f}")

        # Resampling alone should be < 1% of real-time
        assert rtf < 0.01, f"Resample RTF={rtf:.4f} too high (>1%)"

    def test_resample_32ms_chunk_under_1ms(self):
        """32ms chunk resample MUST be under 1ms for real-time streaming."""
        audio = np.random.randn(256).astype(np.float32)  # 32ms at 8kHz
        times = _bench(lambda: resample(audio, 8000, 16000), n_warmup=10, n_runs=1000)
        p99 = np.percentile(times, 99)

        print(f"\n  32ms chunk resample: p99={p99:.3f}ms")

        assert p99 < 1.0, f"32ms resample p99={p99:.3f}ms exceeds 1ms"


# ==================== Full Preprocessor Pipeline ====================


class TestPreprocessorPerformance:
    """Full pipeline: PCM bytes → resampled float32 at model rate."""

    @pytest.mark.parametrize("duration_sec", [0.5, 1.0, 3.0, 5.0])
    def test_preprocessor_latency(self, duration_sec):
        config = AudioConfig(input_sample_rate=8000, model_sample_rate=16000)
        proc = AudioPreprocessor(config)
        n_samples = int(duration_sec * 8000)
        pcm = np.random.randint(-32768, 32767, n_samples, dtype=np.int16).tobytes()

        times = _bench(lambda: proc.process(pcm), n_warmup=3, n_runs=30)
        p50 = np.percentile(times, 50)
        rtf = p50 / (duration_sec * 1000)

        print(f"\n  Preprocessor {duration_sec}s: p50={p50:.2f}ms RTF={rtf:.4f}")

        # Preprocessing RTF should be < 5% of real-time
        assert rtf < 0.05, f"Preprocessor RTF={rtf:.4f} too high (>5%)"
