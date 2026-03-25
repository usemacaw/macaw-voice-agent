"""Performance benchmarks: Model inference.

Extracted from:
- Ollama cmd/bench/bench.go: warmup → epochs → metrics (prefill ns/token, generate ns/token, TTFT)
- Ollama sample/samplers_benchmark_test.go: variable-size benchmarks with b.ResetTimer
- vLLM benchmark_latency.py: warmup runs, p50/p90/p99, statistical aggregation
- NeMo: RTF (Real-Time Factor), streaming chunk latency

All tests run real inference on GPU. No mocks.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from macaw_asr.decode.strategies import GreedyWithEarlyStopping


def _run_inference(model, audio, n_warmup=2, n_runs=5):
    """Run inference with warmup and collect timing. (Ollama bench pattern)"""
    strategy_fn = lambda: GreedyWithEarlyStopping(
        eos_token_id=model.eos_token_id, repetition_window=3
    )

    # Warmup (Ollama: negative epoch numbers)
    for _ in range(n_warmup):
        inputs = model.prepare_inputs(audio)
        model.generate(inputs, strategy_fn())

    # Timed epochs (Ollama: collect metrics per epoch)
    results = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        inputs = model.prepare_inputs(audio)
        prep_ms = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        output = model.generate(inputs, strategy_fn())
        gen_ms = (time.perf_counter() - t0) * 1000

        results.append({
            "prep_ms": prep_ms,
            "prefill_ms": output.timings.get("prefill_ms", 0),
            "decode_ms": output.timings.get("decode_ms", 0),
            "gen_ms": gen_ms,
            "n_tokens": output.n_tokens,
            "total_ms": prep_ms + gen_ms,
        })

    return results


# ==================== Latency by Audio Duration (Ollama: variable-size benchmarks) ====================


class TestLatencyByDuration:
    """From Ollama BenchmarkWeightedSampler: test across multiple sizes.
    From vLLM benchmark_latency.py: warmup + N runs + percentiles."""

    @pytest.mark.parametrize("duration_sec", [0.5, 1.0, 2.0, 3.0, 5.0])
    def test_inference_latency(self, model, duration_sec):
        """Measure real latency for each audio duration."""
        np.random.seed(42)
        audio = np.random.randn(int(16000 * duration_sec)).astype(np.float32) * 0.3

        results = _run_inference(model, audio, n_warmup=2, n_runs=5)

        totals = [r["total_ms"] for r in results]
        p50 = np.percentile(totals, 50)
        p90 = np.percentile(totals, 90)
        prep_p50 = np.percentile([r["prep_ms"] for r in results], 50)
        gen_p50 = np.percentile([r["gen_ms"] for r in results], 50)

        print(
            f"\n  [{duration_sec}s audio] "
            f"prep_p50={prep_p50:.0f}ms gen_p50={gen_p50:.0f}ms "
            f"total_p50={p50:.0f}ms p90={p90:.0f}ms "
            f"tokens={results[0]['n_tokens']}"
        )

        # Gate: total latency must be under 5s for any duration
        assert p50 < 5000, f"p50 latency {p50:.0f}ms exceeds 5000ms for {duration_sec}s audio"


# ==================== Throughput (Ollama: tokens/sec, ns/token) ====================


class TestThroughput:
    """From Ollama bench.go: prefill ns/token, generate ns/token, token/sec."""

    def test_prefill_throughput(self, model):
        """Measure prefill speed in tokens/sec."""
        np.random.seed(42)
        audio = np.random.randn(48000).astype(np.float32) * 0.3  # 3s

        results = _run_inference(model, audio, n_warmup=2, n_runs=5)
        prefill_times = [r["prefill_ms"] for r in results]
        p50 = np.percentile(prefill_times, 50)

        print(f"\n  Prefill p50={p50:.1f}ms")

        # Prefill should be under 200ms on RTX 3090
        assert p50 < 200, f"Prefill p50={p50:.0f}ms exceeds 200ms"

    def test_decode_tokens_per_second(self, model):
        """From Ollama: generate tokens/sec."""
        np.random.seed(42)
        audio = np.random.randn(48000).astype(np.float32) * 0.3

        results = _run_inference(model, audio, n_warmup=2, n_runs=5)
        tps_list = []
        for r in results:
            if r["decode_ms"] > 0 and r["n_tokens"] > 0:
                tps = r["n_tokens"] / (r["decode_ms"] / 1000)
                tps_list.append(tps)

        if tps_list:
            tps_p50 = np.percentile(tps_list, 50)
            print(f"\n  Decode throughput p50={tps_p50:.1f} tokens/sec")

            # Minimum 10 tokens/sec on RTX 3090
            assert tps_p50 >= 10, f"Decode {tps_p50:.1f} tok/s below 10 tok/s"


# ==================== RTF — Real-Time Factor (NeMo pattern) ====================


class TestRealTimeFactor:
    """From NeMo: RTF = processing_time / audio_duration.
    RTF < 1.0 means faster than real-time (required for streaming)."""

    @pytest.mark.parametrize("duration_sec", [1.0, 3.0, 5.0])
    def test_rtf_below_threshold(self, model, duration_sec):
        """RTF must be below 1.0 for real-time streaming viability."""
        np.random.seed(42)
        audio = np.random.randn(int(16000 * duration_sec)).astype(np.float32) * 0.3

        results = _run_inference(model, audio, n_warmup=2, n_runs=5)
        rtfs = [r["total_ms"] / (duration_sec * 1000) for r in results]
        rtf_p50 = np.percentile(rtfs, 50)
        rtfx = 1.0 / rtf_p50 if rtf_p50 > 0 else float("inf")

        print(f"\n  [{duration_sec}s] RTF={rtf_p50:.3f} RTFx={rtfx:.1f}")

        assert rtf_p50 < 1.0, (
            f"RTF={rtf_p50:.3f} >= 1.0 for {duration_sec}s audio — not real-time viable"
        )

    def test_rtf_improves_with_longer_audio(self, model):
        """Longer audio should have better (lower) RTF due to fixed overhead amortization."""
        np.random.seed(42)
        rtfs = {}
        for dur in [1.0, 5.0]:
            audio = np.random.randn(int(16000 * dur)).astype(np.float32) * 0.3
            results = _run_inference(model, audio, n_warmup=1, n_runs=3)
            rtfs[dur] = np.median([r["total_ms"] / (dur * 1000) for r in results])

        print(f"\n  RTF@1s={rtfs[1.0]:.3f} RTF@5s={rtfs[5.0]:.3f}")

        assert rtfs[5.0] <= rtfs[1.0], (
            f"RTF should improve with longer audio: 1s={rtfs[1.0]:.3f}, 5s={rtfs[5.0]:.3f}"
        )


# ==================== Fast Finish Speedup (unique to macaw-asr) ====================


class TestFastFinishPerformance:
    """Benchmark the fast_finish optimization vs full prepare_inputs."""

    def test_fast_finish_speedup_long_audio(self, model):
        """With 10s audio, fast_finish should show clear speedup over full path."""
        np.random.seed(42)
        full_audio = np.random.randn(160000).astype(np.float32) * 0.3  # 10s
        partial = full_audio[:80000]  # 5s

        # Warmup both paths
        model.prepare_inputs(full_audio)
        c = model.prepare_inputs(partial)
        model.fast_finish_inputs(full_audio, c, len(partial))

        # Measure full prepare (5 runs)
        full_times = []
        for _ in range(5):
            t0 = time.perf_counter()
            model.prepare_inputs(full_audio)
            full_times.append((time.perf_counter() - t0) * 1000)

        # Measure fast finish (5 runs)
        fast_times = []
        for _ in range(5):
            cached = model.prepare_inputs(partial)
            t0 = time.perf_counter()
            model.fast_finish_inputs(full_audio, cached, len(partial))
            fast_times.append((time.perf_counter() - t0) * 1000)

        full_p50 = np.percentile(full_times, 50)
        fast_p50 = np.percentile(fast_times, 50)

        print(f"\n  Full prep p50={full_p50:.1f}ms, Fast finish p50={fast_p50:.1f}ms")

        assert fast_p50 < full_p50, (
            f"Fast finish ({fast_p50:.1f}ms) not faster than full ({full_p50:.1f}ms)"
        )
