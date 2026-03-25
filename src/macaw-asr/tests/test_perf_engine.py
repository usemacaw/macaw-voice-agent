"""Performance benchmarks: Engine E2E pipeline.

Extracted from:
- Ollama integration/concurrency_test.go: parallel requests, model thrashing
- Ollama integration/model_perf_test.go: context sizes, prompt eval TPS
- vLLM benchmark_serving.py: requests/sec under load
- NeMo: streaming chunk latency, WER + timing combined
"""

from __future__ import annotations

import asyncio
import time

import numpy as np
import pytest

from macaw_asr.runner.engine import ASREngine
from tests.conftest import make_pcm, make_silence, make_noise


# ==================== E2E Latency (Ollama model_perf_test.go pattern) ====================


class TestE2ELatency:
    """From Ollama model_perf_test.go: measure load time, eval TPS across configs."""

    async def test_first_request_latency(self, engine_config):
        """First request includes model load — must still complete in bounded time.
        From Ollama: load + prefill + generate = total."""
        engine = ASREngine(engine_config)

        t0 = time.perf_counter()
        await engine.start()
        load_ms = (time.perf_counter() - t0) * 1000

        audio = make_pcm(1.0)
        t0 = time.perf_counter()
        text = await engine.transcribe(audio)
        first_req_ms = (time.perf_counter() - t0) * 1000

        print(f"\n  Load={load_ms:.0f}ms, First request={first_req_ms:.0f}ms")

        await engine.stop()

        assert len(text) > 0
        assert first_req_ms < 5000, f"First request took {first_req_ms:.0f}ms"

    async def test_warm_request_latency(self, engine):
        """Subsequent requests (model warm) should be fast.
        From vLLM: measure after warmup."""
        audio = make_pcm(1.0)

        # Warmup
        await engine.transcribe(audio)
        await engine.transcribe(audio)

        # Measure
        times = []
        for _ in range(5):
            t0 = time.perf_counter()
            await engine.transcribe(audio)
            times.append((time.perf_counter() - t0) * 1000)

        p50 = np.percentile(times, 50)
        p90 = np.percentile(times, 90)

        print(f"\n  Warm request: p50={p50:.0f}ms p90={p90:.0f}ms")

        assert p50 < 3000, f"Warm request p50={p50:.0f}ms too slow"


# ==================== Streaming Latency (NeMo chunk latency pattern) ====================


class TestStreamingLatency:
    """From NeMo: measure per-chunk latency and finish latency separately."""

    async def test_chunk_push_latency(self, engine):
        """Each push_audio should return in < 10ms (just accumulation, no inference)."""
        audio = make_pcm(2.0)
        await engine.create_session("chunk-lat")

        chunk_size = 512  # 32ms at 8kHz
        push_times = []
        for i in range(0, len(audio), chunk_size):
            chunk = audio[i:i + chunk_size]
            t0 = time.perf_counter()
            await engine.push_audio("chunk-lat", chunk)
            push_times.append((time.perf_counter() - t0) * 1000)

        await engine.finish_session("chunk-lat")

        p50 = np.percentile(push_times, 50)
        p99 = np.percentile(push_times, 99)

        print(f"\n  Chunk push: p50={p50:.2f}ms p99={p99:.2f}ms ({len(push_times)} chunks)")

        assert p50 < 10, f"Chunk push p50={p50:.1f}ms exceeds 10ms"

    async def test_finish_latency(self, engine):
        """finish_session is the bottleneck — measure it.
        From NeMo: this is where RTF matters."""
        audio = make_pcm(2.0)

        # Warmup
        await engine.create_session("fin-warmup")
        for i in range(0, len(audio), 512):
            await engine.push_audio("fin-warmup", audio[i:i + 512])
        await engine.finish_session("fin-warmup")

        # Measure
        times = []
        for run in range(3):
            sid = f"fin-{run}"
            await engine.create_session(sid)
            for i in range(0, len(audio), 512):
                await engine.push_audio(sid, audio[i:i + 512])
            t0 = time.perf_counter()
            await engine.finish_session(sid)
            times.append((time.perf_counter() - t0) * 1000)

        p50 = np.percentile(times, 50)
        print(f"\n  Finish latency: p50={p50:.0f}ms")

        # 2s audio finish should complete in < 3s
        assert p50 < 3000, f"Finish p50={p50:.0f}ms exceeds 3000ms"

    async def test_streaming_rtf(self, engine):
        """From NeMo: total streaming pipeline RTF must be < 1.0."""
        audio = make_pcm(3.0)

        t0 = time.perf_counter()
        await engine.create_session("rtf-stream")
        for i in range(0, len(audio), 512):
            await engine.push_audio("rtf-stream", audio[i:i + 512])
        text = await engine.finish_session("rtf-stream")
        total_ms = (time.perf_counter() - t0) * 1000

        rtf = total_ms / 3000  # 3 seconds of audio
        print(f"\n  Streaming RTF={rtf:.3f} ({total_ms:.0f}ms for 3s audio)")

        assert rtf < 1.0, f"Streaming RTF={rtf:.3f} >= 1.0"


# ==================== Concurrent Requests (Ollama concurrency_test.go) ====================


class TestConcurrentPerformance:
    """From Ollama TestConcurrentChat: parallel requests to same model."""

    async def test_parallel_transcriptions(self, engine):
        """Multiple concurrent transcribe calls must all succeed."""
        audio = make_pcm(1.0)

        async def transcribe_one(i):
            t0 = time.perf_counter()
            text = await engine.transcribe(audio)
            elapsed = (time.perf_counter() - t0) * 1000
            return i, text, elapsed

        # Warmup
        await engine.transcribe(audio)

        # 3 parallel requests (Ollama: numParallel + 1)
        results = await asyncio.gather(
            transcribe_one(0),
            transcribe_one(1),
            transcribe_one(2),
        )

        times = [r[2] for r in results]
        texts = [r[1] for r in results]

        print(f"\n  Parallel latencies: {[f'{t:.0f}ms' for t in times]}")

        # All must succeed
        for i, text, elapsed in results:
            assert len(text) > 0, f"Request {i} returned empty"

        # All should produce same text (deterministic + same input)
        assert all(t == texts[0] for t in texts), f"Divergent results: {texts}"

    async def test_parallel_streaming_sessions(self, engine):
        """From Ollama TestMultiModelStress: multiple concurrent sessions."""
        audio = make_pcm(1.0)

        async def stream_one(sid):
            await engine.create_session(sid)
            for i in range(0, len(audio), 512):
                await engine.push_audio(sid, audio[i:i + 512])
            t0 = time.perf_counter()
            text = await engine.finish_session(sid)
            elapsed = (time.perf_counter() - t0) * 1000
            return sid, text, elapsed

        # 3 parallel streaming sessions
        results = await asyncio.gather(
            stream_one("par-a"),
            stream_one("par-b"),
            stream_one("par-c"),
        )

        for sid, text, elapsed in results:
            assert len(text) > 0, f"Session {sid} returned empty"
            print(f"\n  Session {sid}: {elapsed:.0f}ms")
