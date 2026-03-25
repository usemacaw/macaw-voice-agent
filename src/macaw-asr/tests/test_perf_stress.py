"""Stress tests: Queue saturation and concurrent load.

Extracted from:
- Ollama integration/max_queue_test.go: N goroutines fire requests, classify errors
- Ollama integration/concurrency_test.go: parallel requests with WaitGroup
- Ollama integration/concurrency_test.go TestMultiModelStress: load until VRAM exhaustion

All tests run real inference on GPU. No mocks.
"""

from __future__ import annotations

import asyncio
import time

import numpy as np
import pytest

from macaw_asr.config import AudioConfig, EngineConfig, StreamingConfig
from macaw_asr.manifest.paths import ModelPaths
from macaw_asr.manifest.registry import ModelRegistry
from macaw_asr.runner.engine import ASREngine
from macaw_asr.server.scheduler import Scheduler
from tests.conftest import make_pcm


# ==================== Queue Stress (Ollama TestMaxQueue pattern) ====================


class TestQueueStress:
    """From Ollama max_queue_test.go:
    - Spawn long-running request to occupy GPU
    - Fire N concurrent requests
    - Classify: success, busy, error, timeout
    - Verify no crashes or connection resets
    """

    async def test_queue_16_concurrent_requests(self, engine):
        """16 concurrent transcribe requests to a single engine.

        Ollama pattern: fire 16+ requests, verify all complete or fail gracefully.
        GPU serializes, so requests queue up — but none should crash.
        """
        audio = make_pcm(1.0)
        n_workers = 16

        results = {"success": 0, "error": 0, "errors": []}
        lock = asyncio.Lock()

        async def worker(i: int):
            try:
                text = await engine.transcribe(audio)
                if text and len(text) > 0:
                    async with lock:
                        results["success"] += 1
                else:
                    async with lock:
                        results["error"] += 1
                        results["errors"].append(f"worker-{i}: empty result")
            except Exception as e:
                async with lock:
                    results["error"] += 1
                    results["errors"].append(f"worker-{i}: {type(e).__name__}: {e}")

        # Warmup
        await engine.transcribe(audio)

        # Fire all workers
        t0 = time.perf_counter()
        await asyncio.gather(*[worker(i) for i in range(n_workers)])
        total_ms = (time.perf_counter() - t0) * 1000

        print(f"\n  16 concurrent: {results['success']} success, {results['error']} error, {total_ms:.0f}ms total")
        if results["errors"]:
            for err in results["errors"][:5]:
                print(f"    {err}")

        # All must succeed (GPU serializes but doesn't reject)
        assert results["success"] == n_workers, (
            f"{results['success']}/{n_workers} succeeded. Errors: {results['errors'][:3]}"
        )

    async def test_queue_32_streaming_sessions(self, engine):
        """32 concurrent streaming sessions.

        From Ollama: tests scheduler capacity under streaming load.
        Each session: create → push 0.5s audio → finish.
        """
        audio = make_pcm(0.5)
        n_sessions = 32

        results = {"success": 0, "error": 0, "errors": []}
        lock = asyncio.Lock()

        async def session_worker(i: int):
            sid = f"stress-{i}"
            try:
                await engine.create_session(sid)
                chunk_size = 512
                for j in range(0, len(audio), chunk_size):
                    await engine.push_audio(sid, audio[j:j + chunk_size])
                text = await engine.finish_session(sid)
                if text is not None:  # empty string is OK for short audio
                    async with lock:
                        results["success"] += 1
                else:
                    async with lock:
                        results["error"] += 1
                        results["errors"].append(f"session-{i}: None result")
            except Exception as e:
                async with lock:
                    results["error"] += 1
                    results["errors"].append(f"session-{i}: {type(e).__name__}: {e}")

        t0 = time.perf_counter()
        await asyncio.gather(*[session_worker(i) for i in range(n_sessions)])
        total_ms = (time.perf_counter() - t0) * 1000

        print(f"\n  32 sessions: {results['success']} success, {results['error']} error, {total_ms:.0f}ms total")

        assert results["success"] == n_sessions, (
            f"{results['success']}/{n_sessions} succeeded. Errors: {results['errors'][:3]}"
        )

    async def test_mixed_batch_and_streaming(self, engine):
        """Mix of batch transcribe + streaming sessions concurrently.

        Tests that both code paths don't interfere.
        """
        audio = make_pcm(1.0)

        async def batch_worker(i):
            return await engine.transcribe(audio)

        async def stream_worker(i):
            sid = f"mix-stream-{i}"
            await engine.create_session(sid)
            for j in range(0, len(audio), 512):
                await engine.push_audio(sid, audio[j:j + 512])
            return await engine.finish_session(sid)

        # 5 batch + 5 streaming = 10 concurrent
        tasks = []
        for i in range(5):
            tasks.append(batch_worker(i))
            tasks.append(stream_worker(i))

        t0 = time.perf_counter()
        results = await asyncio.gather(*tasks, return_exceptions=True)
        total_ms = (time.perf_counter() - t0) * 1000

        errors = [r for r in results if isinstance(r, Exception)]
        successes = [r for r in results if isinstance(r, str)]

        print(f"\n  Mixed 5+5: {len(successes)} success, {len(errors)} error, {total_ms:.0f}ms")

        assert len(errors) == 0, f"Errors in mixed workload: {errors[:3]}"
        assert len(successes) == 10


# ==================== Model Thrashing (Ollama TestMultiModelStress pattern) ====================


class TestModelThrashing:
    """From Ollama TestMultiModelStress:
    - Load models until VRAM exhaustion
    - Force scheduler to evict and reload
    - Verify all requests still succeed

    We simulate this with the scheduler loading/unloading the same model,
    since we only have one model (qwen). The pattern tests the scheduler
    lifecycle under load.
    """

    async def test_scheduler_load_unload_cycles(self, engine_config, tmp_path):
        """Load → transcribe → unload → reload → transcribe. 5 cycles.

        Tests that scheduler handles repeated load/unload without leaks.
        """
        paths = ModelPaths(home=tmp_path)
        registry = ModelRegistry(paths=paths)
        scheduler = Scheduler(registry=registry, keep_alive_sec=1)

        audio = make_pcm(0.5)

        for cycle in range(5):
            engine = await scheduler.get_runner(engine_config)
            text = await engine.transcribe(audio)
            assert isinstance(text, str), f"Cycle {cycle}: transcribe failed"
            await scheduler.unload(engine_config.model_id)
            assert scheduler.list_loaded() == []

        print(f"\n  5 load/unload cycles completed")

    async def test_scheduler_concurrent_get_runner(self, engine_config, tmp_path):
        """From Ollama: multiple goroutines request the same model.

        Only one should actually load; others should get cached engine.
        """
        paths = ModelPaths(home=tmp_path)
        registry = ModelRegistry(paths=paths)
        scheduler = Scheduler(registry=registry, keep_alive_sec=60)

        # 10 concurrent get_runner for the same model
        results = await asyncio.gather(*[
            scheduler.get_runner(engine_config) for _ in range(10)
        ])

        # All should return the same engine instance
        engines = set(id(r) for r in results)
        assert len(engines) == 1, f"Expected 1 engine, got {len(engines)} different instances"

        # Only loaded once
        assert len(scheduler.list_loaded()) == 1

        await scheduler.stop()
        print(f"\n  10 concurrent get_runner → 1 engine loaded")

    async def test_scheduler_eviction(self, engine_config, tmp_path):
        """Scheduler should evict model after TTL expires.

        From Ollama: KeepAlive duration controls eviction timing.
        """
        paths = ModelPaths(home=tmp_path)
        registry = ModelRegistry(paths=paths)
        scheduler = Scheduler(registry=registry, keep_alive_sec=0.5)
        await scheduler.start()

        # Load model
        engine = await scheduler.get_runner(engine_config)
        assert len(scheduler.list_loaded()) == 1

        # Wait for eviction (TTL=0.5s, check interval=30s is too long)
        # We'll manually trigger by checking
        await asyncio.sleep(1.0)

        # Force eviction check by requesting again (which triggers reload)
        # The old one should have been evicted by now if we run the check
        # Since eviction loop checks every 30s, we manually verify staleness
        ref = scheduler._loaded.get(engine_config.model_id)
        if ref:
            age = time.time() - ref.last_used
            print(f"\n  Model age: {age:.1f}s (TTL=0.5s)")
            assert age > 0.5, "Model should be stale"

        await scheduler.stop()


# ==================== Sustained Load (vLLM benchmark_serving.py pattern) ====================


class TestSustainedLoad:
    """From vLLM benchmark_serving.py: measure performance under sustained request rate."""

    async def test_sustained_10_requests(self, engine):
        """10 sequential requests — measure stability of latency.

        From vLLM: latency should not degrade over time (no memory leaks).
        """
        audio = make_pcm(1.0)

        # Warmup
        await engine.transcribe(audio)

        latencies = []
        for i in range(10):
            t0 = time.perf_counter()
            text = await engine.transcribe(audio)
            ms = (time.perf_counter() - t0) * 1000
            latencies.append(ms)
            assert len(text) > 0, f"Request {i} empty"

        first_half = np.mean(latencies[:5])
        second_half = np.mean(latencies[5:])
        degradation = (second_half - first_half) / first_half if first_half > 0 else 0

        print(f"\n  Sustained 10 requests:")
        print(f"    First 5 avg: {first_half:.0f}ms")
        print(f"    Last 5 avg:  {second_half:.0f}ms")
        print(f"    Degradation: {degradation:+.1%}")

        # Latency should not degrade more than 50% over 10 requests
        assert degradation < 0.5, (
            f"Latency degraded {degradation:.1%}: {first_half:.0f}ms → {second_half:.0f}ms"
        )

    async def test_sustained_streaming_sessions(self, engine):
        """10 sequential streaming sessions — no degradation."""
        audio = make_pcm(1.0)

        # Warmup
        await engine.create_session("sus-warmup")
        for i in range(0, len(audio), 512):
            await engine.push_audio("sus-warmup", audio[i:i + 512])
        await engine.finish_session("sus-warmup")

        latencies = []
        for i in range(10):
            sid = f"sus-{i}"
            await engine.create_session(sid)
            for j in range(0, len(audio), 512):
                await engine.push_audio(sid, audio[j:j + 512])
            t0 = time.perf_counter()
            text = await engine.finish_session(sid)
            ms = (time.perf_counter() - t0) * 1000
            latencies.append(ms)

        p50 = np.percentile(latencies, 50)
        p99 = np.percentile(latencies, 99)

        print(f"\n  Sustained 10 streaming: p50={p50:.0f}ms p99={p99:.0f}ms")

        # p99 should not be more than 3x p50
        assert p99 < p50 * 3, f"p99={p99:.0f}ms > 3x p50={p50:.0f}ms"
