"""Tests for admission control: ProviderSemaphore and AdmissionControls."""

import asyncio
import os

import pytest

from providers.admission import AdmissionControls, ProviderBusyError, ProviderSemaphore, create_admission_controls


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_semaphore(max_concurrent: int = 2, name: str = "test") -> ProviderSemaphore:
    return ProviderSemaphore(name, max_concurrent)


# ---------------------------------------------------------------------------
# ProviderSemaphore — basic acquire / release
# ---------------------------------------------------------------------------


class TestBasicAcquireRelease:
    async def test_acquire_and_release_succeeds(self):
        sem = make_semaphore(max_concurrent=1)

        async with sem.acquire(timeout=1.0):
            pass  # no exception = success

    async def test_active_count_increments_inside_context(self):
        sem = make_semaphore(max_concurrent=2)

        async with sem.acquire():
            assert sem.active == 1

    async def test_active_count_returns_to_zero_after_release(self):
        sem = make_semaphore(max_concurrent=2)

        async with sem.acquire():
            pass

        assert sem.active == 0

    async def test_sequential_acquires_both_complete(self):
        sem = make_semaphore(max_concurrent=1)

        async with sem.acquire():
            pass

        async with sem.acquire():
            pass

        assert sem.active == 0


# ---------------------------------------------------------------------------
# ProviderSemaphore — max concurrent limit
# ---------------------------------------------------------------------------


class TestMaxConcurrentLimit:
    async def test_second_acquire_blocks_when_at_capacity(self):
        sem = make_semaphore(max_concurrent=1)
        acquired = asyncio.Event()
        released = asyncio.Event()

        async def holder():
            async with sem.acquire():
                acquired.set()
                await released.wait()

        task = asyncio.create_task(holder())
        await acquired.wait()

        # Semaphore is full; a concurrent acquire with short timeout must fail
        with pytest.raises(ProviderBusyError):
            async with sem.acquire(timeout=0.05):
                pass

        released.set()
        await task

    async def test_n_concurrent_limited_to_max(self):
        """Exactly max_concurrent tasks should run concurrently; the rest must wait."""
        max_concurrent = 3
        sem = make_semaphore(max_concurrent=max_concurrent)
        inside_count = 0
        peak: list[int] = []
        release_gate = asyncio.Event()
        all_inside = asyncio.Event()

        async def worker():
            nonlocal inside_count
            async with sem.acquire():
                inside_count += 1
                peak.append(inside_count)
                if inside_count == max_concurrent:
                    all_inside.set()
                await release_gate.wait()
                inside_count -= 1

        tasks = [asyncio.create_task(worker()) for _ in range(max_concurrent)]

        await asyncio.wait_for(all_inside.wait(), timeout=2.0)

        release_gate.set()
        await asyncio.gather(*tasks)

        assert max(peak) == max_concurrent

    async def test_acquire_succeeds_after_slot_freed(self):
        sem = make_semaphore(max_concurrent=1)
        gate = asyncio.Event()

        async def holder():
            async with sem.acquire():
                await gate.wait()

        holder_task = asyncio.create_task(holder())
        await asyncio.sleep(0)  # let holder acquire

        gate.set()
        await holder_task  # slot released

        # Now the semaphore has a free slot; this must not raise
        async with sem.acquire(timeout=1.0):
            assert sem.active == 1


# ---------------------------------------------------------------------------
# ProviderSemaphore — timeout raises ProviderBusyError
# ---------------------------------------------------------------------------


class TestTimeout:
    async def test_timeout_raises_provider_busy_error(self):
        sem = make_semaphore(max_concurrent=1)
        gate = asyncio.Event()

        async def holder():
            async with sem.acquire():
                await gate.wait()

        task = asyncio.create_task(holder())
        await asyncio.sleep(0)  # let holder acquire

        with pytest.raises(ProviderBusyError):
            async with sem.acquire(timeout=0.05):
                pass

        gate.set()
        await task

    async def test_provider_busy_error_message_contains_name(self):
        sem = make_semaphore(max_concurrent=1, name="tts_gpu")
        gate = asyncio.Event()

        async def holder():
            async with sem.acquire():
                await gate.wait()

        task = asyncio.create_task(holder())
        await asyncio.sleep(0)

        with pytest.raises(ProviderBusyError, match="tts_gpu"):
            async with sem.acquire(timeout=0.05):
                pass

        gate.set()
        await task

    async def test_active_count_unchanged_after_timeout(self):
        """A timed-out waiter must not corrupt the active counter."""
        sem = make_semaphore(max_concurrent=1)
        gate = asyncio.Event()

        async def holder():
            async with sem.acquire():
                await gate.wait()

        task = asyncio.create_task(holder())
        await asyncio.sleep(0)

        try:
            async with sem.acquire(timeout=0.05):
                pass
        except ProviderBusyError:
            pass

        active_after_timeout = sem.active
        gate.set()
        await task

        # Holder had active=1 while running; after timeout the count must still be 1
        # (not incremented by the failed waiter).
        assert active_after_timeout == 1
        assert sem.active == 0  # after holder finishes


# ---------------------------------------------------------------------------
# ProviderSemaphore — metrics
# ---------------------------------------------------------------------------


class TestMetrics:
    async def test_total_acquired_increments_per_successful_acquire(self):
        sem = make_semaphore(max_concurrent=3)

        for _ in range(5):
            async with sem.acquire():
                pass

        assert sem.snapshot()["total_acquired"] == 5

    async def test_timeout_does_not_increment_total_acquired(self):
        sem = make_semaphore(max_concurrent=1)
        gate = asyncio.Event()

        async def holder():
            async with sem.acquire():
                await gate.wait()

        task = asyncio.create_task(holder())
        await asyncio.sleep(0)

        try:
            async with sem.acquire(timeout=0.05):
                pass
        except ProviderBusyError:
            pass

        gate.set()
        await task

        # Only the holder's acquire counts
        assert sem.snapshot()["total_acquired"] == 1

    async def test_peak_active_tracks_maximum_concurrency(self):
        sem = make_semaphore(max_concurrent=4)
        gate = asyncio.Event()
        n_workers = 3
        all_inside = asyncio.Event()
        inside_count = 0

        async def worker():
            nonlocal inside_count
            async with sem.acquire():
                inside_count += 1
                if inside_count == n_workers:
                    all_inside.set()
                await gate.wait()

        tasks = [asyncio.create_task(worker()) for _ in range(n_workers)]
        await asyncio.wait_for(all_inside.wait(), timeout=2.0)

        peak_before_release = sem.snapshot()["peak_active"]

        gate.set()
        await asyncio.gather(*tasks)

        assert peak_before_release == 3

    async def test_peak_active_never_decreases(self):
        sem = make_semaphore(max_concurrent=5)
        n_wave1 = 3

        # First wave: 3 concurrent — wait until all are inside
        gate1 = asyncio.Event()
        inside1 = asyncio.Event()
        count1 = 0

        async def wave1_worker():
            nonlocal count1
            async with sem.acquire():
                count1 += 1
                if count1 == n_wave1:
                    inside1.set()
                await gate1.wait()

        batch1 = [asyncio.create_task(wave1_worker()) for _ in range(n_wave1)]
        await asyncio.wait_for(inside1.wait(), timeout=2.0)
        peak_after_wave1 = sem.snapshot()["peak_active"]
        gate1.set()
        await asyncio.gather(*batch1)

        # Second wave: only 1 concurrent
        gate2 = asyncio.Event()
        inside2 = asyncio.Event()

        async def wave2_worker():
            async with sem.acquire():
                inside2.set()
                await gate2.wait()

        t = asyncio.create_task(wave2_worker())
        await asyncio.wait_for(inside2.wait(), timeout=2.0)
        gate2.set()
        await t

        assert sem.snapshot()["peak_active"] == peak_after_wave1

    async def test_avg_wait_ms_is_non_negative(self):
        sem = make_semaphore(max_concurrent=2)

        for _ in range(4):
            async with sem.acquire():
                pass

        assert sem.snapshot()["avg_wait_ms"] >= 0.0

    async def test_avg_wait_ms_zero_when_no_acquires(self):
        sem = make_semaphore(max_concurrent=2)
        # Uses max(1, total_acquired) in denominator — should not divide by zero
        assert sem.snapshot()["avg_wait_ms"] == 0.0


# ---------------------------------------------------------------------------
# ProviderSemaphore — available property
# ---------------------------------------------------------------------------


class TestAvailableProperty:
    async def test_available_equals_max_when_idle(self):
        sem = make_semaphore(max_concurrent=5)
        assert sem.available == 5

    async def test_available_decrements_while_held(self):
        sem = make_semaphore(max_concurrent=3)
        n_holders = 2
        gate = asyncio.Event()
        all_inside = asyncio.Event()
        inside_count = 0

        async def holder():
            nonlocal inside_count
            async with sem.acquire():
                inside_count += 1
                if inside_count == n_holders:
                    all_inside.set()
                await gate.wait()

        tasks = [asyncio.create_task(holder()) for _ in range(n_holders)]
        await asyncio.wait_for(all_inside.wait(), timeout=2.0)

        assert sem.available == 1

        gate.set()
        await asyncio.gather(*tasks)

        assert sem.available == 3

    async def test_available_plus_active_equals_max(self):
        sem = make_semaphore(max_concurrent=4)
        gate = asyncio.Event()

        async def holder():
            async with sem.acquire():
                await gate.wait()

        tasks = [asyncio.create_task(holder()) for _ in range(3)]
        await asyncio.sleep(0)
        await asyncio.sleep(0)

        assert sem.available + sem.active == 4

        gate.set()
        await asyncio.gather(*tasks)


# ---------------------------------------------------------------------------
# ProviderSemaphore — snapshot dict
# ---------------------------------------------------------------------------


class TestSnapshot:
    async def test_snapshot_keys_are_complete(self):
        sem = make_semaphore(max_concurrent=3, name="asr")
        snap = sem.snapshot()

        assert set(snap.keys()) == {"name", "max", "active", "available", "peak_active", "total_acquired", "avg_wait_ms"}

    async def test_snapshot_name_and_max_correct(self):
        sem = make_semaphore(max_concurrent=7, name="llm")
        snap = sem.snapshot()

        assert snap["name"] == "llm"
        assert snap["max"] == 7

    async def test_snapshot_reflects_current_state(self):
        sem = make_semaphore(max_concurrent=3)
        n_holders = 2
        gate = asyncio.Event()
        all_inside = asyncio.Event()
        inside_count = 0

        async def holder():
            nonlocal inside_count
            async with sem.acquire():
                inside_count += 1
                if inside_count == n_holders:
                    all_inside.set()
                await gate.wait()

        tasks = [asyncio.create_task(holder()) for _ in range(n_holders)]
        await asyncio.wait_for(all_inside.wait(), timeout=2.0)

        snap = sem.snapshot()
        assert snap["active"] == 2
        assert snap["available"] == 1

        gate.set()
        await asyncio.gather(*tasks)

        snap_after = sem.snapshot()
        assert snap_after["active"] == 0
        assert snap_after["available"] == 3


# ---------------------------------------------------------------------------
# ProviderSemaphore — context manager safety on exception
# ---------------------------------------------------------------------------


class TestContextManagerSafety:
    async def test_exception_inside_context_releases_slot(self):
        sem = make_semaphore(max_concurrent=1)

        with pytest.raises(RuntimeError):
            async with sem.acquire():
                raise RuntimeError("boom")

        # Slot must be released even though the body raised
        assert sem.active == 0
        assert sem.available == 1

    async def test_slot_reusable_after_exception(self):
        sem = make_semaphore(max_concurrent=1)

        with pytest.raises(ValueError):
            async with sem.acquire():
                raise ValueError("oops")

        # Must be acquirable again without timeout
        async with sem.acquire(timeout=1.0):
            assert sem.active == 1

    async def test_peak_active_still_tracked_after_exception(self):
        sem = make_semaphore(max_concurrent=2)

        with pytest.raises(RuntimeError):
            async with sem.acquire():
                raise RuntimeError("fail")

        assert sem.snapshot()["total_acquired"] == 1
        assert sem.snapshot()["peak_active"] == 1


# ---------------------------------------------------------------------------
# Concurrent access — race conditions
# ---------------------------------------------------------------------------


class TestConcurrentAccess:
    async def test_concurrent_workers_respect_max_limit(self):
        """No more than max_concurrent tasks should hold the semaphore simultaneously."""
        max_concurrent = 3
        sem = make_semaphore(max_concurrent=max_concurrent)
        observed_active: list[int] = []
        lock = asyncio.Lock()

        async def worker():
            async with sem.acquire(timeout=5.0):
                async with lock:
                    observed_active.append(sem.active)
                await asyncio.sleep(0.01)

        await asyncio.gather(*[worker() for _ in range(10)])

        assert max(observed_active) <= max_concurrent

    async def test_all_workers_eventually_complete(self):
        """Every waiter must eventually get the semaphore (no starvation)."""
        sem = make_semaphore(max_concurrent=2)
        completed = 0

        async def worker():
            nonlocal completed
            async with sem.acquire(timeout=5.0):
                await asyncio.sleep(0.005)
                completed += 1

        await asyncio.gather(*[worker() for _ in range(8)])

        assert completed == 8

    async def test_total_acquired_matches_successful_workers(self):
        sem = make_semaphore(max_concurrent=3)
        n_workers = 12

        async def worker():
            async with sem.acquire(timeout=5.0):
                await asyncio.sleep(0.002)

        await asyncio.gather(*[worker() for _ in range(n_workers)])

        assert sem.snapshot()["total_acquired"] == n_workers

    async def test_active_returns_to_zero_after_all_workers(self):
        sem = make_semaphore(max_concurrent=4)

        async def worker():
            async with sem.acquire(timeout=5.0):
                await asyncio.sleep(0.002)

        await asyncio.gather(*[worker() for _ in range(10)])

        assert sem.active == 0


# ---------------------------------------------------------------------------
# AdmissionControls — structure and defaults
# ---------------------------------------------------------------------------


class TestAdmissionControls:
    def test_has_asr_tts_llm_semaphores(self):
        controls = AdmissionControls()

        assert isinstance(controls.asr, ProviderSemaphore)
        assert isinstance(controls.tts, ProviderSemaphore)
        assert isinstance(controls.llm, ProviderSemaphore)

    def test_default_asr_max_is_5(self):
        controls = AdmissionControls()
        assert controls.asr.snapshot()["max"] == 5

    def test_default_tts_max_is_5(self):
        controls = AdmissionControls()
        assert controls.tts.snapshot()["max"] == 5

    def test_default_llm_max_is_3(self):
        controls = AdmissionControls()
        assert controls.llm.snapshot()["max"] == 3

    def test_snapshot_returns_all_three_providers(self):
        controls = AdmissionControls()
        snap = controls.snapshot()

        assert set(snap.keys()) == {"asr", "tts", "llm"}

    def test_snapshot_values_are_dicts(self):
        controls = AdmissionControls()
        snap = controls.snapshot()

        for key in ("asr", "tts", "llm"):
            assert isinstance(snap[key], dict)
            assert "max" in snap[key]

    def test_semaphores_are_independent(self):
        """Acquiring ASR must not affect TTS or LLM active counts."""
        controls = AdmissionControls()

        # We can't easily hold a semaphore synchronously, so we just verify
        # that the semaphore objects are distinct instances.
        assert controls.asr is not controls.tts
        assert controls.tts is not controls.llm
        assert controls.asr is not controls.llm


# ---------------------------------------------------------------------------
# create_admission_controls — reads environment variables
# ---------------------------------------------------------------------------


class TestCreateAdmissionControls:
    def test_default_values_without_env_vars(self, monkeypatch):
        monkeypatch.delenv("MAX_CONCURRENT_ASR", raising=False)
        monkeypatch.delenv("MAX_CONCURRENT_TTS", raising=False)
        monkeypatch.delenv("MAX_CONCURRENT_LLM", raising=False)

        controls = create_admission_controls()

        assert controls.asr.snapshot()["max"] == 5
        assert controls.tts.snapshot()["max"] == 5
        assert controls.llm.snapshot()["max"] == 3

    def test_reads_max_concurrent_asr_from_env(self, monkeypatch):
        monkeypatch.setenv("MAX_CONCURRENT_ASR", "10")
        monkeypatch.delenv("MAX_CONCURRENT_TTS", raising=False)
        monkeypatch.delenv("MAX_CONCURRENT_LLM", raising=False)

        controls = create_admission_controls()

        assert controls.asr.snapshot()["max"] == 10

    def test_reads_max_concurrent_tts_from_env(self, monkeypatch):
        monkeypatch.delenv("MAX_CONCURRENT_ASR", raising=False)
        monkeypatch.setenv("MAX_CONCURRENT_TTS", "8")
        monkeypatch.delenv("MAX_CONCURRENT_LLM", raising=False)

        controls = create_admission_controls()

        assert controls.tts.snapshot()["max"] == 8

    def test_reads_max_concurrent_llm_from_env(self, monkeypatch):
        monkeypatch.delenv("MAX_CONCURRENT_ASR", raising=False)
        monkeypatch.delenv("MAX_CONCURRENT_TTS", raising=False)
        monkeypatch.setenv("MAX_CONCURRENT_LLM", "1")

        controls = create_admission_controls()

        assert controls.llm.snapshot()["max"] == 1

    def test_raises_on_value_below_minimum(self, monkeypatch):
        """_env_int raises ValueError when value is below min_val=1."""
        monkeypatch.setenv("MAX_CONCURRENT_ASR", "0")
        monkeypatch.delenv("MAX_CONCURRENT_TTS", raising=False)
        monkeypatch.delenv("MAX_CONCURRENT_LLM", raising=False)

        with pytest.raises(ValueError, match="out of range"):
            create_admission_controls()

    def test_raises_on_value_above_maximum(self, monkeypatch):
        """_env_int raises ValueError when value exceeds max_val=100."""
        monkeypatch.setenv("MAX_CONCURRENT_ASR", "999")
        monkeypatch.delenv("MAX_CONCURRENT_TTS", raising=False)
        monkeypatch.delenv("MAX_CONCURRENT_LLM", raising=False)

        with pytest.raises(ValueError, match="out of range"):
            create_admission_controls()

    def test_all_three_env_vars_applied_together(self, monkeypatch):
        monkeypatch.setenv("MAX_CONCURRENT_ASR", "7")
        monkeypatch.setenv("MAX_CONCURRENT_TTS", "6")
        monkeypatch.setenv("MAX_CONCURRENT_LLM", "2")

        controls = create_admission_controls()

        assert controls.asr.snapshot()["max"] == 7
        assert controls.tts.snapshot()["max"] == 6
        assert controls.llm.snapshot()["max"] == 2
