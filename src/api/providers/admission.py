"""
Per-provider admission control using asyncio semaphores.

Prevents resource exhaustion when multiple sessions compete for
shared providers (LLM API, TTS GPU, ASR GPU).

Usage:
    async with ADMISSION.tts.acquire():
        await tts.synthesize(text)
"""

from __future__ import annotations

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field

from config import _env_int

logger = logging.getLogger("open-voice-api.admission")


class ProviderBusyError(Exception):
    """Raised when a provider semaphore times out."""


class ProviderSemaphore:
    """Async semaphore with metrics for a provider resource."""

    def __init__(self, name: str, max_concurrent: int):
        self._name = name
        self._max = max_concurrent
        self._sem = asyncio.Semaphore(max_concurrent)
        # Metrics
        self._active: int = 0
        self._total_acquired: int = 0
        self._total_wait_ms: float = 0.0
        self._peak_active: int = 0

    @asynccontextmanager
    async def acquire(self, timeout: float = 30.0):
        """Acquire the semaphore with a timeout.

        Raises:
            ProviderBusyError: If the semaphore is not available within timeout.
        """
        t0 = time.perf_counter()
        try:
            await asyncio.wait_for(self._sem.acquire(), timeout=timeout)
        except asyncio.TimeoutError:
            logger.warning(
                f"[admission] {self._name} busy: {self._active}/{self._max} active, "
                f"timeout after {timeout:.1f}s"
            )
            raise ProviderBusyError(
                f"{self._name} at capacity ({self._max} concurrent), "
                f"try again later"
            ) from None

        wait_ms = (time.perf_counter() - t0) * 1000
        self._active += 1
        self._total_acquired += 1
        self._total_wait_ms += wait_ms
        if self._active > self._peak_active:
            self._peak_active = self._active

        try:
            yield
        finally:
            self._active -= 1
            self._sem.release()

    @property
    def active(self) -> int:
        return self._active

    @property
    def available(self) -> int:
        return self._max - self._active

    def snapshot(self) -> dict:
        """Return current metrics for observability."""
        return {
            "name": self._name,
            "max": self._max,
            "active": self._active,
            "available": self._max - self._active,
            "peak_active": self._peak_active,
            "total_acquired": self._total_acquired,
            "avg_wait_ms": round(
                self._total_wait_ms / max(1, self._total_acquired), 1
            ),
        }


@dataclass
class AdmissionControls:
    """Container for all provider semaphores."""

    asr: ProviderSemaphore = field(default_factory=lambda: ProviderSemaphore("asr", 5))
    tts: ProviderSemaphore = field(default_factory=lambda: ProviderSemaphore("tts", 5))
    llm: ProviderSemaphore = field(default_factory=lambda: ProviderSemaphore("llm", 3))

    def snapshot(self) -> dict:
        return {
            "asr": self.asr.snapshot(),
            "tts": self.tts.snapshot(),
            "llm": self.llm.snapshot(),
        }


def create_admission_controls() -> AdmissionControls:
    """Create admission controls from environment configuration."""
    max_asr = _env_int("MAX_CONCURRENT_ASR", 5, 1, 100)
    max_tts = _env_int("MAX_CONCURRENT_TTS", 5, 1, 100)
    max_llm = _env_int("MAX_CONCURRENT_LLM", 3, 1, 100)

    controls = AdmissionControls(
        asr=ProviderSemaphore("asr", max_asr),
        tts=ProviderSemaphore("tts", max_tts),
        llm=ProviderSemaphore("llm", max_llm),
    )

    logger.info(
        f"Admission controls: ASR={max_asr}, TTS={max_tts}, LLM={max_llm}"
    )
    return controls


# Module-level singleton — shared across all sessions
ADMISSION = create_admission_controls()
