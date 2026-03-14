"""
System-level metrics for operational visibility.

Samples CPU, memory, and event loop lag periodically.
Exposes a snapshot() for the /metrics endpoint.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field

logger = logging.getLogger("open-voice-api.system-metrics")


@dataclass
class SystemMetrics:
    """Lightweight system metrics collector.

    Call sample_event_loop_lag() periodically to track async health.
    """

    # Event loop lag (ms) — last measured
    event_loop_lag_ms: float = 0.0

    # Session counters (updated externally)
    active_sessions: int = 0
    total_sessions: int = 0

    # Aggregate response counters
    total_responses: int = 0
    total_cancels: int = 0
    total_barge_before_audio: int = 0

    def record_response(self) -> None:
        self.total_responses += 1

    def record_cancel(self) -> None:
        self.total_cancels += 1

    def record_barge_before_audio(self) -> None:
        self.total_barge_before_audio += 1

    async def sample_event_loop_lag(self) -> None:
        """Measure event loop scheduling lag.

        Schedules a callback and measures actual vs expected delay.
        High values (>50ms) indicate the event loop is saturated.
        """
        t0 = time.perf_counter()
        await asyncio.sleep(0)  # Yield to event loop
        self.event_loop_lag_ms = (time.perf_counter() - t0) * 1000

    def snapshot(self) -> dict:
        """Return current metrics as a dict for JSON serialization."""
        result: dict = {
            "event_loop_lag_ms": round(self.event_loop_lag_ms, 2),
            "active_sessions": self.active_sessions,
            "total_sessions": self.total_sessions,
            "total_responses": self.total_responses,
            "total_cancels": self.total_cancels,
            "total_barge_before_audio": self.total_barge_before_audio,
        }

        # Process-level metrics (optional, requires psutil)
        try:
            import psutil
            proc = psutil.Process()
            mem = proc.memory_info()
            result["cpu_percent"] = proc.cpu_percent()
            result["memory_rss_mb"] = round(mem.rss / (1024 * 1024), 1)
        except ImportError:
            pass
        except Exception as e:
            logger.debug(f"Failed to collect process metrics: {e}")

        # Admission control metrics
        try:
            from providers.admission import ADMISSION
            result["admission"] = ADMISSION.snapshot()
        except Exception:
            pass

        return result


# Module-level singleton
SYSTEM_METRICS = SystemMetrics()
