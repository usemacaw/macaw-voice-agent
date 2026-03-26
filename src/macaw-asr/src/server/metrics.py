"""Metrics aggregator for macaw-asr.

Collects per-request timing data, computes percentiles (p50/p95/p99),
and exposes aggregated stats per model. Thread-safe via Lock.

Usage:
    collector = MetricsCollector()
    collector.record("faster-whisper-small", {"e2e_ms": 15.3, "prepare_ms": 0.1})
    stats = collector.get_stats()
"""

from __future__ import annotations

import threading
from collections import defaultdict
from dataclasses import dataclass, field


_MAX_WINDOW = 1000  # Rolling window of recent requests


@dataclass
class _ModelStats:
    """Per-model request history."""
    entries: list[dict[str, float]] = field(default_factory=list)
    total_count: int = 0


class MetricsCollector:
    """Aggregates per-model request metrics with percentile computation."""

    def __init__(self, max_window: int = _MAX_WINDOW) -> None:
        self._max_window = max_window
        self._lock = threading.Lock()
        self._models: dict[str, _ModelStats] = defaultdict(lambda: _ModelStats())

    def record(self, model_name: str, timings: dict[str, float]) -> None:
        """Record timing data from a single request."""
        with self._lock:
            stats = self._models[model_name]
            stats.entries.append(timings)
            stats.total_count += 1
            if len(stats.entries) > self._max_window:
                stats.entries.pop(0)

    def get_stats(self) -> dict:
        """Get aggregated statistics for all models."""
        with self._lock:
            total = sum(s.total_count for s in self._models.values())
            by_model = {}
            for name, stats in self._models.items():
                if not stats.entries:
                    continue
                by_model[name] = {
                    "count": stats.total_count,
                    "window": len(stats.entries),
                    "latency": _compute_percentiles(stats.entries),
                }
            return {"total": total, "by_model": by_model}

    def reset(self) -> None:
        """Reset all collected metrics."""
        with self._lock:
            self._models.clear()


def _compute_percentiles(entries: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    """Compute avg/p50/p95/p99 for each timing key across entries."""
    if not entries:
        return {}

    # Collect all timing keys
    keys = set()
    for e in entries:
        keys.update(e.keys())

    result = {}
    for key in sorted(keys):
        values = sorted(v for e in entries if (v := e.get(key)) is not None)
        if not values:
            continue
        n = len(values)
        result[key] = {
            "avg": round(sum(values) / n, 2),
            "min": round(values[0], 2),
            "max": round(values[-1], 2),
            "p50": round(values[int(n * 0.50)], 2),
            "p95": round(values[min(int(n * 0.95), n - 1)], 2),
            "p99": round(values[min(int(n * 0.99), n - 1)], 2),
        }
    return result
