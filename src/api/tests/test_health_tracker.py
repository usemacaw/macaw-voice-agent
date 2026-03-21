"""Tests for HealthTracker shared across gRPC microservices."""

import pytest
from unittest.mock import MagicMock

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from common.grpc_server import HealthTracker


class FakeHealthServicer:
    """Minimal fake for grpc_health HealthServicer."""

    def __init__(self):
        self.statuses: dict[str, int] = {}

    def set(self, service_name: str, status) -> None:
        self.statuses[service_name] = status


class TestHealthTracker:
    def test_starts_with_zero_errors(self):
        tracker = HealthTracker("my.Service")
        # No exception on initial state
        tracker.record_success()

    def test_record_success_resets_errors(self):
        health = FakeHealthServicer()
        tracker = HealthTracker("my.Service", health, max_errors=3)

        # Accumulate some errors
        tracker.record_error()
        tracker.record_error()

        # Success resets
        tracker.record_success()
        assert "my.Service" in health.statuses  # set to SERVING

    def test_record_success_noop_when_no_errors(self):
        health = FakeHealthServicer()
        tracker = HealthTracker("my.Service", health)

        # No errors accumulated, success should not touch health
        tracker.record_success()
        assert "my.Service" not in health.statuses

    def test_marks_not_serving_after_max_errors(self):
        health = FakeHealthServicer()
        tracker = HealthTracker("my.Service", health, max_errors=3)

        tracker.record_error()
        assert "my.Service" not in health.statuses

        tracker.record_error()
        assert "my.Service" not in health.statuses

        tracker.record_error()  # 3rd error = threshold
        assert "my.Service" in health.statuses  # marked NOT_SERVING

    def test_success_after_degradation_restores_serving(self):
        health = FakeHealthServicer()
        tracker = HealthTracker("my.Service", health, max_errors=2)

        tracker.record_error()
        tracker.record_error()
        # Now degraded
        degraded_status = health.statuses["my.Service"]

        tracker.record_success()
        restored_status = health.statuses["my.Service"]
        assert degraded_status != restored_status

    def test_works_without_health_servicer(self):
        tracker = HealthTracker("my.Service", health_servicer=None)

        # Should not raise even without health servicer
        for _ in range(10):
            tracker.record_error()
        tracker.record_success()

    def test_custom_max_errors(self):
        health = FakeHealthServicer()
        tracker = HealthTracker("my.Service", health, max_errors=1)

        tracker.record_error()  # 1st error = threshold with max_errors=1
        assert "my.Service" in health.statuses
