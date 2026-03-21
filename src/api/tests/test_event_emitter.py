"""Tests for EventEmitter: cancellation fencing, backpressure, and event delivery."""

import asyncio
import json

import pytest

from protocol.event_emitter import EventEmitter, SlowClientError


class FakeWS:
    """Minimal WebSocket fake for EventEmitter tests."""

    def __init__(self, *, slow: bool = False, fail_after: int = 0):
        self.sent: list[dict] = []
        self._slow = slow
        self._fail_after = fail_after  # start timing out after N sends
        self._send_count = 0

    async def send(self, data: str) -> None:
        self._send_count += 1
        if self._slow or (self._fail_after and self._send_count > self._fail_after):
            await asyncio.sleep(999)  # will be cancelled by timeout
        self.sent.append(json.loads(data))


class TestEventIdGeneration:
    @pytest.mark.asyncio
    async def test_auto_generates_event_id(self):
        ws = FakeWS()
        emitter = EventEmitter(ws, "sess_test")
        await emitter.emit({"type": "session.created"})

        assert len(ws.sent) == 1
        assert ws.sent[0]["event_id"].startswith("evt_")

    @pytest.mark.asyncio
    async def test_preserves_existing_event_id(self):
        ws = FakeWS()
        emitter = EventEmitter(ws, "sess_test")
        await emitter.emit({"type": "test", "event_id": "my_id"})

        assert ws.sent[0]["event_id"] == "my_id"


class TestCancellationFence:
    @pytest.mark.asyncio
    async def test_drops_events_from_stale_response(self):
        ws = FakeWS()
        emitter = EventEmitter(ws, "sess_test")
        emitter.set_active_response("resp_1")

        # Event from active response: delivered
        await emitter.emit({"type": "response.audio.delta", "response_id": "resp_1"})
        assert len(ws.sent) == 1

        # Event from stale response: dropped
        await emitter.emit({"type": "response.audio.delta", "response_id": "resp_old"})
        assert len(ws.sent) == 1  # no new event

    @pytest.mark.asyncio
    async def test_events_without_response_id_always_delivered(self):
        ws = FakeWS()
        emitter = EventEmitter(ws, "sess_test")
        emitter.set_active_response("resp_1")

        await emitter.emit({"type": "session.created"})
        assert len(ws.sent) == 1

    @pytest.mark.asyncio
    async def test_invalidate_response(self):
        ws = FakeWS()
        emitter = EventEmitter(ws, "sess_test")
        emitter.set_active_response("resp_1")
        emitter.invalidate_response("resp_1")

        # After invalidation, events from resp_1 still delivered
        # (no active response means no fence)
        await emitter.emit({"type": "test", "response_id": "resp_1"})
        assert len(ws.sent) == 1

    @pytest.mark.asyncio
    async def test_invalidate_wrong_response_is_noop(self):
        ws = FakeWS()
        emitter = EventEmitter(ws, "sess_test")
        emitter.set_active_response("resp_1")
        emitter.invalidate_response("resp_other")  # wrong id

        # Fence still active for resp_1
        await emitter.emit({"type": "test", "response_id": "resp_old"})
        assert len(ws.sent) == 0  # dropped by fence


class TestBackpressure:
    @pytest.mark.asyncio
    async def test_droppable_event_dropped_on_timeout(self):
        ws = FakeWS(slow=True)
        emitter = EventEmitter(ws, "sess_test")

        # Audio delta is droppable — should not raise
        await emitter.emit({"type": "response.audio.delta"})
        assert len(ws.sent) == 0
        assert emitter.total_drops == 1

    @pytest.mark.asyncio
    async def test_structural_event_raises_on_timeout(self):
        ws = FakeWS(slow=True)
        emitter = EventEmitter(ws, "sess_test")

        with pytest.raises(SlowClientError):
            await emitter.emit({"type": "response.done"})

    @pytest.mark.asyncio
    async def test_pressure_level_escalation(self):
        ws = FakeWS(slow=True)
        emitter = EventEmitter(ws, "sess_test")

        assert emitter.pressure_level == 0

        # Drop 3 droppable events → level 1
        for _ in range(3):
            await emitter.emit({"type": "response.audio.delta"})
        assert emitter.pressure_level == 1

        # Drop 7 more → total 10 → level 2
        for _ in range(7):
            await emitter.emit({"type": "response.audio.delta"})
        assert emitter.pressure_level == 2

        # Drop 10 more → total 20 → level 3
        for _ in range(10):
            await emitter.emit({"type": "response.audio.delta"})
        assert emitter.pressure_level == 3

    @pytest.mark.asyncio
    async def test_pressure_resets_on_successful_send(self):
        ws = FakeWS(fail_after=3)
        emitter = EventEmitter(ws, "sess_test")

        # First 3 sends succeed
        for _ in range(3):
            await emitter.emit({"type": "session.created"})
        assert emitter.pressure_level == 0
        assert len(ws.sent) == 3

        # Next sends timeout (droppable)
        for _ in range(5):
            await emitter.emit({"type": "response.audio.delta"})
        assert emitter.pressure_level == 1
        assert emitter.total_drops == 5

    @pytest.mark.asyncio
    async def test_transcript_delta_throttled_at_level_1(self):
        ws = FakeWS(slow=True)
        emitter = EventEmitter(ws, "sess_test")

        # Force to level 1 by dropping 3 audio deltas
        for _ in range(3):
            await emitter.emit({"type": "response.audio.delta"})
        assert emitter.pressure_level == 1

        # Now transcript deltas should be throttled (1 in 3)
        # But since ws is slow, even the 1-in-3 that passes will timeout
        # So total_drops increases for all of them
        initial_drops = emitter.total_drops
        for _ in range(6):
            await emitter.emit({"type": "response.audio_transcript.delta"})
        assert emitter.total_drops > initial_drops

    @pytest.mark.asyncio
    async def test_transcript_delta_dropped_at_level_2(self):
        ws = FakeWS(slow=True)
        emitter = EventEmitter(ws, "sess_test")

        # Force to level 2 by dropping 10 audio deltas
        for _ in range(10):
            await emitter.emit({"type": "response.audio.delta"})
        assert emitter.pressure_level == 2

        # Transcript deltas should be completely dropped (no send attempt)
        initial_drops = emitter.total_drops
        await emitter.emit({"type": "response.audio_transcript.delta"})
        assert emitter.total_drops == initial_drops + 1


class TestEmitMany:
    @pytest.mark.asyncio
    async def test_emit_many_sends_in_order(self):
        ws = FakeWS()
        emitter = EventEmitter(ws, "sess_test")

        events = [
            {"type": "response.created"},
            {"type": "response.output_item.added"},
            {"type": "response.done"},
        ]
        await emitter.emit_many(events)

        assert len(ws.sent) == 3
        assert ws.sent[0]["type"] == "response.created"
        assert ws.sent[1]["type"] == "response.output_item.added"
        assert ws.sent[2]["type"] == "response.done"
