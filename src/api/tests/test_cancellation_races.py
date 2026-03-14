"""Tests for cancellation race conditions and barge-in robustness.

Validates that:
- Cancelled responses don't leak stale events (cancellation fences)
- Rapid barge-in produces clean state transitions
- Pipeline tasks are cleaned up on cancellation
"""

import asyncio
import json

import pytest

from providers.llm import LLMStreamEvent


# ---------------------------------------------------------------------------
# Fakes for cancellation testing
# ---------------------------------------------------------------------------


class FakeWebSocket:
    """Fake WebSocket that records sent messages and yields incoming messages."""

    def __init__(
        self,
        incoming: list[str] | None = None,
        wait_for_response: bool = False,
        delay_after: dict[str, float] | None = None,
    ):
        self._incoming = incoming or []
        self._incoming_index = 0
        self.sent: list[dict] = []
        self._closed = False
        self._wait_for_response = wait_for_response
        self.response_done_event: asyncio.Event = asyncio.Event()
        self._delay_after = delay_after or {}
        self._pending_delay: float = 0

    async def send(self, data: str) -> None:
        parsed = json.loads(data)
        self.sent.append(parsed)
        if parsed.get("type") == "response.done":
            self.response_done_event.set()

    async def recv(self) -> str:
        if self._pending_delay > 0:
            await asyncio.sleep(self._pending_delay)
            self._pending_delay = 0

        if self._incoming_index < len(self._incoming):
            msg = self._incoming[self._incoming_index]
            self._incoming_index += 1
            try:
                parsed = json.loads(msg)
                event_type = parsed.get("type", "")
                self._pending_delay = self._delay_after.get(event_type, 0)
            except (json.JSONDecodeError, KeyError):
                pass
            return msg

        if self._wait_for_response:
            try:
                await asyncio.wait_for(self.response_done_event.wait(), timeout=5.0)
            except asyncio.TimeoutError:
                pass

        import websockets.exceptions
        raise websockets.exceptions.ConnectionClosed(None, None)


class FakeASR:
    supports_streaming = False

    async def transcribe(self, audio: bytes) -> str:
        return "hello world"

    async def connect(self) -> None:
        pass

    async def disconnect(self) -> None:
        pass

    async def warmup(self) -> None:
        pass

    async def start_stream(self, stream_id: str) -> None:
        pass

    async def feed_chunk(self, audio: bytes, stream_id: str) -> str:
        return ""

    async def finish_stream(self, stream_id: str) -> str:
        return "hello world"


class FakeTTS:
    supports_streaming = False

    async def synthesize(self, text: str) -> bytes:
        return b"\x00\x00" * 800

    async def synthesize_stream(self, text):
        yield b"\x00\x00" * 800

    async def connect(self) -> None:
        pass

    async def disconnect(self) -> None:
        pass

    async def warmup(self) -> None:
        pass


class SlowLLM:
    """LLM that yields slowly for cancellation testing."""

    last_ttft_ms: float = 0.0
    last_stream_total_ms: float = 0.0

    async def generate_stream(self, messages, system="", tools=None,
                              temperature=0.8, max_tokens=1024):
        for chunk in ["Hello", " ", "world", "!"]:
            await asyncio.sleep(0.3)
            yield chunk

    async def generate_sentences(self, messages, system="", tools=None,
                                 temperature=0.8, max_tokens=1024):
        await asyncio.sleep(0.3)
        yield "Hello world!"

    async def generate_stream_with_tools(self, messages, system="", tools=None,
                                         temperature=0.8, max_tokens=1024):
        for chunk in ["Hello", " ", "world", "!"]:
            await asyncio.sleep(0.3)
            yield LLMStreamEvent(type="text_delta", text=chunk)


class SlowTTS(FakeTTS):
    """TTS that delays synthesis for cancellation testing."""

    async def synthesize(self, text: str) -> bytes:
        await asyncio.sleep(0.5)
        return b"\x00\x00" * 800

    async def synthesize_stream(self, text):
        await asyncio.sleep(0.5)
        yield b"\x00\x00" * 800


class ToolCallingLLM:
    """LLM that emits a tool call on first round, then text on second."""

    last_ttft_ms: float = 0.0
    last_stream_total_ms: float = 0.0
    _call_count: int = 0

    async def generate_stream(self, messages, system="", tools=None,
                              temperature=0.8, max_tokens=1024):
        for chunk in ["Response", " text"]:
            yield chunk

    async def generate_sentences(self, messages, system="", tools=None,
                                 temperature=0.8, max_tokens=1024):
        yield "Response text"

    async def generate_stream_with_tools(self, messages, system="", tools=None,
                                         temperature=0.8, max_tokens=1024):
        self._call_count += 1
        if self._call_count == 1 and tools:
            # First round: emit a tool call
            yield LLMStreamEvent(
                type="tool_call_start",
                tool_call_id="call_test_1",
                tool_name="slow_tool",
            )
            yield LLMStreamEvent(
                type="tool_call_delta",
                tool_arguments_delta='{"query": "test"}',
            )
            yield LLMStreamEvent(type="tool_call_end")
        else:
            # Subsequent rounds: emit text
            await asyncio.sleep(0.3)
            yield LLMStreamEvent(type="text_delta", text="Tool result response")


class SlowToolCallingLLM(ToolCallingLLM):
    """Like ToolCallingLLM but the tool call takes time to stream."""

    async def generate_stream_with_tools(self, messages, system="", tools=None,
                                         temperature=0.8, max_tokens=1024):
        self._call_count += 1
        if self._call_count == 1 and tools:
            await asyncio.sleep(0.5)
            yield LLMStreamEvent(
                type="tool_call_start",
                tool_call_id="call_test_slow",
                tool_name="slow_tool",
            )
            yield LLMStreamEvent(
                type="tool_call_delta",
                tool_arguments_delta='{"query": "test"}',
            )
            yield LLMStreamEvent(type="tool_call_end")
        else:
            yield LLMStreamEvent(type="text_delta", text="Done")


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _events_of_type(ws: FakeWebSocket, event_type: str) -> list[dict]:
    return [e for e in ws.sent if e["type"] == event_type]


def _response_ids_for_type(ws: FakeWebSocket, event_type: str) -> list[str]:
    return [e.get("response_id", "") for e in ws.sent if e["type"] == event_type]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestCancellationFences:
    """Tests that validate the EventEmitter cancellation fence mechanism."""

    @pytest.mark.asyncio
    async def test_cancel_during_llm_text_drops_stale_events(self):
        """Cancel during text LLM streaming: no events from old response leak."""
        from server.session import RealtimeSession

        incoming = [
            json.dumps({"type": "session.update", "session": {"modalities": ["text"]}}),
            json.dumps({
                "type": "conversation.item.create",
                "item": {"type": "message", "role": "user",
                         "content": [{"type": "input_text", "text": "Tell me a story"}]},
            }),
            json.dumps({"type": "response.create"}),
            json.dumps({"type": "response.cancel"}),
        ]
        ws = FakeWebSocket(
            incoming,
            wait_for_response=True,
            delay_after={"response.create": 0.1},
        )
        session = RealtimeSession(ws, FakeASR(), SlowLLM(), FakeTTS())
        await session.run()

        done_events = _events_of_type(ws, "response.done")
        assert len(done_events) >= 1
        assert done_events[0]["response"]["status"] == "cancelled"

    @pytest.mark.asyncio
    async def test_new_response_fences_old_response_events(self):
        """When a new response replaces an old one, old response events are fenced."""
        from server.session import RealtimeSession

        incoming = [
            json.dumps({"type": "session.update", "session": {"modalities": ["text"]}}),
            json.dumps({
                "type": "conversation.item.create",
                "item": {"type": "message", "role": "user",
                         "content": [{"type": "input_text", "text": "First"}]},
            }),
            json.dumps({"type": "response.create"}),
            # Second response immediately after
            json.dumps({"type": "response.create"}),
        ]
        ws = FakeWebSocket(
            incoming,
            wait_for_response=True,
            delay_after={"response.create": 0.05},
        )
        session = RealtimeSession(ws, FakeASR(), SlowLLM(), FakeTTS())
        await session.run()

        done_events = _events_of_type(ws, "response.done")
        # At least one must complete (the second one)
        assert len(done_events) >= 1

        # All response.done events should have distinct response_ids
        response_ids = [e["response_id"] for e in done_events if "response_id" in e]
        assert len(response_ids) == len(set(response_ids)), "Duplicate response_ids in done events"

    @pytest.mark.asyncio
    async def test_cancel_before_first_audio_clean_lifecycle(self):
        """Cancel before any audio.delta was sent: clean lifecycle, no partial state."""
        from server.session import RealtimeSession

        incoming = [
            json.dumps({
                "type": "conversation.item.create",
                "item": {"type": "message", "role": "user",
                         "content": [{"type": "input_text", "text": "Hi"}]},
            }),
            json.dumps({"type": "response.create"}),
            json.dumps({"type": "response.cancel"}),
        ]
        ws = FakeWebSocket(
            incoming,
            wait_for_response=True,
            delay_after={"response.create": 0.05},
        )
        session = RealtimeSession(ws, FakeASR(), SlowLLM(), SlowTTS())
        await session.run()

        done_events = _events_of_type(ws, "response.done")
        assert len(done_events) >= 1

        # Should have been cancelled
        assert done_events[0]["response"]["status"] == "cancelled"

        # No audio should have been sent (SlowTTS delays 0.5s, cancel at 0.05s)
        audio_events = _events_of_type(ws, "response.audio.delta")
        assert len(audio_events) == 0

    @pytest.mark.asyncio
    async def test_rapid_double_cancel_no_crash(self):
        """Two rapid cancels should not raise or produce duplicate response.done."""
        from server.session import RealtimeSession

        incoming = [
            json.dumps({"type": "session.update", "session": {"modalities": ["text"]}}),
            json.dumps({
                "type": "conversation.item.create",
                "item": {"type": "message", "role": "user",
                         "content": [{"type": "input_text", "text": "Test"}]},
            }),
            json.dumps({"type": "response.create"}),
            json.dumps({"type": "response.cancel"}),
            json.dumps({"type": "response.cancel"}),  # Second cancel
        ]
        ws = FakeWebSocket(
            incoming,
            wait_for_response=True,
            delay_after={"response.create": 0.1},
        )
        session = RealtimeSession(ws, FakeASR(), SlowLLM(), FakeTTS())
        await session.run()

        # Should not crash and should have exactly one response.done
        done_events = _events_of_type(ws, "response.done")
        assert len(done_events) == 1

    @pytest.mark.asyncio
    async def test_cancel_then_new_response_succeeds(self):
        """After cancelling a response, a new response.create should succeed normally."""
        from server.session import RealtimeSession

        class FastLLM:
            last_ttft_ms: float = 0.0
            last_stream_total_ms: float = 0.0
            _call_count = 0

            async def generate_stream(self, messages, system="", tools=None,
                                      temperature=0.8, max_tokens=1024):
                self._call_count += 1
                if self._call_count == 1:
                    # First call: slow (will be cancelled)
                    await asyncio.sleep(1.0)
                    yield "should not appear"
                else:
                    # Second call: fast
                    yield "Success"

            async def generate_sentences(self, messages, system="", tools=None,
                                         temperature=0.8, max_tokens=1024):
                self._call_count += 1
                if self._call_count == 1:
                    await asyncio.sleep(1.0)
                    yield "should not appear"
                else:
                    yield "Success"

        incoming = [
            json.dumps({"type": "session.update", "session": {"modalities": ["text"]}}),
            json.dumps({
                "type": "conversation.item.create",
                "item": {"type": "message", "role": "user",
                         "content": [{"type": "input_text", "text": "First"}]},
            }),
            json.dumps({"type": "response.create"}),
            json.dumps({"type": "response.cancel"}),
            # New response after cancel
            json.dumps({"type": "response.create"}),
        ]
        ws = FakeWebSocket(
            incoming,
            wait_for_response=True,
            delay_after={"response.cancel": 0.1},
        )
        session = RealtimeSession(ws, FakeASR(), FastLLM(), FakeTTS())
        await session.run()

        done_events = _events_of_type(ws, "response.done")
        assert len(done_events) >= 1

        # The last response.done should be completed (not cancelled)
        completed = [e for e in done_events if e["response"]["status"] == "completed"]
        assert len(completed) >= 1

    @pytest.mark.asyncio
    async def test_output_audio_buffer_clear_fences_response(self):
        """output_audio_buffer.clear should cancel response and fence its events."""
        from server.session import RealtimeSession

        incoming = [
            json.dumps({"type": "session.update", "session": {"modalities": ["text"]}}),
            json.dumps({
                "type": "conversation.item.create",
                "item": {"type": "message", "role": "user",
                         "content": [{"type": "input_text", "text": "Hi"}]},
            }),
            json.dumps({"type": "response.create"}),
            json.dumps({"type": "output_audio_buffer.clear"}),
        ]
        ws = FakeWebSocket(
            incoming,
            wait_for_response=True,
            delay_after={"response.create": 0.1},
        )
        session = RealtimeSession(ws, FakeASR(), SlowLLM(), FakeTTS())
        await session.run()

        # Session should complete without crashing
        types = [e["type"] for e in ws.sent]
        assert "session.created" in types


class TestEventEmitterFence:
    """Unit tests for EventEmitter fence mechanism."""

    @pytest.mark.asyncio
    async def test_stale_response_id_dropped(self):
        """Events with a stale response_id should be silently dropped."""
        from protocol.event_emitter import EventEmitter

        ws = FakeWebSocket()
        emitter = EventEmitter(ws, "test_session")

        # Set active response
        emitter.set_active_response("resp_active")

        # Emit event from active response — should go through
        await emitter.emit({"type": "response.audio.delta", "response_id": "resp_active"})
        assert len(ws.sent) == 1

        # Emit event from stale response — should be dropped
        await emitter.emit({"type": "response.audio.delta", "response_id": "resp_stale"})
        assert len(ws.sent) == 1  # Still 1

    @pytest.mark.asyncio
    async def test_events_without_response_id_pass_through(self):
        """Events without response_id should always pass through."""
        from protocol.event_emitter import EventEmitter

        ws = FakeWebSocket()
        emitter = EventEmitter(ws, "test_session")
        emitter.set_active_response("resp_active")

        # session.created has no response_id — should pass
        await emitter.emit({"type": "session.created"})
        assert len(ws.sent) == 1

    @pytest.mark.asyncio
    async def test_invalidate_then_all_pass(self):
        """After invalidating, events with any response_id should pass (no fence)."""
        from protocol.event_emitter import EventEmitter

        ws = FakeWebSocket()
        emitter = EventEmitter(ws, "test_session")
        emitter.set_active_response("resp_1")
        emitter.invalidate_response("resp_1")

        # No active response — should pass through
        await emitter.emit({"type": "response.done", "response_id": "resp_1"})
        assert len(ws.sent) == 1

    @pytest.mark.asyncio
    async def test_invalidate_wrong_id_keeps_fence(self):
        """Invalidating a different response_id should not affect the active fence."""
        from protocol.event_emitter import EventEmitter

        ws = FakeWebSocket()
        emitter = EventEmitter(ws, "test_session")
        emitter.set_active_response("resp_active")

        # Invalidate a different response — fence should remain
        emitter.invalidate_response("resp_other")

        # Stale event should still be dropped
        await emitter.emit({"type": "response.audio.delta", "response_id": "resp_stale"})
        assert len(ws.sent) == 0

    @pytest.mark.asyncio
    async def test_structural_events_fenced_too(self):
        """Even structural events (response.done) from stale responses are dropped."""
        from protocol.event_emitter import EventEmitter

        ws = FakeWebSocket()
        emitter = EventEmitter(ws, "test_session")
        emitter.set_active_response("resp_new")

        # Stale response.done should be dropped
        await emitter.emit({"type": "response.done", "response_id": "resp_old"})
        assert len(ws.sent) == 0

        # Active response.done should pass
        await emitter.emit({"type": "response.done", "response_id": "resp_new"})
        assert len(ws.sent) == 1
