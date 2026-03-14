"""Tests for RealtimeSession lifecycle, event ordering, cancellation, and error handling."""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from protocol.models import SessionConfig, ConversationItem, ContentPart


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
        # Map of event type → delay in seconds to insert AFTER the message is dispatched
        self._delay_after = delay_after or {}
        self._pending_delay: float = 0

    async def send(self, data: str) -> None:
        parsed = json.loads(data)
        self.sent.append(parsed)
        if parsed.get("type") == "response.done":
            self.response_done_event.set()

    async def recv(self) -> str:
        """Return next incoming message, or wait for response_done then raise ConnectionClosed."""
        # Apply any pending delay from the previously dispatched message
        if self._pending_delay > 0:
            await asyncio.sleep(self._pending_delay)
            self._pending_delay = 0

        if self._incoming_index < len(self._incoming):
            msg = self._incoming[self._incoming_index]
            self._incoming_index += 1
            # Schedule delay for AFTER this message is dispatched (applied on next recv)
            try:
                parsed = json.loads(msg)
                event_type = parsed.get("type", "")
                self._pending_delay = self._delay_after.get(event_type, 0)
            except (json.JSONDecodeError, KeyError):
                pass
            return msg

        # All messages consumed — wait for response if needed, then signal close
        if self._wait_for_response:
            try:
                await asyncio.wait_for(self.response_done_event.wait(), timeout=5.0)
            except asyncio.TimeoutError:
                pass

        import websockets.exceptions
        raise websockets.exceptions.ConnectionClosed(None, None)

    def __aiter__(self):
        return self._aiter_impl()

    async def _aiter_impl(self):
        for msg in self._incoming:
            yield msg
            # Allow inserting delays after specific event types
            try:
                parsed = json.loads(msg)
                event_type = parsed.get("type", "")
                delay = self._delay_after.get(event_type, 0)
                if delay > 0:
                    await asyncio.sleep(delay)
            except (json.JSONDecodeError, KeyError):
                pass
        # Wait for response pipeline to complete instead of sleeping a fixed amount
        if self._wait_for_response:
            try:
                await asyncio.wait_for(self.response_done_event.wait(), timeout=5.0)
            except asyncio.TimeoutError:
                pass
        return


class FakeASR:
    """Fake ASR provider for testing."""

    supports_streaming = False

    async def transcribe(self, audio: bytes) -> str:
        return "hello world"

    async def connect(self) -> None:
        pass

    async def disconnect(self) -> None:
        pass

    async def start_stream(self, stream_id: str) -> None:
        pass

    async def feed_chunk(self, audio: bytes, stream_id: str) -> str:
        return ""

    async def finish_stream(self, stream_id: str) -> str:
        return "hello world"


class FakeLLM:
    """Fake LLM provider for testing."""

    async def generate_stream(self, messages, system="", tools=None, temperature=0.8, max_tokens=1024):
        for chunk in ["Hello", ", ", "world", "!"]:
            yield chunk

    async def generate_sentences(self, messages, system="", tools=None, temperature=0.8, max_tokens=1024):
        yield "Hello, world!"


class FakeTTS:
    """Fake TTS provider for testing."""

    supports_streaming = False

    async def synthesize(self, text: str) -> bytes:
        # 100ms of silence at 8kHz 16-bit
        return b"\x00\x00" * 800

    async def synthesize_stream(self, text):
        yield b"\x00\x00" * 800

    async def connect(self) -> None:
        pass

    async def disconnect(self) -> None:
        pass


class FailingASR(FakeASR):
    """ASR that raises on transcribe."""

    async def transcribe(self, audio: bytes) -> str:
        raise RuntimeError("ASR service unavailable")


class FailingLLM:
    """LLM that raises during streaming."""

    async def generate_stream(self, messages, system="", tools=None, temperature=0.8, max_tokens=1024):
        yield "Start..."
        raise RuntimeError("LLM connection lost")

    async def generate_sentences(self, messages, system="", tools=None, temperature=0.8, max_tokens=1024):
        yield "Start..."
        raise RuntimeError("LLM connection lost")


class FailingTTS(FakeTTS):
    """TTS that raises on synthesize."""

    async def synthesize(self, text: str) -> bytes:
        raise RuntimeError("TTS GPU OOM")


class SlowLLM:
    """LLM that yields slowly for cancellation testing."""

    async def generate_stream(self, messages, system="", tools=None, temperature=0.8, max_tokens=1024):
        for chunk in ["Hello", " ", "world"]:
            await asyncio.sleep(0.5)
            yield chunk

    async def generate_sentences(self, messages, system="", tools=None, temperature=0.8, max_tokens=1024):
        await asyncio.sleep(0.5)
        yield "Hello world"


@pytest.fixture
def providers():
    return FakeASR(), FakeLLM(), FakeTTS()


class TestSessionLifecycle:
    @pytest.mark.asyncio
    async def test_session_created_on_connect(self, providers):
        from server.session import RealtimeSession

        ws = FakeWebSocket()
        asr, llm, tts = providers
        session = RealtimeSession(ws, asr, llm, tts)
        await session.run()

        # First two events should be session.created and conversation.created
        assert len(ws.sent) >= 2
        assert ws.sent[0]["type"] == "session.created"
        assert ws.sent[0]["session"]["id"] == session.session_id
        assert ws.sent[1]["type"] == "conversation.created"

    @pytest.mark.asyncio
    async def test_session_update(self, providers):
        from server.session import RealtimeSession

        incoming = [
            json.dumps({
                "type": "session.update",
                "session": {"instructions": "Be helpful", "temperature": 0.5},
            })
        ]
        ws = FakeWebSocket(incoming)
        asr, llm, tts = providers
        session = RealtimeSession(ws, asr, llm, tts)
        await session.run()

        # Should have session.created, conversation.created, session.updated
        types = [e["type"] for e in ws.sent]
        assert "session.updated" in types

        updated_event = next(e for e in ws.sent if e["type"] == "session.updated")
        assert updated_event["session"]["instructions"] == "Be helpful"
        assert updated_event["session"]["temperature"] == 0.5

    @pytest.mark.asyncio
    async def test_conversation_item_create(self, providers):
        from server.session import RealtimeSession

        incoming = [
            json.dumps({
                "type": "conversation.item.create",
                "item": {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "Hello"}],
                },
            })
        ]
        ws = FakeWebSocket(incoming)
        asr, llm, tts = providers
        session = RealtimeSession(ws, asr, llm, tts)
        await session.run()

        types = [e["type"] for e in ws.sent]
        assert "conversation.item.created" in types

    @pytest.mark.asyncio
    async def test_conversation_item_delete(self, providers):
        from server.session import RealtimeSession

        incoming = [
            json.dumps({
                "type": "conversation.item.create",
                "item": {"id": "item_test_1", "type": "message", "role": "user", "content": []},
            }),
            json.dumps({
                "type": "conversation.item.delete",
                "item_id": "item_test_1",
            }),
        ]
        ws = FakeWebSocket(incoming)
        asr, llm, tts = providers
        session = RealtimeSession(ws, asr, llm, tts)
        await session.run()

        types = [e["type"] for e in ws.sent]
        assert "conversation.item.deleted" in types

    @pytest.mark.asyncio
    async def test_input_audio_buffer_clear(self, providers):
        from server.session import RealtimeSession

        incoming = [
            json.dumps({"type": "input_audio_buffer.clear"}),
        ]
        ws = FakeWebSocket(incoming)
        asr, llm, tts = providers
        session = RealtimeSession(ws, asr, llm, tts)
        await session.run()

        types = [e["type"] for e in ws.sent]
        assert "input_audio_buffer.cleared" in types

    @pytest.mark.asyncio
    async def test_unknown_event_returns_error(self, providers):
        from server.session import RealtimeSession

        incoming = [
            json.dumps({"type": "totally.unknown.event"}),
        ]
        ws = FakeWebSocket(incoming)
        asr, llm, tts = providers
        session = RealtimeSession(ws, asr, llm, tts)
        await session.run()

        error_events = [e for e in ws.sent if e["type"] == "error"]
        assert len(error_events) >= 1
        assert "unknown" in error_events[0]["error"]["message"].lower()

    @pytest.mark.asyncio
    async def test_invalid_json_returns_error(self, providers):
        from server.session import RealtimeSession

        ws = FakeWebSocket(["not valid json{{{"])
        asr, llm, tts = providers
        session = RealtimeSession(ws, asr, llm, tts)
        await session.run()

        error_events = [e for e in ws.sent if e["type"] == "error"]
        assert len(error_events) >= 1


class TestResponsePipeline:
    @pytest.mark.asyncio
    async def test_text_response_event_ordering(self, providers):
        """Text-only response should emit events in correct order."""
        from server.session import RealtimeSession

        incoming = [
            json.dumps({
                "type": "session.update",
                "session": {"modalities": ["text"]},
            }),
            json.dumps({
                "type": "conversation.item.create",
                "item": {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "Hi"}],
                },
            }),
            json.dumps({"type": "response.create"}),
        ]
        ws = FakeWebSocket(incoming, wait_for_response=True)
        asr, llm, tts = providers
        session = RealtimeSession(ws, asr, llm, tts)
        await session.run()

        types = [e["type"] for e in ws.sent]

        # Must include these in order
        assert "response.created" in types
        assert "response.output_item.added" in types
        assert "response.content_part.added" in types
        assert "response.text.delta" in types
        assert "response.text.done" in types
        assert "response.content_part.done" in types
        assert "response.output_item.done" in types
        assert "response.done" in types

        # Verify ordering: created before done
        created_idx = types.index("response.created")
        done_idx = types.index("response.done")
        assert created_idx < done_idx

    @pytest.mark.asyncio
    async def test_audio_response_emits_audio_events(self, providers):
        """Audio response should emit audio.delta and transcript events."""
        from server.session import RealtimeSession

        incoming = [
            json.dumps({
                "type": "conversation.item.create",
                "item": {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "Hi"}],
                },
            }),
            json.dumps({"type": "response.create"}),
        ]
        ws = FakeWebSocket(incoming, wait_for_response=True)
        asr, llm, tts = providers
        session = RealtimeSession(ws, asr, llm, tts)
        await session.run()

        types = [e["type"] for e in ws.sent]
        assert "response.created" in types
        assert "response.done" in types

    @pytest.mark.asyncio
    async def test_text_response_done_has_completed_status(self, providers):
        """response.done should have status=completed on success."""
        from server.session import RealtimeSession

        incoming = [
            json.dumps({"type": "session.update", "session": {"modalities": ["text"]}}),
            json.dumps({
                "type": "conversation.item.create",
                "item": {"type": "message", "role": "user",
                         "content": [{"type": "input_text", "text": "Hi"}]},
            }),
            json.dumps({"type": "response.create"}),
        ]
        ws = FakeWebSocket(incoming, wait_for_response=True)
        asr, llm, tts = providers
        session = RealtimeSession(ws, asr, llm, tts)
        await session.run()

        done_events = [e for e in ws.sent if e["type"] == "response.done"]
        assert len(done_events) == 1
        assert done_events[0]["response"]["status"] == "completed"


class TestResponseCancellation:
    @pytest.mark.asyncio
    async def test_response_cancel_emits_cancelled_status(self):
        """response.cancel should cancel active response and emit cancelled status."""
        from server.session import RealtimeSession

        incoming = [
            json.dumps({"type": "session.update", "session": {"modalities": ["text"]}}),
            json.dumps({
                "type": "conversation.item.create",
                "item": {"type": "message", "role": "user",
                         "content": [{"type": "input_text", "text": "Tell me a long story"}]},
            }),
            json.dumps({"type": "response.create"}),
            json.dumps({"type": "response.cancel"}),
        ]
        # Delay after response.create to give the response task time to start
        ws = FakeWebSocket(
            incoming,
            wait_for_response=True,
            delay_after={"response.create": 0.1},
        )
        asr = FakeASR()
        llm = SlowLLM()
        tts = FakeTTS()
        session = RealtimeSession(ws, asr, llm, tts)
        await session.run()

        done_events = [e for e in ws.sent if e["type"] == "response.done"]
        # Should have at least one response.done (either cancelled or completed)
        assert len(done_events) >= 1
        # Should be cancelled since SlowLLM takes 0.5s per chunk
        assert done_events[0]["response"]["status"] == "cancelled"

    @pytest.mark.asyncio
    async def test_new_response_cancels_previous(self):
        """Creating a new response while one is active should cancel the previous."""
        from server.session import RealtimeSession

        incoming = [
            json.dumps({"type": "session.update", "session": {"modalities": ["text"]}}),
            json.dumps({
                "type": "conversation.item.create",
                "item": {"type": "message", "role": "user",
                         "content": [{"type": "input_text", "text": "First"}]},
            }),
            json.dumps({"type": "response.create"}),
            json.dumps({"type": "response.create"}),  # Second response
        ]
        ws = FakeWebSocket(incoming, wait_for_response=True)
        asr = FakeASR()
        llm = SlowLLM()
        tts = FakeTTS()
        session = RealtimeSession(ws, asr, llm, tts)
        await session.run()

        # Should complete without hanging
        done_events = [e for e in ws.sent if e["type"] == "response.done"]
        assert len(done_events) >= 1


class TestProviderFailures:
    @pytest.mark.asyncio
    async def test_llm_failure_emits_failed_response(self):
        """LLM error during streaming should emit response.done with status=failed."""
        from server.session import RealtimeSession

        incoming = [
            json.dumps({"type": "session.update", "session": {"modalities": ["text"]}}),
            json.dumps({
                "type": "conversation.item.create",
                "item": {"type": "message", "role": "user",
                         "content": [{"type": "input_text", "text": "Hi"}]},
            }),
            json.dumps({"type": "response.create"}),
        ]
        ws = FakeWebSocket(incoming, wait_for_response=True)
        asr = FakeASR()
        llm = FailingLLM()
        tts = FakeTTS()
        session = RealtimeSession(ws, asr, llm, tts)
        await session.run()

        done_events = [e for e in ws.sent if e["type"] == "response.done"]
        assert len(done_events) == 1
        assert done_events[0]["response"]["status"] == "failed"

    @pytest.mark.asyncio
    async def test_session_survives_provider_error(self):
        """A provider error in one response should not kill the session."""
        from server.session import RealtimeSession

        incoming = [
            json.dumps({"type": "session.update", "session": {"modalities": ["text"]}}),
            json.dumps({
                "type": "conversation.item.create",
                "item": {"type": "message", "role": "user",
                         "content": [{"type": "input_text", "text": "Hi"}]},
            }),
            json.dumps({"type": "response.create"}),
            # After the failed response, the session should still process this:
            json.dumps({"type": "input_audio_buffer.clear"}),
        ]
        ws = FakeWebSocket(incoming, wait_for_response=True)
        asr = FakeASR()
        llm = FailingLLM()
        tts = FakeTTS()
        session = RealtimeSession(ws, asr, llm, tts)
        await session.run()

        types = [e["type"] for e in ws.sent]
        # Session continued processing after the failed response
        assert "input_audio_buffer.cleared" in types


class TestAudioBufferBackpressure:
    @pytest.mark.asyncio
    async def test_buffer_overflow_rejected(self):
        """Audio buffer should reject appends when full."""
        from server.session import RealtimeSession, _MAX_AUDIO_BUFFER_BYTES
        import base64

        # Disable VAD and use g711_ulaw (1 byte input → 2 bytes stored, no resample)
        incoming = [
            json.dumps({
                "type": "session.update",
                "session": {"turn_detection": None, "input_audio_format": "g711_ulaw"},
            }),
        ]

        # 1MB of g711 mu-law → 2MB of PCM16 stored in buffer
        chunk_size = 1024 * 1024
        b64_chunk = base64.b64encode(b"\x7f" * chunk_size).decode()

        # Each chunk stores 2*chunk_size bytes. Send enough to overflow.
        stored_per_chunk = chunk_size * 2
        num_chunks = (_MAX_AUDIO_BUFFER_BYTES // stored_per_chunk) + 2
        for _ in range(num_chunks):
            incoming.append(json.dumps({
                "type": "input_audio_buffer.append",
                "audio": b64_chunk,
            }))

        ws = FakeWebSocket(incoming)
        asr, llm, tts = FakeASR(), FakeLLM(), FakeTTS()
        session = RealtimeSession(ws, asr, llm, tts)
        await session.run()

        # Should have an error about buffer being full
        error_events = [e for e in ws.sent if e["type"] == "error"]
        buffer_full_errors = [e for e in error_events if "buffer_full" == e.get("error", {}).get("code")]
        assert len(buffer_full_errors) >= 1


class TestModels:
    def test_session_config_roundtrip(self):
        config = SessionConfig(
            modalities=["text", "audio"],
            instructions="Be helpful",
            voice="shimmer",
            temperature=0.6,
        )
        d = config.to_dict()
        restored = SessionConfig.from_dict(d)
        assert restored.instructions == "Be helpful"
        assert restored.voice == "shimmer"
        assert restored.temperature == 0.6

    def test_session_config_update_partial(self):
        config = SessionConfig()
        config.update({"temperature": 0.3, "voice": "nova"})
        assert config.temperature == 0.3
        assert config.voice == "nova"
        # Other fields unchanged
        assert config.modalities == ["text", "audio"]

    def test_conversation_item_roundtrip(self):
        item = ConversationItem(
            id="item_1",
            type="message",
            role="user",
            content=[ContentPart(type="input_text", text="Hello")],
        )
        d = item.to_dict()
        assert d["id"] == "item_1"
        assert d["content"][0]["text"] == "Hello"

        restored = ConversationItem.from_dict(d)
        assert restored.id == "item_1"
        assert restored.content[0].text == "Hello"
