"""Tests for AudioEmitter — unified TTS-to-WebSocket audio emission."""

import asyncio

import pytest

from protocol.models import ContentPart, ConversationItem
from server.audio_emitter import AudioEmitter


class FakeEmitter:
    """Collects emitted events for inspection."""

    def __init__(self):
        self.events: list[dict] = []

    async def emit(self, event: dict) -> None:
        self.events.append(event)


class FakeTTS:
    """Fake TTS that yields deterministic audio chunks."""

    def __init__(self, chunks: list[bytes] | None = None):
        self._chunks = chunks or [b"\x00\x01" * 100]
        self.supports_streaming = True

    async def synthesize_stream(self, text: str):
        for chunk in self._chunks:
            yield chunk


class TestAudioEmitterFromText:
    @pytest.fixture
    def emitter(self):
        return FakeEmitter()

    @pytest.fixture
    def tts(self):
        return FakeTTS()

    async def test_emits_audio_events(self, emitter, tts):
        ae = AudioEmitter(emitter, tts, "pcm16", on_first_audio=None)
        lock = asyncio.Lock()
        items = []
        item_id, transcript = await ae.emit_from_text(
            "Olá mundo", "resp_1", 0, lock, items.append
        )
        assert transcript == "Olá mundo"
        assert item_id.startswith("item_")
        assert len(items) == 1
        assert items[0].role == "assistant"

        # Should have: output_item_added, audio_delta(s), transcript_delta,
        # audio_done, transcript_done, content_part_done, output_item_done
        event_types = [e["type"] for e in emitter.events]
        assert "response.output_item.added" in event_types
        assert "response.audio.delta" in event_types
        assert "response.audio.done" in event_types
        assert "response.output_item.done" in event_types

    async def test_calls_on_first_audio(self, emitter, tts):
        called = []
        ae = AudioEmitter(emitter, tts, "pcm16", on_first_audio=lambda: called.append(True))
        lock = asyncio.Lock()
        await ae.emit_from_text("Test", "resp_1", 0, lock, lambda x: None)
        assert len(called) == 1  # Called exactly once

    async def test_empty_text_no_audio(self, emitter):
        tts = FakeTTS(chunks=[])  # No audio output
        ae = AudioEmitter(emitter, tts, "pcm16")
        lock = asyncio.Lock()
        _, transcript = await ae.emit_from_text("", "resp_1", 0, lock, lambda x: None)
        assert transcript == ""


class TestAudioEmitterFromQueue:
    @pytest.fixture
    def emitter(self):
        return FakeEmitter()

    @pytest.fixture
    def tts(self):
        return FakeTTS()

    async def test_processes_queue_sentences(self, emitter, tts):
        ae = AudioEmitter(emitter, tts, "pcm16", on_first_audio=None)
        lock = asyncio.Lock()
        items = []
        queue: asyncio.Queue = asyncio.Queue()
        await queue.put("Primeira frase.")
        await queue.put("Segunda frase.")
        await queue.put(None)  # Sentinel

        item_id, transcript = await ae.emit_from_queue(
            queue, "resp_1", 0, lock, items.append
        )
        assert "Primeira frase." in transcript
        assert "Segunda frase." in transcript
        assert len(items) == 1

    async def test_empty_queue(self, emitter, tts):
        ae = AudioEmitter(emitter, tts, "pcm16")
        lock = asyncio.Lock()
        queue: asyncio.Queue = asyncio.Queue()
        await queue.put(None)

        _, transcript = await ae.emit_from_queue(
            queue, "resp_1", 0, lock, lambda x: None
        )
        assert transcript == ""

    async def test_first_audio_callback(self, emitter, tts):
        called = []
        ae = AudioEmitter(emitter, tts, "pcm16", on_first_audio=lambda: called.append(1))
        lock = asyncio.Lock()
        queue: asyncio.Queue = asyncio.Queue()
        await queue.put("Test sentence.")
        await queue.put(None)

        await ae.emit_from_queue(queue, "resp_1", 0, lock, lambda x: None)
        assert len(called) == 1
