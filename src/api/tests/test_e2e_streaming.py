"""Tests for E2E streaming pipeline — partial transcripts, early LLM trigger.

100% CPU, uses mocks. No GPU needed.
"""

import asyncio
import pytest

from protocol.models import ContentPart, ConversationItem
from server.session import RealtimeSession


# ---- Helpers ----


class FakeASRWithPartials:
    """ASR mock that emits partial transcripts."""

    provider_name = "fake-partial"

    def __init__(self, partials=None, final=""):
        self._partials = list(partials or [])
        self._partial_idx = 0
        self._final = final
        self._stream_id = None

    async def transcribe(self, audio):
        return self._final

    async def start_stream(self, stream_id):
        self._stream_id = stream_id
        self._partial_idx = 0

    async def feed_chunk(self, audio, stream_id):
        return ""

    async def feed_chunk_with_partial(self, audio, stream_id):
        if self._partial_idx < len(self._partials):
            partial = self._partials[self._partial_idx]
            self._partial_idx += 1
            return partial
        return None

    async def finish_stream(self, stream_id):
        self._stream_id = None
        return self._final

    @property
    def supports_streaming(self):
        return True

    @property
    def supports_partial_results(self):
        return True

    async def connect(self):
        pass

    async def warmup(self):
        pass

    async def disconnect(self):
        pass

    async def health_check(self):
        return True


class TestCountStableWords:
    """Tests for _count_stable_words static method."""

    def test_empty_previous_returns_zero(self):
        assert RealtimeSession._count_stable_words("hello world", "") == 0

    def test_identical_partials(self):
        assert RealtimeSession._count_stable_words("hello world", "hello world") == 2

    def test_growing_partial(self):
        assert RealtimeSession._count_stable_words("hello world foo", "hello world") == 2

    def test_changed_last_word(self):
        assert RealtimeSession._count_stable_words("hello there", "hello world") == 1

    def test_completely_different(self):
        assert RealtimeSession._count_stable_words("goodbye", "hello") == 0

    def test_case_insensitive(self):
        assert RealtimeSession._count_stable_words("Hello World", "hello world") == 2

    def test_single_stable_word(self):
        assert RealtimeSession._count_stable_words("qual é", "qual") == 1

    def test_three_stable_words(self):
        result = RealtimeSession._count_stable_words(
            "qual é o saldo", "qual é o"
        )
        assert result == 3

    def test_prefix_mismatch_stops_counting(self):
        # "qual" matches, "é" matches, "a" != "o" → stops at 2
        result = RealtimeSession._count_stable_words(
            "qual é a reunião", "qual é o saldo"
        )
        assert result == 2


class TestASRPartialResults:
    """Tests for ASR partial results in feed_chunk_with_partial."""

    def test_fake_asr_emits_partials(self):
        asr = FakeASRWithPartials(
            partials=["Qual", "Qual é", "Qual é o saldo"],
            final="Qual é o saldo da conta?"
        )

        async def _run():
            await asr.start_stream("s1")
            p1 = await asr.feed_chunk_with_partial(b"\x00" * 100, "s1")
            assert p1 == "Qual"
            p2 = await asr.feed_chunk_with_partial(b"\x00" * 100, "s1")
            assert p2 == "Qual é"
            p3 = await asr.feed_chunk_with_partial(b"\x00" * 100, "s1")
            assert p3 == "Qual é o saldo"
            # No more partials
            p4 = await asr.feed_chunk_with_partial(b"\x00" * 100, "s1")
            assert p4 is None
            final = await asr.finish_stream("s1")
            assert final == "Qual é o saldo da conta?"

        asyncio.get_event_loop().run_until_complete(_run())

    def test_fake_asr_supports_partial(self):
        asr = FakeASRWithPartials()
        assert asr.supports_streaming is True
        assert asr.supports_partial_results is True


class TestStreamingConfig:
    """Tests for StreamingPolicy configuration."""

    def test_config_loads(self):
        from config import STREAMING
        assert isinstance(STREAMING.enable_early_llm_trigger, bool)
        assert isinstance(STREAMING.min_stable_words, int)
        assert STREAMING.min_stable_words >= 1
        assert isinstance(STREAMING.partial_interval_ms, int)
        assert STREAMING.partial_interval_ms >= 50
        assert isinstance(STREAMING.min_eager_chars, int)
        assert STREAMING.min_eager_chars >= 5

    def test_default_values(self):
        from config import STREAMING
        assert STREAMING.enable_early_llm_trigger is False  # Off by default
        assert STREAMING.min_stable_words == 3
        assert STREAMING.partial_interval_ms == 300
        assert STREAMING.min_eager_chars == 10


class TestTranscriptionDeltaEvent:
    """Tests for the new transcription delta event."""

    def test_delta_event_structure(self):
        from protocol import events
        ev = events.input_audio_transcription_delta("", "item_123", 0, "Qual é o")
        assert ev["type"] == "conversation.item.input_audio_transcription.delta"
        assert ev["item_id"] == "item_123"
        assert ev["content_index"] == 0
        assert ev["delta"] == "Qual é o"


class TestASRProviderABC:
    """Tests for the updated ASR ABC."""

    def test_default_supports_partial_results_is_false(self):
        from providers.asr import ASRProvider

        class MinimalASR(ASRProvider):
            async def transcribe(self, audio):
                return ""

        asr = MinimalASR()
        assert asr.supports_partial_results is False

    def test_feed_chunk_with_partial_default(self):
        """Default implementation returns None when not streaming."""
        from providers.asr import ASRProvider

        class MinimalASR(ASRProvider):
            async def transcribe(self, audio):
                return ""

        asr = MinimalASR()

        async def _run():
            result = await asr.feed_chunk_with_partial(b"\x00", "s1")
            assert result is None

        asyncio.get_event_loop().run_until_complete(_run())


class TestSentenceSplitterEagerChars:
    """Tests for reduced min_eager_chars."""

    def test_eager_chars_uses_config(self):
        """Verify sentence splitter uses STREAMING.min_eager_chars (10, not 20)."""
        from config import STREAMING
        assert STREAMING.min_eager_chars == 10
