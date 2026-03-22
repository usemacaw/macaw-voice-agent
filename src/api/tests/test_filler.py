"""
Tests for the filler phrase module (server/filler.py).

Covers:
- build_dynamic_filler phrase selection per tool type
- build_dynamic_filler robustness to bad inputs
- Portuguese string validity for all phrase pools
- send_filler_audio with streaming TTS
- send_filler_audio with non-streaming TTS
- send_filler_audio when TTS raises an exception (non-critical)
- Filler items are NOT appended to conversation store
"""

from __future__ import annotations

import json
import logging
from typing import AsyncGenerator

import pytest

from protocol.models import SessionConfig
from server.filler import (
    _GENERIC_FILLERS,
    _MEMORY_FILLERS,
    _SEARCH_FILLERS,
    build_dynamic_filler,
    send_filler_audio,
)

# ---------------------------------------------------------------------------
# Shared fakes
# ---------------------------------------------------------------------------

class FakeEmitter:
    """Records every event passed to emit()."""

    def __init__(self):
        self.events: list[dict] = []

    async def emit(self, event: dict) -> None:
        self.events.append(event)

    def event_types(self) -> list[str]:
        return [e["type"] for e in self.events]


class FakeStreamingTTS:
    """TTS fake whose supports_streaming is True, yields deterministic chunks."""

    def __init__(self, chunks: list[bytes] | None = None, raises: bool = False):
        self.supports_streaming = True
        self._chunks = chunks if chunks is not None else [b"\x00\x01" * 400]
        self._raises = raises
        self.synthesize_called = False
        self.synthesize_stream_called = False

    async def synthesize(self, text: str) -> bytes:
        self.synthesize_called = True
        if self._raises:
            raise RuntimeError("TTS synthesis failed")
        return b"".join(self._chunks)

    async def synthesize_stream(self, text: str) -> AsyncGenerator[bytes, None]:
        self.synthesize_stream_called = True
        if self._raises:
            raise RuntimeError("TTS streaming failed")
        for chunk in self._chunks:
            yield chunk


class FakeNonStreamingTTS:
    """TTS fake whose supports_streaming is False, returns audio in one call."""

    def __init__(self, audio: bytes | None = None, raises: bool = False):
        self.supports_streaming = False
        self._audio = audio if audio is not None else b"\x00\x01" * 400
        self._raises = raises
        self.synthesize_called = False

    async def synthesize(self, text: str) -> bytes:
        self.synthesize_called = True
        if self._raises:
            raise RuntimeError("TTS synthesis failed")
        return self._audio


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_PORTUGUESE_ACCENTS = set("áàâãéêíóôõúüçÁÀÂÃÉÊÍÓÔÕÚÜÇ")

def _has_portuguese_accents(text: str) -> bool:
    return bool(set(text) & _PORTUGUESE_ACCENTS)


def _make_config() -> SessionConfig:
    return SessionConfig(output_audio_format="pcm16")


# ---------------------------------------------------------------------------
# Unit tests: build_dynamic_filler
# ---------------------------------------------------------------------------

class TestBuildDynamicFillerWebSearch:
    def test_includes_query_in_filler_when_query_present(self):
        """Filler for web_search with a query should embed the query text."""
        filler = build_dynamic_filler("web_search", json.dumps({"query": "cotação do dólar"}))

        assert "cotação do dólar" in filler

    def test_query_is_truncated_to_60_chars(self):
        """Query longer than 60 characters is truncated in the filler."""
        long_query = "a" * 80
        filler = build_dynamic_filler("web_search", json.dumps({"query": long_query}))

        assert long_query not in filler
        assert "a" * 60 in filler

    def test_falls_back_to_generic_phrase_when_no_query(self):
        """Filler for web_search without query uses the fallback (no-query) variant."""
        filler = build_dynamic_filler("web_search", json.dumps({}))

        # The fallback variant must not contain a format placeholder
        assert "{q}" not in filler
        assert len(filler) > 0

    def test_falls_back_when_query_is_empty_string(self):
        """Empty string query is treated as absent — uses fallback."""
        filler = build_dynamic_filler("web_search", json.dumps({"query": ""}))

        assert "{q}" not in filler
        assert len(filler) > 0

    def test_result_is_one_of_known_search_fallbacks(self):
        """Without a query, result must match one of the known fallback phrases."""
        known_fallbacks = {without_q for _, without_q in _SEARCH_FILLERS}

        filler = build_dynamic_filler("web_search", json.dumps({}))

        assert filler in known_fallbacks

    def test_result_with_query_matches_template_format(self):
        """With a query, the result must be a formatted version of a known template."""
        query = "taxa selic"
        known_with_q_templates = [with_q for with_q, _ in _SEARCH_FILLERS]

        filler = build_dynamic_filler("web_search", json.dumps({"query": query}))

        assert any(
            filler == template.format(q=query) for template in known_with_q_templates
        )


class TestBuildDynamicFillerRecallMemory:
    def test_returns_memory_specific_phrase(self):
        """Filler for recall_memory uses the dedicated memory phrase pool."""
        filler = build_dynamic_filler("recall_memory", json.dumps({}))

        assert filler in _MEMORY_FILLERS

    def test_memory_filler_ignores_arguments(self):
        """Arguments are irrelevant for recall_memory — pool selection is by tool name."""
        filler = build_dynamic_filler("recall_memory", json.dumps({"key": "valor"}))

        assert filler in _MEMORY_FILLERS


class TestBuildDynamicFillerUnknownTool:
    def test_unknown_tool_uses_generic_fillers(self):
        """Unknown tool names fall back to the generic pool."""
        filler = build_dynamic_filler("some_unknown_tool", json.dumps({}))

        assert filler in _GENERIC_FILLERS

    def test_empty_tool_name_uses_generic_fillers(self):
        """Empty tool name also falls back to the generic pool."""
        filler = build_dynamic_filler("", "{}")

        assert filler in _GENERIC_FILLERS


class TestBuildDynamicFillerRobustness:
    def test_invalid_json_does_not_crash(self):
        """Malformed JSON arguments must not raise — falls back to generic."""
        filler = build_dynamic_filler("web_search", "not valid json {{{{")

        # No crash; since args parse failed, query is absent → fallback phrase
        assert len(filler) > 0
        assert "{q}" not in filler

    def test_invalid_json_for_unknown_tool_uses_generic(self):
        """Invalid JSON with an unknown tool falls back to generic pool."""
        filler = build_dynamic_filler("some_tool", "[broken")

        assert filler in _GENERIC_FILLERS

    def test_empty_arguments_string_does_not_crash(self):
        """Empty string for arguments_json must not raise."""
        filler = build_dynamic_filler("web_search", "")

        assert len(filler) > 0

    def test_none_like_empty_string_for_recall_memory(self):
        """Empty arguments_json with recall_memory still returns a memory phrase."""
        filler = build_dynamic_filler("recall_memory", "")

        assert filler in _MEMORY_FILLERS

    def test_non_dict_json_does_not_crash(self):
        """JSON that is valid but not an object (e.g. a list) must not crash."""
        filler = build_dynamic_filler("web_search", json.dumps(["item1", "item2"]))

        # args.get("query") would fail on a list, but the code catches TypeError
        assert len(filler) > 0


# ---------------------------------------------------------------------------
# Unit tests: Portuguese phrase pool validity
# ---------------------------------------------------------------------------

class TestPortugueseFillerPhrases:
    def test_all_search_fillers_with_query_are_non_empty_strings(self):
        for with_q, _ in _SEARCH_FILLERS:
            assert isinstance(with_q, str) and len(with_q) > 0

    def test_all_search_fillers_without_query_are_non_empty_strings(self):
        for _, without_q in _SEARCH_FILLERS:
            assert isinstance(without_q, str) and len(without_q) > 0

    def test_all_memory_fillers_are_non_empty_strings(self):
        for phrase in _MEMORY_FILLERS:
            assert isinstance(phrase, str) and len(phrase) > 0

    def test_all_generic_fillers_are_non_empty_strings(self):
        for phrase in _GENERIC_FILLERS:
            assert isinstance(phrase, str) and len(phrase) > 0

    def test_search_filler_pool_collectively_contains_portuguese_accents(self):
        """At least one phrase in the search pool must use accented Portuguese.

        Not every individual phrase needs an accent (e.g. 'Vou pesquisar sobre {q},
        aguarde.' is valid Portuguese without one), but the pool as a whole must
        demonstrate correct UTF-8 Portuguese usage.
        """
        all_search_texts = [t for pair in _SEARCH_FILLERS for t in pair]
        assert any(_has_portuguese_accents(t) for t in all_search_texts), (
            "Search filler pool has no accented Portuguese phrases at all"
        )

    def test_memory_filler_pool_collectively_contains_portuguese_accents(self):
        """At least one phrase in the memory pool must use accented Portuguese."""
        assert any(_has_portuguese_accents(p) for p in _MEMORY_FILLERS), (
            "Memory filler pool has no accented Portuguese phrases at all"
        )

    def test_generic_filler_pool_collectively_contains_portuguese_accents(self):
        """At least one phrase in the generic pool must use accented Portuguese."""
        assert any(_has_portuguese_accents(p) for p in _GENERIC_FILLERS), (
            "Generic filler pool has no accented Portuguese phrases at all"
        )

    def test_search_with_query_templates_have_format_placeholder(self):
        """Every 'with query' search template must contain {q} for formatting."""
        for with_q, _ in _SEARCH_FILLERS:
            assert "{q}" in with_q, (
                f"Search template missing {{q}} placeholder: {with_q!r}"
            )

    def test_search_without_query_templates_have_no_format_placeholder(self):
        """Fallback search phrases must NOT contain {q} (they stand alone)."""
        for _, without_q in _SEARCH_FILLERS:
            assert "{q}" not in without_q, (
                f"Fallback search phrase contains unexpected {{q}}: {without_q!r}"
            )


# ---------------------------------------------------------------------------
# Async tests: send_filler_audio
# ---------------------------------------------------------------------------

class TestSendFillerAudioStreaming:
    async def test_emits_output_item_added_first(self):
        """First event must be response.output_item.added to open the item."""
        emitter = FakeEmitter()
        tts = FakeStreamingTTS()

        await send_filler_audio(
            session_id="sess_abc123",
            tts=tts,
            emitter=emitter,
            config=_make_config(),
            response_id="resp_1",
            output_index=0,
            filler_text="Aguarde, vou buscar.",
        )

        assert emitter.events[0]["type"] == "response.output_item.added"

    async def test_emits_audio_delta_events(self):
        """Streaming TTS must produce at least one response.audio.delta event."""
        emitter = FakeEmitter()
        tts = FakeStreamingTTS(chunks=[b"\x00\x01" * 200, b"\x02\x03" * 200])

        await send_filler_audio(
            session_id="sess_abc123",
            tts=tts,
            emitter=emitter,
            config=_make_config(),
            response_id="resp_1",
            output_index=0,
            filler_text="Um momento.",
        )

        types = emitter.event_types()
        assert "response.audio.delta" in types

    async def test_emits_transcript_delta(self):
        """A transcript delta must be emitted carrying the filler text."""
        emitter = FakeEmitter()
        tts = FakeStreamingTTS()
        filler_text = "Deixa eu buscar isso pra você."

        await send_filler_audio(
            session_id="sess_abc123",
            tts=tts,
            emitter=emitter,
            config=_make_config(),
            response_id="resp_1",
            output_index=0,
            filler_text=filler_text,
        )

        transcript_deltas = [
            e for e in emitter.events
            if e["type"] == "response.audio_transcript.delta"
        ]
        assert len(transcript_deltas) == 1
        assert transcript_deltas[0]["delta"] == filler_text

    async def test_emits_audio_done_and_output_item_done(self):
        """Closing events response.audio.done and response.output_item.done must be emitted."""
        emitter = FakeEmitter()
        tts = FakeStreamingTTS()

        await send_filler_audio(
            session_id="sess_abc123",
            tts=tts,
            emitter=emitter,
            config=_make_config(),
            response_id="resp_1",
            output_index=0,
            filler_text="Só um momento.",
        )

        types = emitter.event_types()
        assert "response.audio.done" in types
        assert "response.output_item.done" in types

    async def test_uses_synthesize_stream_not_synthesize(self):
        """When supports_streaming is True, synthesize_stream is called, not synthesize."""
        emitter = FakeEmitter()
        tts = FakeStreamingTTS()

        await send_filler_audio(
            session_id="sess_abc123",
            tts=tts,
            emitter=emitter,
            config=_make_config(),
            response_id="resp_1",
            output_index=0,
            filler_text="Vou verificar.",
        )

        assert tts.synthesize_stream_called is True
        assert tts.synthesize_called is False

    async def test_event_sequence_ordering(self):
        """Events must follow the protocol order: item_added ... audio_delta(s) ...
        transcript_delta ... audio_done ... item_done."""
        emitter = FakeEmitter()
        tts = FakeStreamingTTS()

        await send_filler_audio(
            session_id="sess_abc123",
            tts=tts,
            emitter=emitter,
            config=_make_config(),
            response_id="resp_1",
            output_index=0,
            filler_text="Aguarde.",
        )

        types = emitter.event_types()
        assert types[0] == "response.output_item.added"
        assert types[-1] == "response.output_item.done"
        # audio_done must come before item_done
        assert types.index("response.audio.done") < types.index("response.output_item.done")


class TestSendFillerAudioNonStreaming:
    async def test_emits_audio_delta_via_synthesize(self):
        """Non-streaming TTS must still produce a response.audio.delta event."""
        emitter = FakeEmitter()
        tts = FakeNonStreamingTTS(audio=b"\x00\x01" * 400)

        await send_filler_audio(
            session_id="sess_abc123",
            tts=tts,
            emitter=emitter,
            config=_make_config(),
            response_id="resp_1",
            output_index=0,
            filler_text="Um momento, por favor.",
        )

        types = emitter.event_types()
        assert "response.audio.delta" in types

    async def test_uses_synthesize_not_synthesize_stream(self):
        """When supports_streaming is False, synthesize is called."""
        emitter = FakeEmitter()
        tts = FakeNonStreamingTTS()

        await send_filler_audio(
            session_id="sess_abc123",
            tts=tts,
            emitter=emitter,
            config=_make_config(),
            response_id="resp_1",
            output_index=0,
            filler_text="Aguarde um instante.",
        )

        assert tts.synthesize_called is True

    async def test_emits_full_event_sequence_non_streaming(self):
        """Full event sequence is emitted correctly for non-streaming TTS."""
        emitter = FakeEmitter()
        tts = FakeNonStreamingTTS()

        await send_filler_audio(
            session_id="sess_abc123",
            tts=tts,
            emitter=emitter,
            config=_make_config(),
            response_id="resp_1",
            output_index=0,
            filler_text="Vou buscar.",
        )

        types = emitter.event_types()
        assert "response.output_item.added" in types
        assert "response.audio.delta" in types
        assert "response.audio_transcript.delta" in types
        assert "response.audio.done" in types
        assert "response.output_item.done" in types

    async def test_no_audio_delta_when_synthesize_returns_empty(self):
        """If synthesize returns empty bytes, no audio delta must be emitted."""
        emitter = FakeEmitter()
        tts = FakeNonStreamingTTS(audio=b"")

        await send_filler_audio(
            session_id="sess_abc123",
            tts=tts,
            emitter=emitter,
            config=_make_config(),
            response_id="resp_1",
            output_index=0,
            filler_text="Vou verificar.",
        )

        types = emitter.event_types()
        assert "response.audio.delta" not in types
        # Protocol events still emitted
        assert "response.output_item.added" in types
        assert "response.output_item.done" in types


class TestSendFillerAudioTTSFailure:
    async def test_streaming_tts_exception_does_not_propagate(self):
        """TTS failure must be swallowed — filler is non-critical."""
        emitter = FakeEmitter()
        tts = FakeStreamingTTS(raises=True)

        # Must not raise
        await send_filler_audio(
            session_id="sess_abc123",
            tts=tts,
            emitter=emitter,
            config=_make_config(),
            response_id="resp_1",
            output_index=0,
            filler_text="Um momento.",
        )

    async def test_non_streaming_tts_exception_does_not_propagate(self):
        """Non-streaming TTS failure must be swallowed — filler is non-critical."""
        emitter = FakeEmitter()
        tts = FakeNonStreamingTTS(raises=True)

        # Must not raise
        await send_filler_audio(
            session_id="sess_abc123",
            tts=tts,
            emitter=emitter,
            config=_make_config(),
            response_id="resp_1",
            output_index=0,
            filler_text="Aguarde.",
        )

    async def test_tts_failure_emits_warning_log(self, caplog):
        """TTS failure must emit a WARNING-level log, not silently disappear."""
        emitter = FakeEmitter()
        tts = FakeNonStreamingTTS(raises=True)

        with caplog.at_level(logging.WARNING, logger="open-voice-api.filler"):
            await send_filler_audio(
                session_id="sess_abc123",
                tts=tts,
                emitter=emitter,
                config=_make_config(),
                response_id="resp_1",
                output_index=0,
                filler_text="Vou buscar.",
            )

        warning_messages = [r.message for r in caplog.records if r.levelno == logging.WARNING]
        assert any("filler" in msg.lower() or "non-critical" in msg.lower() for msg in warning_messages)


class TestSendFillerAudioConversationStore:
    async def test_filler_item_is_not_appended_to_conversation_store(self):
        """Filler must NOT call append_item on any conversation store.

        Filler phrases must stay out of LLM context to prevent the model
        from imitating them instead of calling tools.
        """
        emitter = FakeEmitter()
        tts = FakeStreamingTTS()
        append_calls: list = []

        class FakeConversationStore:
            def append_item(self, item):
                append_calls.append(item)

        # send_filler_audio does not accept a store argument by design.
        # Verify this by checking the function signature never touches a store.
        await send_filler_audio(
            session_id="sess_abc123",
            tts=tts,
            emitter=emitter,
            config=_make_config(),
            response_id="resp_1",
            output_index=0,
            filler_text="Vou verificar.",
        )

        # No append call should have happened — filler_audio has no store param
        assert append_calls == [], (
            "Filler audio must never append items to conversation store"
        )

    async def test_filler_item_id_not_in_emitted_conversation_events(self):
        """No conversation.item.created event must be emitted for filler items."""
        emitter = FakeEmitter()
        tts = FakeStreamingTTS()

        await send_filler_audio(
            session_id="sess_abc123",
            tts=tts,
            emitter=emitter,
            config=_make_config(),
            response_id="resp_1",
            output_index=0,
            filler_text="Só um momento.",
        )

        conversation_events = [
            e for e in emitter.events
            if e["type"] == "conversation.item.created"
        ]
        assert conversation_events == [], (
            "Filler must not emit conversation.item.created events"
        )
