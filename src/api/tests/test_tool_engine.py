"""Tests for ToolExecutionEngine: stream parsing, tool execution, and event emission."""

from __future__ import annotations

import asyncio
import json
from typing import AsyncGenerator
from unittest.mock import AsyncMock

import pytest

from intelligence.tool_engine import ToolExecutionEngine, ToolRoundResult
from protocol.models import ConversationItem, SessionConfig
from providers.llm import LLMStreamEvent
from providers.tts import TTSProvider


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class FakeWS:
    """Minimal WebSocket that records every sent event."""

    def __init__(self) -> None:
        self.sent: list[dict] = []

    async def send(self, data: str) -> None:
        import json as _json
        self.sent.append(_json.loads(data))


class FakeEmitter:
    """Records emitted events without WebSocket overhead."""

    def __init__(self) -> None:
        self.emitted: list[dict] = []

    async def emit(self, event: dict) -> None:
        self.emitted.append(event)

    async def emit_many(self, events: list[dict]) -> None:
        for event in events:
            await self.emit(event)

    def events_of_type(self, event_type: str) -> list[dict]:
        return [e for e in self.emitted if e.get("type") == event_type]


class FakeTTS(TTSProvider):
    """Returns configurable audio bytes for synthesis."""

    def __init__(self, audio: bytes = b"\x00" * 160) -> None:
        self._audio = audio

    async def synthesize(self, text: str) -> bytes:
        return self._audio

    @property
    def supports_streaming(self) -> bool:
        return False


class FakeToolRegistry:
    """Minimal ToolRegistry fake with configurable per-tool results."""

    def __init__(self) -> None:
        self._results: dict[str, str] = {}
        self.calls: list[tuple[str, str]] = []

    def register_result(self, tool_name: str, result_json: str) -> None:
        self._results[tool_name] = result_json

    async def execute(self, name: str, arguments_json: str) -> str:
        self.calls.append((name, arguments_json))
        return self._results.get(name, json.dumps({"result": "ok"}))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_engine(
    emitter: FakeEmitter | None = None,
    tts: FakeTTS | None = None,
    registry: FakeToolRegistry | None = None,
    config: SessionConfig | None = None,
) -> ToolExecutionEngine:
    return ToolExecutionEngine(
        session_id="sess_test_0001",
        emitter=emitter or FakeEmitter(),
        tts=tts or FakeTTS(),
        config=config or SessionConfig(),
        tool_registry=registry or FakeToolRegistry(),
    )


def _tc(
    tool_id: str = "call_abc",
    name: str = "my_tool",
    args: str = "{}",
) -> dict:
    return {"id": tool_id, "name": name, "arguments": args}


def _stream(*events: LLMStreamEvent) -> list[LLMStreamEvent]:
    return list(events)


def _text(chunk: str) -> LLMStreamEvent:
    return LLMStreamEvent(type="text_delta", text=chunk)


def _tool_start(tool_id: str, name: str) -> LLMStreamEvent:
    return LLMStreamEvent(type="tool_call_start", tool_call_id=tool_id, tool_name=name)


def _tool_delta(delta: str) -> LLMStreamEvent:
    return LLMStreamEvent(type="tool_call_delta", tool_arguments_delta=delta)


def _tool_end() -> LLMStreamEvent:
    return LLMStreamEvent(type="tool_call_end")


# ---------------------------------------------------------------------------
# collect_tool_calls_from_stream
# ---------------------------------------------------------------------------


class TestCollectToolCallsFromStream:
    def test_text_only_returns_full_text_and_no_tool_calls(self):
        """Stream with only text_delta events yields concatenated text and empty tool list."""
        engine = _make_engine()
        events = _stream(_text("Hello "), _text("world"))

        text, tool_calls = engine.collect_tool_calls_from_stream(events)

        assert text == "Hello world"
        assert tool_calls == []

    def test_single_tool_call_parsed_correctly(self):
        """A complete tool_call_start + delta + end sequence yields one tool call dict."""
        engine = _make_engine()
        events = _stream(
            _tool_start("call_001", "web_search"),
            _tool_delta('{"query":'),
            _tool_delta(' "dolar"}'),
            _tool_end(),
        )

        text, tool_calls = engine.collect_tool_calls_from_stream(events)

        assert text == ""
        assert len(tool_calls) == 1
        assert tool_calls[0]["id"] == "call_001"
        assert tool_calls[0]["name"] == "web_search"
        assert tool_calls[0]["arguments"] == '{"query": "dolar"}'

    def test_multiple_tool_calls_all_parsed(self):
        """Two sequential tool calls are both parsed into separate dicts."""
        engine = _make_engine()
        events = _stream(
            _tool_start("call_001", "web_search"),
            _tool_delta('{"query": "cotacao"}'),
            _tool_end(),
            _tool_start("call_002", "get_balance"),
            _tool_delta('{"account_id": "42"}'),
            _tool_end(),
        )

        text, tool_calls = engine.collect_tool_calls_from_stream(events)

        assert len(tool_calls) == 2
        assert tool_calls[0]["name"] == "web_search"
        assert tool_calls[1]["name"] == "get_balance"
        assert tool_calls[1]["id"] == "call_002"

    def test_mixed_text_and_tool_calls_split_correctly(self):
        """Text before a tool call is captured; the tool call is parsed separately."""
        engine = _make_engine()
        events = _stream(
            _text("Vou verificar "),
            _text("isso agora."),
            _tool_start("call_001", "lookup_customer"),
            _tool_delta('{"cpf": "123"}'),
            _tool_end(),
        )

        text, tool_calls = engine.collect_tool_calls_from_stream(events)

        assert text == "Vou verificar isso agora."
        assert len(tool_calls) == 1
        assert tool_calls[0]["name"] == "lookup_customer"

    def test_empty_stream_returns_empty_text_and_no_tool_calls(self):
        """An empty event list produces empty text and no tool calls."""
        engine = _make_engine()

        text, tool_calls = engine.collect_tool_calls_from_stream([])

        assert text == ""
        assert tool_calls == []

    def test_tool_call_delta_without_start_is_ignored(self):
        """tool_call_delta events with no preceding tool_call_start are discarded."""
        engine = _make_engine()
        # Delta arrives before any start — in_tool is False so it is ignored.
        events = _stream(
            _tool_delta('{"stray": "delta"}'),
        )

        text, tool_calls = engine.collect_tool_calls_from_stream(events)

        assert tool_calls == []

    def test_arguments_assembled_from_multiple_deltas(self):
        """Arguments split across many delta events are concatenated faithfully."""
        engine = _make_engine()
        events = _stream(
            _tool_start("c1", "search"),
            _tool_delta("{"),
            _tool_delta('"q"'),
            _tool_delta(": "),
            _tool_delta('"test"'),
            _tool_delta("}"),
            _tool_end(),
        )

        _, tool_calls = engine.collect_tool_calls_from_stream(events)

        assert tool_calls[0]["arguments"] == '{"q": "test"}'


# ---------------------------------------------------------------------------
# execute_server_side
# ---------------------------------------------------------------------------


class TestExecuteServerSide:
    @pytest.mark.asyncio
    async def test_single_tool_success_appends_two_items(self):
        """Successful tool execution appends a function_call item and a function_call_output item."""
        registry = FakeToolRegistry()
        registry.register_result("web_search", json.dumps({"title": "BRL", "snippet": "R$5"}))

        appended: list[ConversationItem] = []

        def append_item(item: ConversationItem) -> None:
            appended.append(item)

        engine = _make_engine(registry=registry)
        result = await engine.execute_server_side(
            response_id="resp_1",
            output_index=0,
            tool_calls=[_tc("call_001", "web_search", '{"query": "dolar"}')],
            has_audio=False,
            ctx_lock=asyncio.Lock(),
            append_item=append_item,
        )

        assert result.all_tools_ok is True
        assert len(appended) == 2
        assert appended[0].type == "function_call"
        assert appended[0].name == "web_search"
        assert appended[1].type == "function_call_output"
        assert "BRL" in appended[1].output

    @pytest.mark.asyncio
    async def test_multiple_tools_timing_metrics_recorded(self):
        """Each executed tool produces a timing entry with name, exec_ms, and ok fields."""
        registry = FakeToolRegistry()
        registry.register_result("tool_a", json.dumps({"val": 1}))
        registry.register_result("tool_b", json.dumps({"val": 2}))

        engine = _make_engine(registry=registry)
        await engine.execute_server_side(
            response_id="resp_1",
            output_index=0,
            tool_calls=[
                _tc("c1", "tool_a", "{}"),
                _tc("c2", "tool_b", "{}"),
            ],
            has_audio=False,
            ctx_lock=asyncio.Lock(),
            append_item=lambda _: None,
        )

        timings = engine.tool_timings
        assert len(timings) == 2
        assert timings[0]["name"] == "tool_a"
        assert timings[1]["name"] == "tool_b"
        assert isinstance(timings[0]["exec_ms"], float)
        assert timings[0]["ok"] is True
        assert timings[1]["ok"] is True

    @pytest.mark.asyncio
    async def test_tool_returning_error_json_sets_all_tools_ok_false(self):
        """When a tool returns a JSON object containing an 'error' key, all_tools_ok is False."""
        registry = FakeToolRegistry()
        registry.register_result(
            "flaky_tool", json.dumps({"error": "timeout", "message": "Service unavailable"})
        )

        engine = _make_engine(registry=registry)
        result = await engine.execute_server_side(
            response_id="resp_1",
            output_index=0,
            tool_calls=[_tc("c1", "flaky_tool", "{}")],
            has_audio=False,
            ctx_lock=asyncio.Lock(),
            append_item=lambda _: None,
        )

        assert result.all_tools_ok is False
        assert engine.tool_timings[0]["ok"] is False

    @pytest.mark.asyncio
    async def test_filler_audio_sent_when_has_audio_true(self):
        """With has_audio=True, filler audio events are emitted before tool execution."""
        emitter = FakeEmitter()
        registry = FakeToolRegistry()
        registry.register_result("web_search", json.dumps({"result": "ok"}))

        engine = _make_engine(emitter=emitter, registry=registry)
        await engine.execute_server_side(
            response_id="resp_1",
            output_index=0,
            tool_calls=[_tc("c1", "web_search", '{"query": "euro"}')],
            has_audio=True,
            ctx_lock=asyncio.Lock(),
            append_item=lambda _: None,
        )

        # Filler synthesizes audio → response.audio.delta event must appear
        audio_deltas = emitter.events_of_type("response.audio.delta")
        assert len(audio_deltas) >= 1

    @pytest.mark.asyncio
    async def test_no_filler_when_has_audio_false(self):
        """With has_audio=False, no filler audio events are emitted."""
        emitter = FakeEmitter()
        registry = FakeToolRegistry()
        registry.register_result("web_search", json.dumps({"result": "ok"}))

        engine = _make_engine(emitter=emitter, registry=registry)
        await engine.execute_server_side(
            response_id="resp_1",
            output_index=0,
            tool_calls=[_tc("c1", "web_search", '{"query": "euro"}')],
            has_audio=False,
            ctx_lock=asyncio.Lock(),
            append_item=lambda _: None,
        )

        audio_deltas = emitter.events_of_type("response.audio.delta")
        assert audio_deltas == []

    @pytest.mark.asyncio
    async def test_no_filler_when_tool_calls_empty(self):
        """When tool_calls is empty, no filler is sent even if has_audio=True."""
        emitter = FakeEmitter()
        engine = _make_engine(emitter=emitter)

        await engine.execute_server_side(
            response_id="resp_1",
            output_index=0,
            tool_calls=[],
            has_audio=True,
            ctx_lock=asyncio.Lock(),
            append_item=lambda _: None,
        )

        audio_deltas = emitter.events_of_type("response.audio.delta")
        assert audio_deltas == []

    @pytest.mark.asyncio
    async def test_output_index_delta_incremented_per_tool(self):
        """output_index_delta in ToolRoundResult equals the number of executed tool calls."""
        registry = FakeToolRegistry()
        registry.register_result("tool_a", json.dumps({"ok": True}))
        registry.register_result("tool_b", json.dumps({"ok": True}))
        registry.register_result("tool_c", json.dumps({"ok": True}))

        engine = _make_engine(registry=registry)
        result = await engine.execute_server_side(
            response_id="resp_1",
            output_index=0,
            tool_calls=[
                _tc("c1", "tool_a", "{}"),
                _tc("c2", "tool_b", "{}"),
                _tc("c3", "tool_c", "{}"),
            ],
            has_audio=False,
            ctx_lock=asyncio.Lock(),
            append_item=lambda _: None,
        )

        assert result.output_index_delta == 3

    @pytest.mark.asyncio
    async def test_correct_events_emitted_for_single_tool(self):
        """Executing one tool emits: output_item_added, arguments_done, output_item_done, conversation_item_created."""
        emitter = FakeEmitter()
        registry = FakeToolRegistry()
        registry.register_result("my_tool", json.dumps({"answer": 42}))

        engine = _make_engine(emitter=emitter, registry=registry)
        await engine.execute_server_side(
            response_id="resp_1",
            output_index=0,
            tool_calls=[_tc("c1", "my_tool", "{}")],
            has_audio=False,
            ctx_lock=asyncio.Lock(),
            append_item=lambda _: None,
        )

        types_emitted = [e["type"] for e in emitter.emitted]
        assert "response.output_item.added" in types_emitted
        assert "response.function_call_arguments.done" in types_emitted
        assert "response.output_item.done" in types_emitted
        assert "conversation.item.created" in types_emitted

    @pytest.mark.asyncio
    async def test_missing_tool_id_generates_fallback_id(self):
        """An empty tool_id triggers a generated fallback id (does not crash)."""
        registry = FakeToolRegistry()
        registry.register_result("my_tool", json.dumps({"ok": True}))

        appended: list[ConversationItem] = []

        engine = _make_engine(registry=registry)
        result = await engine.execute_server_side(
            response_id="resp_1",
            output_index=0,
            tool_calls=[{"id": "", "name": "my_tool", "arguments": "{}"}],
            has_audio=False,
            ctx_lock=asyncio.Lock(),
            append_item=lambda item: appended.append(item),
        )

        assert result.all_tools_ok is True
        fc_item = appended[0]
        # call_id must be non-empty — the engine generated one
        assert fc_item.call_id != ""

    @pytest.mark.asyncio
    async def test_tool_result_stored_in_output_item(self):
        """The function_call_output item's output field contains the exact JSON returned by the tool."""
        result_json = json.dumps({"balance": "R$1.000,00"})
        registry = FakeToolRegistry()
        registry.register_result("get_balance", result_json)

        appended: list[ConversationItem] = []

        engine = _make_engine(registry=registry)
        await engine.execute_server_side(
            response_id="resp_1",
            output_index=0,
            tool_calls=[_tc("c1", "get_balance", '{"account_id": "123"}')],
            has_audio=False,
            ctx_lock=asyncio.Lock(),
            append_item=lambda item: appended.append(item),
        )

        fco_item = appended[1]
        assert fco_item.type == "function_call_output"
        assert fco_item.output == result_json

    @pytest.mark.asyncio
    async def test_non_error_tool_result_leaves_all_tools_ok_true(self):
        """A tool result without an 'error' key keeps all_tools_ok as True."""
        registry = FakeToolRegistry()
        # JSON with a field named "status" but NOT "error"
        registry.register_result("check_status", json.dumps({"status": "ok", "code": 200}))

        engine = _make_engine(registry=registry)
        result = await engine.execute_server_side(
            response_id="resp_1",
            output_index=0,
            tool_calls=[_tc("c1", "check_status", "{}")],
            has_audio=False,
            ctx_lock=asyncio.Lock(),
            append_item=lambda _: None,
        )

        assert result.all_tools_ok is True

    @pytest.mark.asyncio
    async def test_multiple_tools_one_fails_all_tools_ok_false(self):
        """If any tool returns an error, all_tools_ok is False regardless of other tools."""
        registry = FakeToolRegistry()
        registry.register_result("tool_ok", json.dumps({"value": 1}))
        registry.register_result("tool_bad", json.dumps({"error": "not_found"}))

        engine = _make_engine(registry=registry)
        result = await engine.execute_server_side(
            response_id="resp_1",
            output_index=0,
            tool_calls=[
                _tc("c1", "tool_ok", "{}"),
                _tc("c2", "tool_bad", "{}"),
            ],
            has_audio=False,
            ctx_lock=asyncio.Lock(),
            append_item=lambda _: None,
        )

        assert result.all_tools_ok is False

    @pytest.mark.asyncio
    async def test_ctx_lock_acquired_for_append_item(self):
        """append_item is always called with ctx_lock held (no concurrent mutation)."""
        lock_was_locked_on_call: list[bool] = []
        ctx_lock = asyncio.Lock()

        registry = FakeToolRegistry()
        registry.register_result("my_tool", json.dumps({"ok": True}))

        def check_lock(item: ConversationItem) -> None:
            lock_was_locked_on_call.append(ctx_lock.locked())

        engine = _make_engine(registry=registry)
        await engine.execute_server_side(
            response_id="resp_1",
            output_index=0,
            tool_calls=[_tc("c1", "my_tool", "{}")],
            has_audio=False,
            ctx_lock=ctx_lock,
            append_item=check_lock,
        )

        assert all(lock_was_locked_on_call), "append_item was called without holding ctx_lock"


# ---------------------------------------------------------------------------
# emit_tool_calls_for_client
# ---------------------------------------------------------------------------


class TestEmitToolCallsForClient:
    @pytest.mark.asyncio
    async def test_emits_required_events_for_each_tool_call(self):
        """For each tool call, three events are emitted: output_item_added, arguments_done, output_item_done."""
        emitter = FakeEmitter()
        engine = _make_engine(emitter=emitter)

        await engine.emit_tool_calls_for_client(
            response_id="resp_1",
            output_index=0,
            tool_calls=[_tc("c1", "client_tool", '{"arg": "val"}')],
            ctx_lock=asyncio.Lock(),
            append_item=lambda _: None,
        )

        types_emitted = [e["type"] for e in emitter.emitted]
        assert types_emitted.count("response.output_item.added") == 1
        assert types_emitted.count("response.function_call_arguments.done") == 1
        assert types_emitted.count("response.output_item.done") == 1

    @pytest.mark.asyncio
    async def test_emits_events_for_multiple_tool_calls(self):
        """Two tool calls produce two sets of output events (six total)."""
        emitter = FakeEmitter()
        engine = _make_engine(emitter=emitter)

        await engine.emit_tool_calls_for_client(
            response_id="resp_1",
            output_index=0,
            tool_calls=[
                _tc("c1", "tool_alpha", "{}"),
                _tc("c2", "tool_beta", "{}"),
            ],
            ctx_lock=asyncio.Lock(),
            append_item=lambda _: None,
        )

        types_emitted = [e["type"] for e in emitter.emitted]
        assert types_emitted.count("response.output_item.added") == 2
        assert types_emitted.count("response.function_call_arguments.done") == 2
        assert types_emitted.count("response.output_item.done") == 2

    @pytest.mark.asyncio
    async def test_arguments_in_arguments_done_event_match_tool_call(self):
        """The arguments in the arguments_done event match the tool call arguments exactly."""
        emitter = FakeEmitter()
        engine = _make_engine(emitter=emitter)
        args = '{"query": "bitcoin price"}'

        await engine.emit_tool_calls_for_client(
            response_id="resp_1",
            output_index=0,
            tool_calls=[_tc("c1", "web_search", args)],
            ctx_lock=asyncio.Lock(),
            append_item=lambda _: None,
        )

        args_done = emitter.events_of_type("response.function_call_arguments.done")[0]
        assert args_done["arguments"] == args

    @pytest.mark.asyncio
    async def test_function_call_item_appended_to_conversation(self):
        """emit_tool_calls_for_client appends a function_call ConversationItem per tool."""
        appended: list[ConversationItem] = []
        engine = _make_engine()

        await engine.emit_tool_calls_for_client(
            response_id="resp_1",
            output_index=0,
            tool_calls=[_tc("c1", "lookup_customer", '{"cpf": "000"}')],
            ctx_lock=asyncio.Lock(),
            append_item=lambda item: appended.append(item),
        )

        assert len(appended) == 1
        assert appended[0].type == "function_call"
        assert appended[0].name == "lookup_customer"

    @pytest.mark.asyncio
    async def test_empty_tool_id_generates_fallback(self):
        """A tool call with empty id does not crash — a fallback id is generated."""
        emitter = FakeEmitter()
        appended: list[ConversationItem] = []
        engine = _make_engine(emitter=emitter)

        await engine.emit_tool_calls_for_client(
            response_id="resp_1",
            output_index=0,
            tool_calls=[{"id": "", "name": "some_tool", "arguments": "{}"}],
            ctx_lock=asyncio.Lock(),
            append_item=lambda item: appended.append(item),
        )

        assert appended[0].call_id != ""

    @pytest.mark.asyncio
    async def test_no_conversation_item_created_event_emitted(self):
        """Client-side emission does NOT emit conversation.item.created (server-side only)."""
        emitter = FakeEmitter()
        engine = _make_engine(emitter=emitter)

        await engine.emit_tool_calls_for_client(
            response_id="resp_1",
            output_index=0,
            tool_calls=[_tc("c1", "my_tool", "{}")],
            ctx_lock=asyncio.Lock(),
            append_item=lambda _: None,
        )

        assert emitter.events_of_type("conversation.item.created") == []


# ---------------------------------------------------------------------------
# tool_timings property
# ---------------------------------------------------------------------------


class TestToolTimingsProperty:
    @pytest.mark.asyncio
    async def test_tool_timings_empty_before_execution(self):
        """tool_timings is empty on a freshly created engine."""
        engine = _make_engine()
        assert engine.tool_timings == []

    @pytest.mark.asyncio
    async def test_tool_timings_accumulate_across_calls(self):
        """Timings accumulate across multiple execute_server_side calls on the same engine."""
        registry = FakeToolRegistry()
        registry.register_result("tool_x", json.dumps({"ok": True}))

        engine = _make_engine(registry=registry)

        # First call
        await engine.execute_server_side(
            response_id="resp_1",
            output_index=0,
            tool_calls=[_tc("c1", "tool_x", "{}")],
            has_audio=False,
            ctx_lock=asyncio.Lock(),
            append_item=lambda _: None,
        )
        # Second call
        await engine.execute_server_side(
            response_id="resp_2",
            output_index=0,
            tool_calls=[_tc("c2", "tool_x", "{}")],
            has_audio=False,
            ctx_lock=asyncio.Lock(),
            append_item=lambda _: None,
        )

        assert len(engine.tool_timings) == 2

    @pytest.mark.asyncio
    async def test_tool_timing_entry_has_required_fields(self):
        """Each timing entry contains name (str), exec_ms (float), and ok (bool)."""
        registry = FakeToolRegistry()
        registry.register_result("my_tool", json.dumps({"v": 1}))

        engine = _make_engine(registry=registry)
        await engine.execute_server_side(
            response_id="resp_1",
            output_index=0,
            tool_calls=[_tc("c1", "my_tool", "{}")],
            has_audio=False,
            ctx_lock=asyncio.Lock(),
            append_item=lambda _: None,
        )

        timing = engine.tool_timings[0]
        assert isinstance(timing["name"], str)
        assert isinstance(timing["exec_ms"], float)
        assert isinstance(timing["ok"], bool)

    @pytest.mark.asyncio
    async def test_tool_timing_ok_false_for_error_result(self):
        """A tool that returns an error JSON is recorded with ok=False in timings."""
        registry = FakeToolRegistry()
        registry.register_result("bad_tool", json.dumps({"error": "not_found"}))

        engine = _make_engine(registry=registry)
        await engine.execute_server_side(
            response_id="resp_1",
            output_index=0,
            tool_calls=[_tc("c1", "bad_tool", "{}")],
            has_audio=False,
            ctx_lock=asyncio.Lock(),
            append_item=lambda _: None,
        )

        assert engine.tool_timings[0]["ok"] is False
