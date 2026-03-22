"""Tests for intelligence/context_builder.py.

ContextBuilder is the single authority on LLM context construction.
These tests verify its public contract: which messages are produced,
which config values are forwarded, and how it delegates to the
underlying pipeline/conversation.py primitives.
"""

from __future__ import annotations

import pytest

from config import CONTEXT, LLM
from intelligence.context_builder import ContextBuilder
from protocol.models import ContentPart, ConversationItem, SessionConfig


# ---------------------------------------------------------------------------
# Helpers — minimal factories, mirroring the convention in test_conversation_window.py
# ---------------------------------------------------------------------------


def _user(text: str, item_id: str = "") -> ConversationItem:
    return ConversationItem(
        id=item_id or f"u_{text[:10]}",
        type="message",
        role="user",
        content=[ContentPart(type="input_text", text=text)],
    )


def _assistant(text: str, item_id: str = "") -> ConversationItem:
    return ConversationItem(
        id=item_id or f"a_{text[:10]}",
        type="message",
        role="assistant",
        content=[ContentPart(type="text", text=text)],
    )


def _function_call(call_id: str, name: str = "web_search", args: str = "{}") -> ConversationItem:
    return ConversationItem(
        id=f"fc_{call_id}",
        type="function_call",
        call_id=call_id,
        name=name,
        arguments=args,
    )


def _function_call_output(call_id: str, output: str = '{"ok": true}') -> ConversationItem:
    return ConversationItem(
        id=f"fco_{call_id}",
        type="function_call_output",
        call_id=call_id,
        output=output,
    )


def _config(**kwargs) -> SessionConfig:
    """Return a SessionConfig with sensible defaults, overridable via kwargs."""
    defaults = {
        "instructions": "You are a helpful assistant.",
        "temperature": 0.7,
        "max_response_output_tokens": "inf",
    }
    defaults.update(kwargs)
    return SessionConfig(**defaults)


# ---------------------------------------------------------------------------
# build_for_response — return shape
# ---------------------------------------------------------------------------


class TestBuildForResponseShape:
    """build_for_response must return (messages, system, temperature, max_tokens)."""

    def test_returns_four_element_tuple(self):
        builder = ContextBuilder(_config())
        result = builder.build_for_response([_user("Hi")], has_tools=False)

        assert isinstance(result, tuple)
        assert len(result) == 4

    def test_messages_is_list(self):
        builder = ContextBuilder(_config())
        messages, *_ = builder.build_for_response([_user("Hi")], has_tools=False)

        assert isinstance(messages, list)

    def test_system_is_string(self):
        builder = ContextBuilder(_config())
        _, system, *_ = builder.build_for_response([_user("Hi")], has_tools=False)

        assert isinstance(system, str)

    def test_temperature_is_float(self):
        builder = ContextBuilder(_config())
        _, _, temperature, _ = builder.build_for_response([_user("Hi")], has_tools=False)

        assert isinstance(temperature, float)

    def test_max_tokens_is_int(self):
        builder = ContextBuilder(_config())
        _, _, _, max_tokens = builder.build_for_response([_user("Hi")], has_tools=False)

        assert isinstance(max_tokens, int)


# ---------------------------------------------------------------------------
# build_for_response — config values forwarded correctly
# ---------------------------------------------------------------------------


class TestBuildForResponseConfig:
    """Config values must be forwarded unchanged."""

    def test_system_prompt_from_config_instructions(self):
        builder = ContextBuilder(_config(instructions="Speak like a pirate."))
        _, system, _, _ = builder.build_for_response([_user("Hi")], has_tools=False)

        assert system == "Speak like a pirate."

    def test_temperature_from_config(self):
        builder = ContextBuilder(_config(temperature=0.3))
        _, _, temperature, _ = builder.build_for_response([_user("Hi")], has_tools=False)

        assert temperature == pytest.approx(0.3)

    def test_max_tokens_uses_llm_global_when_config_is_inf(self):
        """When max_response_output_tokens is 'inf', fall back to LLM.max_tokens."""
        builder = ContextBuilder(_config(max_response_output_tokens="inf"))
        _, _, _, max_tokens = builder.build_for_response([_user("Hi")], has_tools=False)

        assert max_tokens == LLM.max_tokens

    def test_max_tokens_uses_config_when_explicitly_set(self):
        """When max_response_output_tokens is an int, use it directly."""
        builder = ContextBuilder(_config(max_response_output_tokens=256))
        _, _, _, max_tokens = builder.build_for_response([_user("Hi")], has_tools=False)

        assert max_tokens == 256

    def test_max_tokens_ignores_llm_global_when_config_is_int(self):
        """An explicit integer config value must not be silently replaced by LLM.max_tokens."""
        explicit_limit = 42
        builder = ContextBuilder(_config(max_response_output_tokens=explicit_limit))
        _, _, _, max_tokens = builder.build_for_response([_user("Hi")], has_tools=False)

        assert max_tokens == explicit_limit
        assert max_tokens != LLM.max_tokens or explicit_limit == LLM.max_tokens


# ---------------------------------------------------------------------------
# build_messages — routing to underlying conversion functions
# ---------------------------------------------------------------------------


class TestBuildMessagesRouting:
    """build_messages must route to the correct underlying function based on has_tools."""

    def test_without_tools_converts_all_items(self):
        """has_tools=False uses items_to_messages — all items in history are returned."""
        items = [_user(f"message {i}") for i in range(20)]
        builder = ContextBuilder(_config())

        result = builder.build_messages(items, has_tools=False)

        assert len(result) == 20

    def test_with_tools_applies_budget_windowing(self):
        """has_tools=True uses budget windowing — respects window_fallback cap."""
        # Build a history far larger than the default window_fallback
        items = [_user(f"message {i}") for i in range(CONTEXT.window_fallback * 3)]
        builder = ContextBuilder(_config())

        result = builder.build_messages(items, has_tools=True)

        # Budget windowing should cap the result at or near window_fallback
        # (+ 1 for the pinned first user message if it falls outside the window)
        assert len(result) <= CONTEXT.window_fallback + 1

    def test_without_tools_default_has_tools_false(self):
        """Default value for has_tools must be False (no windowing)."""
        items = [_user(f"msg {i}") for i in range(20)]
        builder = ContextBuilder(_config())

        default_result = builder.build_messages(items)
        explicit_false_result = builder.build_messages(items, has_tools=False)

        assert default_result == explicit_false_result


# ---------------------------------------------------------------------------
# build_messages — message content correctness
# ---------------------------------------------------------------------------


class TestBuildMessagesContent:
    """Messages returned must faithfully represent the conversation items."""

    def test_user_message_role_and_content(self):
        builder = ContextBuilder(_config())
        result = builder.build_messages([_user("Hello there")], has_tools=False)

        assert result == [{"role": "user", "content": "Hello there"}]

    def test_assistant_message_included(self):
        builder = ContextBuilder(_config())
        items = [_user("Hello"), _assistant("Hi!")]
        result = builder.build_messages(items, has_tools=False)

        assert result[1] == {"role": "assistant", "content": "Hi!"}

    def test_tool_pair_produces_assistant_and_tool_messages(self):
        """A function_call + function_call_output pair must yield assistant + tool messages."""
        builder = ContextBuilder(_config())
        items = [
            _user("Search something"),
            _function_call("call_1", "web_search", '{"query": "macaw"}'),
            _function_call_output("call_1", '{"results": ["..."]}'),
            _assistant("Here is what I found."),
        ]
        result = builder.build_messages(items, has_tools=False)

        roles = [m["role"] for m in result]
        assert roles == ["user", "assistant", "tool", "assistant"]

    def test_function_call_message_has_tool_calls_field(self):
        builder = ContextBuilder(_config())
        items = [
            _function_call("call_2", "lookup_customer", '{"id": "123"}'),
            _function_call_output("call_2", '{"name": "Paulo"}'),
        ]
        result = builder.build_messages(items, has_tools=False)

        assistant_msg = next(m for m in result if m["role"] == "assistant")
        assert "tool_calls" in assistant_msg
        assert assistant_msg["tool_calls"][0]["id"] == "call_2"

    def test_function_call_output_message_has_tool_call_id(self):
        builder = ContextBuilder(_config())
        items = [
            _function_call("call_3", "get_balance"),
            _function_call_output("call_3", '{"balance": 100}'),
        ]
        result = builder.build_messages(items, has_tools=False)

        tool_msg = next(m for m in result if m["role"] == "tool")
        assert tool_msg["tool_call_id"] == "call_3"
        assert tool_msg["content"] == '{"balance": 100}'


# ---------------------------------------------------------------------------
# rebuild_after_tool_round
# ---------------------------------------------------------------------------


class TestRebuildAfterToolRound:
    """rebuild_after_tool_round always uses budget windowing (same as has_tools=True)."""

    def test_returns_list_of_dicts(self):
        builder = ContextBuilder(_config())
        result = builder.rebuild_after_tool_round([_user("Hi"), _assistant("Hello")])

        assert isinstance(result, list)
        assert all(isinstance(m, dict) for m in result)

    def test_identical_to_build_messages_with_has_tools_true(self):
        """rebuild_after_tool_round must produce the same output as build_messages(has_tools=True)."""
        items = [
            _user("Start"),
            _assistant("Ok"),
            _function_call("call_r", "web_search"),
            _function_call_output("call_r", '{"data": "x"}'),
            _assistant("Done"),
        ]
        builder = ContextBuilder(_config())

        via_rebuild = builder.rebuild_after_tool_round(items)
        via_build = builder.build_messages(items, has_tools=True)

        assert via_rebuild == via_build

    def test_applies_budget_windowing_on_large_history(self):
        """Even after a tool round, history should be capped by window_fallback."""
        items = [_user(f"turn {i}") for i in range(CONTEXT.window_fallback * 3)]
        builder = ContextBuilder(_config())

        result = builder.rebuild_after_tool_round(items)

        assert len(result) <= CONTEXT.window_fallback + 1

    def test_tool_pairs_kept_together_after_rebuild(self):
        """A function_call/output pair must never be split by the rebuild windowing."""
        items = [
            _user("first message"),
            *[_assistant(f"filler {i}") for i in range(10)],
            _function_call("call_last", "web_search"),
            _function_call_output("call_last", '{"ok": true}'),
        ]
        builder = ContextBuilder(_config())

        result = builder.rebuild_after_tool_round(items)

        has_tool_result = any(m.get("role") == "tool" for m in result)
        has_tool_call = any(m.get("tool_calls") for m in result)

        if has_tool_result:
            assert has_tool_call, "Orphan tool result without its function_call"
        if has_tool_call:
            assert has_tool_result, "Dangling function_call without its tool result"


# ---------------------------------------------------------------------------
# Edge cases — empty and minimal inputs
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_empty_items_build_messages_without_tools(self):
        builder = ContextBuilder(_config())
        result = builder.build_messages([], has_tools=False)

        assert result == []

    def test_empty_items_build_messages_with_tools(self):
        builder = ContextBuilder(_config())
        result = builder.build_messages([], has_tools=True)

        assert result == []

    def test_empty_items_rebuild_after_tool_round(self):
        builder = ContextBuilder(_config())
        result = builder.rebuild_after_tool_round([])

        assert result == []

    def test_empty_items_build_for_response(self):
        """build_for_response on empty history must still return the correct tuple shape."""
        builder = ContextBuilder(_config(instructions="sys", temperature=0.5))
        messages, system, temperature, max_tokens = builder.build_for_response([], has_tools=False)

        assert messages == []
        assert system == "sys"
        assert temperature == pytest.approx(0.5)
        assert isinstance(max_tokens, int)

    def test_single_user_item(self):
        builder = ContextBuilder(_config())
        result = builder.build_messages([_user("Only message")], has_tools=False)

        assert len(result) == 1
        assert result[0]["content"] == "Only message"

    def test_orphan_function_call_output_cleaned_in_budget_path(self):
        """When has_tools=True, an orphan tool result (no matching call) must be dropped.

        _clean_orphan_tool_messages is applied by items_to_budget_messages but NOT
        by items_to_messages, so we verify the clean-up through the has_tools=True path.
        """
        items = [
            _user("Hello"),
            _function_call_output("call_orphan", '{"data": "x"}'),
            _assistant("Done"),
        ]
        builder = ContextBuilder(_config())

        result = builder.build_messages(items, has_tools=True)

        tool_messages = [m for m in result if m.get("role") == "tool"]
        assert tool_messages == [], "Orphan tool result must be removed by budget windowing"

    def test_orphan_function_call_cleaned_in_budget_path(self):
        """When has_tools=True, a dangling function_call (no output) must be dropped.

        _clean_orphan_tool_messages is applied by items_to_budget_messages but NOT
        by items_to_messages, so we verify the clean-up through the has_tools=True path.
        """
        items = [
            _user("Hello"),
            _function_call("call_dangling", "web_search"),
            _assistant("Done"),
        ]
        builder = ContextBuilder(_config())

        result = builder.build_messages(items, has_tools=True)

        tool_calls_messages = [m for m in result if m.get("tool_calls")]
        assert tool_calls_messages == [], "Dangling function_call must be removed by budget windowing"

    def test_accepts_sequence_not_only_list(self):
        """build_messages must accept any Sequence, not just lists."""
        builder = ContextBuilder(_config())
        items_tuple = (_user("Hi"), _assistant("Hello"))

        result = builder.build_messages(items_tuple, has_tools=False)

        assert len(result) == 2

    def test_empty_instructions_forwarded_as_empty_string(self):
        builder = ContextBuilder(_config(instructions=""))
        _, system, _, _ = builder.build_for_response([_user("Hi")], has_tools=False)

        assert system == ""


# ---------------------------------------------------------------------------
# First user message pinning (budget windowing behaviour surfaced through ContextBuilder)
# ---------------------------------------------------------------------------


class TestFirstUserMessagePinning:
    """Budget windowing pins the first user message. ContextBuilder must surface this."""

    def test_first_user_message_present_in_windowed_output(self):
        """The first user message must be included even when well outside the window."""
        items = [
            _user("My name is Paulo and I need help with billing", item_id="first_user"),
            *[_assistant(f"reply {i}") for i in range(20)],
            _user("Latest question"),
        ]
        builder = ContextBuilder(_config())

        result = builder.build_messages(items, has_tools=True)

        texts = [m.get("content", "") for m in result if m.get("role") == "user"]
        assert any("Paulo" in t for t in texts), "First user message must be pinned in windowed context"
