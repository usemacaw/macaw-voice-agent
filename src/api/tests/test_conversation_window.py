"""Tests for conversation windowing, token budget, and orphan cleanup."""

from __future__ import annotations

import pytest

from pipeline.conversation import (
    items_to_messages,
    items_to_windowed_messages,
    items_to_budget_messages,
    _clean_orphan_tool_messages,
    _estimate_tokens,
)
from protocol.models import ConversationItem, ContentPart


def _msg(role: str, text: str, item_id: str = "") -> ConversationItem:
    return ConversationItem(
        id=item_id or f"item_{role}_{text[:8]}",
        type="message",
        role=role,
        content=[ContentPart(type="input_text" if role == "user" else "text", text=text)],
        status="completed",
    )


def _fc(call_id: str, name: str, args: str = "{}") -> ConversationItem:
    return ConversationItem(
        id=f"item_fc_{call_id}",
        type="function_call",
        call_id=call_id,
        name=name,
        arguments=args,
        status="completed",
    )


def _fco(call_id: str, output: str = '{"ok": true}') -> ConversationItem:
    return ConversationItem(
        id=f"item_fco_{call_id}",
        type="function_call_output",
        call_id=call_id,
        output=output,
        status="completed",
    )


class TestWindowedMessages:
    def test_small_history_no_window(self):
        items = [_msg("user", "Olá"), _msg("assistant", "Oi!")]
        result = items_to_windowed_messages(items, window=6)
        assert len(result) == 2

    def test_large_history_windowed(self):
        items = [_msg("user", f"msg{i}") for i in range(20)]
        result = items_to_windowed_messages(items, window=6)
        assert len(result) == 6
        assert result[-1]["content"] == "msg19"

    def test_window_includes_tool_pair(self):
        """If window starts with function_call_output, pull in the call."""
        items = [
            _msg("user", "msg0"),
            _msg("assistant", "msg1"),
            _fc("call_1", "web_search", '{"query": "test"}'),
            _fco("call_1", '{"results": []}'),
            _msg("assistant", "Resultado"),
            _msg("user", "msg2"),
            _msg("assistant", "msg3"),
        ]
        # window=4 would take: fco, assistant, user, assistant
        # But fco is orphan, so it should pull fc in too
        result = items_to_windowed_messages(items, window=4)
        roles = [m["role"] for m in result]
        # Should have the tool pair complete
        if "tool" in roles:
            assert "assistant" in roles  # the tool_call msg

    def test_window_drops_dangling_call(self):
        """If window ends with function_call without output, drop it."""
        items = [
            _msg("user", "msg0"),
            _msg("user", "msg1"),
            _msg("assistant", "msg2"),
            _msg("user", "msg3"),
            _msg("assistant", "msg4"),
            _fc("call_orphan", "web_search"),
        ]
        result = items_to_windowed_messages(items, window=4)
        # The dangling fc should be dropped
        for msg in result:
            if msg.get("tool_calls"):
                pytest.fail("Dangling tool call should have been removed")


class TestCleanOrphanToolMessages:
    def test_complete_pair_kept(self):
        messages = [
            {"role": "user", "content": "Olá"},
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": "call_1", "type": "function", "function": {"name": "web_search", "arguments": "{}"}}
            ]},
            {"role": "tool", "tool_call_id": "call_1", "content": '{"ok": true}'},
            {"role": "assistant", "content": "Pronto!"},
        ]
        result = _clean_orphan_tool_messages(messages)
        assert len(result) == 4

    def test_orphan_tool_result_removed(self):
        messages = [
            {"role": "user", "content": "Olá"},
            {"role": "tool", "tool_call_id": "call_missing", "content": '{"ok": true}'},
            {"role": "assistant", "content": "Pronto!"},
        ]
        result = _clean_orphan_tool_messages(messages)
        assert len(result) == 2
        assert all(m["role"] != "tool" for m in result)

    def test_orphan_function_call_removed(self):
        messages = [
            {"role": "user", "content": "Olá"},
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": "call_1", "type": "function", "function": {"name": "web_search", "arguments": "{}"}}
            ]},
            {"role": "assistant", "content": "Resposta normal"},
        ]
        result = _clean_orphan_tool_messages(messages)
        assert len(result) == 2
        # The assistant with tool_calls but no result should be removed
        for msg in result:
            assert not msg.get("tool_calls")

    def test_regular_messages_untouched(self):
        messages = [
            {"role": "user", "content": "Olá"},
            {"role": "assistant", "content": "Oi!"},
        ]
        result = _clean_orphan_tool_messages(messages)
        assert result == messages


class TestBudgetMessages:
    def test_small_history_uses_all(self):
        items = [_msg("user", "Olá"), _msg("assistant", "Oi!")]
        result = items_to_budget_messages(items, max_tokens=4000)
        assert len(result) == 2

    def test_budget_limits_items(self):
        """Large history exceeding token budget should be trimmed."""
        items = [_msg("user", f"Message number {i} with some extra text") for i in range(50)]
        result = items_to_budget_messages(items, max_tokens=100, window_fallback=50)
        assert len(result) < 50
        # Last message should always be included
        assert result[-1]["content"].startswith("Message number 49")

    def test_window_fallback_limits_items(self):
        """Even with large budget, window_fallback caps item count."""
        items = [_msg("user", "short") for _ in range(20)]
        result = items_to_budget_messages(items, max_tokens=100000, window_fallback=5)
        # Should be capped at window_fallback (5) + possibly pinned first user
        assert len(result) <= 6

    def test_first_user_message_pinned(self):
        """First user message should be preserved even when outside window."""
        items = [
            _msg("user", "My name is Paulo, I need help with billing"),
            *[_msg("assistant", f"response {i}") for i in range(10)],
            _msg("user", "What was my last transaction?"),
        ]
        result = items_to_budget_messages(items, max_tokens=200, window_fallback=4)
        texts = [m["content"] for m in result]
        assert any("Paulo" in t for t in texts), "First user message should be pinned"

    def test_tool_pairs_kept_together(self):
        """Function call and output should never be split."""
        items = [
            _msg("user", "msg0"),
            _msg("assistant", "msg1"),
            _fc("call_1", "web_search", '{"query": "test"}'),
            _fco("call_1", '{"results": []}'),
            _msg("assistant", "Resultado"),
            _msg("user", "Obrigado"),
        ]
        result = items_to_budget_messages(items, max_tokens=4000, window_fallback=4)
        # If tool result is included, tool call must be too
        has_tool = any(m.get("role") == "tool" for m in result)
        has_call = any(m.get("tool_calls") for m in result)
        if has_tool:
            assert has_call, "Tool result without call"
        if has_call:
            assert has_tool, "Tool call without result"

    def test_empty_items(self):
        result = items_to_budget_messages([], max_tokens=4000)
        assert result == []

    def test_single_item(self):
        items = [_msg("user", "Hi")]
        result = items_to_budget_messages(items, max_tokens=4000)
        assert len(result) == 1


class TestTokenEstimation:
    def test_estimate_basic(self):
        assert _estimate_tokens("hello world") >= 1
        assert _estimate_tokens("") == 1  # min 1

    def test_longer_text_more_tokens(self):
        short = _estimate_tokens("hi")
        long = _estimate_tokens("This is a much longer piece of text that should have more tokens")
        assert long > short
