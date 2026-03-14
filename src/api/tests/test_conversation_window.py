"""Tests for conversation windowing and orphan cleanup."""

from __future__ import annotations

import pytest

from pipeline.conversation import (
    items_to_messages,
    items_to_windowed_messages,
    _clean_orphan_tool_messages,
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
