"""Tests for pipeline/conversation.py — items_to_messages conversion."""

import pytest

from pipeline.conversation import items_to_messages
from protocol.models import ConversationItem, ContentPart


class TestItemsToMessages:
    """Test conversion of ConversationItems to LLM messages format."""

    def test_empty_items(self):
        assert items_to_messages([]) == []

    def test_text_message(self):
        item = ConversationItem(
            id="item_1", type="message", role="user",
            content=[ContentPart(type="input_text", text="Hello")],
        )
        result = items_to_messages([item])
        assert len(result) == 1
        assert result[0] == {"role": "user", "content": "Hello"}

    def test_audio_message_with_transcript(self):
        item = ConversationItem(
            id="item_1", type="message", role="user",
            content=[ContentPart(type="input_audio", transcript="Hi there")],
        )
        result = items_to_messages([item])
        assert len(result) == 1
        assert result[0] == {"role": "user", "content": "Hi there"}

    def test_audio_message_without_transcript_skipped(self):
        item = ConversationItem(
            id="item_1", type="message", role="user",
            content=[ContentPart(type="input_audio", transcript=None)],
        )
        result = items_to_messages([item])
        assert result == []

    def test_assistant_audio_message(self):
        item = ConversationItem(
            id="item_1", type="message", role="assistant",
            content=[ContentPart(type="audio", transcript="Olá, como posso ajudar?")],
        )
        result = items_to_messages([item])
        assert len(result) == 1
        assert result[0]["role"] == "assistant"
        assert result[0]["content"] == "Olá, como posso ajudar?"

    def test_multiple_content_parts_joined(self):
        item = ConversationItem(
            id="item_1", type="message", role="user",
            content=[
                ContentPart(type="input_text", text="Hello"),
                ContentPart(type="input_text", text="world"),
            ],
        )
        result = items_to_messages([item])
        assert result[0]["content"] == "Hello world"

    def test_message_without_role_skipped(self):
        item = ConversationItem(
            id="item_1", type="message", role=None,
            content=[ContentPart(type="text", text="orphan")],
        )
        result = items_to_messages([item])
        assert result == []

    def test_function_call_item(self):
        item = ConversationItem(
            id="item_1", type="function_call",
            call_id="call_123", name="get_weather",
            arguments='{"city": "SP"}',
        )
        result = items_to_messages([item])
        assert len(result) == 1
        msg = result[0]
        assert msg["role"] == "assistant"
        assert msg["content"] is None
        assert len(msg["tool_calls"]) == 1
        tc = msg["tool_calls"][0]
        assert tc["id"] == "call_123"
        assert tc["type"] == "function"
        assert tc["function"]["name"] == "get_weather"
        assert tc["function"]["arguments"] == '{"city": "SP"}'

    def test_function_call_output_item(self):
        item = ConversationItem(
            id="item_2", type="function_call_output",
            call_id="call_123", output='{"temp": 25}',
        )
        result = items_to_messages([item])
        assert len(result) == 1
        msg = result[0]
        assert msg["role"] == "tool"
        assert msg["tool_call_id"] == "call_123"
        assert msg["content"] == '{"temp": 25}'

    def test_full_conversation_ordering(self):
        items = [
            ConversationItem(
                id="1", type="message", role="user",
                content=[ContentPart(type="input_text", text="What's the weather?")],
            ),
            ConversationItem(
                id="2", type="function_call",
                call_id="call_1", name="get_weather",
                arguments='{"city": "SP"}',
            ),
            ConversationItem(
                id="3", type="function_call_output",
                call_id="call_1", output='{"temp": 25}',
            ),
            ConversationItem(
                id="4", type="message", role="assistant",
                content=[ContentPart(type="text", text="It's 25°C in SP.")],
            ),
        ]
        result = items_to_messages(items)
        assert len(result) == 4
        assert result[0]["role"] == "user"
        assert result[1]["role"] == "assistant"
        assert result[2]["role"] == "tool"
        assert result[3]["role"] == "assistant"

    def test_empty_content_parts_skipped(self):
        item = ConversationItem(
            id="item_1", type="message", role="user",
            content=[ContentPart(type="text", text=""), ContentPart(type="text", text=None)],
        )
        result = items_to_messages([item])
        assert result == []

    def test_function_call_defaults(self):
        """Function call with no call_id falls back to item id."""
        item = ConversationItem(
            id="item_fallback", type="function_call",
            name="do_something",
        )
        result = items_to_messages([item])
        assert result[0]["tool_calls"][0]["id"] == "item_fallback"
        assert result[0]["tool_calls"][0]["function"]["arguments"] == "{}"
