"""Tests for ConversationStore: items CRUD, eviction, and memory indexing."""

import pytest

from protocol.models import ConversationItem, ContentPart
from server.conversation_store import ConversationStore


def _make_item(item_id: str, role: str = "user", text: str = "hello") -> ConversationItem:
    return ConversationItem(
        id=item_id,
        type="message",
        role=role,
        content=[ContentPart(type="input_text", text=text)],
    )


class TestAppend:
    def test_append_adds_item(self):
        store = ConversationStore()
        item = _make_item("item_1")
        store.append(item)

        assert len(store.items) == 1
        assert store.items[0].id == "item_1"

    def test_append_feeds_memory(self):
        store = ConversationStore()
        store.append(_make_item("item_1", text="Python programming"))

        results = store.memory.search("Python")
        assert len(results) >= 1

    def test_append_evicts_oldest_when_full(self):
        store = ConversationStore()
        # Fill beyond maxlen (128 from contract)
        from protocol.contract import MAX_CONVERSATION_ITEMS

        for i in range(MAX_CONVERSATION_ITEMS + 5):
            store.append(_make_item(f"item_{i}"))

        assert len(store.items) == MAX_CONVERSATION_ITEMS
        # First items evicted
        assert store.items[0].id == "item_5"

    def test_append_ignores_non_message_for_memory(self):
        store = ConversationStore()
        item = ConversationItem(
            id="tool_1",
            type="function_call_output",
            role="tool",
            content=[ContentPart(type="text", text="tool result data")],
        )
        store.append(item)

        # Tool results not indexed in memory
        results = store.memory.search("tool result")
        assert len(results) == 0


class TestDelete:
    def test_delete_existing_item(self):
        store = ConversationStore()
        store.append(_make_item("item_1"))
        store.append(_make_item("item_2"))

        found = store.delete("item_1")
        assert found is True
        assert len(store.items) == 1
        assert store.items[0].id == "item_2"

    def test_delete_nonexistent_item(self):
        store = ConversationStore()
        store.append(_make_item("item_1"))

        found = store.delete("item_nonexistent")
        assert found is False
        assert len(store.items) == 1

    def test_delete_from_empty_store(self):
        store = ConversationStore()
        found = store.delete("item_1")
        assert found is False


class TestFind:
    def test_find_existing_item(self):
        store = ConversationStore()
        store.append(_make_item("item_1"))
        store.append(_make_item("item_2"))

        found = store.find("item_2")
        assert found is not None
        assert found.id == "item_2"

    def test_find_nonexistent_returns_none(self):
        store = ConversationStore()
        store.append(_make_item("item_1"))

        assert store.find("missing") is None

    def test_find_in_empty_store(self):
        store = ConversationStore()
        assert store.find("anything") is None


class TestLastId:
    def test_last_id_with_items(self):
        store = ConversationStore()
        store.append(_make_item("item_1"))
        store.append(_make_item("item_2"))

        assert store.last_id() == "item_2"

    def test_last_id_empty_store(self):
        store = ConversationStore()
        assert store.last_id() == ""


class TestMemoryIntegration:
    def test_user_message_indexed(self):
        store = ConversationStore()
        store.append(_make_item("item_1", role="user", text="What is the weather?"))

        results = store.memory.search("weather")
        assert len(results) >= 1

    def test_assistant_message_indexed(self):
        store = ConversationStore()
        store.append(_make_item("item_1", role="assistant", text="The weather is sunny"))

        results = store.memory.search("sunny")
        assert len(results) >= 1

    def test_transcript_part_indexed(self):
        store = ConversationStore()
        item = ConversationItem(
            id="item_1",
            type="message",
            role="user",
            content=[ContentPart(type="audio", transcript="spoken words here")],
        )
        store.append(item)

        results = store.memory.search("spoken words")
        assert len(results) >= 1
