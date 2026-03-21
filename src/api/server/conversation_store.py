"""
ConversationStore — manages conversation items and memory.

Extracted from RealtimeSession to isolate conversation state
from protocol handling and response orchestration.
"""

from __future__ import annotations

import asyncio
from collections import deque
from typing import TYPE_CHECKING

from protocol.contract import MAX_CONVERSATION_ITEMS
from tools.recall_memory import ConversationMemory

if TYPE_CHECKING:
    from protocol.models import ConversationItem


class ConversationStore:
    """Thread-safe conversation item storage with memory indexing.

    Owns:
    - Conversation items (deque with auto-eviction)
    - ConversationMemory for keyword search
    - State lock for serialized mutations

    Does NOT own:
    - Session config, providers, or response tasks
    """

    def __init__(self):
        self._items: deque[ConversationItem] = deque(maxlen=MAX_CONVERSATION_ITEMS)
        self._memory = ConversationMemory()
        self._lock = asyncio.Lock()

    @property
    def items(self) -> deque[ConversationItem]:
        return self._items

    @property
    def memory(self) -> ConversationMemory:
        return self._memory

    @property
    def lock(self) -> asyncio.Lock:
        return self._lock

    def append(self, item: ConversationItem) -> None:
        """Append item and feed conversation memory.

        Must be called under self.lock.
        """
        self._items.append(item)
        self._feed_memory(item)

    def delete(self, item_id: str) -> bool:
        """Delete item by ID. Returns True if found and deleted.

        Must be called under self.lock.
        """
        original_len = len(self._items)
        filtered = deque(
            (i for i in self._items if i.id != item_id),
            maxlen=MAX_CONVERSATION_ITEMS,
        )
        found = len(filtered) < original_len
        self._items = filtered
        return found

    def find(self, item_id: str) -> ConversationItem | None:
        """Find item by ID. Can be called under self.lock."""
        for item in self._items:
            if item.id == item_id:
                return item
        return None

    def last_id(self) -> str:
        """Return ID of last item, or empty string if empty."""
        return self._items[-1].id if self._items else ""

    def _feed_memory(self, item: ConversationItem) -> None:
        """Extract text from item and add to conversation memory."""
        if item.type != "message" or item.role not in ("user", "assistant"):
            return
        text = ""
        for part in item.content:
            if part.type in ("input_text", "text") and part.text:
                text += part.text + " "
            elif part.type in ("input_audio", "audio") and part.transcript:
                text += part.transcript + " "
        if text.strip():
            self._memory.add(item.role, text.strip())
