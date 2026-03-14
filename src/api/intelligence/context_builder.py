"""
ContextBuilder — single authority on LLM context construction.

Encapsulates ALL decisions about which conversation items enter the LLM,
in what format, and with what invariants preserved. No other module
should build LLM messages directly.

Design note: this module wraps pipeline/conversation.py (low-level
conversion) with higher-level policies (windowing for tools, full
context for text, orphan cleanup).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from config import CONTEXT, LLM
from pipeline.conversation import items_to_budget_messages, items_to_messages, items_to_windowed_messages

if TYPE_CHECKING:
    from protocol.models import ConversationItem, SessionConfig


class ContextBuilder:
    """Builds LLM message lists from conversation history.

    Single authority on:
    - which items enter the context
    - in what format (OpenAI messages)
    - which invariants are preserved (tool call pairs, orphan cleanup)
    - windowing policy per response type
    """

    def __init__(self, config: SessionConfig):
        self._config = config

    def build_for_response(
        self,
        items: Sequence[ConversationItem],
        has_tools: bool,
    ) -> tuple[list[dict], str, float, int]:
        """Build complete LLM call parameters for a response.

        Returns:
            (messages, system, temperature, max_tokens) ready for LLM call.
        """
        messages = self.build_messages(items, has_tools=has_tools)
        system = self._config.instructions
        temperature = self._config.temperature
        max_tokens = (
            self._config.max_response_output_tokens
            if isinstance(self._config.max_response_output_tokens, int)
            else LLM.max_tokens
        )
        return messages, system, temperature, max_tokens

    def build_messages(
        self,
        items: Sequence[ConversationItem],
        has_tools: bool = False,
    ) -> list[dict]:
        """Build LLM messages from conversation items.

        Args:
            items: Conversation items (will be converted to list internally).
            has_tools: If True, use token-budget windowed context.

        Returns:
            List of OpenAI-format message dicts.
        """
        item_list = list(items)
        if has_tools:
            return items_to_budget_messages(
                item_list,
                max_tokens=CONTEXT.max_context_tokens,
                window_fallback=CONTEXT.window_fallback,
            )
        return items_to_messages(item_list)

    def rebuild_after_tool_round(
        self,
        items: Sequence[ConversationItem],
    ) -> list[dict]:
        """Rebuild messages after tool execution added new items.

        Always uses token-budget windowed context since we're in a tool-calling flow.
        """
        return items_to_budget_messages(
            list(items),
            max_tokens=CONTEXT.max_context_tokens,
            window_fallback=CONTEXT.window_fallback,
        )

    def merge_tool_schemas(
        self,
        config_tools: list[dict] | None,
        server_schemas: list[dict],
    ) -> list[dict]:
        """Merge session config tools with server-side tool schemas.

        Avoids duplicates by name.
        """
        tools = list(config_tools) if config_tools else []
        existing_names = {
            t.get("function", {}).get("name")
            for t in tools
            if isinstance(t, dict)
        }
        for schema in server_schemas:
            name = schema.get("function", {}).get("name", "")
            if name not in existing_names:
                tools.append(schema)
        return tools
