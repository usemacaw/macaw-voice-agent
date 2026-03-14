"""
Conversation memory recall tool.

Allows the LLM to search older conversation history when the recent
window doesn't contain enough context. This enables sending only a
small window of recent messages to the LLM while still giving it
access to the full conversation when needed.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tools.registry import ToolRegistry

logger = logging.getLogger("open-voice-api.tools.recall")


class ConversationMemory:
    """Stores full conversation history and provides keyword search."""

    def __init__(self) -> None:
        self._entries: list[dict] = []  # {role, content, index}

    def add(self, role: str, content: str) -> None:
        if not content or not content.strip():
            return
        self._entries.append({
            "role": role,
            "content": content.strip(),
            "index": len(self._entries),
        })

    def search(self, query: str, max_results: int = 5) -> list[dict]:
        """Simple keyword search over conversation history."""
        if not query:
            return []

        keywords = query.lower().split()
        scored: list[tuple[int, dict]] = []

        for entry in self._entries:
            text = entry["content"].lower()
            score = sum(1 for kw in keywords if kw in text)
            if score > 0:
                scored.append((score, entry))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [
            {"role": e["role"], "content": e["content"][:200]}
            for _, e in scored[:max_results]
        ]

    def get_recent(self, n: int = 5) -> list[dict]:
        """Get the N most recent entries."""
        return [
            {"role": e["role"], "content": e["content"][:200]}
            for e in self._entries[-n:]
        ]

    @property
    def size(self) -> int:
        return len(self._entries)


async def recall_memory(query: str, max_results: int = 3) -> dict:
    """Placeholder — replaced at registration with bound instance."""
    return {"error": "Memory not initialized."}


def create_recall_handler(memory: ConversationMemory):
    """Create a recall handler bound to a specific ConversationMemory instance."""

    async def _recall(query: str, max_results: int = 3) -> dict:
        """Busca no historico da conversa por informacoes anteriores."""
        if not query:
            return {"error": "Informe o que deseja lembrar."}

        results = memory.search(query, max_results=max_results)
        if not results:
            return {
                "query": query,
                "results": [],
                "message": "Nao encontrei isso no historico da conversa.",
            }

        return {"query": query, "results": results}

    return _recall


_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "recall_memory",
        "description": (
            "Busca no historico anterior da conversa. Use quando o usuario "
            "mencionar algo que foi dito antes, como 'voce lembra?', "
            "'o que eu perguntei?', 'como era mesmo?'. NAO use para "
            "perguntas novas — apenas para relembrar o que ja foi conversado."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Palavras-chave do que buscar no historico (ex: 'dolar', 'guerra')",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Numero maximo de resultados (padrao: 3)",
                },
            },
            "required": ["query"],
        },
    },
}


def register_recall_handler(registry: ToolRegistry, memory: ConversationMemory) -> None:
    """Register the recall_memory tool bound to a ConversationMemory instance."""
    handler = create_recall_handler(memory)
    registry.register(
        "recall_memory",
        handler=handler,
        schema=_TOOL_SCHEMA,
        filler_phrase="Deixa eu lembrar...",
    )
    logger.info("Registered recall_memory tool")
