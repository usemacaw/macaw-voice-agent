"""Tests for conversation memory recall tool."""

from __future__ import annotations

import json
import pytest

from tools.recall_memory import (
    ConversationMemory,
    create_recall_handler,
    register_recall_handler,
)
from tools.registry import ToolRegistry


class TestConversationMemory:
    def test_add_and_search(self):
        mem = ConversationMemory()
        mem.add("user", "Qual a cotação do dólar hoje?")
        mem.add("assistant", "O dólar está a R$5,20 hoje.")
        mem.add("user", "E as notícias sobre política?")
        mem.add("assistant", "O congresso aprovou nova lei fiscal.")

        results = mem.search("dólar")
        assert len(results) >= 1
        assert any("dólar" in r["content"].lower() for r in results)

    def test_search_returns_max_results(self):
        mem = ConversationMemory()
        for i in range(10):
            mem.add("user", f"Mensagem {i} sobre python")

        results = mem.search("python", max_results=3)
        assert len(results) == 3

    def test_search_empty_query(self):
        mem = ConversationMemory()
        mem.add("user", "teste")
        assert mem.search("") == []

    def test_search_no_match(self):
        mem = ConversationMemory()
        mem.add("user", "Olá bom dia")
        results = mem.search("xylophone")
        assert results == []

    def test_add_ignores_empty(self):
        mem = ConversationMemory()
        mem.add("user", "")
        mem.add("user", "   ")
        assert mem.size == 0

    def test_get_recent(self):
        mem = ConversationMemory()
        mem.add("user", "Mensagem 1")
        mem.add("assistant", "Resposta 1")
        mem.add("user", "Mensagem 2")

        recent = mem.get_recent(2)
        assert len(recent) == 2
        assert recent[0]["content"] == "Resposta 1"
        assert recent[1]["content"] == "Mensagem 2"

    def test_content_truncated_to_200(self):
        mem = ConversationMemory()
        mem.add("user", "x" * 500)
        results = mem.search("x", max_results=1)
        assert len(results[0]["content"]) == 200


class TestRecallHandler:
    @pytest.mark.asyncio
    async def test_recall_returns_results(self):
        mem = ConversationMemory()
        mem.add("user", "Quanto custa o dólar?")
        mem.add("assistant", "O dólar está R$5,20.")

        handler = create_recall_handler(mem)
        result = await handler(query="dólar")
        assert result["query"] == "dólar"
        assert len(result["results"]) > 0

    @pytest.mark.asyncio
    async def test_recall_empty_query(self):
        mem = ConversationMemory()
        handler = create_recall_handler(mem)
        result = await handler(query="")
        assert "error" in result

    @pytest.mark.asyncio
    async def test_recall_no_match(self):
        mem = ConversationMemory()
        mem.add("user", "Olá")
        handler = create_recall_handler(mem)
        result = await handler(query="xylophone")
        assert result["results"] == []
        assert "message" in result


class TestRecallRegistration:
    def test_register_recall_handler(self):
        registry = ToolRegistry()
        mem = ConversationMemory()
        register_recall_handler(registry, mem)

        assert registry.has_tool("recall_memory")
        schemas = registry.get_schemas()
        names = [s["function"]["name"] for s in schemas]
        assert "recall_memory" in names

    @pytest.mark.asyncio
    async def test_registered_recall_executes(self):
        registry = ToolRegistry()
        mem = ConversationMemory()
        mem.add("user", "Meu nome é Paulo")
        register_recall_handler(registry, mem)

        result_json = await registry.execute(
            "recall_memory", '{"query": "nome"}'
        )
        result = json.loads(result_json)
        assert "results" in result
        assert len(result["results"]) > 0
