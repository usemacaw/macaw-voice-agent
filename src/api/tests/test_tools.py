"""Tests for server-side tool calling."""

from __future__ import annotations

import asyncio
import json

import pytest

from tools.registry import ToolRegistry, ToolExecutionError
from tools.handlers import (
    mock_get_account_balance,
    mock_get_card_info,
    mock_lookup_customer,
    mock_get_recent_transactions,
    mock_create_support_ticket,
    mock_transfer_to_human,
    register_mock_handlers,
)


# ---------------------------------------------------------------------------
# ToolRegistry unit tests
# ---------------------------------------------------------------------------


class TestToolRegistry:
    def setup_method(self):
        self.registry = ToolRegistry(timeout=5.0, max_rounds=3)

    def test_empty_registry_has_no_tools(self):
        assert not self.registry.has_server_tools
        assert self.registry.get_schemas() == []

    def test_register_tool(self):
        async def dummy(**kwargs):
            return {"ok": True}

        schema = {
            "type": "function",
            "function": {"name": "test", "description": "A test tool"},
        }
        self.registry.register("test", handler=dummy, schema=schema)

        assert self.registry.has_server_tools
        assert self.registry.has_tool("test")
        assert not self.registry.has_tool("nonexistent")
        assert len(self.registry.get_schemas()) == 1

    @pytest.mark.asyncio
    async def test_execute_tool(self):
        async def greet(name: str = "World"):
            return {"greeting": f"Hello, {name}!"}

        schema = {
            "type": "function",
            "function": {"name": "greet", "description": "Greet"},
        }
        self.registry.register("greet", handler=greet, schema=schema)

        result_json = await self.registry.execute("greet", '{"name": "Paulo"}')
        result = json.loads(result_json)
        assert result["greeting"] == "Hello, Paulo!"

    @pytest.mark.asyncio
    async def test_execute_unknown_tool_raises(self):
        with pytest.raises(ToolExecutionError, match="not registered"):
            await self.registry.execute("nonexistent", "{}")

    @pytest.mark.asyncio
    async def test_execute_invalid_json_raises(self):
        async def dummy(**kwargs):
            return {}

        self.registry.register(
            "test",
            handler=dummy,
            schema={"type": "function", "function": {"name": "test"}},
        )
        with pytest.raises(ToolExecutionError, match="Invalid JSON"):
            await self.registry.execute("test", "not json")

    @pytest.mark.asyncio
    async def test_execute_timeout_returns_error(self):
        async def slow_tool(**kwargs):
            await asyncio.sleep(10)
            return {"result": "too late"}

        self.registry = ToolRegistry(timeout=0.1)
        self.registry.register(
            "slow",
            handler=slow_tool,
            schema={"type": "function", "function": {"name": "slow"}},
        )

        result_json = await self.registry.execute("slow", "{}")
        result = json.loads(result_json)
        assert result["error"] == "timeout"

    @pytest.mark.asyncio
    async def test_execute_handler_exception_returns_error(self):
        async def broken(**kwargs):
            raise ValueError("something went wrong")

        self.registry.register(
            "broken",
            handler=broken,
            schema={"type": "function", "function": {"name": "broken"}},
        )

        result_json = await self.registry.execute("broken", "{}")
        result = json.loads(result_json)
        assert result["error"] == "execution_failed"

    def test_get_filler_default(self):
        async def dummy(**kwargs):
            return {}

        self.registry.register(
            "test",
            handler=dummy,
            schema={"type": "function", "function": {"name": "test"}},
        )
        # No custom filler — returns default
        filler = self.registry.get_filler("test")
        assert filler  # Non-empty default

    def test_get_filler_custom(self):
        async def dummy(**kwargs):
            return {}

        self.registry.register(
            "test",
            handler=dummy,
            schema={"type": "function", "function": {"name": "test"}},
            filler_phrase="Consultando agora...",
        )
        assert self.registry.get_filler("test") == "Consultando agora..."

    def test_max_rounds(self):
        assert self.registry.max_rounds == 3


# ---------------------------------------------------------------------------
# Mock handler tests
# ---------------------------------------------------------------------------


class TestMockHandlers:
    @pytest.mark.asyncio
    async def test_lookup_customer_with_phone(self):
        result = await mock_lookup_customer(phone="11987654321")
        assert result["customer_id"]
        assert result["status"] == "active"

    @pytest.mark.asyncio
    async def test_lookup_customer_no_args_returns_error(self):
        result = await mock_lookup_customer()
        assert "error" in result

    @pytest.mark.asyncio
    async def test_get_account_balance(self):
        result = await mock_get_account_balance(account_id="ACC-123")
        assert result["account_id"] == "ACC-123"
        assert isinstance(result["balance"], float)
        assert result["currency"] == "BRL"

    @pytest.mark.asyncio
    async def test_get_card_info(self):
        result = await mock_get_card_info(card_number="1234567890123456")
        assert "3456" in result["card_number"]
        assert result["limit"] > 0

    @pytest.mark.asyncio
    async def test_get_recent_transactions(self):
        result = await mock_get_recent_transactions(account_id="ACC-123", limit=3)
        assert len(result["transactions"]) == 3
        assert result["total_shown"] == 3

    @pytest.mark.asyncio
    async def test_create_support_ticket(self):
        result = await mock_create_support_ticket(
            category="billing", description="Cobranca indevida"
        )
        assert result["ticket_id"].startswith("TKT-")
        assert result["status"] == "open"

    @pytest.mark.asyncio
    async def test_transfer_to_human(self):
        result = await mock_transfer_to_human(department="billing")
        assert result["status"] == "transferring"
        assert result["department"] == "billing"


class TestMockRegistration:
    def test_register_mock_handlers(self):
        registry = ToolRegistry()
        register_mock_handlers(registry)

        assert registry.has_server_tools
        assert registry.has_tool("lookup_customer")
        assert registry.has_tool("get_account_balance")
        assert registry.has_tool("get_card_info")
        assert registry.has_tool("get_recent_transactions")
        assert registry.has_tool("create_support_ticket")
        assert registry.has_tool("transfer_to_human")
        assert len(registry.get_schemas()) == 6

    @pytest.mark.asyncio
    async def test_registered_handlers_execute(self):
        registry = ToolRegistry()
        register_mock_handlers(registry)

        result_json = await registry.execute(
            "get_account_balance", '{"account_id": "ACC-999"}'
        )
        result = json.loads(result_json)
        assert result["account_id"] == "ACC-999"
        assert isinstance(result["balance"], float)
