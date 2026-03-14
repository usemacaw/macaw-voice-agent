"""
Tool Registry — server-side tool registration and execution.

Tools registered here are executed server-side when the LLM emits tool calls,
instead of being forwarded to the client for execution.

Usage:
    registry = ToolRegistry()
    registry.register("get_balance", handler=my_handler, schema={...})
    result = await registry.execute("get_balance", '{"account_id": "123"}')
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Coroutine

from config import TOOL_CONFIG

logger = logging.getLogger("open-voice-api.tools")

# Type alias for tool handler: async function that takes kwargs and returns dict
ToolHandlerFn = Callable[..., Coroutine[Any, Any, dict]]


@dataclass
class ToolDef:
    """Definition of a registered tool."""

    name: str
    handler: ToolHandlerFn
    schema: dict
    filler_phrase: str = ""


class ToolExecutionError(Exception):
    """Raised when a tool fails to execute."""

    def __init__(self, tool_name: str, message: str):
        self.tool_name = tool_name
        super().__init__(f"Tool '{tool_name}' failed: {message}")


class ToolRegistry:
    """Registry of server-side tools with async execution."""

    def __init__(self, timeout: float = 10.0, max_rounds: int = 5):
        self._tools: dict[str, ToolDef] = {}
        self._timeout = timeout
        self._max_rounds = max_rounds
        self._default_filler = TOOL_CONFIG.get(
            "default_filler", "Um momento, por favor."
        )

    def fork(self) -> ToolRegistry:
        """Create a copy of this registry for per-session customization.

        The new registry shares the same tool definitions but can have
        additional tools registered without affecting the original.
        """
        clone = ToolRegistry(timeout=self._timeout, max_rounds=self._max_rounds)
        clone._tools = dict(self._tools)
        clone._default_filler = self._default_filler
        return clone

    @property
    def has_server_tools(self) -> bool:
        """True if any tools are registered for server-side execution."""
        return len(self._tools) > 0

    @property
    def max_rounds(self) -> int:
        return self._max_rounds

    def register(
        self,
        name: str,
        handler: ToolHandlerFn,
        schema: dict,
        filler_phrase: str = "",
    ) -> None:
        """Register a tool for server-side execution.

        Args:
            name: Tool name (must match the function name in LLM tool schema).
            handler: Async function that executes the tool. Receives kwargs from
                     parsed JSON arguments. Must return a dict.
            schema: OpenAI function tool schema (type, function, name, description, parameters).
            filler_phrase: Custom filler phrase for TTS while this tool executes.
                          Falls back to default_filler if empty.
        """
        if name in self._tools:
            logger.warning(f"Tool '{name}' already registered — overwriting")

        self._tools[name] = ToolDef(
            name=name,
            handler=handler,
            schema=schema,
            filler_phrase=filler_phrase,
        )
        logger.info(f"Tool registered: {name}")

    def has_tool(self, name: str) -> bool:
        return name in self._tools

    def get_filler(self, name: str) -> str:
        """Get filler phrase for a tool (custom or default)."""
        tool = self._tools.get(name)
        if tool and tool.filler_phrase:
            return tool.filler_phrase
        return self._default_filler

    def get_schemas(self) -> list[dict]:
        """Get all tool schemas in OpenAI function format."""
        return [t.schema for t in self._tools.values()]

    async def execute(self, name: str, arguments_json: str) -> str:
        """Execute a tool by name with JSON arguments.

        Args:
            name: Tool name.
            arguments_json: JSON string with tool arguments.

        Returns:
            JSON string with tool result.

        Raises:
            ToolExecutionError: If tool not found, arguments invalid, or execution fails.
        """
        tool = self._tools.get(name)
        if tool is None:
            raise ToolExecutionError(name, f"Tool not registered: {name}")

        # Parse arguments
        try:
            args = json.loads(arguments_json) if arguments_json else {}
        except json.JSONDecodeError as e:
            raise ToolExecutionError(name, f"Invalid JSON arguments: {e}") from e

        if not isinstance(args, dict):
            raise ToolExecutionError(
                name, f"Arguments must be a JSON object, got {type(args).__name__}"
            )

        # Execute with timeout
        t0 = time.perf_counter()
        try:
            result = await asyncio.wait_for(
                tool.handler(**args), timeout=self._timeout
            )
        except asyncio.TimeoutError:
            elapsed_ms = (time.perf_counter() - t0) * 1000
            logger.error(
                f"Tool '{name}' timed out after {elapsed_ms:.0f}ms "
                f"(limit={self._timeout}s)"
            )
            return json.dumps(
                {
                    "error": "timeout",
                    "message": f"A consulta demorou mais que o esperado. Tente novamente.",
                }
            )
        except Exception as e:
            elapsed_ms = (time.perf_counter() - t0) * 1000
            logger.error(
                f"Tool '{name}' failed after {elapsed_ms:.0f}ms: {e}", exc_info=True
            )
            return json.dumps(
                {
                    "error": "execution_failed",
                    "message": f"Nao foi possivel completar a consulta no momento.",
                }
            )

        elapsed_ms = (time.perf_counter() - t0) * 1000
        logger.info(f"Tool '{name}' executed in {elapsed_ms:.0f}ms")

        if not isinstance(result, dict):
            result = {"result": result}

        return json.dumps(result, ensure_ascii=False)


def create_tool_registry() -> ToolRegistry:
    """Create a ToolRegistry and register handlers based on config."""
    timeout = TOOL_CONFIG.get("timeout", 10.0)
    max_rounds = TOOL_CONFIG.get("max_rounds", 5)
    registry = ToolRegistry(timeout=timeout, max_rounds=max_rounds)

    if TOOL_CONFIG.get("enable_mock_tools", False):
        from tools.handlers import register_mock_handlers

        register_mock_handlers(registry)
        logger.info("Mock tool handlers registered")

    if TOOL_CONFIG.get("enable_web_search", False):
        from tools.web_search import register_web_search_handlers

        register_web_search_handlers(registry)
        logger.info("Web search tools registered")

    return registry
