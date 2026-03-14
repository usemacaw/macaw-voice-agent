"""
Server-side tool calling for voice agents.

Provides a ToolRegistry for registering and executing tools server-side,
eliminating the need for client-side execution and enabling filler phrases
during tool execution.
"""

from tools.registry import ToolRegistry, create_tool_registry

__all__ = ["ToolRegistry", "create_tool_registry"]
