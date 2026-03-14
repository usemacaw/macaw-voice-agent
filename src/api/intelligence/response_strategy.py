"""
ResponseStrategy — policy selection for response generation.

Replaces branching in ResponseRunner with explicit strategy objects.
Each strategy encapsulates the "how" of generating a specific type
of response (text-only, audio streaming, tool calling).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import TYPE_CHECKING

from config import SLO

if TYPE_CHECKING:
    from protocol.models import SessionConfig
    from tools.registry import ToolRegistry


class ResponseMode(Enum):
    """The type of response to generate."""
    TEXT_ONLY = auto()
    AUDIO_STREAMING = auto()
    TOOL_CALLING = auto()
    TOOL_CALLING_AUDIO = auto()


@dataclass(frozen=True)
class ResponsePlan:
    """Describes how a response should be executed.

    Created by select_strategy() before execution begins.
    Immutable — the plan doesn't change during execution.
    """
    mode: ResponseMode
    has_audio: bool
    has_tools: bool
    server_side_tools: bool
    tools: list[dict]
    max_rounds: int
    max_first_audio_ms: float = 0.0  # SLO target (0 = no SLO)


def select_strategy(
    config: SessionConfig,
    tool_registry: ToolRegistry | None,
) -> ResponsePlan:
    """Select response strategy based on config and available tools.

    This is the single place where the system decides HOW to respond.
    All branching logic that was scattered in ResponseRunner.run()
    is now concentrated here.
    """
    has_audio = "audio" in config.modalities
    has_tools = bool(config.tools) or (
        tool_registry is not None and tool_registry.has_server_tools
    )
    server_side = (
        tool_registry is not None
        and tool_registry.has_server_tools
    )
    max_rounds = tool_registry.max_rounds if tool_registry else 5

    # Merge tool schemas
    tools: list[dict] = []
    if has_tools:
        tools = list(config.tools) if config.tools else []
        if server_side:
            existing_names = {
                t.get("function", {}).get("name")
                for t in tools
                if isinstance(t, dict)
            }
            for schema in tool_registry.get_schemas():
                name = schema.get("function", {}).get("name", "")
                if name not in existing_names:
                    tools.append(schema)

    if has_tools:
        mode = ResponseMode.TOOL_CALLING_AUDIO if has_audio else ResponseMode.TOOL_CALLING
    elif has_audio:
        mode = ResponseMode.AUDIO_STREAMING
    else:
        mode = ResponseMode.TEXT_ONLY

    # SLO target: tool responses get more budget
    slo_target = SLO.first_audio_tool_ms if has_tools else SLO.first_audio_ms

    return ResponsePlan(
        mode=mode,
        has_audio=has_audio,
        has_tools=has_tools,
        server_side_tools=server_side,
        tools=tools,
        max_rounds=max_rounds,
        max_first_audio_ms=slo_target,
    )
