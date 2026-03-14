"""
ToolExecutionEngine — server-side tool call detection and execution.

Extracted from ResponseRunner to isolate tool-calling logic.
Handles: LLM stream consumption, tool call parsing, server-side
execution with timeout, filler audio, function item creation,
and multi-round orchestration.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from protocol import events
from protocol.models import ContentPart, ConversationItem
from server.filler import build_dynamic_filler, send_filler_audio

if TYPE_CHECKING:
    from protocol.event_emitter import EventEmitter
    from protocol.models import SessionConfig
    from providers.llm import LLMProvider, LLMStreamEvent
    from tools.registry import ToolRegistry


logger = logging.getLogger("open-voice-api.tool-engine")


@dataclass
class ToolRoundResult:
    """Result of a single LLM + tool execution round."""

    collected_text: str = ""
    tool_calls: list[dict] = field(default_factory=list)
    all_tools_ok: bool = True
    output_index_delta: int = 0


class ToolExecutionEngine:
    """Executes server-side tools with filler audio and conversation updates.

    Responsibilities:
    - Parse tool calls from LLM stream events
    - Execute tools with timeout via ToolRegistry
    - Send filler audio during execution
    - Create function_call and function_call_output items
    - Track per-tool timing metrics
    """

    def __init__(
        self,
        session_id: str,
        emitter: EventEmitter,
        tts: object,  # TTSProvider
        config: SessionConfig,
        tool_registry: ToolRegistry,
    ):
        self._sid = session_id
        self._emitter = emitter
        self._tts = tts
        self._config = config
        self._registry = tool_registry
        self._tool_timings: list[dict] = []

    @property
    def tool_timings(self) -> list[dict]:
        return self._tool_timings

    def collect_tool_calls_from_stream(
        self, stream_events: list[LLMStreamEvent],
    ) -> tuple[str, list[dict]]:
        """Parse collected LLM stream events into text and tool calls.

        Args:
            stream_events: List of LLMStreamEvent from generate_stream_with_tools.

        Returns:
            (collected_text, tool_calls) where tool_calls is a list of
            {"id": str, "name": str, "arguments": str}.
        """
        collected_text = ""
        tool_calls: list[dict] = []
        current_id = ""
        current_name = ""
        args_buffer = ""
        in_tool = False

        for event in stream_events:
            if event.type == "text_delta":
                collected_text += event.text
            elif event.type == "tool_call_start":
                current_id = event.tool_call_id
                current_name = event.tool_name
                args_buffer = ""
                in_tool = True
            elif event.type == "tool_call_delta" and in_tool:
                args_buffer += event.tool_arguments_delta
            elif event.type == "tool_call_end" and in_tool:
                tool_calls.append({
                    "id": current_id,
                    "name": current_name,
                    "arguments": args_buffer,
                })
                in_tool = False

        return collected_text, tool_calls

    async def execute_server_side(
        self,
        response_id: str,
        output_index: int,
        tool_calls: list[dict],
        has_audio: bool,
        ctx_lock: object,  # asyncio.Lock
        append_item: object,  # Callable
    ) -> ToolRoundResult:
        """Execute tool calls server-side with filler TTS.

        Args:
            response_id: Current response ID.
            output_index: Starting output index for items.
            tool_calls: Parsed tool calls from LLM.
            has_audio: Whether to send filler audio.
            ctx_lock: State lock for appending items.
            append_item: Callback to append items under lock.

        Returns:
            ToolRoundResult with execution outcome.
        """
        result = ToolRoundResult()

        # Send filler audio for the first tool call
        if has_audio and tool_calls:
            first_tool = tool_calls[0]
            filler = build_dynamic_filler(first_tool["name"], first_tool["arguments"])
            await send_filler_audio(
                self._sid, self._tts, self._emitter, self._config,
                response_id, output_index, filler,
            )

        for tc in tool_calls:
            tc_id = tc["id"] or f"call_{uuid.uuid4().hex[:12]}"
            tc_name = tc["name"]
            tc_args = tc["arguments"]

            # Create function_call item
            fc_item_id = f"item_{uuid.uuid4().hex[:24]}"
            fc_item = ConversationItem(
                id=fc_item_id,
                type="function_call",
                status="completed",
                call_id=tc_id,
                name=tc_name,
                arguments=tc_args,
            )
            await self._emitter.emit(
                events.response_output_item_added(
                    "", response_id, output_index, fc_item
                )
            )
            async with ctx_lock:
                append_item(fc_item)

            await self._emitter.emit(
                events.response_function_call_arguments_done(
                    "", response_id, fc_item_id, output_index, tc_id, tc_args,
                )
            )
            await self._emitter.emit(
                events.response_output_item_done(
                    "", response_id, output_index, fc_item
                )
            )

            # Execute tool with timing
            tool_t0 = time.perf_counter()
            result_json = await self._registry.execute(tc_name, tc_args)
            tool_exec_ms = (time.perf_counter() - tool_t0) * 1000
            logger.info(
                f"[{self._sid[:8]}] Tool '{tc_name}' result ({tool_exec_ms:.0f}ms): "
                f"{result_json[:200]}"
            )

            # Check for errors
            tool_ok = True
            try:
                result_data = json.loads(result_json)
                if isinstance(result_data, dict) and "error" in result_data:
                    result.all_tools_ok = False
                    tool_ok = False
            except (json.JSONDecodeError, TypeError):
                pass
            self._tool_timings.append({
                "name": tc_name,
                "exec_ms": round(tool_exec_ms, 1),
                "ok": tool_ok,
            })

            # Create function_call_output item
            fco_item_id = f"item_{uuid.uuid4().hex[:24]}"
            fco_item = ConversationItem(
                id=fco_item_id,
                type="function_call_output",
                status="completed",
                call_id=tc_id,
                output=result_json,
            )
            async with ctx_lock:
                append_item(fco_item)
            await self._emitter.emit(
                events.conversation_item_created("", fc_item_id, fco_item)
            )

            output_index += 1
            result.output_index_delta += 1

        return result

    async def emit_tool_calls_for_client(
        self,
        response_id: str,
        output_index: int,
        tool_calls: list[dict],
        ctx_lock: object,
        append_item: object,
    ) -> None:
        """Emit tool call events for client-side execution (fallback mode)."""
        for tc in tool_calls:
            tc_id = tc["id"] or f"call_{uuid.uuid4().hex[:12]}"
            fc_item_id = f"item_{uuid.uuid4().hex[:24]}"
            fc_item = ConversationItem(
                id=fc_item_id,
                type="function_call",
                status="completed",
                call_id=tc_id,
                name=tc["name"],
                arguments=tc["arguments"],
            )
            await self._emitter.emit(
                events.response_output_item_added(
                    "", response_id, output_index, fc_item
                )
            )
            async with ctx_lock:
                append_item(fc_item)

            await self._emitter.emit(
                events.response_function_call_arguments_done(
                    "", response_id, fc_item_id, output_index,
                    tc_id, tc["arguments"],
                )
            )
            await self._emitter.emit(
                events.response_output_item_done(
                    "", response_id, output_index, fc_item
                )
            )
            output_index += 1
