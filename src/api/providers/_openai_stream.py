"""
Shared OpenAI-compatible streaming parser for tool calls.

Used by both OpenAILLM and VLLMProvider to avoid duplicating the
delta-accumulation logic for the OpenAI streaming protocol.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator, Callable

from providers.llm import LLMStreamEvent


async def parse_openai_tool_stream(
    stream,
    on_first_token: Callable[[], None] | None = None,
) -> AsyncGenerator[LLMStreamEvent, None]:
    """Parse an OpenAI-compatible streaming response into LLMStreamEvents.

    Args:
        stream: Async iterable of chat completion chunks.
        on_first_token: Optional callback invoked once on the first chunk
                        with a delta (useful for TTFT measurement).

    Yields:
        LLMStreamEvent for text deltas and tool call lifecycle.
    """
    first_token = False
    active_tool_calls: dict[int, dict] = {}

    async for chunk in stream:
        delta = chunk.choices[0].delta if chunk.choices else None
        if not delta:
            continue

        if not first_token:
            first_token = True
            if on_first_token:
                on_first_token()

        if delta.content:
            yield LLMStreamEvent(type="text_delta", text=delta.content)

        if delta.tool_calls:
            for tc in delta.tool_calls:
                idx = tc.index
                if idx not in active_tool_calls:
                    tc_id = tc.id or ""
                    tc_name = (
                        tc.function.name
                        if tc.function and tc.function.name
                        else ""
                    )
                    active_tool_calls[idx] = {"id": tc_id, "name": tc_name}
                    yield LLMStreamEvent(
                        type="tool_call_start",
                        tool_call_id=tc_id,
                        tool_name=tc_name,
                    )
                if tc.function and tc.function.arguments:
                    info = active_tool_calls[idx]
                    yield LLMStreamEvent(
                        type="tool_call_delta",
                        tool_call_id=info["id"],
                        tool_name=info["name"],
                        tool_arguments_delta=tc.function.arguments,
                    )

    for info in active_tool_calls.values():
        yield LLMStreamEvent(
            type="tool_call_end",
            tool_call_id=info["id"],
            tool_name=info["name"],
        )
