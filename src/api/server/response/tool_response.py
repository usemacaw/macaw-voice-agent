"""Tool-calling response path: LLM → tools → inline TTS → loop.

Handles responses that involve function/tool calling, including:
- Multi-round tool execution loops
- Inline TTS streaming during LLM generation
- Fallback to pipelined LLM→TTS for final round
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

from audio.text_cleaning import clean_for_voice
from intelligence.tool_engine import ToolExecutionEngine
from pipeline.sentence_splitter import IncrementalSplitter
from protocol import events
from protocol.metrics import ResponseMetrics
from protocol.models import ContentPart, ConversationItem
from providers.admission import ADMISSION
from server.audio_emitter import AudioEmitter

if TYPE_CHECKING:
    from intelligence.context_builder import ContextBuilder
    from intelligence.response_strategy import ResponsePlan
    from protocol.event_emitter import EventEmitter
    from protocol.models import SessionConfig
    from providers.llm import LLMProvider
    from providers.tts import TTSProvider
    from server.response_runner import ResponseContext
    from tools.registry import ToolRegistry

logger = logging.getLogger("open-voice-api.response-runner.tools")


@dataclass
class ToolResponseContext:
    """All state needed to run a tool-calling response.

    Replaces 17 keyword-only parameters with a single typed object.
    """

    response_id: str
    messages: list[dict]
    system: str
    temperature: float
    max_tokens: int
    plan: ResponsePlan
    session_id: str
    emitter: EventEmitter
    llm: LLMProvider
    tts: TTSProvider
    config: SessionConfig
    tool_registry: ToolRegistry | None
    context_builder: ContextBuilder
    ctx: ResponseContext
    metrics: ResponseMetrics
    on_first_audio: Callable[[], None]
    capture_llm_timing: Callable[[int], None]
    run_audio_response: Callable


async def run_with_tools(tc: ToolResponseContext) -> None:
    """Run response with function calling support."""
    # Unpack frequently used fields for readability
    response_id = tc.response_id
    messages = tc.messages
    plan = tc.plan
    sid = tc.session_id
    emitter = tc.emitter
    metrics = tc.metrics

    await emitter.emit(events.response_created("", response_id))

    response_start = time.perf_counter()
    output_index = 0

    tool_engine = None
    if tc.tool_registry:
        tool_engine = ToolExecutionEngine(
            session_id=sid,
            emitter=emitter,
            tts=tc.tts,
            config=tc.config,
            tool_registry=tc.tool_registry,
        )

    ctx = tc.ctx
    tools_used = 0
    tool_round = 0
    for tool_round in range(plan.max_rounds + 1):
        allow_tools = plan.tools if tools_used < plan.max_rounds else None
        round_max_tokens = tc.max_tokens if allow_tools else min(tc.max_tokens, 40)

        # Final round (no tools) + audio: use pipelined LLM->TTS
        use_pipelined = not allow_tools or (tool_round == 0 and not plan.tools)
        if use_pipelined and plan.has_audio:
            logger.info(
                f"[{sid[:8]}] LLM call round {tool_round}: "
                f"msgs={len(messages)}, tools=no (pipelined)"
            )
            await _emit_tool_response_audio_streamed(
                response_id=response_id,
                output_index=output_index,
                messages=messages,
                system=tc.system,
                temperature=tc.temperature,
                max_tokens=round_max_tokens,
                emitter=emitter,
                config=tc.config,
                ctx=ctx,
                on_first_audio=tc.on_first_audio,
                metrics=metrics,
                run_audio_response=tc.run_audio_response,
            )
            break

        logger.info(
            f"[{sid[:8]}] LLM call round {tool_round}: "
            f"msgs={len(messages)}, tools={'yes' if allow_tools else 'no'}"
        )

        # Stream LLM with optional inline TTS for text deltas
        tts_task = None
        try:
            collected_text, collected_tool_calls, tts_task, tts_queue = (
                await _stream_llm_with_inline_tts(
                    response_id=response_id,
                    output_index=output_index,
                    plan=plan,
                    messages=messages,
                    system=tc.system,
                    temperature=tc.temperature,
                    max_tokens=round_max_tokens,
                    allow_tools=allow_tools,
                    llm=tc.llm,
                    tts=tc.tts,
                    config=tc.config,
                    emitter=emitter,
                    ctx=ctx,
                    on_first_audio=tc.on_first_audio,
                )
            )
        except asyncio.CancelledError:
            if tts_task and not tts_task.done():
                tts_task.cancel()
                try:
                    await tts_task
                except (asyncio.CancelledError, Exception):
                    pass
            raise
        saw_tool_call = bool(collected_tool_calls)

        # Wait for inline TTS to finish
        if tts_task and not saw_tool_call:
            await tts_task
            tc.capture_llm_timing(tool_round)
            break  # Response done via inline TTS
        elif tts_task:
            if not tts_task.done():
                await tts_task  # Already signaled None
            output_index += 1  # TTS item already emitted

        tc.capture_llm_timing(tool_round)

        # No tool calls: emit fallback response if inline TTS didn't run
        collected_text = clean_for_voice(collected_text)
        if not collected_tool_calls:
            if collected_text:
                await _emit_fallback_response(
                    response_id=response_id,
                    output_index=output_index,
                    text=collected_text,
                    has_audio=plan.has_audio,
                    sid=sid,
                    emitter=emitter,
                    tts=tc.tts,
                    config=tc.config,
                    ctx=ctx,
                    on_first_audio=tc.on_first_audio,
                )
            break

        # Execute tool calls
        logger.info(
            f"[{sid[:8]}] Tool round {tool_round}: "
            f"{len(collected_tool_calls)} tool call(s): "
            f"{[call['name'] for call in collected_tool_calls]}"
        )

        if plan.server_side_tools and tool_engine:
            for call in collected_tool_calls:
                metrics.tools_used.append(call["name"])

            result = await tool_engine.execute_server_side(
                response_id, output_index, collected_tool_calls,
                plan.has_audio, ctx.state_lock, ctx.append_item,
            )
            output_index += result.output_index_delta
            tools_used += 1

            async with ctx.state_lock:
                messages = tc.context_builder.rebuild_after_tool_round(ctx.items)

            if not result.all_tools_ok:
                logger.info(
                    f"[{sid[:8]}] Tool error detected, "
                    "forcing text response on next call"
                )
                tools_used = plan.max_rounds
        else:
            if tool_engine:
                await tool_engine.emit_tool_calls_for_client(
                    response_id, output_index, collected_tool_calls,
                    ctx.state_lock, ctx.append_item,
                )
            break
    else:
        logger.warning(
            f"[{sid[:8]}] Tool execution exceeded max rounds ({plan.max_rounds})"
        )

    response_ms = (time.perf_counter() - response_start) * 1000
    logger.info(
        f"[{sid[:8]}] Tool response completed: {response_ms:.0f}ms, "
        f"rounds={tool_round + 1}"
    )

    # Populate tool-specific metrics; response.done and macaw.metrics
    # are emitted by ResponseRunner (single lifecycle owner).
    metrics.tool_rounds = tool_round + 1
    if tool_engine:
        metrics.tool_timings = tool_engine.tool_timings


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


async def _stream_llm_with_inline_tts(
    *,
    response_id: str,
    output_index: int,
    plan: ResponsePlan,
    messages: list[dict],
    system: str,
    temperature: float,
    max_tokens: int,
    allow_tools: list[dict] | None,
    llm: LLMProvider,
    tts: TTSProvider,
    config: SessionConfig,
    emitter: EventEmitter,
    ctx: ResponseContext,
    on_first_audio: Callable[[], None],
) -> tuple[str, list[dict], asyncio.Task | None, asyncio.Queue | None]:
    """Stream LLM events, feed text to inline TTS, collect tool calls."""
    from config import STREAMING

    collected_text = ""
    collected_tool_calls: list[dict] = []
    saw_tool_call = False

    # Tool call accumulation state
    current_tool_call_id = ""
    current_tool_name = ""
    tool_arguments_buffer = ""
    in_tool_call = False

    # Inline TTS setup
    tts_queue: asyncio.Queue | None = None
    tts_task: asyncio.Task | None = None

    if plan.has_audio and tts.supports_streaming:
        tts_queue = asyncio.Queue(maxsize=10)
        audio_emitter = AudioEmitter(
            emitter=emitter,
            tts=tts,
            output_audio_format=config.output_audio_format,
            on_first_audio=on_first_audio,
        )
        tts_task = asyncio.create_task(
            audio_emitter.emit_from_queue(
                tts_queue, response_id, output_index,
                ctx.state_lock, ctx.append_item,
            )
        )

    splitter = IncrementalSplitter(min_eager_chars=STREAMING.min_eager_chars)

    async with ADMISSION.llm.acquire():
        async for event in llm.generate_stream_with_tools(
            messages, system=system,
            tools=allow_tools or None,
            temperature=temperature,
            max_tokens=max_tokens,
        ):
            if event.type == "text_delta":
                collected_text += event.text
                if tts_queue and not saw_tool_call:
                    for sentence in splitter.feed(event.text):
                        clean = clean_for_voice(sentence)
                        if clean:
                            await tts_queue.put(clean)

            elif event.type == "tool_call_start":
                current_tool_call_id = event.tool_call_id
                current_tool_name = event.tool_name
                tool_arguments_buffer = ""
                in_tool_call = True
                saw_tool_call = True
                if tts_queue:
                    await tts_queue.put(None)
            elif event.type == "tool_call_delta" and in_tool_call:
                tool_arguments_buffer += event.tool_arguments_delta
            elif event.type == "tool_call_end" and in_tool_call:
                collected_tool_calls.append({
                    "id": current_tool_call_id,
                    "name": current_tool_name,
                    "arguments": tool_arguments_buffer,
                })
                in_tool_call = False

    # Flush remaining text to TTS
    if tts_queue and not saw_tool_call:
        remaining = splitter.flush()
        if remaining:
            clean = clean_for_voice(remaining)
            if clean:
                await tts_queue.put(clean)
        await tts_queue.put(None)

    return collected_text, collected_tool_calls, tts_task, tts_queue


async def _emit_tool_response_audio_streamed(
    *,
    response_id: str,
    output_index: int,
    messages: list[dict],
    system: str,
    temperature: float,
    max_tokens: int,
    emitter: EventEmitter,
    config: SessionConfig,
    ctx: ResponseContext,
    on_first_audio: Callable[[], None],
    metrics: ResponseMetrics,
    run_audio_response: Callable,
) -> None:
    """Final tool round: pipelined LLM->TTS via SentencePipeline."""
    item_id = f"item_{uuid.uuid4().hex[:24]}"
    content_index = 0

    assistant_item = ConversationItem(
        id=item_id,
        type="message",
        role="assistant",
        status="in_progress",
        content=[ContentPart(type="audio", audio="", transcript="")],
    )
    await emitter.emit(
        events.response_output_item_added(
            "", response_id, output_index, assistant_item
        )
    )
    async with ctx.state_lock:
        ctx.append_item(assistant_item)

    full_transcript = await run_audio_response(
        response_id=response_id,
        item_id=item_id,
        output_index=output_index,
        content_index=content_index,
        assistant_item=assistant_item,
        messages=messages,
        system=system,
        temperature=temperature,
        max_tokens=max_tokens,
        tools=None,
        config=config,
        emitter=emitter,
        on_first_audio=on_first_audio,
        metrics=metrics,
    )

    await emitter.emit(
        events.response_content_part_done(
            "", response_id, item_id, output_index, content_index,
            assistant_item.content[content_index],
        )
    )
    assistant_item.status = "completed"
    await emitter.emit(
        events.response_output_item_done(
            "", response_id, output_index, assistant_item
        )
    )


async def _emit_fallback_response(
    *,
    response_id: str,
    output_index: int,
    text: str,
    has_audio: bool,
    sid: str,
    emitter: EventEmitter,
    tts: TTSProvider,
    config: SessionConfig,
    ctx: ResponseContext,
    on_first_audio: Callable[[], None],
) -> None:
    """Emit response when inline TTS didn't run (fallback path)."""
    if has_audio:
        audio_emitter = AudioEmitter(
            emitter=emitter,
            tts=tts,
            output_audio_format=config.output_audio_format,
            on_first_audio=on_first_audio,
        )
        logger.info(
            f"[{sid[:8]}] TTS streaming synth: {text[:100]!r}"
        )
        await audio_emitter.emit_from_text(
            text, response_id, output_index,
            ctx.state_lock, ctx.append_item,
        )
    else:
        await _emit_text_item(
            response_id=response_id,
            output_index=output_index,
            text=text,
            emitter=emitter,
            ctx=ctx,
        )


async def _emit_text_item(
    *,
    response_id: str,
    output_index: int,
    text: str,
    emitter: EventEmitter,
    ctx: ResponseContext,
) -> None:
    """Emit collected text as a text response item."""
    item_id = f"item_{uuid.uuid4().hex[:24]}"
    text_item = ConversationItem(
        id=item_id,
        type="message",
        role="assistant",
        status="completed",
        content=[ContentPart(type="text", text=text)],
    )
    await emitter.emit(
        events.response_output_item_added(
            "", response_id, output_index, text_item
        )
    )
    async with ctx.state_lock:
        ctx.append_item(text_item)

    await emitter.emit(
        events.response_text_done(
            "", response_id, item_id, output_index, 0, text
        )
    )
    await emitter.emit(
        events.response_output_item_done(
            "", response_id, output_index, text_item
        )
    )
