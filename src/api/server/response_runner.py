"""
ResponseOrchestrator (ResponseRunner) — delegates response execution to strategies.

Orchestrates the response cycle: selects strategy, builds context,
delegates tool execution and audio synthesis to specialized modules.

Kept as ResponseRunner for backward compatibility with existing imports.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

from audio.codec import encode_audio_for_client
from intelligence.context_builder import ContextBuilder
from intelligence.response_strategy import ResponseMode, select_strategy
from intelligence.tool_engine import ToolExecutionEngine
from pipeline.sentence_pipeline import SentencePipeline
from protocol import events
from protocol.event_emitter import SlowClientError
from protocol.models import ContentPart, ConversationItem
from providers.admission import ADMISSION

if TYPE_CHECKING:
    from protocol.event_emitter import EventEmitter
    from protocol.models import SessionConfig
    from providers.llm import LLMProvider
    from providers.tts import TTSProvider
    from tools.registry import ToolRegistry

logger = logging.getLogger("open-voice-api.response-runner")

# ---------------------------------------------------------------------------
# Text cleaning
# ---------------------------------------------------------------------------

_THINK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)

_EMOJI_RE = re.compile(
    "["
    "\U0001f600-\U0001f64f"
    "\U0001f300-\U0001f5ff"
    "\U0001f680-\U0001f6ff"
    "\U0001f1e0-\U0001f1ff"
    "\U0001f900-\U0001f9ff"
    "\U0001fa00-\U0001fa6f"
    "\U0001fa70-\U0001faff"
    "\U00002702-\U000027b0"
    "\U000024c2-\U0001f251"
    "\U0000fe0f"
    "\U0000200d"
    "]+",
)


def _clean_for_voice(text: str) -> str:
    """Strip thinking blocks and emojis from LLM output."""
    text = _THINK_RE.sub("", text)
    text = _EMOJI_RE.sub("", text)
    return text.strip()


# ---------------------------------------------------------------------------
# Response context (shared session state)
# ---------------------------------------------------------------------------


@dataclass
class ResponseContext:
    """Mutable session state passed to the response runner."""

    items: Sequence[ConversationItem]
    state_lock: asyncio.Lock
    append_item: Callable[[ConversationItem], None]  # must be called under state_lock
    speech_stopped_at: float | None
    turn_count: int
    session_start: float
    barge_in_count: int


# ---------------------------------------------------------------------------
# ResponseRunner (orchestrator)
# ---------------------------------------------------------------------------


class ResponseRunner:
    """Orchestrates a complete response: selects strategy, delegates execution.

    Created per-response. Holds no state between responses.
    Uses ContextBuilder for LLM context, ToolExecutionEngine for tools,
    and ResponsePlan for strategy selection.
    """

    def __init__(
        self,
        session_id: str,
        emitter: EventEmitter,
        llm: LLMProvider,
        tts: TTSProvider,
        config: SessionConfig,
        tool_registry: ToolRegistry | None = None,
    ):
        self._sid = session_id
        self._emitter = emitter
        self._llm = llm
        self._tts = tts
        self._config = config
        self._tool_registry = tool_registry
        self._context_builder = ContextBuilder(config)

        # Populated during run(), read by caller after completion
        self.metrics: dict[str, object] = {}

        # Set by caller, used for E2E latency measurement
        self._speech_stopped_at: float | None = None
        self._slo_target_ms: float = 0.0

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    async def run(
        self,
        response_id: str,
        ctx: ResponseContext,
        prior_metrics: dict[str, object],
    ) -> None:
        """Execute a full response cycle."""
        self._ctx = ctx
        self._speech_stopped_at = ctx.speech_stopped_at

        # Select strategy based on config and tools
        plan = select_strategy(self._config, self._tool_registry)
        self._slo_target_ms = plan.max_first_audio_ms

        item_id = f"item_{uuid.uuid4().hex[:24]}"
        output_index = 0
        content_index = 0
        response_start = time.perf_counter()
        assistant_item = None

        # Initialize per-response metrics
        self.metrics = {
            "response_id": response_id,
            "turn": ctx.turn_count,
            "session_duration_s": round(time.perf_counter() - ctx.session_start, 1),
            "barge_in_count": ctx.barge_in_count,
            "tools_used": [],
            "tool_rounds": 0,
        }
        for key in (
            "asr_ms", "speech_ms", "asr_mode", "input_chars", "speech_rms",
            "vad_silence_wait_ms", "smart_turn_inference_ms", "smart_turn_waits",
        ):
            if key in prior_metrics:
                self.metrics[key] = prior_metrics[key]

        logger.info(
            f"[{self._sid[:8]}] RESPONSE START: "
            f"items={len(ctx.items)}, mode={plan.mode.name}"
        )

        try:
            # Build context via ContextBuilder
            async with ctx.state_lock:
                messages, system, temperature, max_tokens = (
                    self._context_builder.build_for_response(
                        ctx.items, has_tools=plan.has_tools
                    )
                )

            if plan.has_tools:
                await self._run_with_tools(
                    response_id, messages, system, temperature, max_tokens,
                    plan,
                )
            else:
                assistant_item = await self._setup_response(
                    response_id, item_id, output_index, content_index, plan.has_audio
                )

                if plan.has_audio:
                    full_transcript = await self._run_audio_response(
                        response_id, item_id, output_index, content_index,
                        assistant_item, messages, system, temperature, max_tokens,
                    )
                    full_text = ""
                else:
                    full_text = await self._run_text_response(
                        response_id, item_id, output_index, content_index,
                        assistant_item, messages, system, temperature, max_tokens,
                    )
                    full_transcript = ""

                await self._finalize_response(
                    response_id, output_index, content_index,
                    assistant_item, response_start, full_transcript, full_text,
                )

        except asyncio.CancelledError:
            if assistant_item is not None:
                assistant_item.status = "incomplete"
            try:
                await self._emitter.emit(
                    events.response_done("", response_id, status="cancelled")
                )
            except Exception:
                pass
            raise

        except SlowClientError:
            raise

        except Exception as e:
            logger.error(f"[{self._sid[:8]}] Response error: {e}", exc_info=True)
            await self._emitter.emit(
                events.response_done("", response_id, status="failed")
            )

    # ------------------------------------------------------------------
    # Tool calling loop (delegates to ToolExecutionEngine)
    # ------------------------------------------------------------------

    async def _run_with_tools(
        self,
        response_id: str,
        messages: list[dict],
        system: str,
        temperature: float,
        max_tokens: int,
        plan,
    ) -> None:
        """Run response with function calling support."""
        await self._emitter.emit(events.response_created("", response_id))

        response_start = time.perf_counter()
        output_index = 0

        # Create tool engine if server-side tools
        tool_engine = None
        if plan.server_side_tools and self._tool_registry:
            tool_engine = ToolExecutionEngine(
                session_id=self._sid,
                emitter=self._emitter,
                tts=self._tts,
                config=self._config,
                tool_registry=self._tool_registry,
            )

        tools_used = 0
        tool_round = 0
        for tool_round in range(plan.max_rounds + 1):
            allow_tools = plan.tools if tools_used < plan.max_rounds else None
            round_max_tokens = max_tokens if allow_tools else min(max_tokens, 40)

            # Final round (no tools) + audio: use pipelined LLM->TTS
            # Also use pipelined path on round 0 when no tools are registered,
            # to avoid collecting full LLM text before TTS starts (batch anti-pattern).
            use_pipelined = not allow_tools or (tool_round == 0 and not plan.tools)
            if use_pipelined and plan.has_audio:
                logger.info(
                    f"[{self._sid[:8]}] LLM call round {tool_round}: "
                    f"msgs={len(messages)}, tools=no (pipelined)"
                )
                await self._emit_tool_response_audio_streamed(
                    response_id, output_index, response_start,
                    messages, system, temperature, round_max_tokens,
                )
                break

            logger.info(
                f"[{self._sid[:8]}] LLM call round {tool_round}: "
                f"msgs={len(messages)}, tools={'yes' if allow_tools else 'no'}"
            )

            # Collect full LLM stream: text + tool calls
            # While streaming, also send text to TTS via sentence pipeline
            # so audio starts flowing DURING LLM generation (not after).
            collected_text = ""
            collected_tool_calls: list[dict] = []
            current_tool_call_id = ""
            current_tool_name = ""
            tool_arguments_buffer = ""
            in_tool_call = False
            saw_tool_call = False

            # Sentence buffer for inline TTS streaming
            tts_sentence_buffer = ""
            tts_queue: asyncio.Queue | None = None
            tts_task: asyncio.Task | None = None
            first_audio_sent_inline = False

            # Setup inline TTS pipeline if audio mode
            if plan.has_audio and self._tts.supports_streaming:
                tts_queue = asyncio.Queue(maxsize=10)

                async def _inline_tts_worker(q, rid, oidx):
                    """Synthesize sentences from queue as they arrive."""
                    nonlocal first_audio_sent_inline
                    item_id = f"item_{uuid.uuid4().hex[:24]}"
                    cidx = 0
                    assistant_item = ConversationItem(
                        id=item_id, type="message", role="assistant",
                        status="in_progress",
                        content=[ContentPart(type="audio", audio="", transcript="")],
                    )
                    await self._emitter.emit(
                        events.response_output_item_added("", rid, oidx, assistant_item)
                    )
                    async with self._ctx.state_lock:
                        self._ctx.append_item(assistant_item)

                    full_text = ""
                    while True:
                        sentence = await q.get()
                        if sentence is None:
                            break
                        full_text += (" " if full_text else "") + sentence
                        async for audio_chunk in self._tts.synthesize_stream(sentence):
                            if not audio_chunk:
                                continue
                            audio_b64 = encode_audio_for_client(
                                audio_chunk, self._config.output_audio_format
                            )
                            await self._emitter.emit(
                                events.response_audio_delta(
                                    "", rid, item_id, oidx, cidx, audio_b64
                                )
                            )
                            if not first_audio_sent_inline:
                                first_audio_sent_inline = True
                                self._record_e2e_latency()
                        await self._emitter.emit(
                            events.response_audio_transcript_delta(
                                "", rid, item_id, oidx, cidx, sentence
                            )
                        )

                    # Finalize
                    assistant_item.content[cidx] = ContentPart(
                        type="audio", transcript=full_text
                    )
                    await self._emitter.emit(
                        events.response_audio_done("", rid, item_id, oidx, cidx)
                    )
                    await self._emitter.emit(
                        events.response_audio_transcript_done(
                            "", rid, item_id, oidx, cidx, full_text
                        )
                    )
                    await self._emitter.emit(
                        events.response_content_part_done(
                            "", rid, item_id, oidx, cidx,
                            assistant_item.content[cidx],
                        )
                    )
                    assistant_item.status = "completed"
                    await self._emitter.emit(
                        events.response_output_item_done("", rid, oidx, assistant_item)
                    )
                    return item_id, full_text

                tts_task = asyncio.create_task(
                    _inline_tts_worker(tts_queue, response_id, output_index)
                )

            # Sentence splitting regex (inline, matches sentence_splitter.py)
            import re
            _sent_end = re.compile(r'[.!?]\s*$')
            _clause_break = re.compile(r'[,;:\u2014\u2013]\s*$')
            from config import STREAMING
            _min_eager = STREAMING.min_eager_chars
            _first_sentence_sent = False

            async with ADMISSION.llm.acquire():
                async for event in self._llm.generate_stream_with_tools(
                    messages, system=system,
                    tools=allow_tools or None,
                    temperature=temperature,
                    max_tokens=round_max_tokens,
                ):
                    if event.type == "text_delta":
                        collected_text += event.text
                        # Feed sentences to TTS inline (while LLM still generating)
                        if tts_queue and not saw_tool_call:
                            tts_sentence_buffer += event.text
                            # Check for sentence boundary
                            sent_match = _sent_end.search(tts_sentence_buffer)
                            if sent_match:
                                sentence = _clean_for_voice(tts_sentence_buffer[:sent_match.end()].strip())
                                tts_sentence_buffer = tts_sentence_buffer[sent_match.end():]
                                if sentence:
                                    await tts_queue.put(sentence)
                                    _first_sentence_sent = True
                            elif not _first_sentence_sent and len(tts_sentence_buffer) >= _min_eager:
                                clause_match = _clause_break.search(tts_sentence_buffer)
                                if clause_match:
                                    sentence = _clean_for_voice(tts_sentence_buffer[:clause_match.end()].strip())
                                    tts_sentence_buffer = tts_sentence_buffer[clause_match.end():]
                                    if sentence:
                                        await tts_queue.put(sentence)
                                        _first_sentence_sent = True

                    elif event.type == "tool_call_start":
                        current_tool_call_id = event.tool_call_id
                        current_tool_name = event.tool_name
                        tool_arguments_buffer = ""
                        in_tool_call = True
                        saw_tool_call = True
                        # Cancel inline TTS — tool calls need different handling
                        if tts_queue:
                            await tts_queue.put(None)  # Signal end
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
                remaining = _clean_for_voice(tts_sentence_buffer.strip())
                if remaining:
                    await tts_queue.put(remaining)
                await tts_queue.put(None)  # Signal end

            # Wait for TTS to finish
            if tts_task and not saw_tool_call:
                await tts_task
                # Capture LLM timing
                if tool_round == 0:
                    self.metrics["llm_ttft_ms"] = round(self._llm.last_ttft_ms, 1)
                self.metrics["llm_total_ms"] = round(self._llm.last_stream_total_ms, 1)
                break  # Response done via inline TTS
            else:
                # Cancel TTS task if tool call happened
                if tts_task:
                    if not tts_task.done():
                        await tts_task  # Already signaled None
                    output_index += 1  # TTS item already emitted

            # Capture LLM timing
            if tool_round == 0:
                self.metrics["llm_ttft_ms"] = round(self._llm.last_ttft_ms, 1)
            self.metrics["llm_total_ms"] = round(self._llm.last_stream_total_ms, 1)

            # No tool calls: response already emitted via inline TTS above
            collected_text = _clean_for_voice(collected_text)
            if not collected_tool_calls:
                if not first_audio_sent_inline:
                    # Fallback: TTS didn't run (e.g. audio disabled)
                    if collected_text:
                        if plan.has_audio:
                            await self._emit_tool_response_audio(
                                response_id, output_index, response_start,
                                collected_text,
                            )
                        else:
                            await self._emit_tool_response_text(
                                response_id, output_index, collected_text,
                            )
                break

            # Has tool calls
            logger.info(
                f"[{self._sid[:8]}] Tool round {tool_round}: "
                f"{len(collected_tool_calls)} tool call(s): "
                f"{[tc['name'] for tc in collected_tool_calls]}"
            )

            if plan.server_side_tools and tool_engine:
                for tc in collected_tool_calls:
                    self.metrics.setdefault("tools_used", []).append(tc["name"])

                result = await tool_engine.execute_server_side(
                    response_id, output_index, collected_tool_calls,
                    plan.has_audio, self._ctx.state_lock, self._ctx.append_item,
                )
                output_index += result.output_index_delta
                tools_used += 1

                # Rebuild messages with tool results via ContextBuilder
                async with self._ctx.state_lock:
                    messages = self._context_builder.rebuild_after_tool_round(
                        self._ctx.items
                    )

                if not result.all_tools_ok:
                    logger.info(
                        f"[{self._sid[:8]}] Tool error detected, "
                        "forcing text response on next call"
                    )
                    tools_used = plan.max_rounds
            else:
                if tool_engine:
                    await tool_engine.emit_tool_calls_for_client(
                        response_id, output_index, collected_tool_calls,
                        self._ctx.state_lock, self._ctx.append_item,
                    )
                else:
                    await self._emit_tool_calls_for_client_legacy(
                        response_id, output_index, collected_tool_calls,
                    )
                break
        else:
            logger.warning(
                f"[{self._sid[:8]}] Tool execution exceeded max rounds ({plan.max_rounds})"
            )

        response_ms = (time.perf_counter() - response_start) * 1000
        logger.info(
            f"[{self._sid[:8]}] RESPONSE DONE (tools): {response_ms:.0f}ms, "
            f"rounds={tool_round + 1}"
        )
        await self._emitter.emit(
            events.response_done("", response_id, status="completed")
        )

        self.metrics["total_ms"] = round(response_ms, 1)
        self.metrics["tool_rounds"] = tool_round + 1
        self.metrics["backpressure_level"] = self._emitter.pressure_level
        self.metrics["events_dropped"] = self._emitter.total_drops
        if tool_engine:
            self.metrics["tool_timings"] = tool_engine.tool_timings
        await self._emitter.emit(
            events.macaw_metrics(response_id, self.metrics)
        )

    # ------------------------------------------------------------------
    # Response emission helpers
    # ------------------------------------------------------------------

    async def _emit_tool_response_audio(
        self,
        response_id: str,
        output_index: int,
        response_start: float,
        collected_text: str,
    ) -> None:
        """Synthesize already-collected text through TTS (no extra LLM call)."""
        item_id = f"item_{uuid.uuid4().hex[:24]}"
        content_index = 0

        assistant_item = ConversationItem(
            id=item_id,
            type="message",
            role="assistant",
            status="in_progress",
            content=[ContentPart(type="audio", audio="", transcript="")],
        )
        await self._emitter.emit(
            events.response_output_item_added(
                "", response_id, output_index, assistant_item
            )
        )
        async with self._ctx.state_lock:
            self._ctx.append_item(assistant_item)

        full_transcript = collected_text.strip()
        first_audio_sent = False

        logger.info(
            f"[{self._sid[:8]}] TTS streaming synth: {full_transcript[:100]!r}"
        )
        # Always use streaming TTS — emit audio chunks as they're generated
        async for audio_chunk in self._tts.synthesize_stream(full_transcript):
            if not audio_chunk:
                continue
            audio_b64 = encode_audio_for_client(
                audio_chunk, self._config.output_audio_format
            )
            await self._emitter.emit(
                events.response_audio_delta(
                    "", response_id, item_id, output_index, content_index,
                    audio_b64,
                )
            )
            if not first_audio_sent:
                first_audio_sent = True
                self._record_e2e_latency()

        if full_transcript:
            await self._emitter.emit(
                events.response_audio_transcript_delta(
                    "", response_id, item_id, output_index, content_index,
                    full_transcript,
                )
            )

        assistant_item.content[content_index] = ContentPart(
            type="audio", transcript=full_transcript
        )

        await self._emitter.emit(
            events.response_audio_done(
                "", response_id, item_id, output_index, content_index
            )
        )
        await self._emitter.emit(
            events.response_audio_transcript_done(
                "", response_id, item_id, output_index, content_index,
                full_transcript,
            )
        )
        await self._emitter.emit(
            events.response_content_part_done(
                "", response_id, item_id, output_index, content_index,
                assistant_item.content[content_index],
            )
        )
        assistant_item.status = "completed"
        await self._emitter.emit(
            events.response_output_item_done(
                "", response_id, output_index, assistant_item
            )
        )

    async def _emit_tool_response_audio_streamed(
        self,
        response_id: str,
        output_index: int,
        response_start: float,
        messages: list[dict],
        system: str,
        temperature: float,
        max_tokens: int,
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
        await self._emitter.emit(
            events.response_output_item_added(
                "", response_id, output_index, assistant_item
            )
        )
        async with self._ctx.state_lock:
            self._ctx.append_item(assistant_item)

        full_transcript = await self._run_audio_response(
            response_id, item_id, output_index, content_index,
            assistant_item, messages, system, temperature, max_tokens,
            tools=None,
        )

        await self._emitter.emit(
            events.response_content_part_done(
                "", response_id, item_id, output_index, content_index,
                assistant_item.content[content_index],
            )
        )
        assistant_item.status = "completed"
        await self._emitter.emit(
            events.response_output_item_done(
                "", response_id, output_index, assistant_item
            )
        )

    async def _emit_tool_response_text(
        self,
        response_id: str,
        output_index: int,
        text: str,
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
        await self._emitter.emit(
            events.response_output_item_added(
                "", response_id, output_index, text_item
            )
        )
        async with self._ctx.state_lock:
            self._ctx.append_item(text_item)

        await self._emitter.emit(
            events.response_text_done(
                "", response_id, item_id, output_index, 0, text
            )
        )
        await self._emitter.emit(
            events.response_output_item_done(
                "", response_id, output_index, text_item
            )
        )

    async def _emit_tool_calls_for_client_legacy(
        self,
        response_id: str,
        output_index: int,
        tool_calls: list[dict],
    ) -> None:
        """Emit tool call events for client-side execution (no ToolEngine)."""
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
            async with self._ctx.state_lock:
                self._ctx.append_item(fc_item)

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

    # ------------------------------------------------------------------
    # Non-tool response helpers
    # ------------------------------------------------------------------

    async def _setup_response(
        self,
        response_id: str,
        item_id: str,
        output_index: int,
        content_index: int,
        has_audio: bool,
    ) -> ConversationItem:
        """Emit initial response events and create the assistant item."""
        await self._emitter.emit(events.response_created("", response_id))

        assistant_item = ConversationItem(
            id=item_id,
            type="message",
            role="assistant",
            status="in_progress",
        )

        if has_audio:
            assistant_item.content = [
                ContentPart(type="audio", audio="", transcript=""),
            ]
        else:
            assistant_item.content = [ContentPart(type="text", text="")]

        await self._emitter.emit(
            events.response_output_item_added("", response_id, output_index, assistant_item)
        )
        async with self._ctx.state_lock:
            prev_id = self._ctx.items[-1].id if self._ctx.items else ""
            self._ctx.append_item(assistant_item)
        await self._emitter.emit(
            events.conversation_item_created("", prev_id, assistant_item)
        )

        part = assistant_item.content[content_index]
        await self._emitter.emit(
            events.response_content_part_added(
                "", response_id, item_id, output_index, content_index, part
            )
        )

        return assistant_item

    async def _run_audio_response(
        self,
        response_id: str,
        item_id: str,
        output_index: int,
        content_index: int,
        assistant_item: ConversationItem,
        messages: list[dict],
        system: str,
        temperature: float,
        max_tokens: int,
        tools: list[dict] | None = ...,
    ) -> str:
        """Run LLM->TTS pipeline, stream audio events. Returns full transcript."""
        if tools is ...:
            tools = self._config.tools or None

        full_transcript = ""
        first_audio_sent = False

        pipeline = SentencePipeline(self._llm, self._tts)
        async for sentence, audio_chunk, is_new_sentence in pipeline.process_streaming(
            messages,
            system=system,
            tools=tools,
            temperature=temperature,
            max_tokens=max_tokens,
        ):
            audio_b64 = encode_audio_for_client(
                audio_chunk, self._config.output_audio_format
            )
            await self._emitter.emit(
                events.response_audio_delta(
                    "", response_id, item_id, output_index, content_index, audio_b64
                )
            )

            if not first_audio_sent:
                first_audio_sent = True
                self._record_e2e_latency()

            if is_new_sentence and sentence:
                delta = sentence
                if full_transcript:
                    delta = " " + sentence
                await self._emitter.emit(
                    events.response_audio_transcript_delta(
                        "", response_id, item_id, output_index, content_index, delta
                    )
                )
                full_transcript += delta

        # Capture pipeline metrics
        pm = pipeline.metrics
        response_audio_s = (pm.audio_chunks_produced * 0.1) if pm.audio_chunks_produced else 0
        self.metrics.update({
            "llm_ttft_ms": round(pm.llm_ttft_ms, 1),
            "llm_total_ms": round(pm.llm_total_ms, 1),
            "llm_first_sentence_ms": round(pm.first_sentence_latency_ms, 1),
            "pipeline_first_audio_ms": round(pm.first_audio_latency_ms, 1),
            "pipeline_total_ms": round(pm.total_latency_ms, 1),
            "tts_synth_ms": round(pm.tts_synth_ms, 1),
            "tts_wait_ms": round(pm.tts_wait_ms, 1),
            "tts_first_chunk_ms": round(pm.tts_first_chunk_ms, 1),
            "tts_queue_max_depth": pm.tts_queue_max_depth,
            "tts_calls": pm.tts_calls,
            "sentences": pm.sentences_generated,
            "audio_chunks": pm.audio_chunks_produced,
            "output_chars": len(full_transcript),
            "response_audio_ms": round(response_audio_s * 1000, 1),
        })

        assistant_item.content[content_index] = ContentPart(
            type="audio", transcript=full_transcript
        )

        await self._emitter.emit(
            events.response_audio_done("", response_id, item_id, output_index, content_index)
        )
        await self._emitter.emit(
            events.response_audio_transcript_done(
                "", response_id, item_id, output_index, content_index, full_transcript
            )
        )

        return full_transcript

    async def _run_text_response(
        self,
        response_id: str,
        item_id: str,
        output_index: int,
        content_index: int,
        assistant_item: ConversationItem,
        messages: list[dict],
        system: str,
        temperature: float,
        max_tokens: int,
    ) -> str:
        """Run LLM text-only streaming. Returns full text."""
        full_text = ""

        async for chunk in self._llm.generate_stream(
            messages,
            system=system,
            tools=self._config.tools or None,
            temperature=temperature,
            max_tokens=max_tokens,
        ):
            await self._emitter.emit(
                events.response_text_delta(
                    "", response_id, item_id, output_index, content_index, chunk
                )
            )
            full_text += chunk

        assistant_item.content[content_index] = ContentPart(
            type="text", text=full_text
        )
        await self._emitter.emit(
            events.response_text_done(
                "", response_id, item_id, output_index, content_index, full_text
            )
        )

        return full_text

    async def _finalize_response(
        self,
        response_id: str,
        output_index: int,
        content_index: int,
        assistant_item: ConversationItem,
        response_start: float,
        full_transcript: str,
        full_text: str,
    ) -> None:
        """Emit final response lifecycle events."""
        await self._emitter.emit(
            events.response_content_part_done(
                "", response_id, assistant_item.id, output_index, content_index,
                assistant_item.content[content_index],
            )
        )

        assistant_item.status = "completed"
        await self._emitter.emit(
            events.response_output_item_done("", response_id, output_index, assistant_item)
        )

        response_ms = (time.perf_counter() - response_start) * 1000
        logger.info(
            f"[{self._sid[:8]}] RESPONSE DONE: {response_ms:.0f}ms, "
            f"transcript=\"{full_transcript[:60] or full_text[:60]}\""
        )
        await self._emitter.emit(
            events.response_done(
                "", response_id, status="completed",
                output=[assistant_item.to_dict()],
            )
        )

        # Enrich with LLM timing (for non-tool path)
        if "llm_ttft_ms" not in self.metrics:
            self.metrics["llm_ttft_ms"] = round(self._llm.last_ttft_ms, 1)
            self.metrics["llm_total_ms"] = round(self._llm.last_stream_total_ms, 1)
        if "output_chars" not in self.metrics:
            self.metrics["output_chars"] = len(full_transcript or full_text)

        self.metrics["total_ms"] = round(response_ms, 1)
        self.metrics["backpressure_level"] = self._emitter.pressure_level
        self.metrics["events_dropped"] = self._emitter.total_drops
        await self._emitter.emit(
            events.macaw_metrics(response_id, self.metrics)
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _record_e2e_latency(self) -> None:
        """Record E2E latency on first audio chunk sent. Check SLO compliance."""
        if self._speech_stopped_at is not None:
            e2e_ms = (time.perf_counter() - self._speech_stopped_at) * 1000
            self.metrics["e2e_ms"] = round(e2e_ms, 1)

            # SLO compliance
            if self._slo_target_ms > 0:
                slo_met = e2e_ms <= self._slo_target_ms
                self.metrics["slo_target_ms"] = self._slo_target_ms
                self.metrics["slo_met"] = slo_met
                if not slo_met:
                    logger.warning(
                        f"[{self._sid[:8]}] SLO BREACH: e2e={e2e_ms:.0f}ms > "
                        f"target={self._slo_target_ms:.0f}ms"
                    )

            logger.info(
                f"[{self._sid[:8]}] E2E LATENCY: {e2e_ms:.0f}ms "
                f"(speech_stopped -> first_audio_sent)"
            )
            self._speech_stopped_at = None
