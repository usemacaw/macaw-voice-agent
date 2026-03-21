"""
ResponseOrchestrator (ResponseRunner) — delegates response execution to strategies.

Orchestrates the response cycle: selects strategy, builds context,
delegates tool execution and audio synthesis to specialized modules.

Kept as ResponseRunner for backward compatibility with existing imports.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

from audio.codec import encode_audio_for_client
from audio.text_cleaning import clean_for_voice
from intelligence.context_builder import ContextBuilder
from intelligence.response_strategy import select_strategy
from intelligence.tool_engine import ToolExecutionEngine
from pipeline.sentence_pipeline import SentencePipeline
from pipeline.sentence_splitter import IncrementalSplitter
from protocol import events
from protocol.event_emitter import SlowClientError
from protocol.models import ContentPart, ConversationItem
from providers.admission import ADMISSION
from server.audio_emitter import AudioEmitter

if TYPE_CHECKING:
    from protocol.event_emitter import EventEmitter
    from protocol.models import SessionConfig
    from providers.llm import LLMProvider
    from providers.tts import TTSProvider
    from tools.registry import ToolRegistry

logger = logging.getLogger("open-voice-api.response-runner")

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

        tool_engine = None
        if self._tool_registry:
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

            # Stream LLM with optional inline TTS for text deltas
            collected_text, collected_tool_calls, tts_task, tts_queue = (
                await self._stream_llm_with_inline_tts(
                    response_id, output_index, plan,
                    messages, system, temperature, round_max_tokens,
                    allow_tools,
                )
            )
            saw_tool_call = bool(collected_tool_calls)

            # Wait for inline TTS to finish
            if tts_task and not saw_tool_call:
                await tts_task
                self._capture_llm_timing(tool_round)
                break  # Response done via inline TTS
            elif tts_task:
                if not tts_task.done():
                    await tts_task  # Already signaled None
                output_index += 1  # TTS item already emitted

            self._capture_llm_timing(tool_round)

            # No tool calls: emit fallback response if inline TTS didn't run
            collected_text = clean_for_voice(collected_text)
            if not collected_tool_calls:
                if collected_text:
                    await self._emit_fallback_response(
                        response_id, output_index, response_start,
                        collected_text, plan.has_audio,
                    )
                break

            # Execute tool calls
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
    # LLM streaming with inline TTS (extracted from _run_with_tools)
    # ------------------------------------------------------------------

    async def _stream_llm_with_inline_tts(
        self,
        response_id: str,
        output_index: int,
        plan,
        messages: list[dict],
        system: str,
        temperature: float,
        max_tokens: int,
        allow_tools: list[dict] | None,
    ) -> tuple[str, list[dict], asyncio.Task | None, asyncio.Queue | None]:
        """Stream LLM events, feed text to inline TTS, collect tool calls.

        Returns (collected_text, collected_tool_calls, tts_task, tts_queue).
        """
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

        if plan.has_audio and self._tts.supports_streaming:
            tts_queue = asyncio.Queue(maxsize=10)
            audio_emitter = AudioEmitter(
                emitter=self._emitter,
                tts=self._tts,
                output_audio_format=self._config.output_audio_format,
                on_first_audio=self._record_e2e_latency,
            )
            tts_task = asyncio.create_task(
                audio_emitter.emit_from_queue(
                    tts_queue, response_id, output_index,
                    self._ctx.state_lock, self._ctx.append_item,
                )
            )

        splitter = IncrementalSplitter(min_eager_chars=STREAMING.min_eager_chars)

        async with ADMISSION.llm.acquire():
            async for event in self._llm.generate_stream_with_tools(
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

    def _capture_llm_timing(self, tool_round: int) -> None:
        """Capture LLM TTFT and total streaming time."""
        if tool_round == 0:
            self.metrics["llm_ttft_ms"] = round(self._llm.last_ttft_ms, 1)
        self.metrics["llm_total_ms"] = round(self._llm.last_stream_total_ms, 1)

    async def _emit_fallback_response(
        self,
        response_id: str,
        output_index: int,
        response_start: float,
        text: str,
        has_audio: bool,
    ) -> None:
        """Emit response when inline TTS didn't run (fallback path)."""
        if has_audio:
            audio_emitter = AudioEmitter(
                emitter=self._emitter,
                tts=self._tts,
                output_audio_format=self._config.output_audio_format,
                on_first_audio=self._record_e2e_latency,
            )
            logger.info(
                f"[{self._sid[:8]}] TTS streaming synth: {text[:100]!r}"
            )
            await audio_emitter.emit_from_text(
                text, response_id, output_index,
                self._ctx.state_lock, self._ctx.append_item,
            )
        else:
            await self._emit_tool_response_text(
                response_id, output_index, text,
            )

    # ------------------------------------------------------------------
    # Response emission helpers
    # ------------------------------------------------------------------

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
