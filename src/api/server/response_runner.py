"""
ResponseOrchestrator (ResponseRunner) — delegates response execution to strategies.

Thin orchestrator that:
- Selects strategy via ResponsePlan
- Delegates to audio_response, text_response, or tool_response
- Manages response lifecycle events (created, done)
- Captures metrics and SLO compliance

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

from intelligence.context_builder import ContextBuilder
from intelligence.response_strategy import select_strategy
from protocol import events
from protocol.event_emitter import SlowClientError
from protocol.models import ContentPart, ConversationItem
from server.response.audio_response import run_audio_response
from server.response.text_response import run_text_response
from server.response.tool_response import run_with_tools

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
    Uses ContextBuilder for LLM context, and delegates to specialized modules:
    - audio_response: LLM → SentencePipeline → audio events
    - text_response: LLM → text deltas
    - tool_response: LLM → tools → inline TTS → loop
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
                await run_with_tools(
                    response_id=response_id,
                    messages=messages,
                    system=system,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    plan=plan,
                    session_id=self._sid,
                    emitter=self._emitter,
                    llm=self._llm,
                    tts=self._tts,
                    config=self._config,
                    tool_registry=self._tool_registry,
                    context_builder=self._context_builder,
                    ctx=ctx,
                    metrics=self.metrics,
                    on_first_audio=self._record_e2e_latency,
                    capture_llm_timing=self._capture_llm_timing,
                    run_audio_response=self._run_audio_response,
                )
            else:
                assistant_item = await self._setup_response(
                    response_id, item_id, output_index, content_index, plan.has_audio
                )

                if plan.has_audio:
                    full_transcript = await self._run_audio_response(
                        response_id=response_id,
                        item_id=item_id,
                        output_index=output_index,
                        content_index=content_index,
                        assistant_item=assistant_item,
                        messages=messages,
                        system=system,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        tools=self._config.tools or None,
                        config=self._config,
                        emitter=self._emitter,
                        on_first_audio=self._record_e2e_latency,
                        metrics=self.metrics,
                    )
                    full_text = ""
                else:
                    full_text = await run_text_response(
                        response_id=response_id,
                        item_id=item_id,
                        output_index=output_index,
                        content_index=content_index,
                        assistant_item=assistant_item,
                        messages=messages,
                        system=system,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        config=self._config,
                        llm=self._llm,
                        emitter=self._emitter,
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
    # Response lifecycle helpers
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

    async def _run_audio_response(self, **kwargs) -> str:
        """Delegate to audio_response module."""
        return await run_audio_response(
            llm=self._llm,
            tts=self._tts,
            **kwargs,
        )

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

    def _capture_llm_timing(self, tool_round: int) -> None:
        """Capture LLM TTFT and total streaming time."""
        if tool_round == 0:
            self.metrics["llm_ttft_ms"] = round(self._llm.last_ttft_ms, 1)
        self.metrics["llm_total_ms"] = round(self._llm.last_stream_total_ms, 1)

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
