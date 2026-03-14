"""
Response execution engine: LLM -> tools -> TTS -> audio events.

Extracted from RealtimeSession to isolate the response pipeline
from session lifecycle management. Each response creates a fresh
ResponseRunner instance.
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
import re
import time
import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

from audio.codec import encode_audio_for_client
from config import LLM_CONFIG
from pipeline.conversation import items_to_messages, items_to_windowed_messages
from pipeline.sentence_pipeline import SentencePipeline
from protocol import events
from protocol.event_emitter import SlowClientError
from protocol.models import ContentPart, ConversationItem

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
# Dynamic filler phrases
# ---------------------------------------------------------------------------

_SEARCH_FILLERS = [
    ("Vou pesquisar sobre {q}, aguarde.", "Vou pesquisar, aguarde."),
    ("Deixa eu buscar sobre {q}.", "Deixa eu buscar isso pra você."),
    ("Vou verificar sobre {q}, um momento.", "Vou verificar, um momento."),
    ("Vou buscar informações sobre {q}.", "Vou buscar informações pra você."),
    ("Um momento, vou procurar sobre {q}.", "Um momento, vou procurar."),
    ("Espere um pouco, vou pesquisar sobre {q}.", "Espere um pouco, vou pesquisar."),
    ("Aguarde, vou buscar sobre {q}.", "Aguarde, vou buscar."),
]

_MEMORY_FILLERS = [
    "Deixa eu verificar, um momento.",
    "Vou checar, aguarde.",
    "Um momento, vou lembrar.",
]

_GENERIC_FILLERS = [
    "Um momento, por favor.",
    "Aguarde um instante.",
    "Só um momento.",
]


def _build_dynamic_filler(tool_name: str, arguments_json: str) -> str:
    """Build a contextual filler phrase based on tool name and arguments."""
    try:
        args = json.loads(arguments_json) if arguments_json else {}
    except (json.JSONDecodeError, TypeError):
        args = {}

    if tool_name == "web_search":
        query = args.get("query", "")
        with_q, without_q = random.choice(_SEARCH_FILLERS)
        if query:
            short = query[:60].rstrip()
            return with_q.format(q=short)
        return without_q

    if tool_name == "recall_memory":
        return random.choice(_MEMORY_FILLERS)

    return random.choice(_GENERIC_FILLERS)


# ---------------------------------------------------------------------------
# Response context (shared session state)
# ---------------------------------------------------------------------------


@dataclass
class ResponseContext:
    """Mutable session state passed to the response runner."""

    items: list[ConversationItem]
    state_lock: asyncio.Lock
    append_item: Callable[[ConversationItem], None]  # must be called under state_lock
    speech_stopped_at: float | None
    turn_count: int
    session_start: float
    barge_in_count: int


# ---------------------------------------------------------------------------
# ResponseRunner
# ---------------------------------------------------------------------------


class ResponseRunner:
    """Executes a complete response: LLM -> (tools ->)* TTS -> audio events.

    Created per-response. Holds no state between responses.
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

        # Populated during run(), read by caller after completion
        self.metrics: dict[str, object] = {}

        # Set by caller, used for E2E latency measurement
        self._speech_stopped_at: float | None = None

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    async def run(
        self,
        response_id: str,
        ctx: ResponseContext,
        prior_metrics: dict[str, object],
    ) -> None:
        """Execute a full response cycle.

        Args:
            response_id: Unique response identifier.
            ctx: Shared session state (items, lock, callbacks).
            prior_metrics: Metrics from the speech/ASR phase to carry over.
        """
        self._ctx = ctx
        self._speech_stopped_at = ctx.speech_stopped_at

        item_id = f"item_{uuid.uuid4().hex[:24]}"
        output_index = 0
        content_index = 0
        response_start = time.perf_counter()
        has_audio = "audio" in self._config.modalities
        has_tools = bool(self._config.tools) or (
            self._tool_registry is not None and self._tool_registry.has_server_tools
        )
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
        # Carry over metrics from speech/ASR phase
        for key in ("asr_ms", "speech_ms", "asr_mode", "input_chars", "speech_rms"):
            if key in prior_metrics:
                self.metrics[key] = prior_metrics[key]

        logger.info(
            f"[{self._sid[:8]}] RESPONSE START: "
            f"items={len(ctx.items)}, has_audio={has_audio}, has_tools={has_tools}"
        )

        try:
            # Build messages from conversation history (windowed to reduce context)
            async with ctx.state_lock:
                if has_tools:
                    messages = items_to_windowed_messages(list(ctx.items), window=8)
                else:
                    messages = items_to_messages(list(ctx.items))
            system = self._config.instructions
            temperature = self._config.temperature
            max_tokens = (
                self._config.max_response_output_tokens
                if isinstance(self._config.max_response_output_tokens, int)
                else LLM_CONFIG["max_tokens"]
            )

            if has_tools:
                await self._run_with_tools(
                    response_id, messages, system, temperature, max_tokens, has_audio,
                )
            else:
                assistant_item = await self._setup_response(
                    response_id, item_id, output_index, content_index, has_audio
                )

                if has_audio:
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
    # Tool calling loop
    # ------------------------------------------------------------------

    async def _run_with_tools(
        self,
        response_id: str,
        messages: list[dict],
        system: str,
        temperature: float,
        max_tokens: int,
        has_audio: bool,
    ) -> None:
        """Run response with function calling support."""
        from providers.llm import LLMStreamEvent  # noqa: F811

        server_side = (
            self._tool_registry is not None
            and self._tool_registry.has_server_tools
        )

        await self._emitter.emit(events.response_created("", response_id))

        response_start = time.perf_counter()
        output_index = 0
        max_rounds = (
            self._tool_registry.max_rounds if self._tool_registry else 5
        )

        # Merge tool schemas: session config tools + server-side tool schemas
        tools = list(self._config.tools) if self._config.tools else []
        if server_side:
            server_schemas = self._tool_registry.get_schemas()
            existing_names = {
                t.get("function", {}).get("name") for t in tools if isinstance(t, dict)
            }
            for schema in server_schemas:
                name = schema.get("function", {}).get("name", "")
                if name not in existing_names:
                    tools.append(schema)

        tools_used = 0
        tool_round = 0
        for tool_round in range(max_rounds + 1):
            allow_tools = tools if tools_used < max_rounds else None
            round_max_tokens = max_tokens if allow_tools else min(max_tokens, 40)

            # Final round (no tools) + audio: use pipelined LLM->TTS
            if not allow_tools and has_audio:
                logger.info(
                    f"[{self._sid[:8]}] LLM call round {tool_round}: "
                    f"msgs={len(messages)}, tools=no (pipelined), "
                    f"messages={[m.get('role', '?') + ':' + str(m.get('content') or '')[:40] for m in messages[-4:]]}"
                )
                await self._emit_tool_response_audio_streamed(
                    response_id, output_index, response_start,
                    messages, system, temperature, round_max_tokens,
                )
                break

            logger.info(
                f"[{self._sid[:8]}] LLM call round {tool_round}: "
                f"msgs={len(messages)}, tools={'yes' if allow_tools else 'no'}, "
                f"messages={[m.get('role', '?') + ':' + str(m.get('content') or '')[:40] for m in messages[-4:]]}"
            )

            # Collect full LLM stream: text + tool calls
            collected_text = ""
            collected_tool_calls: list[dict] = []

            current_tool_call_id = ""
            current_tool_name = ""
            tool_arguments_buffer = ""
            in_tool_call = False

            async for event in self._llm.generate_stream_with_tools(
                messages, system=system,
                tools=allow_tools or None,
                temperature=temperature,
                max_tokens=round_max_tokens,
            ):
                if event.type == "text_delta":
                    collected_text += event.text
                elif event.type == "tool_call_start":
                    current_tool_call_id = event.tool_call_id
                    current_tool_name = event.tool_name
                    tool_arguments_buffer = ""
                    in_tool_call = True
                elif event.type == "tool_call_delta" and in_tool_call:
                    tool_arguments_buffer += event.tool_arguments_delta
                elif event.type == "tool_call_end" and in_tool_call:
                    collected_tool_calls.append({
                        "id": current_tool_call_id,
                        "name": current_tool_name,
                        "arguments": tool_arguments_buffer,
                    })
                    in_tool_call = False

            # Capture LLM timing
            if tool_round == 0:
                self.metrics["llm_ttft_ms"] = round(self._llm.last_ttft_ms, 1)
            self.metrics["llm_total_ms"] = round(self._llm.last_stream_total_ms, 1)

            # No tool calls: emit final response
            collected_text = _clean_for_voice(collected_text)
            if not collected_tool_calls:
                if collected_text:
                    logger.info(
                        f"[{self._sid[:8]}] LLM text (round {tool_round}): "
                        f"{collected_text[:200]!r}"
                    )
                    if has_audio:
                        await self._emit_tool_response_audio(
                            response_id, output_index, response_start,
                            collected_text,
                        )
                    else:
                        await self._emit_tool_response_text(
                            response_id, output_index, collected_text,
                        )
                else:
                    logger.warning(
                        f"[{self._sid[:8]}] Tool round {tool_round}: "
                        "no text and no tool calls"
                    )
                break

            # Has tool calls
            logger.info(
                f"[{self._sid[:8]}] Tool round {tool_round}: "
                f"{len(collected_tool_calls)} tool call(s): "
                f"{[tc['name'] for tc in collected_tool_calls]}"
            )

            if server_side:
                for tc in collected_tool_calls:
                    self.metrics.setdefault("tools_used", []).append(tc["name"])

                all_ok = await self._execute_tools_server_side(
                    response_id, output_index, collected_tool_calls, has_audio,
                )
                output_index += len(collected_tool_calls)
                tools_used += 1

                # Rebuild messages with tool results
                async with self._ctx.state_lock:
                    messages = items_to_windowed_messages(list(self._ctx.items), window=8)

                if not all_ok:
                    logger.info(
                        f"[{self._sid[:8]}] Tool error detected, "
                        "forcing text response on next call"
                    )
                    tools_used = max_rounds
            else:
                await self._emit_tool_calls_for_client(
                    response_id, output_index, collected_tool_calls,
                )
                break
        else:
            logger.warning(
                f"[{self._sid[:8]}] Tool execution exceeded max rounds ({max_rounds})"
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
        await self._emitter.emit(
            events.macaw_metrics(response_id, self.metrics)
        )

    # ------------------------------------------------------------------
    # Server-side tool execution
    # ------------------------------------------------------------------

    async def _execute_tools_server_side(
        self,
        response_id: str,
        output_index: int,
        tool_calls: list[dict],
        has_audio: bool,
    ) -> bool:
        """Execute tool calls server-side with filler TTS.

        Returns True if all tools succeeded, False if any returned an error.
        """
        any_error = False
        # Send filler audio for the first tool call
        if has_audio and tool_calls:
            first_tool = tool_calls[0]
            filler = _build_dynamic_filler(first_tool["name"], first_tool["arguments"])
            await self._send_filler_audio(response_id, output_index, filler)

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
            async with self._ctx.state_lock:
                self._ctx.append_item(fc_item)

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
            result_json = await self._tool_registry.execute(tc_name, tc_args)
            tool_exec_ms = (time.perf_counter() - tool_t0) * 1000
            logger.info(
                f"[{self._sid[:8]}] Tool '{tc_name}' result ({tool_exec_ms:.0f}ms): "
                f"{result_json[:200]}"
            )

            # Store per-tool timing
            tool_timings = self.metrics.setdefault("tool_timings", [])
            tool_ok = True
            try:
                result_data = json.loads(result_json)
                if isinstance(result_data, dict) and "error" in result_data:
                    any_error = True
                    tool_ok = False
            except (json.JSONDecodeError, TypeError):
                pass
            tool_timings.append({
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
            async with self._ctx.state_lock:
                self._ctx.append_item(fco_item)
            await self._emitter.emit(
                events.conversation_item_created("", fc_item_id, fco_item)
            )

            output_index += 1

        return not any_error

    async def _send_filler_audio(
        self, response_id: str, output_index: int, filler_text: str
    ) -> None:
        """Synthesize and send a filler phrase via TTS while tools execute."""
        try:
            filler_item_id = f"item_{uuid.uuid4().hex[:24]}"
            filler_item = ConversationItem(
                id=filler_item_id,
                type="message",
                role="assistant",
                status="in_progress",
                content=[ContentPart(type="audio", audio="", transcript="")],
            )
            await self._emitter.emit(
                events.response_output_item_added(
                    "", response_id, output_index, filler_item
                )
            )
            # NOTE: filler is NOT added to items on purpose.
            # Storing it would pollute the LLM context.

            if self._tts.supports_streaming:
                async for chunk in self._tts.synthesize_stream(filler_text):
                    if chunk:
                        audio_b64 = encode_audio_for_client(
                            chunk, self._config.output_audio_format
                        )
                        await self._emitter.emit(
                            events.response_audio_delta(
                                "", response_id, filler_item_id, output_index, 0,
                                audio_b64,
                            )
                        )
            else:
                audio = await self._tts.synthesize(filler_text)
                if audio:
                    audio_b64 = encode_audio_for_client(
                        audio, self._config.output_audio_format
                    )
                    await self._emitter.emit(
                        events.response_audio_delta(
                            "", response_id, filler_item_id, output_index, 0,
                            audio_b64,
                        )
                    )

            await self._emitter.emit(
                events.response_audio_transcript_delta(
                    "", response_id, filler_item_id, output_index, 0, filler_text,
                )
            )

            filler_item.content[0] = ContentPart(
                type="audio", transcript=filler_text
            )
            filler_item.status = "completed"
            await self._emitter.emit(
                events.response_audio_done(
                    "", response_id, filler_item_id, output_index, 0
                )
            )
            await self._emitter.emit(
                events.response_output_item_done(
                    "", response_id, output_index, filler_item
                )
            )
            logger.info(f"[{self._sid[:8]}] Filler sent: \"{filler_text}\"")
        except Exception as e:
            logger.warning(
                f"[{self._sid[:8]}] Filler TTS failed (non-critical): {e}"
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
            f"[{self._sid[:8]}] TTS direct synth: {full_transcript[:100]!r}"
        )
        if self._tts.supports_streaming:
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
        else:
            audio = await self._tts.synthesize(full_transcript)
            if audio:
                audio_b64 = encode_audio_for_client(
                    audio, self._config.output_audio_format
                )
                await self._emitter.emit(
                    events.response_audio_delta(
                        "", response_id, item_id, output_index, content_index,
                        audio_b64,
                    )
                )
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

    async def _emit_tool_calls_for_client(
        self,
        response_id: str,
        output_index: int,
        tool_calls: list[dict],
    ) -> None:
        """Emit tool call events for client-side execution."""
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
        await self._emitter.emit(
            events.macaw_metrics(response_id, self.metrics)
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _record_e2e_latency(self) -> None:
        """Record E2E latency on first audio chunk sent."""
        if self._speech_stopped_at is not None:
            e2e_ms = (time.perf_counter() - self._speech_stopped_at) * 1000
            self.metrics["e2e_ms"] = round(e2e_ms, 1)
            logger.info(
                f"[{self._sid[:8]}] E2E LATENCY: {e2e_ms:.0f}ms "
                f"(speech_stopped -> first_audio_sent)"
            )
            # Only record once
            self._speech_stopped_at = None
