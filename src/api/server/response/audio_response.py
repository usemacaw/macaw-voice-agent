"""Audio response path: LLM → SentencePipeline → audio events.

Handles the non-tool audio streaming path where LLM output is split into
sentences, synthesized via TTS, and streamed as audio deltas.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Callable

from audio.codec import encode_audio_for_client
from pipeline.sentence_pipeline import SentencePipeline
from protocol import events
from protocol.metrics import ResponseMetrics
from protocol.models import ContentPart

if TYPE_CHECKING:
    from protocol.event_emitter import EventEmitter
    from protocol.models import ConversationItem, SessionConfig
    from providers.llm import LLMProvider
    from providers.tts import TTSProvider

logger = logging.getLogger("open-voice-api.response-runner.audio")


async def run_audio_response(
    *,
    response_id: str,
    item_id: str,
    output_index: int,
    content_index: int,
    assistant_item: ConversationItem,
    messages: list[dict],
    system: str,
    temperature: float,
    max_tokens: int,
    tools: list[dict] | None,
    config: SessionConfig,
    llm: LLMProvider,
    tts: TTSProvider,
    emitter: EventEmitter,
    on_first_audio: Callable[[], None],
    metrics: ResponseMetrics,
) -> str:
    """Run LLM->TTS pipeline, stream audio events. Returns full transcript."""
    full_transcript = ""
    first_audio_sent = False

    pipeline = SentencePipeline(llm, tts)
    async for sentence, audio_chunk, is_new_sentence in pipeline.process_streaming(
        messages,
        system=system,
        tools=tools,
        temperature=temperature,
        max_tokens=max_tokens,
    ):
        audio_b64 = encode_audio_for_client(
            audio_chunk, config.output_audio_format
        )
        await emitter.emit(
            events.response_audio_delta(
                "", response_id, item_id, output_index, content_index, audio_b64
            )
        )

        if not first_audio_sent:
            first_audio_sent = True
            on_first_audio()

        if is_new_sentence and sentence:
            delta = sentence
            if full_transcript:
                delta = " " + sentence
            await emitter.emit(
                events.response_audio_transcript_delta(
                    "", response_id, item_id, output_index, content_index, delta
                )
            )
            full_transcript += delta

    # Capture pipeline metrics
    pm = pipeline.metrics
    metrics.llm_ttft_ms = round(pm.llm_ttft_ms, 1)
    metrics.llm_total_ms = round(pm.llm_total_ms, 1)
    metrics.llm_first_sentence_ms = round(pm.first_sentence_latency_ms, 1)
    metrics.pipeline_first_audio_ms = round(pm.first_audio_latency_ms, 1)
    metrics.tts_synth_ms = round(pm.tts_synth_ms, 1)
    metrics.tts_wait_ms = round(pm.tts_wait_ms, 1)
    metrics.tts_first_chunk_ms = round(pm.tts_first_chunk_ms, 1)
    metrics.output_chars = len(full_transcript)

    assistant_item.content[content_index] = ContentPart(
        type="audio", transcript=full_transcript
    )

    await emitter.emit(
        events.response_audio_done("", response_id, item_id, output_index, content_index)
    )
    await emitter.emit(
        events.response_audio_transcript_done(
            "", response_id, item_id, output_index, content_index, full_transcript
        )
    )

    return full_transcript
