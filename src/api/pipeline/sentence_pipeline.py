"""
Sentence-level streaming pipeline: LLM → TTS.

Adapted from ai-agent/pipeline/sentence_pipeline.py for OpenVoiceAPI.
Key difference: LLM is stateless — receives full messages each call.

Architecture:
    [LLM] → sentence_queue → [TTS Worker] → audio_queue → [Consumer yield]
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import AsyncGenerator, Optional, TYPE_CHECKING

from audio.codec import INTERNAL_SAMPLE_RATE, SAMPLE_WIDTH
from audio.text_cleaning import strip_emojis
from config import PIPELINE
from pipeline.sentence_splitter import generate_sentences, split_long_sentence
from providers.admission import ADMISSION

if TYPE_CHECKING:
    from providers.llm import LLMProvider
    from providers.tts import TTSProvider

logger = logging.getLogger("open-voice-api.sentence-pipeline")

# 100ms chunk at internal sample rate
_CHUNK_DURATION_S = 0.1
_CHUNK_SIZE_BYTES = int(INTERNAL_SAMPLE_RATE * _CHUNK_DURATION_S * SAMPLE_WIDTH)


@dataclass
class PipelineMetrics:
    sentences_generated: int = 0
    audio_chunks_produced: int = 0
    first_audio_latency_ms: float = 0.0
    total_latency_ms: float = 0.0
    first_sentence_latency_ms: float = 0.0
    tts_total_ms: float = 0.0
    tts_synth_ms: float = 0.0
    tts_wait_ms: float = 0.0
    tts_first_chunk_ms: float = 0.0
    tts_queue_max_depth: int = 0
    tts_calls: int = 0
    llm_ttft_ms: float = 0.0
    llm_total_ms: float = 0.0


@dataclass
class _AudioItem:
    sentence: str
    audio: bytes
    new_sentence: bool = True


class SentencePipeline:
    """Streaming sentence-level pipeline: LLM → TTS in parallel."""

    def __init__(
        self,
        llm: LLMProvider,
        tts: TTSProvider,
        queue_size: int = 6,
        prefetch_size: int | None = None,
    ):
        self._llm = llm
        self._tts = tts
        self._queue_size = queue_size
        self._prefetch_size = prefetch_size or PIPELINE.tts_prefetch_size
        self._metrics = PipelineMetrics()

    @property
    def metrics(self) -> PipelineMetrics:
        return self._metrics

    async def process_streaming(
        self,
        messages: list[dict],
        system: str = "",
        tools: list[dict] | None = None,
        temperature: float = 0.8,
        max_tokens: int = 1024,
    ) -> AsyncGenerator[tuple[str, bytes, bool], None]:
        """Process messages through LLM→TTS pipeline.

        Yields (sentence, audio_chunk, is_new_sentence) tuples where:
        - sentence: the text being spoken
        - audio_chunk: ~100ms PCM 8kHz bytes
        - is_new_sentence: True only on the first chunk of a new sentence
        """
        self._metrics = PipelineMetrics()
        start_time = time.perf_counter()
        first_audio_time: Optional[float] = None

        sentence_queue: asyncio.Queue[Optional[str]] = asyncio.Queue(
            maxsize=self._queue_size
        )

        audio_queue_size = self._prefetch_size
        if self._tts.supports_streaming:
            audio_queue_size = self._prefetch_size * 50
        audio_queue: asyncio.Queue[Optional[_AudioItem]] = asyncio.Queue(
            maxsize=audio_queue_size
        )

        producer_task = asyncio.create_task(
            self._produce_sentences(
                messages, sentence_queue, system, tools, temperature, max_tokens
            )
        )
        tts_worker_task = asyncio.create_task(
            self._tts_worker(sentence_queue, audio_queue)
        )

        try:
            while True:
                try:
                    item = await asyncio.wait_for(
                        audio_queue.get(),
                        timeout=PIPELINE.tts_timeout,
                    )
                except asyncio.TimeoutError:
                    logger.warning("Timeout waiting for TTS audio")
                    break

                if item is None:
                    break

                if item.new_sentence:
                    self._metrics.sentences_generated += 1

                audio = item.audio

                # When TTS supports true streaming, chunks are already
                # the right size (~83ms per frame). Skip re-chunking.
                if self._tts.supports_streaming:
                    if first_audio_time is None:
                        first_audio_time = time.perf_counter()
                        self._metrics.first_audio_latency_ms = (
                            (first_audio_time - start_time) * 1000
                        )
                        logger.info(
                            f"First audio in {self._metrics.first_audio_latency_ms:.0f}ms"
                        )
                    self._metrics.audio_chunks_produced += 1
                    yield item.sentence, audio, item.new_sentence
                else:
                    # Fallback: re-chunk into ~100ms pieces
                    first_sub_chunk = True
                    for i in range(0, len(audio), _CHUNK_SIZE_BYTES):
                        chunk = audio[i:i + _CHUNK_SIZE_BYTES]
                        if not chunk:
                            continue

                        if first_audio_time is None:
                            first_audio_time = time.perf_counter()
                            self._metrics.first_audio_latency_ms = (
                                (first_audio_time - start_time) * 1000
                            )
                            logger.info(
                                f"First audio in {self._metrics.first_audio_latency_ms:.0f}ms"
                            )

                        self._metrics.audio_chunks_produced += 1
                        is_new = item.new_sentence and first_sub_chunk
                        yield item.sentence, chunk, is_new
                        first_sub_chunk = False

        finally:
            for task in (producer_task, tts_worker_task):
                if not task.done():
                    task.cancel()
            for task in (producer_task, tts_worker_task):
                if not task.done():
                    try:
                        await task
                    except (asyncio.CancelledError, Exception) as e:
                        if not isinstance(e, asyncio.CancelledError):
                            logger.warning(f"Pipeline task cleanup error: {e}")

            self._metrics.total_latency_ms = (time.perf_counter() - start_time) * 1000
            logger.info(
                f"Pipeline: {self._metrics.sentences_generated} sentences, "
                f"{self._metrics.audio_chunks_produced} chunks, "
                f"first_audio={self._metrics.first_audio_latency_ms:.0f}ms, "
                f"total={self._metrics.total_latency_ms:.0f}ms"
            )

    async def _produce_sentences(
        self,
        messages: list[dict],
        queue: asyncio.Queue,
        system: str,
        tools: list[dict] | None,
        temperature: float,
        max_tokens: int,
    ) -> None:
        produce_start = time.perf_counter()
        sentence_index = 0
        max_chars = PIPELINE.max_sentence_chars

        try:
            async with ADMISSION.llm.acquire():
                async for sentence in generate_sentences(
                    self._llm, messages, system=system, tools=tools,
                    temperature=temperature, max_tokens=max_tokens,
                ):
                    sub_sentences = split_long_sentence(sentence, max_chars)
                    for sub in sub_sentences:
                        clean = strip_emojis(sub).strip()
                        if not clean:
                            continue
                        sentence_index += 1
                        await queue.put(clean)

                        if sentence_index == 1:
                            self._metrics.first_sentence_latency_ms = (
                                (time.perf_counter() - produce_start) * 1000
                            )
                            logger.info(
                                f"First sentence in {self._metrics.first_sentence_latency_ms:.0f}ms"
                            )
        finally:
            # Capture LLM-level timing from the provider (snapshot)
            timing = self._llm.get_last_timing()
            self._metrics.llm_ttft_ms = timing.ttft_ms
            self._metrics.llm_total_ms = timing.total_ms
            await queue.put(None)

    async def _tts_worker(
        self,
        sentence_queue: asyncio.Queue,
        audio_queue: asyncio.Queue,
    ) -> None:
        use_streaming = self._tts.supports_streaming
        first_chunk_recorded = False
        try:
            while True:
                wait_start = time.perf_counter()
                try:
                    sentence = await asyncio.wait_for(
                        sentence_queue.get(),
                        timeout=PIPELINE.sentence_timeout,
                    )
                except asyncio.TimeoutError:
                    break

                wait_ms = (time.perf_counter() - wait_start) * 1000
                self._metrics.tts_wait_ms += wait_ms

                if sentence is None:
                    break

                # Track queue depth high watermark
                depth = sentence_queue.qsize()
                if depth > self._metrics.tts_queue_max_depth:
                    self._metrics.tts_queue_max_depth = depth

                try:
                    synth_start = time.perf_counter()
                    self._metrics.tts_calls += 1

                    async with ADMISSION.tts.acquire():
                        if use_streaming:
                            chunk_idx = 0
                            async for tts_chunk in self._tts.synthesize_stream(sentence):
                                if tts_chunk:
                                    # Record first chunk latency (once per pipeline)
                                    if not first_chunk_recorded:
                                        first_chunk_recorded = True
                                        self._metrics.tts_first_chunk_ms = (
                                            (time.perf_counter() - synth_start) * 1000
                                        )
                                    await audio_queue.put(
                                        _AudioItem(sentence, tts_chunk, new_sentence=(chunk_idx == 0))
                                    )
                                    chunk_idx += 1
                        else:
                            audio = await self._tts.synthesize(sentence)
                            if audio:
                                if not first_chunk_recorded:
                                    first_chunk_recorded = True
                                    self._metrics.tts_first_chunk_ms = (
                                        (time.perf_counter() - synth_start) * 1000
                                    )
                                await audio_queue.put(_AudioItem(sentence, audio))

                    synth_ms = (time.perf_counter() - synth_start) * 1000
                    self._metrics.tts_synth_ms += synth_ms
                    self._metrics.tts_total_ms += synth_ms

                except asyncio.CancelledError:
                    raise
                except Exception as e:
                    logger.error(f"TTS error for '{sentence[:30]}...': {e}")
        finally:
            await audio_queue.put(None)
