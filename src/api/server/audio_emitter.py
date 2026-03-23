"""
AudioEmitter — unified TTS-to-WebSocket audio emission.

Encapsulates the common pattern: TTS stream → encode → emit protocol events.
Eliminates duplication between inline TTS (tool-calling path) and batch TTS
(fallback path) which previously had separate implementations.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from typing import TYPE_CHECKING

from audio.codec import encode_audio_for_client
from protocol import events
from protocol.models import ContentPart, ConversationItem

if TYPE_CHECKING:
    from protocol.event_emitter import EventEmitter
    from providers.tts import TTSProvider

# Timeout waiting for next sentence from queue.
# Prevents hanging forever if producer is cancelled without sending sentinel.
_QUEUE_SENTINEL_TIMEOUT_S = 10.0

logger = logging.getLogger("open-voice-api.audio-emitter")


class AudioEmitter:
    """Synthesize text via TTS and emit protocol-compliant audio events.

    Handles: item creation, audio delta emission, transcript deltas,
    and finalization events (audio.done, transcript.done, content_part.done,
    output_item.done).

    Created per use. Not reusable across responses.
    """

    def __init__(
        self,
        emitter: EventEmitter,
        tts: TTSProvider,
        output_audio_format: str,
        on_first_audio: callable | None = None,
    ):
        self._emitter = emitter
        self._tts = tts
        self._output_audio_format = output_audio_format
        self._on_first_audio = on_first_audio
        self._first_audio_sent = False

    async def emit_from_text(
        self,
        text: str,
        response_id: str,
        output_index: int,
        state_lock: asyncio.Lock,
        append_item: callable,
    ) -> tuple[str, str]:
        """Synthesize text and emit all audio events.

        Returns (item_id, full_transcript).
        """
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
        async with state_lock:
            append_item(assistant_item)

        full_transcript = text.strip()

        async for audio_chunk in self._tts.synthesize_stream(full_transcript):
            if not audio_chunk:
                continue
            audio_b64 = encode_audio_for_client(
                audio_chunk, self._output_audio_format
            )
            await self._emitter.emit(
                events.response_audio_delta(
                    "", response_id, item_id, output_index, content_index,
                    audio_b64,
                )
            )
            if not self._first_audio_sent:
                self._first_audio_sent = True
                if self._on_first_audio:
                    self._on_first_audio()

        if full_transcript:
            await self._emitter.emit(
                events.response_audio_transcript_delta(
                    "", response_id, item_id, output_index, content_index,
                    full_transcript,
                )
            )

        await self._finalize(
            response_id, item_id, output_index, content_index,
            assistant_item, full_transcript,
        )

        return item_id, full_transcript

    async def emit_from_queue(
        self,
        queue: asyncio.Queue,
        response_id: str,
        output_index: int,
        state_lock: asyncio.Lock,
        append_item: callable,
    ) -> tuple[str, str]:
        """Consume sentences from queue, synthesize each, emit audio events.

        Queue protocol: str sentences, None as sentinel to stop.
        Returns (item_id, full_transcript).
        """
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
        async with state_lock:
            append_item(assistant_item)

        full_text = ""
        while True:
            try:
                sentence = await asyncio.wait_for(
                    queue.get(), timeout=_QUEUE_SENTINEL_TIMEOUT_S,
                )
            except asyncio.TimeoutError:
                logger.warning("TTS queue sentinel timeout, likely cancelled producer")
                break
            if sentence is None:
                break
            full_text += (" " if full_text else "") + sentence

            async for audio_chunk in self._tts.synthesize_stream(sentence):
                if not audio_chunk:
                    continue
                audio_b64 = encode_audio_for_client(
                    audio_chunk, self._output_audio_format
                )
                await self._emitter.emit(
                    events.response_audio_delta(
                        "", response_id, item_id, output_index, content_index,
                        audio_b64,
                    )
                )
                if not self._first_audio_sent:
                    self._first_audio_sent = True
                    if self._on_first_audio:
                        self._on_first_audio()

            await self._emitter.emit(
                events.response_audio_transcript_delta(
                    "", response_id, item_id, output_index, content_index,
                    sentence,
                )
            )

        await self._finalize(
            response_id, item_id, output_index, content_index,
            assistant_item, full_text,
        )

        return item_id, full_text

    async def _finalize(
        self,
        response_id: str,
        item_id: str,
        output_index: int,
        content_index: int,
        assistant_item: ConversationItem,
        full_transcript: str,
    ) -> None:
        """Emit finalization events for an audio response item."""
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
