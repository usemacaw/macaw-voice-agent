"""
Filler phrase management for tool calling.

Generates contextual filler phrases and synthesizes them through TTS
while server-side tools execute. Filler audio is sent to the client
to fill silence gaps, but is NOT stored in conversation history.
"""

from __future__ import annotations

import json
import logging
import random
import uuid
from typing import TYPE_CHECKING

from audio.codec import encode_audio_for_client
from protocol import events
from protocol.models import ContentPart, ConversationItem

if TYPE_CHECKING:
    from protocol.event_emitter import EventEmitter
    from protocol.models import SessionConfig
    from providers.tts import TTSProvider

logger = logging.getLogger("open-voice-api.filler")

# ---------------------------------------------------------------------------
# Filler phrase pools (Portuguese)
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


def build_dynamic_filler(tool_name: str, arguments_json: str) -> str:
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


async def send_filler_audio(
    session_id: str,
    tts: TTSProvider,
    emitter: EventEmitter,
    config: SessionConfig,
    response_id: str,
    output_index: int,
    filler_text: str,
) -> None:
    """Synthesize and send a filler phrase via TTS while tools execute.

    NOTE: Filler items are NOT added to conversation history on purpose.
    Storing them would pollute the LLM context and cause it to imitate
    filler phrases instead of calling tools.
    """
    try:
        filler_item_id = f"item_{uuid.uuid4().hex[:24]}"
        filler_item = ConversationItem(
            id=filler_item_id,
            type="message",
            role="assistant",
            status="in_progress",
            content=[ContentPart(type="audio", audio="", transcript="")],
        )
        await emitter.emit(
            events.response_output_item_added(
                "", response_id, output_index, filler_item
            )
        )

        if tts.supports_streaming:
            async for chunk in tts.synthesize_stream(filler_text):
                if chunk:
                    audio_b64 = encode_audio_for_client(
                        chunk, config.output_audio_format
                    )
                    await emitter.emit(
                        events.response_audio_delta(
                            "", response_id, filler_item_id, output_index, 0,
                            audio_b64,
                        )
                    )
        else:
            audio = await tts.synthesize(filler_text)
            if audio:
                audio_b64 = encode_audio_for_client(
                    audio, config.output_audio_format
                )
                await emitter.emit(
                    events.response_audio_delta(
                        "", response_id, filler_item_id, output_index, 0,
                        audio_b64,
                    )
                )

        await emitter.emit(
            events.response_audio_transcript_delta(
                "", response_id, filler_item_id, output_index, 0, filler_text,
            )
        )

        filler_item.content[0] = ContentPart(
            type="audio", transcript=filler_text
        )
        filler_item.status = "completed"
        await emitter.emit(
            events.response_audio_done(
                "", response_id, filler_item_id, output_index, 0
            )
        )
        await emitter.emit(
            events.response_output_item_done(
                "", response_id, output_index, filler_item
            )
        )
        logger.info(f"[{session_id[:8]}] Filler sent: \"{filler_text}\"")
    except Exception as e:
        logger.warning(
            f"[{session_id[:8]}] Filler TTS failed (non-critical): {e}"
        )
