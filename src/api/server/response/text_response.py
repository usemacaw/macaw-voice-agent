"""Text-only response path: LLM → text deltas.

Handles the non-audio response path where LLM output is streamed
as text deltas without TTS synthesis.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from protocol import events
from protocol.models import ContentPart

if TYPE_CHECKING:
    from protocol.event_emitter import EventEmitter
    from protocol.models import ConversationItem, SessionConfig
    from providers.llm import LLMProvider

__all__ = ["run_text_response"]


async def run_text_response(
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
    config: SessionConfig,
    llm: LLMProvider,
    emitter: EventEmitter,
) -> str:
    """Run LLM text-only streaming. Returns full text."""
    full_text = ""

    async for chunk in llm.generate_stream(
        messages,
        system=system,
        tools=config.tools or None,
        temperature=temperature,
        max_tokens=max_tokens,
    ):
        await emitter.emit(
            events.response_text_delta(
                "", response_id, item_id, output_index, content_index, chunk
            )
        )
        full_text += chunk

    assistant_item.content[content_index] = ContentPart(
        type="text", text=full_text
    )
    await emitter.emit(
        events.response_text_done(
            "", response_id, item_id, output_index, content_index, full_text
        )
    )

    return full_text
