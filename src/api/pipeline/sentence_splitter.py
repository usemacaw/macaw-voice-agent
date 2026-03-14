"""
Sentence splitting utilities for the LLM→TTS pipeline.

Splits LLM streaming output into complete sentences for TTS synthesis.
Uses eager-first logic: yields first sentence at clause boundaries
to minimize latency, then sentence-end boundaries for the rest.
"""

from __future__ import annotations

import re
from collections.abc import AsyncGenerator
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from providers.llm import LLMProvider

# Sentence-ending punctuation
_SENTENCE_END = re.compile(r'[.!?]\s*$')
# Clause break points for eager first sentence
_CLAUSE_BREAK = re.compile(r'[,;:\u2014\u2013]\s*$')
# Break points for splitting long sentences
_BREAK_POINTS = re.compile(r'[,;:\u2014\u2013\u2015]|\.\.\.')


async def generate_sentences(
    llm: LLMProvider,
    messages: list[dict],
    system: str = "",
    tools: list[dict] | None = None,
    temperature: float = 0.8,
    max_tokens: int = 1024,
) -> AsyncGenerator[str, None]:
    """Stream complete sentences from the LLM.

    Uses eager-first logic: yields first sentence at clause boundaries
    to minimize latency, then sentence-end boundaries for rest.
    """
    buffer = ""
    first_sentence = True
    min_eager_chars = 20

    async for chunk in llm.generate_stream(
        messages, system=system, tools=tools,
        temperature=temperature, max_tokens=max_tokens,
    ):
        buffer += chunk

        while True:
            # Look for sentence end
            match = _SENTENCE_END.search(buffer)
            if match:
                sentence = buffer[:match.end()].strip()
                buffer = buffer[match.end():]
                if sentence:
                    yield sentence
                    first_sentence = False
                continue

            # Eager first sentence: yield at clause break if long enough
            if first_sentence and len(buffer) >= min_eager_chars:
                match = _CLAUSE_BREAK.search(buffer)
                if match:
                    sentence = buffer[:match.end()].strip()
                    buffer = buffer[match.end():]
                    if sentence:
                        yield sentence
                        first_sentence = False
                    continue

            break

    # Flush remaining buffer
    remaining = buffer.strip()
    if remaining:
        yield remaining


def split_long_sentence(sentence: str, max_chars: int) -> list[str]:
    """Split long sentence at natural break points."""
    sentence = sentence.strip()
    if not sentence or len(sentence) <= max_chars:
        return [sentence] if sentence else []

    best_pos = -1
    for match in _BREAK_POINTS.finditer(sentence, 0, max_chars):
        best_pos = match.end()

    if best_pos > 0:
        left = sentence[:best_pos].strip()
        right = sentence[best_pos:].strip()
        if left:
            return [left] + split_long_sentence(right, max_chars)

    space_pos = sentence.rfind(' ', 0, max_chars)
    if space_pos > 0:
        left = sentence[:space_pos].strip()
        right = sentence[space_pos:].strip()
        if left:
            return [left] + split_long_sentence(right, max_chars)

    left = sentence[:max_chars].strip()
    right = sentence[max_chars:].strip()
    result = [left] if left else []
    return result + split_long_sentence(right, max_chars)
