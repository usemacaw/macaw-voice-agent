"""
LLM Provider ABC + factory with auto-discovery.

Unlike ai-agent LLM providers, these are STATELESS: conversation history
is managed by RealtimeSession and passed as messages each call.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import AsyncGenerator

from providers.registry import ProviderRegistry


@dataclass
class LLMStreamEvent:
    """Tagged event from LLM stream — text delta or tool call parts."""
    type: str  # "text_delta", "tool_call_start", "tool_call_delta", "tool_call_end"
    text: str = ""
    tool_call_id: str = ""
    tool_name: str = ""
    tool_arguments_delta: str = ""

_registry: ProviderRegistry[LLMProvider] = ProviderRegistry("LLM", {
    "anthropic": "providers.llm_anthropic",
    "openai": "providers.llm_openai",
    "vllm": "providers.llm_vllm",
})

# Sentence-ending punctuation
_SENTENCE_END = re.compile(r'[.!?]\s*$')
# Clause break points for eager first sentence
_CLAUSE_BREAK = re.compile(r'[,;:\u2014\u2013]\s*$')

# Break points for splitting long sentences
_BREAK_POINTS = re.compile(r'[,;:\u2014\u2013\u2015]|\.\.\.')


class LLMProvider(ABC):
    """Abstract base class for LLM providers.

    Stateless: receives full messages list each call.
    Conversation history is managed by RealtimeSession.
    """

    provider_name: str = ""

    async def connect(self) -> None:
        """Connect to the LLM service. No-op by default."""
        pass

    async def disconnect(self) -> None:
        """Disconnect from the LLM service. No-op by default."""
        pass

    @abstractmethod
    async def generate_stream(
        self,
        messages: list[dict],
        system: str = "",
        tools: list[dict] | None = None,
        temperature: float = 0.8,
        max_tokens: int = 1024,
    ) -> AsyncGenerator[str, None]:
        """Stream text tokens from the LLM.

        Args:
            messages: Full conversation messages in OpenAI format.
            system: System prompt.
            tools: Function/tool definitions.
            temperature: Sampling temperature.
            max_tokens: Max output tokens.

        Yields:
            Text chunks as they arrive.
        """
        yield ""  # pragma: no cover

    async def generate_stream_with_tools(
        self,
        messages: list[dict],
        system: str = "",
        tools: list[dict] | None = None,
        temperature: float = 0.8,
        max_tokens: int = 1024,
    ) -> AsyncGenerator[LLMStreamEvent, None]:
        """Stream LLM output as tagged events (text deltas + tool calls).

        Default implementation wraps generate_stream() for text-only output.
        Override in providers that support tool calling.
        """
        async for chunk in self.generate_stream(
            messages, system=system, tools=tools,
            temperature=temperature, max_tokens=max_tokens,
        ):
            yield LLMStreamEvent(type="text_delta", text=chunk)

    async def generate_sentences(
        self,
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

        async for chunk in self.generate_stream(
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


def register_llm_provider(name: str, cls: type[LLMProvider]) -> None:
    _registry.register(name, cls)


def create_llm_provider(name: str) -> LLMProvider:
    return _registry.create(name)
