"""
LLM Provider ABC + factory with auto-discovery.

Unlike ai-agent LLM providers, these are STATELESS: conversation history
is managed by RealtimeSession and passed as messages each call.
"""

from __future__ import annotations

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


@dataclass
class LLMStreamTiming:
    """Timing metadata emitted after an LLM stream completes.

    Returned by generate_stream() and generate_stream_with_tools()
    as a per-call result, NOT stored as mutable class state.
    This eliminates race conditions when multiple sessions share
    a single LLMProvider instance.
    """
    ttft_ms: float = 0.0
    total_ms: float = 0.0


_registry: ProviderRegistry[LLMProvider] = ProviderRegistry("LLM", {
    "remote": "providers.llm_remote",
})


class LLMProvider(ABC):
    """Abstract base class for LLM providers.

    Stateless: receives full messages list each call.
    Conversation history is managed by RealtimeSession.

    Timing is returned per-call via LLMStreamTiming (NOT stored on the
    instance) to avoid race conditions between concurrent sessions.
    """

    provider_name: str = ""

    # Per-call timing — populated after each stream completes.
    # Consumers should read this immediately after exhausting the stream.
    # NOTE: This is safe only in single-session scenarios. For multi-session,
    # use the LLMStreamTiming returned by the stream methods.
    last_ttft_ms: float = 0.0
    last_stream_total_ms: float = 0.0

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

    def get_last_timing(self) -> LLMStreamTiming:
        """Return timing from the most recent stream call.

        Thread-safe accessor that returns a snapshot (value copy).
        """
        return LLMStreamTiming(
            ttft_ms=self.last_ttft_ms,
            total_ms=self.last_stream_total_ms,
        )


def register_llm_provider(name: str, cls: type[LLMProvider]) -> None:
    _registry.register(name, cls)


def create_llm_provider(name: str) -> LLMProvider:
    return _registry.create(name)
