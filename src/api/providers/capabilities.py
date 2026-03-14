"""
ProviderCapabilities — formalized provider feature flags.

Replaces scattered `if hasattr(...)` and `if provider.supports_streaming`
checks with a structured capabilities object.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TTSCapabilities:
    """What a TTS provider can do."""
    streaming: bool = False

    @classmethod
    def from_provider(cls, provider) -> TTSCapabilities:
        return cls(streaming=getattr(provider, "supports_streaming", False))


@dataclass(frozen=True)
class ASRCapabilities:
    """What an ASR provider can do."""
    streaming: bool = False

    @classmethod
    def from_provider(cls, provider) -> ASRCapabilities:
        return cls(streaming=getattr(provider, "supports_streaming", False))


@dataclass(frozen=True)
class LLMCapabilities:
    """What an LLM provider can do."""
    tool_calling: bool = True
    thinking_mode: bool = False

    @classmethod
    def from_provider(cls, provider) -> LLMCapabilities:
        return cls(
            tool_calling=hasattr(provider, "generate_stream_with_tools"),
            thinking_mode=getattr(provider, "provider_name", "") == "vllm",
        )
