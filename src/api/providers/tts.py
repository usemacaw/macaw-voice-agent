"""
TTS Provider ABC + factory with auto-discovery.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import AsyncGenerator

from providers.registry import ProviderRegistry

_registry: ProviderRegistry[TTSProvider] = ProviderRegistry("TTS", {
    "remote": "providers.tts_remote",
    "qwen": "providers.tts_qwen",
    "kokoro": "providers.tts_kokoro",
    "edge": "providers.tts_edge",
})


class TTSProvider(ABC):
    """Abstract base class for TTS (text-to-speech) providers."""

    provider_name: str = ""

    @abstractmethod
    async def synthesize(self, text: str) -> bytes:
        """Synthesize text to PCM 8kHz 16-bit mono audio.

        Args:
            text: Text to synthesize.

        Returns:
            PCM audio bytes.
        """

    async def synthesize_stream(self, text: str) -> AsyncGenerator[bytes, None]:
        """Streaming synthesis — yields audio chunks as they're generated."""
        audio = await self.synthesize(text)
        if audio:
            # Default: yield in ~100ms chunks
            chunk_size = 8000 * 2 // 10  # 100ms at 8kHz 16-bit
            for i in range(0, len(audio), chunk_size):
                yield audio[i:i + chunk_size]

    @property
    def supports_streaming(self) -> bool:
        return False

    async def connect(self) -> None:
        """Optional: connect to remote service."""

    async def warmup(self) -> None:
        """Optional: pre-warm connections to reduce first-call latency."""

    async def disconnect(self) -> None:
        """Optional: disconnect from remote service."""


def register_tts_provider(name: str, cls: type[TTSProvider]) -> None:
    _registry.register(name, cls)


def create_tts_provider(name: str) -> TTSProvider:
    return _registry.create(name)
