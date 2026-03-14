"""
ASR Provider ABC + factory with auto-discovery.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from providers.registry import ProviderRegistry

_registry: ProviderRegistry[ASRProvider] = ProviderRegistry("ASR", {
    "remote": "providers.asr_remote",
    "qwen": "providers.asr_qwen",
    "whisper": "providers.asr_whisper",
})


class ASRProvider(ABC):
    """Abstract base class for ASR (speech-to-text) providers."""

    provider_name: str = ""

    @abstractmethod
    async def transcribe(self, audio: bytes) -> str:
        """Transcribe complete audio buffer to text.

        Args:
            audio: PCM 16-bit 8kHz mono bytes.

        Returns:
            Transcribed text.
        """

    async def start_stream(self, stream_id: str) -> None:
        """Start a streaming transcription session."""
        raise NotImplementedError("Streaming not supported by this provider")

    async def feed_chunk(self, audio: bytes, stream_id: str) -> str:
        """Feed audio chunk, return partial transcript."""
        raise NotImplementedError("Streaming not supported by this provider")

    async def finish_stream(self, stream_id: str) -> str:
        """Finish streaming session, return final transcript."""
        raise NotImplementedError("Streaming not supported by this provider")

    @property
    def supports_streaming(self) -> bool:
        return False

    async def connect(self) -> None:
        """Optional: connect to remote service."""

    async def warmup(self) -> None:
        """Optional: pre-warm connections to reduce first-call latency."""

    async def disconnect(self) -> None:
        """Optional: disconnect from remote service."""


def register_asr_provider(name: str, cls: type[ASRProvider]) -> None:
    _registry.register(name, cls)


def create_asr_provider(name: str) -> ASRProvider:
    return _registry.create(name)
