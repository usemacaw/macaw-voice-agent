"""
ASR Provider ABC + factory with auto-discovery.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from providers.registry import ProviderRegistry

_registry: ProviderRegistry[ASRProvider] = ProviderRegistry("ASR", {
    "remote": "providers.asr_remote",
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

    async def feed_chunk_with_partial(self, audio: bytes, stream_id: str) -> str | None:
        """Feed audio chunk and return partial transcript if available.

        Returns None if no new partial is available yet, or the updated
        partial text. Called every ~100ms with new audio during speech.

        Override this for true streaming ASR with incremental partial results.
        Default implementation delegates to feed_chunk() for backward compat.
        """
        if self.supports_streaming:
            result = await self.feed_chunk(audio, stream_id)
            return result if result else None
        return None

    async def finish_stream(self, stream_id: str) -> str:
        """Finish streaming session, return final transcript."""
        raise NotImplementedError("Streaming not supported by this provider")

    @property
    def supports_streaming(self) -> bool:
        return False

    @property
    def supports_partial_results(self) -> bool:
        """True if provider can emit partial transcripts during speech."""
        return False

    async def connect(self) -> None:
        """Optional: connect to remote service."""

    async def warmup(self) -> None:
        """Optional: pre-warm connections to reduce first-call latency."""

    async def disconnect(self) -> None:
        """Optional: disconnect from remote service."""

    async def health_check(self) -> bool:
        """Return True if provider is healthy. Override for custom checks."""
        return True


def register_asr_provider(name: str, cls: type[ASRProvider]) -> None:
    _registry.register(name, cls)


def create_asr_provider(name: str) -> ASRProvider:
    return _registry.create(name)
