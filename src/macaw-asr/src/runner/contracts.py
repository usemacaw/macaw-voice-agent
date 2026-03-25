"""Rigid contracts for the runner layer.

IEngine  — model lifecycle + transcription orchestration
ISession — per-connection streaming state
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class IEngine(ABC):
    """Contract: inference orchestrator."""

    @abstractmethod
    async def start(self) -> None: ...

    @abstractmethod
    async def stop(self) -> None: ...

    @property
    @abstractmethod
    def is_started(self) -> bool: ...

    @abstractmethod
    async def transcribe(self, pcm_data: bytes) -> str: ...

    @abstractmethod
    async def create_session(self, session_id: str = "") -> None: ...

    @abstractmethod
    async def push_audio(self, session_id: str, pcm_chunk: bytes) -> str: ...

    @abstractmethod
    async def finish_session(self, session_id: str) -> str: ...

    @abstractmethod
    async def has_session(self, session_id: str) -> bool: ...

    @abstractmethod
    def create_strategy(self): ...

    @abstractmethod
    def transcribe_stream(self, audio_model_rate, strategy):
        """Return generator of (delta, is_done, output) for SSE streaming."""


class ISession(ABC):
    """Contract: per-connection streaming session."""

    @property
    @abstractmethod
    def session_id(self) -> str: ...

    @property
    @abstractmethod
    def text(self) -> str: ...

    @abstractmethod
    async def push_audio(self, pcm_chunk: bytes) -> str: ...

    @abstractmethod
    async def finish(self) -> str: ...

    @abstractmethod
    async def cancel(self) -> None: ...
