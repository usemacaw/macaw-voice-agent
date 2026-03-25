"""API types for macaw-asr.

Request/response types shared between client and server.
Equivalent to Ollama's api/types.go — single source of truth
for the wire format.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# ==================== Model Management ====================


@dataclass
class PullRequest:
    """Request to download a model."""

    model: str
    """Model identifier (e.g. 'Qwen/Qwen3-ASR-0.6B')."""


@dataclass
class PullResponse:
    """Progress update during model download."""

    status: str
    """Current status ('downloading', 'verifying', 'complete', 'error')."""

    digest: str = ""
    """Content digest (sha256)."""

    total: int = 0
    """Total bytes to download."""

    completed: int = 0
    """Bytes downloaded so far."""


@dataclass
class ModelInfo:
    """Information about a locally available model."""

    name: str
    """Registry key (e.g. 'qwen')."""

    model_id: str
    """HuggingFace model ID or local path."""

    size_bytes: int = 0
    """Total size on disk."""

    family: str = ""
    """Model family (e.g. 'qwen3-asr', 'whisper')."""

    parameters: str = ""
    """Parameter count (e.g. '0.6B')."""


# ==================== Transcription ====================


@dataclass
class TranscribeRequest:
    """Request for batch transcription."""

    model: str = ""
    """Model to use. Empty = default loaded model."""

    audio: bytes = b""
    """PCM 16-bit audio at input_sample_rate."""

    language: str = ""
    """ISO-639-1 language code. Empty = model default."""

    options: dict[str, Any] = field(default_factory=dict)
    """Model-specific options (temperature, etc.)."""


@dataclass
class TranscribeResponse:
    """Result of a transcription."""

    text: str = ""
    """Transcribed text."""

    model: str = ""
    """Model that performed the transcription."""

    total_duration_ms: float = 0.0
    """Total time including preprocessing."""

    load_duration_ms: float = 0.0
    """Time spent loading model (0 if already loaded)."""

    preprocess_duration_ms: float = 0.0
    """Audio preprocessing time."""

    inference_duration_ms: float = 0.0
    """Model inference time."""

    tokens: int = 0
    """Tokens generated."""


# ==================== Streaming ====================


@dataclass
class StreamStartRequest:
    """Request to start a streaming session."""

    model: str = ""
    """Model to use."""

    session_id: str = ""
    """Client-provided session ID. Auto-generated if empty."""

    language: str = ""
    """ISO-639-1 language code."""


@dataclass
class StreamChunk:
    """Audio chunk for streaming transcription."""

    session_id: str = ""
    """Session to push audio to."""

    audio: bytes = b""
    """PCM 16-bit audio chunk."""

    end_of_stream: bool = False
    """If True, finish the session and return final text."""


@dataclass
class StreamFinishResponse:
    """Final result of a streaming session."""

    text: str = ""
    """Final transcribed text."""

    session_id: str = ""
    model: str = ""

    total_duration_ms: float = 0.0
    preprocess_duration_ms: float = 0.0
    inference_duration_ms: float = 0.0
    bg_decode_count: int = 0
    tokens: int = 0
