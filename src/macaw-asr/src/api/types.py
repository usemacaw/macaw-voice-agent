"""API types for macaw-asr (Pydantic models).

Wire format matches Ollama conventions:
- Durations in nanoseconds (int)
- Errors as {"error": "message"}
- Streaming via NDJSON (application/x-ndjson)
- Optional `stream` parameter (default: false for transcribe)
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


# ==================== Transcribe (Ollama /api/generate equivalent) ====================


# ==================== Model Info (used by manifest registry) ====================


class ModelInfo(BaseModel):
    model_config = {"protected_namespaces": ()}
    name: str = ""
    model_id: str = ""
    size_bytes: int = 0
    family: str = ""
    parameters: str = ""


# ==================== Transcribe (Ollama /api/generate equivalent) ====================


class TranscribeRequest(BaseModel):
    model: str = ""
    audio: str = ""  # base64-encoded PCM16
    language: str = ""
    stream: bool = False
    keep_alive: str = ""  # "5m", "10s", "0" = unload immediately
    options: dict[str, Any] = Field(default_factory=dict)


class TranscribeResponse(BaseModel):
    model: str = ""
    created_at: str = ""
    text: str = ""
    done: bool = False
    total_duration: int = 0  # nanoseconds
    load_duration: int = 0
    prompt_eval_duration: int = 0  # prepare_inputs time
    eval_count: int = 0  # tokens generated
    eval_duration: int = 0  # decode time


# ==================== Show (Ollama /api/show) ====================


class ShowRequest(BaseModel):
    model: str


class ShowResponse(BaseModel):
    model_config = {"protected_namespaces": ()}
    model_info: dict[str, Any] = Field(default_factory=dict)
    details: dict[str, Any] = Field(default_factory=dict)


# ==================== Pull (Ollama /api/pull) ====================


class PullRequest(BaseModel):
    model: str
    stream: bool = True


class PullResponse(BaseModel):
    status: str = ""
    digest: str = ""
    total: int = 0
    completed: int = 0


# ==================== Delete (Ollama /api/delete) ====================


class DeleteRequest(BaseModel):
    model: str


# ==================== List Models (Ollama /api/tags) ====================


class ModelDetails(BaseModel):
    family: str = ""
    parameter_size: str = ""
    quantization_level: str = ""


class ModelEntry(BaseModel):
    name: str = ""
    model: str = ""
    size: int = 0
    details: ModelDetails = Field(default_factory=ModelDetails)


class ListResponse(BaseModel):
    models: list[ModelEntry] = Field(default_factory=list)


# ==================== List Running (Ollama /api/ps) ====================


class RunningModel(BaseModel):
    name: str = ""
    model: str = ""
    size: int = 0
    size_vram: int = 0
    expires_at: str = ""


class PsResponse(BaseModel):
    models: list[RunningModel] = Field(default_factory=list)


# ==================== Version (Ollama /api/version) ====================


class VersionResponse(BaseModel):
    version: str = ""


# ==================== Stream Start/Push/Finish ====================


class StreamStartRequest(BaseModel):
    model: str = ""
    language: str = ""
    session_id: str = ""


class StreamPushRequest(BaseModel):
    session_id: str
    audio: str = ""  # base64
    end_of_stream: bool = False


class StreamFinishResponse(BaseModel):
    text: str = ""
    session_id: str = ""
    model: str = ""
    total_duration: int = 0
