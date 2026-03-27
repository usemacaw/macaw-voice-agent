"""Shared types for model layer.

Value objects shared across all model implementations.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# Standardized timing keys
TIMING_PREPARE_MS = "prepare_ms"
TIMING_PREFILL_MS = "prefill_ms"
TIMING_DECODE_MS = "decode_ms"
TIMING_DECODE_PER_TOKEN_MS = "decode_per_token_ms"
TIMING_TOTAL_MS = "total_ms"


@dataclass
class ModelOutput:
    """Result of a single ASR inference."""

    text: str
    raw_text: str
    n_tokens: int
    timings: dict[str, float] = field(default_factory=dict)


@dataclass
class TranscribeResult:
    """Result of transcription with full timing breakdown."""

    text: str
    timings: dict[str, float] = field(default_factory=dict)


class InputsWrapper:
    """Wraps a dict to support **splat like HuggingFace BatchEncoding."""

    def __init__(self, data: dict) -> None:
        self._data = data
        self._prepare_ms: float = 0.0

    def __getitem__(self, key):
        return self._data[key]

    def __contains__(self, key):
        return key in self._data

    def keys(self):
        return self._data.keys()

    def items(self):
        return self._data.items()

    def values(self):
        return self._data.values()

    def get(self, key, default=None):
        return self._data.get(key, default)
