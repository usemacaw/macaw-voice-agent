"""Decode strategies for autoregressive ASR models.

Separates the decode loop control logic from model-specific inference.
Shared across models — Qwen, Whisper, and future autoregressive models
all benefit from early stopping and repetition detection.

Pattern: Strategy (GoF) — engine/model delegates stop decision to strategy.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class DecodeContext:
    """State passed to strategy at each decode step."""

    step: int
    """Current decode step (0-indexed)."""

    max_steps: int
    """Maximum allowed steps."""

    recent_tokens: list[int] = field(default_factory=list)
    """Recent token IDs for repetition detection."""


class DecodeStrategy(ABC):
    """Controls when to stop the manual decode loop.

    Models call should_stop() after each token generation step.
    """

    @abstractmethod
    def should_stop(self, token_id: int, context: DecodeContext) -> bool:
        """Returns True if decoding should stop.

        Args:
            token_id: Just-generated token ID.
            context: Current decode state.
        """


class GreedyWithEarlyStopping(DecodeStrategy):
    """Greedy decode with EOS detection and repetition detection.

    Extracted from QwenNativeStreamingSTT._run_generate():
    - Stop on EOS token
    - Stop on N consecutive identical tokens (repetition)
    - Stop at max_steps (via context)

    Args:
        eos_token_id: Token ID for end-of-sequence.
        repetition_window: Stop if last N tokens are identical. Default: 3.
    """

    def __init__(self, eos_token_id: int, repetition_window: int = 3) -> None:
        self._eos_id = eos_token_id
        self._repetition_window = repetition_window

    def should_stop(self, token_id: int, context: DecodeContext) -> bool:
        # EOS → always stop
        if token_id == self._eos_id:
            return True

        # Repetition detection: N consecutive identical tokens
        context.recent_tokens.append(token_id)
        if len(context.recent_tokens) >= self._repetition_window:
            window = context.recent_tokens[-self._repetition_window:]
            if all(t == window[0] for t in window):
                return True

        # Max steps
        if context.step >= context.max_steps - 1:
            return True

        return False
