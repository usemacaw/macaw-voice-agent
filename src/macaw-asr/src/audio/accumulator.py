"""Chunk accumulator for streaming ASR.

Manages audio buffer with threshold-based triggering for background
computation. Extracted from the streaming state pattern in
QwenNativeStreamingSTT._StreamState.
"""

from __future__ import annotations

import numpy as np


class ChunkAccumulator:
    """Accumulates audio chunks and triggers when threshold is reached.

    Thread-safe snapshot via copy for background computation.

    Usage:
        acc = ChunkAccumulator(trigger_samples=16000)  # 1s at 16kHz
        for chunk in audio_stream:
            should_trigger = acc.push(chunk)
            if should_trigger:
                snapshot = acc.snapshot()
                # send snapshot to background compute
        final_audio = acc.get_all()
    """

    def __init__(self, trigger_samples: int) -> None:
        """
        Args:
            trigger_samples: Number of samples to accumulate before triggering.
        """
        if trigger_samples <= 0:
            raise ValueError(f"trigger_samples must be positive, got {trigger_samples}")
        self._trigger_samples = trigger_samples
        self._buffer: np.ndarray = np.zeros(0, dtype=np.float32)
        self._samples_since_trigger: int = 0

    def push(self, samples: np.ndarray) -> bool:
        """Append samples to buffer. Returns True if trigger threshold reached.

        Args:
            samples: Float32 audio samples (already at model sample rate).

        Returns:
            True if accumulated samples since last trigger >= trigger_samples.
        """
        if samples.size == 0:
            return False

        if self._buffer.size == 0:
            self._buffer = samples.astype(np.float32)
        else:
            self._buffer = np.concatenate([self._buffer, samples.astype(np.float32)])

        self._samples_since_trigger += len(samples)

        if self._samples_since_trigger >= self._trigger_samples:
            self._samples_since_trigger = 0
            return True
        return False

    def snapshot(self) -> np.ndarray:
        """Return a copy of the current buffer for safe background processing."""
        return self._buffer.copy()

    def get_all(self) -> np.ndarray:
        """Return the full accumulated buffer (no copy — final use only)."""
        return self._buffer

    @property
    def total_samples(self) -> int:
        """Total samples accumulated so far."""
        return self._buffer.size

    @property
    def is_empty(self) -> bool:
        return self._buffer.size == 0

    def reset(self) -> None:
        """Clear the buffer and reset trigger counter."""
        self._buffer = np.zeros(0, dtype=np.float32)
        self._samples_since_trigger = 0
