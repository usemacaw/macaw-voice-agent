"""Hann crossfade between streaming audio chunks.

Based on rekuenkdr implementation. Eliminates clicks/pops at chunk boundaries.
Uses Hann window (cos²) for perfect energy preservation.
"""

from __future__ import annotations

import numpy as np


class HannCrossfader:
    """Stateful crossfader using Hann window between streaming audio chunks.

    Pre-computes fade curves on init (zero per-chunk cost).
    Maintains previous chunk tail for overlap-add blending.

    Args:
        overlap_samples: Number of samples to overlap (512 = ~21ms @ 24kHz).
    """

    def __init__(self, overlap_samples: int = 512):
        if overlap_samples < 0:
            raise ValueError(f"overlap_samples must be >= 0, got {overlap_samples}")

        self._overlap = overlap_samples
        self._prev_tail: np.ndarray | None = None

        # Pre-compute Hann fade curves (done once, reused for every chunk)
        if overlap_samples > 0:
            t = np.arange(overlap_samples, dtype=np.float32) / max(overlap_samples - 1, 1)
            self._fade_in = 0.5 * (1.0 - np.cos(np.pi * t))  # 0 → 1
            self._fade_out = 1.0 - self._fade_in              # 1 → 0
        else:
            self._fade_in = np.array([], dtype=np.float32)
            self._fade_out = np.array([], dtype=np.float32)

    def process(
        self,
        chunk: np.ndarray,
        is_first: bool = False,
        is_last: bool = False,
    ) -> np.ndarray:
        """Apply crossfade to chunk. Returns processed audio.

        Args:
            chunk: Audio samples (float32).
            is_first: Apply fade-in at start (prevents startup click).
            is_last: Apply fade-out at end (prevents termination click).

        Returns:
            Processed chunk with crossfade applied.
        """
        if len(chunk) == 0:
            return chunk

        ov = self._overlap
        result = chunk.copy()

        # First chunk: apply fade-in to prevent startup pop
        if is_first and ov > 0:
            fade_len = min(ov, len(result))
            result[:fade_len] *= self._fade_in[:fade_len]

        # Crossfade with previous chunk's tail
        if self._prev_tail is not None and ov > 0:
            n = min(ov, len(self._prev_tail), len(result))
            if n > 0:
                blended = (
                    self._prev_tail[-n:] * self._fade_out[-n:]
                    + result[:n] * self._fade_in[-n:]
                )
                result[:n] = blended

        # Save tail for next crossfade
        if ov > 0 and not is_last:
            self._prev_tail = result[-ov:].copy() if len(result) >= ov else result.copy()
            # Trim the tail region (will be replaced by next chunk's crossfade)
            result = result[:-ov] if len(result) > ov else result
        else:
            self._prev_tail = None

        # Last chunk: apply fade-out to prevent termination pop
        if is_last and ov > 0:
            fade_len = min(ov, len(result))
            result[-fade_len:] *= self._fade_out[-fade_len:]

        return result

    def drain(self) -> np.ndarray:
        """Return remaining crossfade tail with fade-out applied.

        Called at end-of-stream when no more chunks will follow, to flush the
        overlap samples that process() withheld from the last intermediate chunk.
        Without drain(), the last ~overlap_samples are silently lost.

        Returns:
            Float32 audio with fade-out, or empty array if no tail.
        """
        if self._prev_tail is None:
            return np.array([], dtype=np.float32)

        tail = self._prev_tail.copy()
        self._prev_tail = None

        if self._overlap > 0:
            fade_len = min(self._overlap, len(tail))
            tail[-fade_len:] *= self._fade_out[-fade_len:]

        return tail

    def reset(self) -> None:
        """Clear state between utterances."""
        self._prev_tail = None

    @property
    def overlap_samples(self) -> int:
        return self._overlap

    @property
    def has_pending_tail(self) -> bool:
        """True if there are unreturned overlap samples from a previous process() call."""
        return self._prev_tail is not None
