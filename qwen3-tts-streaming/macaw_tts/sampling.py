"""Token sampling and repetition penalty for streaming TTS.

Based on faster-qwen3-tts sampling.py + rekuenkdr circular buffer GPU.
"""

from __future__ import annotations

from typing import Iterable, Optional

import torch
import torch.nn.functional as F


def sample_logits(
    logits: torch.Tensor,
    *,
    temperature: float = 0.9,
    top_k: int = 50,
    top_p: float = 1.0,
    do_sample: bool = True,
    suppress_mask: Optional[torch.Tensor] = None,
    suppress_tokens: Optional[Iterable[int]] = None,
) -> torch.Tensor:
    """Sample a token from logits.

    Order: suppress -> temperature -> top-k -> top-p -> sample.

    Args:
        logits: [1, vocab] or [vocab] tensor.
        temperature: Sampling temperature (0.9 default for TTS).
        top_k: Top-K filtering (50 default).
        top_p: Nucleus sampling threshold (1.0 = disabled).
        do_sample: If False, use argmax (greedy).
        suppress_mask: Boolean mask of tokens to suppress.
        suppress_tokens: Explicit list of token IDs to suppress.

    Returns:
        Sampled token ID tensor.
    """
    logits = logits.clone()

    if suppress_mask is not None:
        logits[..., suppress_mask] = float("-inf")
    if suppress_tokens:
        logits[..., list(suppress_tokens)] = float("-inf")

    if not do_sample:
        return torch.argmax(logits, dim=-1)

    logits = logits / temperature

    if top_k > 0:
        k = min(top_k, logits.size(-1))
        topk_vals, _ = torch.topk(logits, k)
        logits = torch.where(
            logits < topk_vals[..., -1:],
            torch.full_like(logits, float("-inf")),
            logits,
        )

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        probs = F.softmax(sorted_logits, dim=-1)
        cumulative_probs = torch.cumsum(probs, dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 0] = False
        sorted_logits[sorted_indices_to_remove] = float("-inf")
        logits = torch.full_like(logits, float("-inf"))
        logits.scatter_(-1, sorted_indices, sorted_logits)

    return torch.multinomial(F.softmax(logits, dim=-1), 1).squeeze(-1)


class CircularRepetitionPenalty:
    """GPU-resident circular buffer for repetition penalty.

    Based on rekuenkdr implementation. O(1) per update, no CPU<->GPU sync.

    Args:
        window: Number of recent tokens to track (256 default).
        penalty: HF-style penalty (>1.0 reduces repeated token probability).
        vocab_size: Codec vocabulary size (3072 for Qwen3-TTS).
        device: Torch device.
    """

    def __init__(
        self,
        window: int = 256,
        penalty: float = 1.05,
        vocab_size: int = 3072,
        device: str | torch.device = "cpu",
    ):
        self.window = window
        self.penalty = penalty
        self.vocab_size = vocab_size
        self.device = torch.device(device)

        # Circular buffer: fill with vocab_size (invalid ID = no penalty)
        self._history = torch.full(
            (1, window), vocab_size, dtype=torch.long, device=self.device
        )
        self._step = 0

    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        """Apply penalty to logits in-place. Returns modified logits."""
        if self.penalty == 1.0:
            return logits

        # Build presence mask from circular buffer
        presence = torch.zeros(
            1, self.vocab_size + 1, dtype=torch.bool, device=logits.device
        )
        presence.scatter_(1, self._history, True)
        penalty_mask = presence[:, : self.vocab_size]

        penalized = torch.where(
            logits > 0,
            logits / self.penalty,
            logits * self.penalty,
        )
        return torch.where(penalty_mask, penalized, logits)

    def update(self, token: torch.Tensor) -> None:
        """Add token to circular buffer. Pure tensor op, no CPU↔GPU sync."""
        pos = self._step % self.window
        t = token.view(-1)[0] if token.dim() > 0 else token
        self._history[0, pos] = t
        self._step += 1

    def reset(self) -> None:
        """Clear history for new sequence."""
        self._history.fill_(self.vocab_size)
        self._step = 0
