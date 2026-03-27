"""Qwen3-ASR decoder — unified decode loop for batch and streaming.

DRY: Single _decode_loop() generator consumed by both decode() and decode_stream().
No duplicated prefill/decode code.
"""

from __future__ import annotations

import logging
import time as _time
from typing import Any, Generator

from macaw_asr.decode.strategies import DecodeContext, DecodeStrategy, GreedyWithEarlyStopping
from macaw_asr.models.contracts import IDecoder, IStreamDecoder
from macaw_asr.models.types import (
    TIMING_DECODE_MS, TIMING_DECODE_PER_TOKEN_MS,
    TIMING_PREFILL_MS, TIMING_PREPARE_MS, TIMING_TOTAL_MS,
    ModelOutput,
)

logger = logging.getLogger("macaw-asr.models.qwen.decoder")


def _sync_gpu():
    """Synchronize CUDA if available. No-op on CPU."""
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except ImportError:
        pass


class QwenDecoder(IDecoder, IStreamDecoder):
    """Autoregressive decoder with KV cache.

    Single _decode_loop() generator (Template Method) yields each token.
    decode() and decode_stream() both consume this — zero duplication.
    """

    def __init__(self, thinker, processor, eos_id: int, max_new_tokens: int) -> None:
        self._thinker = thinker
        self._processor = processor
        self._eos_id = eos_id
        self._max_tokens = max_new_tokens

    def decode(self, inputs: Any, strategy: DecodeStrategy | None = None) -> ModelOutput:
        """Batch: collect all tokens, return complete text."""
        import torch

        strategy = strategy or self._default_strategy()
        t_total = _time.perf_counter()

        all_tokens = None
        pf_ms = dc_ms = 0.0

        for token_ids, prefill_ms, decode_ms, is_last in self._decode_loop(inputs, strategy):
            all_tokens = token_ids
            pf_ms = prefill_ms
            dc_ms = decode_ms

        total_ms = (_time.perf_counter() - t_total) * 1000
        n_tok = len(all_tokens) if all_tokens is not None else 0
        decoded = self._processor.batch_decode([all_tokens], skip_special_tokens=True)[0] if all_tokens is not None else ""

        return ModelOutput(
            text=decoded, raw_text=decoded, n_tokens=n_tok,
            timings=self._build_timings(inputs, pf_ms, dc_ms, n_tok, total_ms),
        )

    def decode_stream(
        self, inputs: Any, strategy: DecodeStrategy | None = None,
    ) -> Generator[tuple[str, bool, ModelOutput | None], None, None]:
        """Streaming: yield (delta, is_done, output) per token."""
        strategy = strategy or self._default_strategy()
        t_total = _time.perf_counter()
        text_so_far = ""

        for token_ids, pf_ms, dc_ms, is_last in self._decode_loop(inputs, strategy):
            full_text = self._processor.batch_decode([token_ids], skip_special_tokens=True)[0]
            delta = full_text[len(text_so_far):]
            text_so_far = full_text

            if is_last:
                total_ms = (_time.perf_counter() - t_total) * 1000
                n_tok = len(token_ids)
                output = ModelOutput(
                    text=text_so_far, raw_text=text_so_far, n_tokens=n_tok,
                    timings=self._build_timings(inputs, pf_ms, dc_ms, n_tok, total_ms),
                )
                yield delta, True, output
                return

            if delta:
                yield delta, False, None

    # ==================== Single Decode Loop (DRY) ====================

    def _decode_loop(self, inputs: Any, strategy: DecodeStrategy):
        """Unified decode loop. Yields (all_token_ids, prefill_ms, decode_ms, is_last) per step.

        Template Method: callers decide how to consume tokens.
        """
        import torch

        thinker = self._thinker
        max_tokens = self._max_tokens

        # Prefill
        _sync_gpu()
        t_pf = _time.perf_counter()
        with torch.no_grad():
            out = thinker.generate(**inputs, max_new_tokens=1, return_dict_in_generate=True, use_cache=True)
        _sync_gpu()
        pf_ms = (_time.perf_counter() - t_pf) * 1000

        kv = out.past_key_values
        gen_ids = torch.zeros(max_tokens, dtype=torch.long, device=out.sequences.device)
        gen_ids[0] = out.sequences[0, -1]
        last_token = out.sequences[:, -1:]

        # Decode step-by-step (no sync per token — avoids draining GPU pipeline)
        _sync_gpu()
        t_dc = _time.perf_counter()
        n_tok = 0
        context = DecodeContext(step=0, max_steps=max_tokens - 1)

        for step in range(max_tokens - 1):
            with torch.no_grad():
                out_step = thinker(input_ids=last_token, past_key_values=kv, use_cache=True)
            logits = out_step.logits[:, -1, :]
            next_token = torch.argmax(logits, dim=-1, keepdim=True)
            tok_id = next_token.item()

            n_tok += 1
            gen_ids[n_tok] = tok_id
            last_token = next_token
            kv = out_step.past_key_values

            context.step = step
            done = strategy.should_stop(tok_id, context)

            dc_ms = (_time.perf_counter() - t_dc) * 1000
            yield gen_ids[:n_tok + 1], pf_ms, dc_ms, done

            if done:
                return

        # Max tokens reached
        _sync_gpu()
        dc_ms = (_time.perf_counter() - t_dc) * 1000
        yield gen_ids[:n_tok + 1], pf_ms, dc_ms, True

    # ==================== Helpers ====================

    def _default_strategy(self) -> DecodeStrategy:
        return GreedyWithEarlyStopping(eos_token_id=self._eos_id, repetition_window=3)

    def _build_timings(self, inputs, pf_ms, dc_ms, n_tok, total_ms) -> dict[str, float]:
        return {
            TIMING_PREPARE_MS: getattr(inputs, "_prepare_ms", 0.0),
            TIMING_PREFILL_MS: pf_ms,
            TIMING_DECODE_MS: dc_ms,
            TIMING_DECODE_PER_TOKEN_MS: dc_ms / n_tok if n_tok > 0 else 0,
            TIMING_TOTAL_MS: total_ms,
        }
