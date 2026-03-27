"""Whisper decoder — uses HuggingFace generate() (no manual decode loop).

SRP: Runs Whisper inference via model.generate().
Whisper is encoder-decoder, not autoregressive with manual KV cache.
DecodeStrategy is NOT used — HF generate handles stopping internally.
"""

from __future__ import annotations

import logging
import time as _time
from typing import Any, Generator

from macaw_asr.decode.strategies import DecodeStrategy
from macaw_asr.models.contracts import IDecoder, IStreamDecoder
from macaw_asr.models.types import (
    TIMING_DECODE_MS, TIMING_DECODE_PER_TOKEN_MS,
    TIMING_PREFILL_MS, TIMING_PREPARE_MS, TIMING_TOTAL_MS,
    ModelOutput,
)

logger = logging.getLogger("macaw-asr.models.whisper.decoder")


class WhisperDecoder(IDecoder, IStreamDecoder):
    """Whisper decoder using HuggingFace generate().

    Unlike Qwen (manual decode loop), Whisper uses model.generate()
    which handles encoder + decoder + beam search internally.
    """

    def __init__(self, model, processor, device: str) -> None:
        self._model = model
        self._processor = processor
        self._device = device

    def decode(self, inputs: Any, strategy: DecodeStrategy | None = None) -> ModelOutput:
        """Batch inference via HF generate(). Strategy is ignored."""
        import torch

        t_total = _time.perf_counter()
        input_features = inputs["input_features"]
        forced_decoder_ids = inputs.get("forced_decoder_ids")

        # Generate
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t_gen = _time.perf_counter()

        generate_kwargs = {}
        if forced_decoder_ids:
            generate_kwargs["forced_decoder_ids"] = forced_decoder_ids

        with torch.no_grad():
            predicted_ids = self._model.generate(
                input_features, **generate_kwargs
            )

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        gen_ms = (_time.perf_counter() - t_gen) * 1000

        # Decode tokens to text
        text = self._processor.batch_decode(predicted_ids, skip_special_tokens=True)[0].strip()
        n_tokens = predicted_ids.shape[-1]
        total_ms = (_time.perf_counter() - t_total) * 1000

        return ModelOutput(
            text=text, raw_text=text, n_tokens=n_tokens,
            timings={
                TIMING_PREPARE_MS: getattr(inputs, "_prepare_ms", 0.0),
                TIMING_PREFILL_MS: gen_ms,  # Whisper: prefill ≈ full generate
                TIMING_DECODE_MS: 0.0,  # No separate decode loop
                TIMING_DECODE_PER_TOKEN_MS: gen_ms / n_tokens if n_tokens > 0 else 0,
                TIMING_TOTAL_MS: total_ms,
            },
        )

    def decode_stream(
        self, inputs: Any, strategy: DecodeStrategy | None = None,
    ) -> Generator[tuple[str, bool, ModelOutput | None], None, None]:
        """Whisper doesn't support token-by-token streaming.

        Falls back to batch: generate full text, yield as single done event.
        """
        output = self.decode(inputs, strategy)
        yield output.text, True, output
