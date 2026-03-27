"""Parakeet decoder — uses NeMo model.transcribe() (no manual decode loop).

SRP: Runs NeMo inference. No DecodeStrategy needed — NeMo handles
CTC/TDT/RNNT decoding internally.
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

logger = logging.getLogger("macaw-asr.models.parakeet.decoder")


def _extract_text(output) -> str:
    """Extract text from NeMo transcribe output (handles various formats).

    NeMo v3 returns: (['text1'], ['text1']) — tuple of two lists
    NeMo v2 returns: [Hypothesis(text='...')] — list of Hypothesis objects
    """
    # Tuple: NeMo v3 returns (hypotheses_list, hypotheses_list)
    if isinstance(output, tuple) and len(output) > 0:
        output = output[0]  # Take first element of tuple

    if isinstance(output, list) and len(output) > 0:
        item = output[0]
        # NeMo Hypothesis object
        if hasattr(item, "text"):
            return item.text.strip()
        # Raw string
        if isinstance(item, str):
            return item.strip()
        # List of strings (NeMo v3)
        if isinstance(item, list) and len(item) > 0:
            return str(item[0]).strip()

    if isinstance(output, str):
        return output.strip()
    return str(output).strip()


class ParakeetDecoder(IDecoder, IStreamDecoder):
    """Parakeet decoder via NeMo model.transcribe().

    No manual decode loop. NeMo handles CTC/TDT/RNNT internally.
    DecodeStrategy is ignored.
    """

    def __init__(self, model) -> None:
        self._model = model

    def decode(self, inputs: Any, strategy: DecodeStrategy | None = None) -> ModelOutput:
        """Batch inference via NeMo transcribe(). Strategy is ignored."""
        import torch

        t_total = _time.perf_counter()
        audio = inputs["audio"]

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t_gen = _time.perf_counter()

        # MultiTask models (Canary) need source_lang/target_lang
        kwargs = {"batch_size": 1}
        language = inputs.get("language")
        if language and hasattr(self._model, "cfg") and hasattr(self._model.cfg, "target_lang"):
            kwargs["source_lang"] = language
            kwargs["target_lang"] = language

        with torch.no_grad():
            output = self._model.transcribe([audio], **kwargs)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        gen_ms = (_time.perf_counter() - t_gen) * 1000

        text = _extract_text(output)
        total_ms = (_time.perf_counter() - t_total) * 1000

        return ModelOutput(
            text=text, raw_text=text,
            n_tokens=len(text.split()),  # word count as proxy
            timings={
                TIMING_PREPARE_MS: getattr(inputs, "_prepare_ms", 0.0),
                TIMING_PREFILL_MS: gen_ms,  # Parakeet: single-pass inference
                TIMING_DECODE_MS: 0.0,
                TIMING_DECODE_PER_TOKEN_MS: 0.0,
                TIMING_TOTAL_MS: total_ms,
            },
        )

    def decode_stream(
        self, inputs: Any, strategy: DecodeStrategy | None = None,
    ) -> Generator[tuple[str, bool, ModelOutput | None], None, None]:
        """Parakeet doesn't support token-by-token streaming.
        Fallback: generate full text, yield as single done event."""
        output = self.decode(inputs, strategy)
        yield output.text, True, output
