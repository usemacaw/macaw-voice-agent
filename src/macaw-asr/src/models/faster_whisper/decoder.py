"""Faster-Whisper decoder — CTranslate2 inference.

SRP: Runs Whisper inference via faster-whisper's transcribe().
DecodeStrategy is accepted but ignored — faster-whisper handles
its own stopping via beam search and VAD internally.
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

logger = logging.getLogger("macaw-asr.models.faster_whisper.decoder")


class FasterWhisperDecoder(IDecoder, IStreamDecoder):
    """Decoder using faster-whisper (CTranslate2).

    Uses model.transcribe() which returns a generator of segments.
    Strategy is ignored — faster-whisper handles stopping internally.
    """

    def __init__(self, model) -> None:
        self._model = model

    def decode(self, inputs: Any, strategy: DecodeStrategy | None = None) -> ModelOutput:
        """Batch inference via faster-whisper. Strategy is ignored."""
        t_total = _time.perf_counter()

        audio = inputs["audio"]
        language = inputs.get("language")

        segments, info = self._model.transcribe(
            audio,
            language=language,
            beam_size=5,
            vad_filter=True,
        )

        segment_list = list(segments)
        text = " ".join(seg.text.strip() for seg in segment_list).strip()

        total_ms = (_time.perf_counter() - t_total) * 1000
        # Use actual token count if available, otherwise word count as proxy
        n_tokens = sum(len(getattr(seg, "tokens", []) or seg.text.split()) for seg in segment_list)

        return ModelOutput(
            text=text, raw_text=text, n_tokens=max(n_tokens, 1),
            timings={
                TIMING_PREPARE_MS: getattr(inputs, "_prepare_ms", 0.0),
                TIMING_PREFILL_MS: total_ms,
                TIMING_DECODE_MS: 0.0,
                TIMING_DECODE_PER_TOKEN_MS: total_ms / max(n_tokens, 1),
                TIMING_TOTAL_MS: total_ms,
            },
        )

    def decode_stream(
        self, inputs: Any, strategy: DecodeStrategy | None = None,
    ) -> Generator[tuple[str, bool, ModelOutput | None], None, None]:
        """No token-by-token streaming. Falls back to batch."""
        output = self.decode(inputs, strategy)
        yield output.text, True, output
