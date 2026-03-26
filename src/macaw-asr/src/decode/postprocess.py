"""Text postprocessing for ASR model output.

Handles model-specific tag stripping and cleanup.
"""

from __future__ import annotations

import re

_ASR_TAG = "<asr_text>"
_LANGUAGE_PREFIX_RE = re.compile(r"^language\s+\w+\s*")
# NeMo special tokens: <|en|>, <|pnc|>, <|endoftext|>, <|emo:undefined|>, etc.
_NEMO_SPECIAL_RE = re.compile(r"<\|[^|]*\|>")


def clean_asr_text(raw: str) -> str:
    """Clean raw ASR model output into usable text.

    Strips:
    - <asr_text> tags (Qwen format)
    - "language X" prefixes
    - NeMo special tokens (<|en|>, <|endoftext|>, <|pnc|>, etc.)
    - Leading/trailing whitespace

    Args:
        raw: Raw decoded text from the model.

    Returns:
        Cleaned transcription text.
    """
    if not raw:
        return ""

    text = raw

    # Strip <asr_text> tag and everything before it
    if _ASR_TAG in text:
        text = text.split(_ASR_TAG, 1)[1]

    # Strip NeMo special tokens (Canary, FastConformer MultiTask)
    text = _NEMO_SPECIAL_RE.sub("", text)

    # Strip "language Portuguese" prefix if present
    text = _LANGUAGE_PREFIX_RE.sub("", text)

    return text.strip()
