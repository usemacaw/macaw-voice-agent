"""
Text cleaning for voice output.

Single source of truth for stripping emojis, thinking blocks,
and other non-speakable content from LLM output before TTS.
"""

from __future__ import annotations

import re

# Qwen3 thinking blocks: <think>...</think>
_THINK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)

# Unicode emoji ranges (comprehensive)
_EMOJI_RE = re.compile(
    "["
    "\U0001f600-\U0001f64f"
    "\U0001f300-\U0001f5ff"
    "\U0001f680-\U0001f6ff"
    "\U0001f1e0-\U0001f1ff"
    "\U0001f900-\U0001f9ff"
    "\U0001fa00-\U0001fa6f"
    "\U0001fa70-\U0001faff"
    "\U00002702-\U000027b0"
    "\U000024c2-\U0001f251"
    "\U0000fe0f"
    "\U0000200d"
    "]+",
)


def clean_for_voice(text: str) -> str:
    """Strip thinking blocks and emojis from LLM output for TTS."""
    text = _THINK_RE.sub("", text)
    text = _EMOJI_RE.sub("", text)
    return text.strip()


def strip_emojis(text: str) -> str:
    """Strip only emojis (no thinking block removal)."""
    return _EMOJI_RE.sub("", text)
