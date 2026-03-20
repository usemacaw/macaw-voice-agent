"""macaw-qwen3-tts-streaming — True streaming TTS for Qwen3-TTS.

Combines CUDA graphs (Talker + Predictor), two-phase latency,
torch.compile decoder, and Hann crossfade for ~88ms TTFA.
"""

from macaw_tts.crossfade import HannCrossfader
from macaw_tts.sampling import sample_logits, CircularRepetitionPenalty

__all__ = [
    "HannCrossfader",
    "sample_logits",
    "CircularRepetitionPenalty",
]

__version__ = "0.1.0"
