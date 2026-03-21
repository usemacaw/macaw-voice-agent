"""macaw-qwen3-tts-streaming — True streaming TTS for Qwen3-TTS.

Combines CUDA graphs (Talker + Predictor), two-phase latency,
torch.compile decoder, and Hann crossfade for ~88ms TTFA.
"""

from macaw_tts.audio import (
    CODEC_SAMPLE_RATE,
    INTERNAL_SAMPLE_RATE,
    SAMPLES_PER_FRAME,
    float32_to_pcm16,
    pcm16_to_float32,
    resample,
    resample_to_internal,
)
from macaw_tts.crossfade import HannCrossfader
from macaw_tts.decoder import StreamingDecoder
from macaw_tts.model import MacawTTS
from macaw_tts.sampling import CircularRepetitionPenalty, sample_logits
from macaw_tts.streaming import ChunkMetadata

__all__ = [
    # High-level API
    "MacawTTS",
    # Streaming types
    "ChunkMetadata",
    # Components
    "StreamingDecoder",
    "HannCrossfader",
    "CircularRepetitionPenalty",
    "sample_logits",
    # Audio utilities
    "CODEC_SAMPLE_RATE",
    "INTERNAL_SAMPLE_RATE",
    "SAMPLES_PER_FRAME",
    "float32_to_pcm16",
    "pcm16_to_float32",
    "resample",
    "resample_to_internal",
]

__version__ = "0.1.0"
