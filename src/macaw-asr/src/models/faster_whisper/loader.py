"""Faster-Whisper model loader — CTranslate2 weight lifecycle.

SRP: Only loads/unloads faster-whisper model.
No PyTorch dependency — uses CTranslate2 C++ runtime.
"""

from __future__ import annotations

import logging

from macaw_asr.config import EngineConfig
from macaw_asr.models.contracts import IModelLoader

logger = logging.getLogger("macaw-asr.models.faster_whisper.loader")

_COMPUTE_TYPE_MAP = {
    "float16": "float16",
    "bfloat16": "float16",  # CTranslate2 doesn't support bfloat16
    "float32": "float32",
    "int8": "int8",
    "int8_float16": "int8_float16",
}

# Map HuggingFace IDs to faster-whisper size names.
# faster-whisper accepts short names ("tiny", "small") and auto-downloads
# CTranslate2 models from Systran/faster-whisper-*.
_HF_TO_SIZE: dict[str, str] = {
    "openai/whisper-tiny": "tiny",
    "openai/whisper-small": "small",
    "openai/whisper-medium": "medium",
    "openai/whisper-large-v3": "large-v3",
    "openai/whisper-large": "large-v3",
}


def _parse_device(device_str: str) -> tuple[str, int]:
    """Parse 'cuda:0' → ('cuda', 0), 'cpu' → ('cpu', 0)."""
    if device_str.startswith("cuda"):
        parts = device_str.split(":")
        index = int(parts[1]) if len(parts) > 1 else 0
        return "cuda", index
    return "cpu", 0


class FasterWhisperModelLoader(IModelLoader):
    """Loads Whisper via faster-whisper (CTranslate2 runtime)."""

    def __init__(self) -> None:
        self.model = None

    def load(self, config: EngineConfig) -> None:
        from faster_whisper import WhisperModel

        device, device_index = _parse_device(config.device)
        compute_type = _COMPUTE_TYPE_MAP.get(config.dtype, "float16")

        # Resolve HuggingFace ID to faster-whisper size name
        model_size = _HF_TO_SIZE.get(config.model_id, config.model_id)

        self.model = WhisperModel(
            model_size,
            device=device,
            device_index=device_index,
            compute_type=compute_type,
        )

        logger.info(
            "Faster-Whisper loaded: model=%s (size=%s) device=%s:%d compute_type=%s",
            config.model_id, model_size, device, device_index, compute_type,
        )

    def unload(self) -> None:
        if self.model is not None:
            model = self.model
            self.model = None
            del model
            logger.info("Faster-Whisper unloaded")

    def is_loaded(self) -> bool:
        return self.model is not None
