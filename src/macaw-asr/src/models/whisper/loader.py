"""Whisper model loader — weight lifecycle management.

SRP: Only loads/unloads HuggingFace Whisper model weights.
Supports: tiny, small, medium, large variants.
"""

from __future__ import annotations

import logging

from macaw_asr.config import EngineConfig
from macaw_asr.models.contracts import IModelLoader

logger = logging.getLogger("macaw-asr.models.whisper.loader")

# Whisper model ID mapping: short name → HuggingFace ID
WHISPER_VARIANTS = {
    "openai/whisper-tiny": "openai/whisper-tiny",
    "openai/whisper-small": "openai/whisper-small",
    "openai/whisper-medium": "openai/whisper-medium",
    "openai/whisper-large-v3": "openai/whisper-large-v3",
    "openai/whisper-large-v3-turbo": "openai/whisper-large-v3-turbo",
}


class WhisperModelLoader(IModelLoader):
    """Loads HuggingFace Whisper model + processor."""

    def __init__(self) -> None:
        self.model = None
        self.processor = None
        self.eos_id: int | None = None

    def load(self, config: EngineConfig) -> None:
        import torch
        from transformers import WhisperForConditionalGeneration, WhisperProcessor

        dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
        torch_dtype = dtype_map.get(config.dtype, torch.float16)

        self.processor = WhisperProcessor.from_pretrained(config.model_id)
        self.model = WhisperForConditionalGeneration.from_pretrained(
            config.model_id, torch_dtype=torch_dtype,
        ).to(config.device)

        self.eos_id = self.processor.tokenizer.eos_token_id or 50257

        logger.info("Whisper loaded: %s device=%s dtype=%s", config.model_id, config.device, config.dtype)

    def unload(self) -> None:
        if self.model is not None:
            model = self.model
            self.model = None
            self.processor = None
            self.eos_id = None
            del model
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass
            logger.info("Whisper unloaded")

    def is_loaded(self) -> bool:
        return self.model is not None
