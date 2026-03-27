"""Qwen3-ASR model loader — weight lifecycle management.

SRP: Only responsible for loading/unloading model weights.
"""

from __future__ import annotations

import logging
from typing import Any

from macaw_asr.config import EngineConfig
from macaw_asr.models.contracts import IModelLoader

logger = logging.getLogger("macaw-asr.models.qwen.loader")


class QwenModelLoader(IModelLoader):
    """Loads Qwen3-ASR weights via HuggingFace transformers."""

    def __init__(self) -> None:
        self.thinker = None
        self.processor = None
        self.eos_id: int | None = None

    def load(self, config: EngineConfig) -> None:
        import torch
        from qwen_asr.core.transformers_backend import (
            Qwen3ASRConfig, Qwen3ASRForConditionalGeneration, Qwen3ASRProcessor,
        )
        from transformers import AutoConfig, AutoModel, AutoProcessor

        try:
            AutoConfig.register("qwen3_asr", Qwen3ASRConfig)
            AutoModel.register(Qwen3ASRConfig, Qwen3ASRForConditionalGeneration)
            AutoProcessor.register(Qwen3ASRConfig, Qwen3ASRProcessor)
        except ValueError:
            pass  # Already registered — expected on reload

        dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
        torch_dtype = dtype_map.get(config.dtype, torch.bfloat16)

        model = AutoModel.from_pretrained(config.model_id, device_map=config.device, torch_dtype=torch_dtype)
        self.processor = AutoProcessor.from_pretrained(config.model_id, fix_mistral_regex=True)
        self.thinker = model.thinker
        self.eos_id = self.processor.tokenizer.eos_token_id

        logger.info("Qwen3-ASR loaded: model=%s device=%s dtype=%s", config.model_id, config.device, config.dtype)

    def unload(self) -> None:
        if self.thinker is not None:
            thinker = self.thinker
            self.thinker = None
            self.processor = None
            self.eos_id = None
            del thinker
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass
            logger.info("Qwen3-ASR unloaded")

    def is_loaded(self) -> bool:
        return self.thinker is not None
