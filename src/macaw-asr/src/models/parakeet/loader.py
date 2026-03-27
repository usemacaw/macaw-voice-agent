"""Parakeet model loader — NeMo ASR weight lifecycle.

SRP: Only loads/unloads NeMo Parakeet model.
Requires: nemo_toolkit[asr]
"""

from __future__ import annotations

import logging

from macaw_asr.config import EngineConfig
from macaw_asr.models.contracts import IModelLoader

logger = logging.getLogger("macaw-asr.models.parakeet.loader")


class ParakeetModelLoader(IModelLoader):
    """Loads NVIDIA Parakeet via NeMo ASRModel.from_pretrained()."""

    def __init__(self) -> None:
        self.model = None
        self.eos_id: int = 0  # Parakeet CTC/TDT doesn't have traditional EOS

    def load(self, config: EngineConfig) -> None:
        # Suppress verbose NeMo/PyTorch logs
        for name in ["nemo", "pytorch_lightning", "nemo.collections", "nemo.utils"]:
            logging.getLogger(name).setLevel(logging.WARNING)

        import torch
        import nemo.collections.asr as nemo_asr

        device = config.device if config.device else "cpu"

        # Load on CPU first, then move — avoids cuDNN init errors
        # with Hybrid models (RNN modules need explicit device transfer)
        self.model = nemo_asr.models.ASRModel.from_pretrained(
            model_name=config.model_id,
            map_location="cpu",
        )

        if device != "cpu":
            self.model = self.model.to(torch.device(device))

        self.model.eval()
        logger.info("NeMo loaded: %s device=%s", config.model_id, config.device)

    def unload(self) -> None:
        if self.model is not None:
            model = self.model
            self.model = None
            del model
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass
            logger.info("Parakeet unloaded")

    def is_loaded(self) -> bool:
        return self.model is not None
