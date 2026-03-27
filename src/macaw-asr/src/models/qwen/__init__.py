"""Qwen3-ASR model module. Auto-registers on import."""

from macaw_asr.models.factory import register_model
from macaw_asr.models.qwen.model import QwenASRModel

register_model("qwen", QwenASRModel)

__all__ = ["QwenASRModel"]
