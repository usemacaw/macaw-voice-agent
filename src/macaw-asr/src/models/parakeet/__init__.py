"""Parakeet model module. Auto-registers on import."""

from macaw_asr.models.factory import register_model
from macaw_asr.models.parakeet.model import ParakeetASRModel

register_model("parakeet", ParakeetASRModel)
register_model("parakeet-tdt", ParakeetASRModel)
register_model("parakeet-ctc", ParakeetASRModel)
register_model("fastconformer-pt", ParakeetASRModel)
register_model("canary", ParakeetASRModel)

__all__ = ["ParakeetASRModel"]
