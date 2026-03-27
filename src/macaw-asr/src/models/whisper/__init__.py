"""Whisper model module. Auto-registers all variants on import."""

from macaw_asr.models.factory import register_model
from macaw_asr.models.whisper.model import WhisperASRModel

# Register all Whisper variants under the same class
# The model_id in EngineConfig determines which variant loads
register_model("whisper", WhisperASRModel)
register_model("whisper-tiny", WhisperASRModel)
register_model("whisper-small", WhisperASRModel)
register_model("whisper-medium", WhisperASRModel)
register_model("whisper-large", WhisperASRModel)

__all__ = ["WhisperASRModel"]
