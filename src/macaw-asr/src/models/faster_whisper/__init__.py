"""Faster-Whisper model module (CTranslate2 backend). Auto-registers all variants on import."""

from macaw_asr.models.factory import register_model
from macaw_asr.models.faster_whisper.model import FasterWhisperASRModel

register_model("faster-whisper-tiny", FasterWhisperASRModel)
register_model("faster-whisper-small", FasterWhisperASRModel)
register_model("faster-whisper-medium", FasterWhisperASRModel)
register_model("faster-whisper-large", FasterWhisperASRModel)

__all__ = ["FasterWhisperASRModel"]
