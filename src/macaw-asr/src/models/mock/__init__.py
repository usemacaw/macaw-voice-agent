"""Mock model module. Auto-registers on import."""

from macaw_asr.models.factory import register_model
from macaw_asr.models.mock.model import MockASRModel

register_model("mock", MockASRModel)

__all__ = ["MockASRModel"]
