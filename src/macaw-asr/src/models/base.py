"""Backward-compatibility shim. Old imports still work."""

from macaw_asr.models.contracts import IASRModel as ASRModel
from macaw_asr.models.factory import ModelFactory, register_model
from macaw_asr.models.registry import list_names
from macaw_asr.models.types import (
    TIMING_DECODE_MS, TIMING_DECODE_PER_TOKEN_MS,
    TIMING_PREFILL_MS, TIMING_PREPARE_MS, TIMING_TOTAL_MS,
    ModelOutput,
)


def create_model(name: str) -> ASRModel:
    return ModelFactory.create(name)


def list_models() -> list[str]:
    return ModelFactory.list_models()
