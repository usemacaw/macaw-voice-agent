"""Model layer public API."""

from macaw_asr.models.contracts import IASRModel
from macaw_asr.models.factory import ModelFactory, register_model
from macaw_asr.models.registry import list_all, list_names, get as get_model_meta, is_known
from macaw_asr.models.types import (
    TIMING_DECODE_MS, TIMING_DECODE_PER_TOKEN_MS,
    TIMING_PREFILL_MS, TIMING_PREPARE_MS, TIMING_TOTAL_MS,
    ModelOutput,
)

# Backward-compat aliases
ASRModel = IASRModel
create_model = ModelFactory.create
list_models = ModelFactory.list_models

__all__ = [
    "ASRModel", "IASRModel", "ModelFactory", "ModelOutput",
    "TIMING_DECODE_MS", "TIMING_DECODE_PER_TOKEN_MS",
    "TIMING_PREFILL_MS", "TIMING_PREPARE_MS", "TIMING_TOTAL_MS",
    "create_model", "get_model_meta", "is_known",
    "list_all", "list_models", "list_names", "register_model",
]
