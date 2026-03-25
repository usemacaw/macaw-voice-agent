"""RED/GREEN tests: Decode strategies with real model tokens.

Tests run against the real Qwen tokenizer — no fake token IDs.
"""

from __future__ import annotations

import numpy as np
import pytest

from macaw_asr.decode.postprocess import clean_asr_text
from macaw_asr.decode.strategies import DecodeContext, GreedyWithEarlyStopping
from macaw_asr.models.qwen import QwenASRModel
from macaw_asr.config import EngineConfig


@pytest.fixture(scope="module")
def model():
    config = EngineConfig(
        model_name="qwen", model_id="Qwen/Qwen3-ASR-0.6B",
        device="cuda:0", dtype="bfloat16", language="pt",
    )
    m = QwenASRModel()
    m.load(config)
    m.warmup()
    yield m
    m.unload()


class TestEosStopsDecoding:
    """Strategy with real EOS must stop the decode loop."""

    def test_real_eos_stops_immediately(self, model):
        eos = model._eos_id
        strategy = GreedyWithEarlyStopping(eos_token_id=eos)
        context = DecodeContext(step=0, max_steps=32)
        assert strategy.should_stop(eos, context) is True

    def test_wrong_eos_does_not_stop(self, model):
        """Token 0 is NOT the real EOS — should not stop."""
        strategy = GreedyWithEarlyStopping(eos_token_id=0)
        context = DecodeContext(step=0, max_steps=32)
        real_eos = model._eos_id
        # Real EOS token should NOT trigger stop when strategy has wrong EOS
        assert strategy.should_stop(real_eos, context) is False

    def test_engine_strategy_has_correct_eos(self, model):
        """RED: Engine creates strategy with eos=0 placeholder.

        The engine MUST use the model's real EOS token.
        Currently relies on fragile override in generate().
        This test validates the contract: strategy._eos_id == model's EOS.
        """
        # This is what the engine does today:
        from macaw_asr.runner.engine import ASREngine
        from macaw_asr.config import EngineConfig, AudioConfig

        config = EngineConfig(
            model_name="qwen", model_id="Qwen/Qwen3-ASR-0.6B",
            device="cuda:0", dtype="bfloat16",
        )
        engine = ASREngine(config)
        strategy = engine._create_strategy()

        # RED: strategy has placeholder EOS=0, not the real 151645
        # This should fail until we fix the engine to use model's EOS
        assert strategy._eos_id == model._eos_id, (
            f"Engine strategy has EOS={strategy._eos_id}, "
            f"model has EOS={model._eos_id}"
        )


class TestContextIsolation:
    """DecodeContext must not leak state between decode loops."""

    def test_separate_contexts_independent(self):
        ctx1 = DecodeContext(step=0, max_steps=10)
        ctx2 = DecodeContext(step=0, max_steps=10)
        ctx1.recent_tokens.append(42)
        assert 42 not in ctx2.recent_tokens


class TestPostprocessWithRealOutput:
    """Test postprocessing with actual model output strings."""

    def test_strip_asr_tags(self):
        raw = "language Portuguese<asr_text>Olá mundo"
        assert clean_asr_text(raw) == "Olá mundo"

    def test_clean_normal_text(self):
        assert clean_asr_text("Hello world") == "Hello world"

    def test_empty(self):
        assert clean_asr_text("") == ""
