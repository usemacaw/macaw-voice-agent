"""Real inference tests: Decode strategies with real model tokens.

Model selected via MACAW_ASR_TEST_MODEL env var (Ollama pattern).
Uses `model` and `engine` fixtures from conftest.py.
"""

from __future__ import annotations

import pytest

from macaw_asr.decode.postprocess import clean_asr_text
from macaw_asr.decode.strategies import DecodeContext, GreedyWithEarlyStopping


class TestEosStopsDecoding:

    def test_real_eos_stops_immediately(self, model):
        eos = model.eos_token_id
        strategy = GreedyWithEarlyStopping(eos_token_id=eos)
        context = DecodeContext(step=0, max_steps=32)
        assert strategy.should_stop(eos, context) is True

    def test_wrong_eos_does_not_stop(self, model):
        strategy = GreedyWithEarlyStopping(eos_token_id=0)
        context = DecodeContext(step=0, max_steps=32)
        assert strategy.should_stop(model.eos_token_id, context) is False

    async def test_engine_strategy_has_correct_eos(self, engine, model):
        strategy = engine._create_strategy()
        assert strategy._eos_id == model.eos_token_id


class TestContextIsolation:

    def test_separate_contexts_independent(self):
        ctx1 = DecodeContext(step=0, max_steps=10)
        ctx2 = DecodeContext(step=0, max_steps=10)
        ctx1.recent_tokens.append(42)
        assert 42 not in ctx2.recent_tokens


class TestPostprocessWithRealOutput:

    def test_strip_asr_tags(self):
        assert clean_asr_text("language Portuguese<asr_text>Olá mundo") == "Olá mundo"

    def test_clean_normal_text(self):
        assert clean_asr_text("Hello world") == "Hello world"

    def test_empty(self):
        assert clean_asr_text("") == ""
