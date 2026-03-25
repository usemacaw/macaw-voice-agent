"""Real inference tests: Model contract on GPU.

Model selected via MACAW_ASR_TEST_MODEL env var (Ollama pattern).
No mocks. Uses `model` fixture from conftest.py.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from macaw_asr.decode.strategies import GreedyWithEarlyStopping


class TestModelEosToken:

    def test_eos_token_id_is_not_zero(self, model):
        """Real model must have a real EOS token, not the placeholder 0."""
        assert hasattr(model, "eos_token_id"), "ASRModel must expose eos_token_id property"
        assert model.eos_token_id != 0


class TestModelGenerate:

    def test_silence_produces_minimal_tokens(self, model):
        audio = np.zeros(16000, dtype=np.float32)
        inputs = model.prepare_inputs(audio)
        strategy = GreedyWithEarlyStopping(eos_token_id=model.eos_token_id)
        output = model.generate(inputs, strategy)
        assert output.n_tokens <= 5

    def test_output_has_timing_keys(self, model):
        audio = np.zeros(16000, dtype=np.float32)
        inputs = model.prepare_inputs(audio)
        strategy = GreedyWithEarlyStopping(eos_token_id=model.eos_token_id)
        output = model.generate(inputs, strategy)

        assert "prefill_ms" in output.timings
        assert "decode_ms" in output.timings
        assert output.timings["prefill_ms"] > 0

    def test_deterministic_same_input(self, model):
        audio = np.zeros(16000, dtype=np.float32)

        inputs1 = model.prepare_inputs(audio)
        out1 = model.generate(inputs1, GreedyWithEarlyStopping(eos_token_id=model.eos_token_id))

        inputs2 = model.prepare_inputs(audio)
        out2 = model.generate(inputs2, GreedyWithEarlyStopping(eos_token_id=model.eos_token_id))

        assert out1.text == out2.text
        assert out1.n_tokens == out2.n_tokens


class TestFastFinish:

    def test_fast_finish_matches_full(self, model):
        np.random.seed(42)
        full_audio = np.random.randn(48000).astype(np.float32) * 0.3
        partial_audio = full_audio[:24000]

        full_inputs = model.prepare_inputs(full_audio)
        out_full = model.generate(full_inputs, GreedyWithEarlyStopping(eos_token_id=model.eos_token_id))

        cached = model.prepare_inputs(partial_audio)
        fast_inputs = model.fast_finish_inputs(full_audio, cached, len(partial_audio))

        if fast_inputs is None:
            pytest.skip("Model does not support fast_finish_inputs")

        out_fast = model.generate(fast_inputs, GreedyWithEarlyStopping(eos_token_id=model.eos_token_id))
        assert out_fast.text == out_full.text

    def test_fast_finish_is_faster_than_full(self, model):
        np.random.seed(123)
        full_audio = np.random.randn(160000).astype(np.float32) * 0.3
        partial_audio = full_audio[:80000]

        # Warmup
        model.prepare_inputs(full_audio)
        c = model.prepare_inputs(partial_audio)
        if model.fast_finish_inputs(full_audio, c, len(partial_audio)) is None:
            pytest.skip("Model does not support fast_finish_inputs")

        full_times = []
        for _ in range(3):
            t0 = time.perf_counter()
            model.prepare_inputs(full_audio)
            full_times.append((time.perf_counter() - t0) * 1000)

        fast_times = []
        for _ in range(3):
            cached = model.prepare_inputs(partial_audio)
            t0 = time.perf_counter()
            model.fast_finish_inputs(full_audio, cached, len(partial_audio))
            fast_times.append((time.perf_counter() - t0) * 1000)

        assert sorted(fast_times)[1] < sorted(full_times)[1]
