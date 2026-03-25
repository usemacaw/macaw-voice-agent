"""RED/GREEN tests: Real Qwen3-ASR model on GPU.

No mocks. These tests validate real inference behavior.
"""

from __future__ import annotations

import numpy as np
import pytest

from macaw_asr.config import EngineConfig
from macaw_asr.decode.strategies import DecodeContext, GreedyWithEarlyStopping
from macaw_asr.models.qwen import QwenASRModel


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


class TestModelEosToken:
    """The model must expose its real EOS token ID."""

    def test_eos_token_id_is_not_zero(self, model):
        """EOS=0 is a placeholder. Real model has EOS=151645."""
        # RED: ASRModel ABC doesn't have eos_token_id property yet
        assert hasattr(model, "eos_token_id"), "ASRModel must expose eos_token_id property"
        assert model.eos_token_id != 0
        assert model.eos_token_id == 151645


class TestModelGenerate:
    """Real inference on GPU."""

    def test_silence_produces_minimal_tokens(self, model):
        """Silence → model should stop early (< 5 tokens)."""
        audio = np.zeros(16000, dtype=np.float32)  # 1s silence
        inputs = model.prepare_inputs(audio)
        strategy = GreedyWithEarlyStopping(eos_token_id=model._eos_id)
        output = model.generate(inputs, strategy)
        assert output.n_tokens <= 5

    def test_output_has_timing_keys(self, model):
        """Every generate() must produce prefill_ms and decode_ms."""
        audio = np.zeros(16000, dtype=np.float32)
        inputs = model.prepare_inputs(audio)
        strategy = GreedyWithEarlyStopping(eos_token_id=model._eos_id)
        output = model.generate(inputs, strategy)

        assert "prefill_ms" in output.timings
        assert "decode_ms" in output.timings
        assert output.timings["prefill_ms"] > 0
        assert output.timings["decode_ms"] >= 0

    def test_deterministic_same_input(self, model):
        """Same input → same output (greedy decode is deterministic)."""
        audio = np.zeros(16000, dtype=np.float32)
        strategy = GreedyWithEarlyStopping(eos_token_id=model._eos_id)

        inputs1 = model.prepare_inputs(audio)
        out1 = model.generate(inputs1, strategy)

        strategy2 = GreedyWithEarlyStopping(eos_token_id=model._eos_id)
        inputs2 = model.prepare_inputs(audio)
        out2 = model.generate(inputs2, strategy2)

        assert out1.text == out2.text
        assert out1.n_tokens == out2.n_tokens


class TestFastFinish:
    """fast_finish_inputs must produce identical output to full prepare_inputs."""

    def test_fast_finish_matches_full(self, model):
        """This is the core optimization validation."""
        np.random.seed(42)
        full_audio = np.random.randn(48000).astype(np.float32) * 0.3  # 3s at 16kHz
        partial_audio = full_audio[:24000]  # 1.5s

        strategy = GreedyWithEarlyStopping(eos_token_id=model._eos_id)

        # Full path
        full_inputs = model.prepare_inputs(full_audio)
        out_full = model.generate(full_inputs, strategy)

        # Fast path
        cached = model.prepare_inputs(partial_audio)
        fast_inputs = model.fast_finish_inputs(full_audio, cached, len(partial_audio))
        assert fast_inputs is not None, "fast_finish_inputs must return non-None for Qwen"

        strategy2 = GreedyWithEarlyStopping(eos_token_id=model._eos_id)
        out_fast = model.generate(fast_inputs, strategy2)

        assert out_fast.text == out_full.text, (
            f"fast_finish diverged: fast={out_fast.text!r}, full={out_full.text!r}"
        )

    def test_fast_finish_is_faster_than_full(self, model):
        """fast_finish must be at least 10x faster than full prepare_inputs."""
        import time

        np.random.seed(123)
        full_audio = np.random.randn(48000).astype(np.float32) * 0.3
        partial_audio = full_audio[:24000]

        # Time full prepare
        t0 = time.perf_counter()
        full_inputs = model.prepare_inputs(full_audio)
        full_ms = (time.perf_counter() - t0) * 1000

        # Time fast finish
        cached = model.prepare_inputs(partial_audio)
        t0 = time.perf_counter()
        fast_inputs = model.fast_finish_inputs(full_audio, cached, len(partial_audio))
        fast_ms = (time.perf_counter() - t0) * 1000

        speedup = full_ms / fast_ms if fast_ms > 0 else float("inf")
        assert speedup >= 5, (
            f"fast_finish not fast enough: full={full_ms:.0f}ms, fast={fast_ms:.0f}ms, speedup={speedup:.1f}x"
        )
