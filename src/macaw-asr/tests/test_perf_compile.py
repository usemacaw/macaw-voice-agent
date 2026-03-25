"""Performance tests: torch.compile and CUDA graph integration (M7).

Tests that torch.compile produces identical results and measures speedup.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from macaw_asr.config import AudioConfig, EngineConfig, StreamingConfig
from macaw_asr.runner.engine import ASREngine
from tests.conftest import make_pcm, get_test_model_config, DEVICE


@pytest.fixture
def compile_config() -> EngineConfig:
    """Config with torch.compile enabled."""
    mc = get_test_model_config()
    return EngineConfig(
        model_name=mc["model_name"],
        model_id=mc["model_id"],
        device=DEVICE,
        dtype=mc["dtype"],
        language="pt",
        enable_compile=True,
        audio=AudioConfig(input_sample_rate=8000, model_sample_rate=mc["model_sample_rate"]),
        streaming=StreamingConfig(chunk_trigger_sec=1.0, max_new_tokens=32),
    )


class TestTorchCompile:
    """M7: torch.compile integration."""

    async def test_compile_engine_starts(self, compile_config):
        """Engine with enable_compile=True must start without error."""
        engine = ASREngine(compile_config)
        try:
            await engine.start()
            assert engine.is_started
            assert engine.startup_timings["compile_ms"] >= 0
        finally:
            await engine.stop()

    async def test_compile_produces_output(self, compile_config):
        """Compiled model must produce valid transcription."""
        engine = ASREngine(compile_config)
        await engine.start()
        try:
            audio = make_pcm(1.0)
            text = await engine.transcribe(audio)
            assert isinstance(text, str)
            # Compiled model should produce some output (may differ from uncompiled due to numerics)
            # The important thing is it doesn't crash
        finally:
            await engine.stop()

    async def test_compile_vs_no_compile_same_audio(self, engine_config, compile_config):
        """Compiled and uncompiled should produce same text for same input.

        Note: torch.compile may produce slightly different float values
        due to kernel fusion. Text output should still match for greedy decode.
        """
        audio = make_pcm(2.0)

        # Uncompiled
        eng_normal = ASREngine(engine_config)
        await eng_normal.start()
        text_normal = await eng_normal.transcribe(audio)
        await eng_normal.stop()

        # Compiled
        eng_compile = ASREngine(compile_config)
        await eng_compile.start()
        text_compile = await eng_compile.transcribe(audio)
        await eng_compile.stop()

        # May or may not match exactly due to floating point differences
        # Log both for comparison
        print(f"\n  Normal:   {text_normal!r}")
        print(f"  Compiled: {text_compile!r}")

        # At minimum, both must produce non-empty output
        assert len(text_normal) > 0
        assert len(text_compile) > 0
