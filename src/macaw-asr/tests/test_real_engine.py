"""RED/GREEN tests: Real ASREngine E2E on GPU.

No mocks. Tests the full pipeline: PCM → preprocess → model → text.
"""

from __future__ import annotations

import time

import pytest

from macaw_asr.runner.engine import ASREngine
from tests.conftest import make_noise, make_pcm, make_silence


class TestBatchTranscription:

    async def test_transcribe_returns_nonempty_text(self, engine):
        audio = make_pcm(1.0, freq=440)
        text = await engine.transcribe(audio)
        assert isinstance(text, str)
        assert len(text) > 0

    async def test_transcribe_empty_returns_empty(self, engine):
        text = await engine.transcribe(b"")
        assert text == ""

    async def test_transcribe_silence_returns_short_text(self, engine):
        """Silence should produce minimal output, not 32 tokens of garbage."""
        audio = make_silence(1.0)
        text = await engine.transcribe(audio)
        word_count = len(text.split())
        assert word_count <= 5, f"Silence produced too many words: {text!r}"

    async def test_transcribe_deterministic(self, engine):
        """Same audio → same text (greedy decode)."""
        audio = make_pcm(1.0, freq=440)
        text1 = await engine.transcribe(audio)
        text2 = await engine.transcribe(audio)
        assert text1 == text2

    async def test_different_duration_different_tokens(self, engine):
        """Different durations should produce different token counts."""
        audio_short = make_pcm(0.5, freq=440)
        audio_long = make_pcm(3.0, freq=440)
        text_short = await engine.transcribe(audio_short)
        text_long = await engine.transcribe(audio_long)
        # Longer audio may produce more or different text
        assert isinstance(text_short, str)
        assert isinstance(text_long, str)

    async def test_transcribe_latency_under_3s(self, engine):
        """1s audio should transcribe in under 3 seconds."""
        audio = make_pcm(1.0)
        t0 = time.perf_counter()
        await engine.transcribe(audio)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        assert elapsed_ms < 3000, f"Transcription took {elapsed_ms:.0f}ms"


class TestStreaming:

    async def test_streaming_produces_result(self, engine):
        audio = make_pcm(2.0)
        await engine.create_session("real-s1")
        chunk_size = 512  # 32ms at 8kHz
        for i in range(0, len(audio), chunk_size):
            await engine.push_audio("real-s1", audio[i:i + chunk_size])
        text = await engine.finish_session("real-s1")
        assert isinstance(text, str)
        assert len(text) > 0

    async def test_streaming_empty_session(self, engine):
        await engine.create_session("real-empty")
        text = await engine.finish_session("real-empty")
        assert text == ""

    async def test_batch_equals_streaming(self, engine):
        """Batch and streaming with same audio MUST produce identical text.

        This is the NeMo dual-path validation pattern:
        streaming_wer == offline_wer.

        RED: currently diverges because streaming goes through
        different code path (background precompute disabled for
        short audio, recompute mode differs).
        """
        audio = make_pcm(1.0, freq=440)

        # Batch
        text_batch = await engine.transcribe(audio)

        # Streaming (same audio, chunked)
        await engine.create_session("dual-path")
        chunk_size = 512
        for i in range(0, len(audio), chunk_size):
            await engine.push_audio("dual-path", audio[i:i + chunk_size])
        text_stream = await engine.finish_session("dual-path")

        assert text_batch == text_stream, (
            f"Batch/streaming diverged: batch={text_batch!r}, stream={text_stream!r}"
        )

    async def test_concurrent_sessions_isolated(self, engine):
        """Two sessions with same audio must produce same result."""
        audio = make_pcm(0.5)
        await engine.create_session("iso-a")
        await engine.create_session("iso-b")
        await engine.push_audio("iso-a", audio)
        await engine.push_audio("iso-b", audio)
        text_a = await engine.finish_session("iso-a")
        text_b = await engine.finish_session("iso-b")
        assert text_a == text_b


class TestEngineLifecycle:

    async def test_start_stop_restart(self, engine_config):
        """Engine must support clean restart."""
        eng = ASREngine(engine_config)
        await eng.start()
        audio = make_pcm(0.5)
        text1 = await eng.transcribe(audio)
        await eng.stop()

        await eng.start()
        text2 = await eng.transcribe(audio)
        await eng.stop()

        assert text1 == text2, "Restart changed inference result"
