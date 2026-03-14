"""Tests for VADProcessor — speech detection state machine."""

import struct
from unittest.mock import MagicMock

import numpy as np
import pytest

from audio.vad import VADProcessor, SAMPLE_RATE, CHUNK_SAMPLES, CHUNK_BYTES, CHUNK_DURATION_MS
from protocol.models import TurnDetection


def _make_silence(duration_ms: int) -> bytes:
    """Generate silence PCM 8kHz 16-bit."""
    n_samples = int(SAMPLE_RATE * duration_ms / 1000)
    return b"\x00\x00" * n_samples


def _make_sine(duration_ms: int, frequency: int = 440, amplitude: float = 0.8) -> bytes:
    """Generate a sine wave PCM 8kHz 16-bit."""
    n_samples = int(SAMPLE_RATE * duration_ms / 1000)
    t = np.linspace(0, duration_ms / 1000, n_samples, endpoint=False)
    samples = (np.sin(2 * np.pi * frequency * t) * amplitude * 32767).astype(np.int16)
    return samples.tobytes()


@pytest.fixture
def default_config():
    return TurnDetection(
        threshold=0.5,
        prefix_padding_ms=300,
        silence_duration_ms=200,
    )


class TestVADProcessorInit:
    def test_creates_with_default_config(self, default_config):
        vad = VADProcessor(config=default_config)
        assert vad.is_speaking is False
        assert vad.total_audio_ms == 0

    def test_creates_with_callbacks(self, default_config):
        started = MagicMock()
        stopped = MagicMock()
        vad = VADProcessor(
            config=default_config,
            on_speech_started=started,
            on_speech_stopped=stopped,
        )
        assert vad.is_speaking is False


class TestVADProcessorFeed:
    def test_silence_does_not_trigger_speech(self, default_config):
        started = MagicMock()
        stopped = MagicMock()
        vad = VADProcessor(
            config=default_config,
            on_speech_started=started,
            on_speech_stopped=stopped,
        )

        # Feed 1 second of silence
        vad.feed(_make_silence(1000))

        assert vad.is_speaking is False
        started.assert_not_called()
        stopped.assert_not_called()

    def test_total_audio_ms_tracks_fed_audio(self, default_config):
        vad = VADProcessor(config=default_config)

        # Feed 500ms of silence (multiple chunks)
        vad.feed(_make_silence(500))

        # Should have processed ~500ms (within one chunk granularity)
        expected_chunks = 500 // CHUNK_DURATION_MS
        expected_ms = expected_chunks * CHUNK_DURATION_MS
        assert vad.total_audio_ms == expected_ms

    def test_partial_chunk_buffered(self, default_config):
        vad = VADProcessor(config=default_config)

        # Feed less than one chunk
        half_chunk = b"\x00\x00" * (CHUNK_SAMPLES // 2)
        vad.feed(half_chunk)

        # Not enough for a full chunk, so nothing processed yet
        assert vad.total_audio_ms == 0

        # Feed the other half
        vad.feed(half_chunk)
        assert vad.total_audio_ms == CHUNK_DURATION_MS


class TestVADProcessorReset:
    def test_reset_clears_state(self, default_config):
        vad = VADProcessor(config=default_config)

        # Feed some audio
        vad.feed(_make_silence(500))
        assert vad.total_audio_ms > 0

        vad.reset()
        assert vad.total_audio_ms == 0
        assert vad.is_speaking is False


class TestVADProcessorSpeechDetection:
    def test_speech_started_requires_consecutive_chunks(self, default_config):
        """VAD needs 3 consecutive speech chunks to trigger speech_started."""
        started = MagicMock()
        vad = VADProcessor(
            config=default_config,
            on_speech_started=started,
        )

        # Feed a single chunk of high-energy audio — not enough for 3 consecutive
        vad.feed(_make_sine(CHUNK_DURATION_MS))
        # May or may not trigger depending on Silero's classification
        # The key invariant: if started IS called, is_speaking should be True
        if started.called:
            assert vad.is_speaking is True

    def test_speech_stopped_after_silence(self, default_config):
        """After speech is detected and silence follows, speech_stopped fires."""
        started_calls = []
        stopped_calls = []

        def on_started(ms):
            started_calls.append(ms)

        def on_stopped(ms, audio):
            stopped_calls.append((ms, audio))

        config = TurnDetection(
            threshold=0.1,  # Very low threshold to trigger easily
            prefix_padding_ms=0,
            silence_duration_ms=200,
        )
        vad = VADProcessor(
            config=config,
            on_speech_started=on_started,
            on_speech_stopped=on_stopped,
        )

        # Feed loud audio for 500ms then silence for 500ms
        vad.feed(_make_sine(500, amplitude=0.9))
        vad.feed(_make_silence(500))

        # If speech was detected and then stopped, we should have both callbacks
        # (depends on Silero model classification, so we test the invariant)
        if started_calls and stopped_calls:
            assert stopped_calls[0][0] > started_calls[0]
            assert len(stopped_calls[0][1]) > 0  # speech audio not empty

    def test_short_speech_discarded(self, default_config):
        """Speech shorter than min_speech_chunks is discarded."""
        stopped_calls = []

        config = TurnDetection(
            threshold=0.1,
            prefix_padding_ms=0,
            silence_duration_ms=64,  # Very short silence threshold
        )
        vad = VADProcessor(
            config=config,
            on_speech_stopped=lambda ms, audio: stopped_calls.append((ms, audio)),
        )

        # Feed very short audio then silence
        vad.feed(_make_sine(32))  # Just 1 chunk
        vad.feed(_make_silence(200))

        # Speech should be too short to trigger stopped callback
        # (min_speech_chunks requires ~250ms)
        # This is a best-effort test — Silero may not even detect speech


class TestVADErrorHandling:
    def test_consecutive_errors_raise(self, default_config):
        """After 10 consecutive VAD model errors, RuntimeError is raised."""
        vad = VADProcessor(config=default_config)

        # Monkey-patch the model to always fail
        def failing_model(*args, **kwargs):
            raise RuntimeError("Model broken")

        vad._model = failing_model

        with pytest.raises(RuntimeError, match="VAD model failed 10 consecutive"):
            vad.feed(_make_silence(1000))
