"""Tests for AudioInputHandler: VAD callbacks, ASR transcription, RMS, barge-in."""

import asyncio
import struct
from unittest.mock import AsyncMock, MagicMock

import pytest

from protocol.models import SessionConfig, TurnDetection
from server.audio_input import AudioInputCallbacks, AudioInputHandler


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_callbacks() -> AudioInputCallbacks:
    return AudioInputCallbacks(
        cancel_active_response=AsyncMock(return_value=True),
        append_user_item_and_respond=AsyncMock(),
        emit=AsyncMock(),
    )


def _make_asr(*, supports_streaming: bool = False, transcript: str = "hello world") -> MagicMock:
    asr = MagicMock()
    asr.supports_streaming = supports_streaming
    asr.transcribe = AsyncMock(return_value=transcript)
    asr.start_stream = AsyncMock()
    asr.feed_chunk = AsyncMock()
    asr.finish_stream = AsyncMock(return_value=transcript)
    return asr


def _make_config(*, vad: bool = False) -> SessionConfig:
    config = SessionConfig()
    if not vad:
        config.turn_detection = None
    return config


def _make_emitter() -> MagicMock:
    emitter = MagicMock()
    emitter.emit = AsyncMock()
    return emitter


def _pcm_silence(duration_ms: int, sample_rate: int = 8000) -> bytes:
    """Generate PCM16 silence (all zeros)."""
    n_samples = int(sample_rate * duration_ms / 1000)
    return b"\x00\x00" * n_samples


def _pcm_tone(duration_ms: int, amplitude: int = 10000, sample_rate: int = 8000) -> bytes:
    """Generate PCM16 square wave at given amplitude."""
    n_samples = int(sample_rate * duration_ms / 1000)
    return struct.pack(f"<{n_samples}h", *([amplitude] * n_samples))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestRmsComputation:
    def test_rms_silence(self):
        """RMS of silence should be 0."""
        audio = _pcm_silence(100)
        assert AudioInputHandler._compute_rms(audio) == 0.0

    def test_rms_speech(self):
        """RMS of a tone should match the amplitude."""
        amplitude = 10000
        audio = _pcm_tone(100, amplitude=amplitude)
        rms = AudioInputHandler._compute_rms(audio)
        assert rms == pytest.approx(amplitude, rel=0.01)

    def test_rms_empty(self):
        """RMS of empty audio should be 0."""
        assert AudioInputHandler._compute_rms(b"") == 0.0

    def test_rms_odd_bytes(self):
        """RMS should handle odd byte count (truncate last byte)."""
        audio = _pcm_tone(100) + b"\x00"  # extra byte
        rms = AudioInputHandler._compute_rms(audio)
        assert rms > 0


class TestLowRmsDiscardsSpeech:
    @pytest.mark.asyncio
    async def test_low_rms_does_not_trigger_response(self):
        """Speech with RMS below threshold should be discarded."""
        callbacks = _make_callbacks()
        asr = _make_asr()
        emitter = _make_emitter()
        config = _make_config()

        handler = AudioInputHandler(
            asr=asr, config=config, callbacks=callbacks,
            emitter=emitter, session_id="sess_test",
        )

        # Simulate VAD speech_stopped with silence (low RMS)
        silence = _pcm_silence(500)
        await handler._handle_speech_stopped_with_rms(1000, silence, "item_test")

        # Should NOT have called ASR or callbacks
        asr.transcribe.assert_not_called()
        callbacks.append_user_item_and_respond.assert_not_called()


class TestSpeechTriggersTranscription:
    @pytest.mark.asyncio
    async def test_speech_triggers_full_flow(self):
        """Real speech should trigger ASR transcription and callback."""
        callbacks = _make_callbacks()
        asr = _make_asr(transcript="olá mundo")
        emitter = _make_emitter()
        config = _make_config()

        handler = AudioInputHandler(
            asr=asr, config=config, callbacks=callbacks,
            emitter=emitter, session_id="sess_test",
        )

        # Simulate VAD speech_stopped with loud audio
        loud_audio = _pcm_tone(500, amplitude=5000)
        await handler._handle_speech_stopped_with_rms(1000, loud_audio, "item_test")

        # Should have transcribed via batch ASR
        asr.transcribe.assert_called_once_with(loud_audio)

        # Should have called callback with item and transcript
        callbacks.append_user_item_and_respond.assert_called_once()
        call_args = callbacks.append_user_item_and_respond.call_args
        item = call_args[0][0]
        transcript = call_args[0][1]
        assert item.id == "item_test"
        assert item.role == "user"
        assert transcript == "olá mundo"


class TestBargeIn:
    @pytest.mark.asyncio
    async def test_barge_in_calls_cancel(self):
        """Speech during active response should trigger cancel callback."""
        callbacks = _make_callbacks()
        asr = _make_asr()
        emitter = _make_emitter()
        config = _make_config()
        # Enable interrupt_response
        config.turn_detection = TurnDetection(
            type="server_vad", interrupt_response=True
        )

        handler = AudioInputHandler(
            asr=asr, config=config, callbacks=callbacks,
            emitter=emitter, session_id="sess_test",
        )

        loud_audio = _pcm_tone(500, amplitude=5000)
        await handler._handle_speech_stopped_with_rms(1000, loud_audio, "item_test")

        # Cancel should have been called
        callbacks.cancel_active_response.assert_called_once()
        assert handler.barge_in_count == 1

    @pytest.mark.asyncio
    async def test_barge_in_not_counted_when_no_active_response(self):
        """Barge-in count should not increment when no response was active."""
        callbacks = _make_callbacks()
        callbacks.cancel_active_response = AsyncMock(return_value=False)
        asr = _make_asr()
        emitter = _make_emitter()
        config = _make_config()
        config.turn_detection = TurnDetection(
            type="server_vad", interrupt_response=True
        )

        handler = AudioInputHandler(
            asr=asr, config=config, callbacks=callbacks,
            emitter=emitter, session_id="sess_test",
        )

        loud_audio = _pcm_tone(500, amplitude=5000)
        await handler._handle_speech_stopped_with_rms(1000, loud_audio, "item_test")

        callbacks.cancel_active_response.assert_called_once()
        assert handler.barge_in_count == 0

    @pytest.mark.asyncio
    async def test_no_barge_in_when_disabled(self):
        """No cancel when interrupt_response is False."""
        callbacks = _make_callbacks()
        asr = _make_asr()
        emitter = _make_emitter()
        config = _make_config()
        config.turn_detection = TurnDetection(
            type="server_vad", interrupt_response=False
        )

        handler = AudioInputHandler(
            asr=asr, config=config, callbacks=callbacks,
            emitter=emitter, session_id="sess_test",
        )

        loud_audio = _pcm_tone(500, amplitude=5000)
        await handler._handle_speech_stopped_with_rms(1000, loud_audio, "item_test")

        callbacks.cancel_active_response.assert_not_called()
        assert handler.barge_in_count == 0


class TestStreamingAsrFallback:
    @pytest.mark.asyncio
    async def test_fallback_to_batch_when_streaming_empty(self):
        """When streaming ASR returns empty, should fallback to batch."""
        callbacks = _make_callbacks()
        asr = _make_asr(supports_streaming=True, transcript="batch result")
        # Streaming returns empty, batch returns text
        asr.finish_stream = AsyncMock(return_value="")
        asr.transcribe = AsyncMock(return_value="batch result")
        emitter = _make_emitter()
        config = _make_config()

        handler = AudioInputHandler(
            asr=asr, config=config, callbacks=callbacks,
            emitter=emitter, session_id="sess_test",
        )

        # Set stream_id to simulate active streaming
        handler._asr_stream_id = "item_test"

        loud_audio = _pcm_tone(500, amplitude=5000)
        await handler._handle_speech_stopped_with_rms(1000, loud_audio, "item_test")

        # Should have tried streaming first, then fallen back to batch
        asr.finish_stream.assert_called_once_with("item_test")
        asr.transcribe.assert_called_once_with(loud_audio)

        # Callback should have received the batch result
        call_args = callbacks.append_user_item_and_respond.call_args
        assert call_args[0][1] == "batch result"


class TestCleanup:
    @pytest.mark.asyncio
    async def test_cleanup_finishes_asr_stream(self):
        """Cleanup should finish any active ASR stream."""
        callbacks = _make_callbacks()
        asr = _make_asr(supports_streaming=True)
        emitter = _make_emitter()
        config = _make_config()

        handler = AudioInputHandler(
            asr=asr, config=config, callbacks=callbacks,
            emitter=emitter, session_id="sess_test",
        )
        handler._asr_stream_id = "stream_123"

        await handler.cleanup()

        asr.finish_stream.assert_called_once_with("stream_123")
        assert handler._asr_stream_id is None

    @pytest.mark.asyncio
    async def test_cleanup_handles_asr_error(self):
        """Cleanup should not raise if ASR stream finish fails."""
        callbacks = _make_callbacks()
        asr = _make_asr()
        asr.finish_stream = AsyncMock(side_effect=RuntimeError("connection lost"))
        emitter = _make_emitter()
        config = _make_config()

        handler = AudioInputHandler(
            asr=asr, config=config, callbacks=callbacks,
            emitter=emitter, session_id="sess_test",
        )
        handler._asr_stream_id = "stream_123"

        # Should not raise
        await handler.cleanup()
        assert handler._asr_stream_id is None


class TestEmptyTranscriptSkipsResponse:
    @pytest.mark.asyncio
    async def test_empty_transcript_does_not_respond(self):
        """When ASR returns empty string, should not call response callback."""
        callbacks = _make_callbacks()
        asr = _make_asr(transcript="")
        emitter = _make_emitter()
        config = _make_config()

        handler = AudioInputHandler(
            asr=asr, config=config, callbacks=callbacks,
            emitter=emitter, session_id="sess_test",
        )

        loud_audio = _pcm_tone(500, amplitude=5000)
        await handler._handle_speech_stopped_with_rms(1000, loud_audio, "item_test")

        # ASR was called but returned empty — no response should be triggered
        asr.transcribe.assert_called_once()
        callbacks.append_user_item_and_respond.assert_not_called()
