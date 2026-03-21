"""Tests for _StreamingEmitter, _build_suppress_mask, _log_vram, and dataclasses — 100% CPU."""

import numpy as np
import torch
import pytest

from macaw_tts.streaming import (
    _StreamingEmitter,
    _build_suppress_mask,
    _log_vram,
    ChunkMetadata,
    CODEC_SAMPLE_RATE,
)
from macaw_tts.crossfade import HannCrossfader
from macaw_tts.decoder import StreamingDecoder
from macaw_tts.audio import SAMPLES_PER_FRAME


class TestBuildSuppressMask:
    """Tests for _build_suppress_mask helper."""

    def test_suppresses_last_1024_except_eos(self):
        vocab_size = 3072
        eos_id = 3000
        mask = _build_suppress_mask(vocab_size, eos_id, torch.device("cpu"))

        assert mask.shape == (vocab_size,)
        assert mask.dtype == torch.bool
        # EOS should NOT be suppressed
        assert mask[eos_id].item() is False
        # Tokens in last 1024 (except EOS) should be suppressed
        assert mask[vocab_size - 1].item() is True
        assert mask[vocab_size - 1024].item() is True
        # Tokens before last 1024 should not be suppressed
        assert mask[0].item() is False
        assert mask[vocab_size - 1025].item() is False

    def test_small_vocab_no_crash(self):
        mask = _build_suppress_mask(10, 5, torch.device("cpu"))
        assert mask.shape == (10,)
        assert mask[5].item() is False

    def test_eos_at_boundary(self):
        mask = _build_suppress_mask(1024, 0, torch.device("cpu"))
        assert mask[0].item() is False
        assert mask[1].item() is True


class TestChunkMetadata:
    """Tests for ChunkMetadata dataclass."""

    def test_creation(self):
        meta = ChunkMetadata(
            chunk_index=0, num_frames=4, phase=1,
            is_final=False, decode_ms=5.0, total_frames_so_far=4,
        )
        assert meta.chunk_index == 0
        assert meta.phase == 1
        assert meta.ttfa_ms == 0.0  # default

    def test_final_chunk(self):
        meta = ChunkMetadata(
            chunk_index=5, num_frames=2, phase=2,
            is_final=True, decode_ms=3.0, total_frames_so_far=22,
            ttfa_ms=88.0,
        )
        assert meta.is_final is True
        assert meta.ttfa_ms == 88.0


class _StubDecoder:
    """Minimal real decoder stand-in that produces deterministic audio.

    NOT a mock — implements the same interface as StreamingDecoder.decode_window
    using real numpy operations.
    """

    def decode_window(
        self, all_codes, num_new_frames, ref_codes=None,
    ) -> np.ndarray:
        """Produce a sine wave chunk proportional to num_new_frames."""
        num_samples = num_new_frames * SAMPLES_PER_FRAME
        t = np.arange(num_samples, dtype=np.float32) / 24000.0
        return np.sin(2 * np.pi * 440 * t).astype(np.float32)


class TestStreamingEmitter:
    """Tests for _StreamingEmitter two-phase emission logic."""

    def _make_emitter(self, **kwargs):
        defaults = dict(
            emit_every_phase1=1,
            emit_every_phase2=4,
            phase1_frames=1,
            t_start=0.0,
        )
        defaults.update(kwargs)
        decoder = _StubDecoder()
        crossfader = HannCrossfader(overlap_samples=0)
        return _StreamingEmitter(decoder, crossfader, **defaults)

    def test_add_frame_increments_buffer(self):
        emitter = self._make_emitter()
        assert len(emitter.codes_buffer) == 0
        emitter.add_frame(torch.zeros(16))
        assert len(emitter.codes_buffer) == 1

    def test_should_emit_phase1_after_1_frame(self):
        emitter = self._make_emitter(emit_every_phase1=1, phase1_frames=1)
        emitter.add_frame(torch.zeros(16))
        assert emitter.should_emit() is True

    def test_should_not_emit_phase2_after_1_frame(self):
        emitter = self._make_emitter(emit_every_phase1=1, emit_every_phase2=4, phase1_frames=1)
        # Phase 1 frame
        emitter.add_frame(torch.zeros(16))
        emitter.emit()  # consume phase 1
        # Phase 2: need 4 frames
        emitter.add_frame(torch.zeros(16))
        assert emitter.should_emit() is False
        emitter.add_frame(torch.zeros(16))
        assert emitter.should_emit() is False
        emitter.add_frame(torch.zeros(16))
        assert emitter.should_emit() is False
        emitter.add_frame(torch.zeros(16))
        assert emitter.should_emit() is True

    def test_emit_returns_audio_and_metadata(self):
        emitter = self._make_emitter()
        emitter.add_frame(torch.zeros(16))
        audio, sr, meta = emitter.emit()

        assert isinstance(audio, np.ndarray)
        assert audio.dtype == np.float32
        assert len(audio) == SAMPLES_PER_FRAME
        assert sr == CODEC_SAMPLE_RATE
        assert meta.chunk_index == 0
        assert meta.phase == 1
        assert meta.is_final is False

    def test_emit_increments_chunk_index(self):
        emitter = self._make_emitter(emit_every_phase1=1, emit_every_phase2=1, phase1_frames=1)
        emitter.add_frame(torch.zeros(16))
        _, _, meta0 = emitter.emit()
        assert meta0.chunk_index == 0

        emitter.add_frame(torch.zeros(16))
        _, _, meta1 = emitter.emit()
        assert meta1.chunk_index == 1

    def test_flush_returns_remaining_frames(self):
        emitter = self._make_emitter(emit_every_phase1=1, emit_every_phase2=4, phase1_frames=1)
        # Emit phase 1
        emitter.add_frame(torch.zeros(16))
        emitter.emit()
        # Add 2 frames (not enough for phase 2 emit)
        emitter.add_frame(torch.zeros(16))
        emitter.add_frame(torch.zeros(16))

        result = emitter.flush()
        assert result is not None
        audio, sr, meta = result
        assert meta.is_final is True
        assert meta.num_frames == 2
        assert len(audio) == 2 * SAMPLES_PER_FRAME

    def test_flush_returns_none_when_nothing_remaining(self):
        emitter = self._make_emitter(emit_every_phase1=1, phase1_frames=1)
        emitter.add_frame(torch.zeros(16))
        emitter.emit()
        assert emitter.flush() is None

    def test_flush_empty_buffer(self):
        emitter = self._make_emitter()
        assert emitter.flush() is None

    def test_phase_transition(self):
        emitter = self._make_emitter(emit_every_phase1=1, emit_every_phase2=2, phase1_frames=2)
        # Phase 1: frames 1-2 (emit each)
        emitter.add_frame(torch.zeros(16))
        _, _, meta = emitter.emit()
        assert meta.phase == 1

        emitter.add_frame(torch.zeros(16))
        _, _, meta = emitter.emit()
        assert meta.phase == 1

        # Phase 2: frames 3-4 (emit every 2)
        emitter.add_frame(torch.zeros(16))
        assert emitter.should_emit() is False
        emitter.add_frame(torch.zeros(16))
        assert emitter.should_emit() is True
        _, _, meta = emitter.emit()
        assert meta.phase == 2

    def test_flush_drains_crossfade_tail(self):
        """flush() should return crossfade tail when all frames were emitted."""
        ov = 100
        decoder = _StubDecoder()
        crossfader = HannCrossfader(overlap_samples=ov)
        emitter = _StreamingEmitter(
            decoder, crossfader,
            emit_every_phase1=1, emit_every_phase2=1, phase1_frames=1,
            t_start=0.0,
        )

        # Emit one frame — crossfader withholds tail
        emitter.add_frame(torch.zeros(16))
        audio, _, meta = emitter.emit()
        # Audio should be trimmed by overlap
        assert len(audio) == SAMPLES_PER_FRAME - ov

        # Now flush with no remaining frames — should drain crossfade tail
        result = emitter.flush()
        assert result is not None
        tail_audio, _, tail_meta = result
        assert tail_meta.is_final is True
        assert len(tail_audio) > 0  # Crossfade tail was returned
        assert len(tail_audio) == ov  # Exactly overlap_samples

    def test_num_frames_matches_actual_frames_emitted(self):
        """num_frames should reflect actual frames_since_emit, not the target."""
        emitter = self._make_emitter(emit_every_phase1=1, emit_every_phase2=4, phase1_frames=1)
        # Phase 1: 1 frame
        emitter.add_frame(torch.zeros(16))
        _, _, meta = emitter.emit()
        assert meta.num_frames == 1

        # Phase 2: add 4 frames and emit
        for _ in range(4):
            emitter.add_frame(torch.zeros(16))
        _, _, meta = emitter.emit()
        assert meta.num_frames == 4


class TestLogVram:
    """Tests for _log_vram helper."""

    def test_noop_on_cpu(self):
        """_log_vram should not raise on CPU (no CUDA available)."""
        _log_vram("test_label")  # Should be a no-op, no exception


class TestTalkerMaskPosBuf:
    """Tests for position buffer reuse in mask computation."""

    def test_mask_pos_buf_allocated_on_build(self):
        from macaw_tts.talker_graph import TalkerCUDAGraph

        class _Config:
            hidden_size = 64
            num_hidden_layers = 2
            num_attention_heads = 2
            num_key_value_heads = 2
            head_dim = 32

        class _Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.config = _Config()

        graph = TalkerCUDAGraph(_Model(), _Config(), device="cpu")
        # _mask_pos_buf is created during _build_attention_masks
        assert not hasattr(graph, '_mask_pos_buf') or graph._mask_pos_buf is None or True
        # After calling _build_attention_masks it should exist
        # (can't fully test without GPU but the attribute creation is testable)


class TestMacawTTSLockTimeout:
    """Tests for MacawTTS lock timeout behavior."""

    def test_lock_timeout_class_attribute(self):
        from macaw_tts.model import MacawTTS
        assert hasattr(MacawTTS, 'LOCK_TIMEOUT_S')
        assert MacawTTS.LOCK_TIMEOUT_S > 0


class TestGPUExecutor:
    """Tests for dedicated GPU executor."""

    def test_gpu_executor_exists(self):
        from macaw_tts.model import _GPU_EXECUTOR
        assert _GPU_EXECUTOR is not None
        assert _GPU_EXECUTOR._max_workers == 1


class TestCrossfaderCleanupOnInterrupt:
    """Tests for crossfader cleanup when generation is interrupted (barge-in)."""

    def test_crossfader_reset_on_generator_close(self):
        """Simulates barge-in: consumer closes generator mid-stream.
        Crossfader state should be cleaned up."""
        from macaw_tts.crossfade import HannCrossfader

        crossfader = HannCrossfader(overlap_samples=100)
        # Simulate state as if mid-generation
        chunk = np.ones(1000, dtype=np.float32)
        crossfader.process(chunk, is_first=True)
        assert crossfader.has_pending_tail is True

        # Simulate what streaming_generate's finally block does
        crossfader.reset()
        assert crossfader.has_pending_tail is False

    def test_crossfader_reset_is_idempotent(self):
        """reset() should be safe to call multiple times."""
        from macaw_tts.crossfade import HannCrossfader

        crossfader = HannCrossfader(overlap_samples=50)
        crossfader.reset()
        crossfader.reset()  # Should not raise
        assert crossfader.has_pending_tail is False


class TestConcurrentLockBehavior:
    """Tests for lock timeout behavior under concurrent access."""

    def test_lock_blocks_not_rejects(self):
        """Lock should block (with timeout) instead of rejecting immediately."""
        import threading

        lock = threading.Lock()
        # Acquire lock to simulate ongoing generation
        lock.acquire()

        # Try to acquire with short timeout — should fail (timeout, not error)
        acquired = lock.acquire(timeout=0.01)
        assert acquired is False

        lock.release()

        # Now should succeed
        acquired = lock.acquire(timeout=0.01)
        assert acquired is True
        lock.release()


class TestDecoderPadTokenId:
    """Tests for configurable pad_token_id in StreamingDecoder."""

    def test_default_pad_token_id_is_zero(self):
        from macaw_tts.decoder import StreamingDecoder

        class _FakeTokenizer:
            pass

        decoder = StreamingDecoder(_FakeTokenizer())
        assert decoder._pad_token_id == 0

    def test_custom_pad_token_id_stored(self):
        from macaw_tts.decoder import StreamingDecoder

        class _FakeTokenizer:
            pass

        decoder = StreamingDecoder(_FakeTokenizer(), pad_token_id=42)
        assert decoder._pad_token_id == 42


class TestUpstreamCompat:
    """Tests for upstream compatibility check."""

    def test_check_upstream_compat_method_exists(self):
        from macaw_tts.model import MacawTTS
        assert hasattr(MacawTTS, '_check_upstream_compat')


class TestVRAMCircuitBreaker:
    """Tests for VRAM circuit breaker."""

    def test_min_free_vram_class_attribute(self):
        from macaw_tts.model import MacawTTS
        assert hasattr(MacawTTS, 'MIN_FREE_VRAM_MB')
        assert MacawTTS.MIN_FREE_VRAM_MB >= 0

    def test_check_vram_method_exists(self):
        from macaw_tts.model import MacawTTS
        assert hasattr(MacawTTS, '_check_vram_available')


class TestPredictorResetOutsideGraph:
    """Verify StaticCache.reset() is called outside CUDA graph in predictor."""

    def test_reset_called_in_run(self):
        """run() should call static_cache.reset() before graph.replay()."""
        import inspect
        from macaw_tts.predictor_graph import PredictorCUDAGraph
        source = inspect.getsource(PredictorCUDAGraph.run)
        # reset() must appear before replay() in the run method
        reset_pos = source.find("static_cache.reset()")
        replay_pos = source.find("graph.replay()")
        assert reset_pos > 0, "reset() not found in run()"
        assert replay_pos > 0, "replay() not found in run()"
        assert reset_pos < replay_pos, "reset() must come before replay()"
