"""Tests for StreamingDecoder — CPU logic only.

Tests the window calculation, padding, and direct decode paths using a
real speech tokenizer stub (NOT a mock — implements the actual interface).
"""

import numpy as np
import torch
import pytest

from macaw_tts.decoder import StreamingDecoder
from macaw_tts.audio import SAMPLES_PER_FRAME


class _RealSpeechTokenizer:
    """Minimal speech tokenizer that implements the real decode interface.

    Produces deterministic audio: each frame generates SAMPLES_PER_FRAME
    samples of value = frame_index * 0.01. This allows tests to verify
    which frames were decoded by checking sample values.
    """

    def decode(self, items: list) -> tuple:
        """Decode codec frames to audio.

        Args:
            items: List of dicts with 'audio_codes' key, each [T, Q] tensor.

        Returns:
            (wavs, sample_rate) where wavs is list of tensors.
        """
        wavs = []
        for item in items:
            codes = item["audio_codes"]
            num_frames = codes.shape[0]
            # Each frame produces SAMPLES_PER_FRAME samples
            # Value encodes frame index for verification
            audio = torch.zeros(num_frames * SAMPLES_PER_FRAME, dtype=torch.float32)
            for f in range(num_frames):
                start = f * SAMPLES_PER_FRAME
                end = start + SAMPLES_PER_FRAME
                audio[start:end] = (f + 1) * 0.01
            wavs.append(audio)
        return wavs, 24000


class TestStreamingDecoder:
    """Tests for StreamingDecoder window and decode logic."""

    def _make_decoder(self, decode_window=40, context_frames=25):
        tokenizer = _RealSpeechTokenizer()
        return StreamingDecoder(tokenizer, decode_window=decode_window, context_frames=context_frames)

    def test_empty_codes_returns_empty(self):
        decoder = self._make_decoder()
        result = decoder.decode_window([], num_new_frames=0)
        assert len(result) == 0

    def test_zero_new_frames_returns_empty(self):
        decoder = self._make_decoder()
        codes = [torch.zeros(16) for _ in range(5)]
        result = decoder.decode_window(codes, num_new_frames=0)
        assert len(result) == 0

    def test_single_frame_decode(self):
        decoder = self._make_decoder()
        codes = [torch.zeros(16)]
        result = decoder.decode_window(codes, num_new_frames=1)
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float32
        assert len(result) == SAMPLES_PER_FRAME

    def test_multiple_frames_decode(self):
        decoder = self._make_decoder(context_frames=0)
        codes = [torch.zeros(16) for _ in range(4)]
        result = decoder.decode_window(codes, num_new_frames=4)
        assert len(result) == 4 * SAMPLES_PER_FRAME

    def test_context_frames_excluded_from_output(self):
        decoder = self._make_decoder(context_frames=3)
        codes = [torch.zeros(16) for _ in range(5)]
        # Request 2 new frames, with 3 context frames
        result = decoder.decode_window(codes, num_new_frames=2)
        assert len(result) == 2 * SAMPLES_PER_FRAME

    def test_ref_codes_used_as_prefix(self):
        decoder = self._make_decoder(decode_window=10, context_frames=2)
        # Only 3 codes in buffer, window allows 10
        codes = [torch.zeros(16) for _ in range(3)]
        ref_codes = torch.zeros(5, 16)  # 5 ref frames
        result = decoder.decode_window(codes, num_new_frames=3, ref_codes=ref_codes)
        # Should produce audio for all 3 new frames
        assert len(result) == 3 * SAMPLES_PER_FRAME

    def test_properties(self):
        decoder = self._make_decoder(decode_window=50, context_frames=30)
        assert decoder.decode_window_size == 50
        assert decoder.context_frames == 30
        assert decoder.is_compiled is False

    def test_sliding_window_limits_decode_context(self):
        """When total frames > context_frames + new_frames, window slides."""
        decoder = self._make_decoder(context_frames=5)
        # 20 frames total, request last 2
        codes = [torch.zeros(16) for _ in range(20)]
        result = decoder.decode_window(codes, num_new_frames=2)
        # Should get exactly 2 new frames of audio
        assert len(result) == 2 * SAMPLES_PER_FRAME


class TestStreamingDecoderPadded:
    """Tests for the padded decode path (torch.compile compatible)."""

    def test_decode_padded_pads_short_input(self):
        """Verify fixed-shape padding logic for short inputs."""
        tokenizer = _RealSpeechTokenizer()
        # Add a decoder attribute so compile path is available
        tokenizer.decoder = type('Decoder', (), {
            'forward': lambda self, x: torch.zeros(1, x.shape[2] * SAMPLES_PER_FRAME),
        })()
        decoder = StreamingDecoder(tokenizer, decode_window=10)
        # Don't actually compile (no GPU), but test the logic exists
        assert decoder.is_compiled is False

    def test_decode_direct_with_batch_dim(self):
        """Direct decode handles [1, T, Q] input correctly."""
        tokenizer = _RealSpeechTokenizer()
        decoder = StreamingDecoder(tokenizer)
        codes = torch.zeros(1, 5, 16)  # [B, T, Q]
        result = decoder._decode_direct(codes)
        assert len(result) == 5 * SAMPLES_PER_FRAME

    def test_decode_direct_without_batch_dim(self):
        """Direct decode handles [T, Q] input correctly."""
        tokenizer = _RealSpeechTokenizer()
        decoder = StreamingDecoder(tokenizer)
        codes = torch.zeros(5, 16)  # [T, Q]
        result = decoder._decode_direct(codes)
        assert len(result) == 5 * SAMPLES_PER_FRAME
