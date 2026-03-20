"""Tests for Hann crossfade — 100% CPU, no GPU needed."""

import numpy as np
import pytest

from macaw_tts.crossfade import HannCrossfader


class TestHannCrossfader:
    """Tests for HannCrossfader."""

    def test_init_with_zero_overlap(self):
        cf = HannCrossfader(overlap_samples=0)
        assert cf.overlap_samples == 0

    def test_init_rejects_negative_overlap(self):
        with pytest.raises(ValueError):
            HannCrossfader(overlap_samples=-1)

    def test_fade_curves_are_complementary(self):
        """Hann fade_in + fade_out should sum to 1.0 at every point."""
        cf = HannCrossfader(overlap_samples=100)
        total = cf._fade_in + cf._fade_out
        np.testing.assert_allclose(total, 1.0, atol=1e-6)

    def test_fade_in_starts_at_zero_ends_at_one(self):
        cf = HannCrossfader(overlap_samples=100)
        assert cf._fade_in[0] == pytest.approx(0.0, abs=1e-6)
        assert cf._fade_in[-1] == pytest.approx(1.0, abs=1e-6)

    def test_fade_out_starts_at_one_ends_at_zero(self):
        cf = HannCrossfader(overlap_samples=100)
        assert cf._fade_out[0] == pytest.approx(1.0, abs=1e-6)
        assert cf._fade_out[-1] == pytest.approx(0.0, abs=1e-6)

    def test_first_chunk_gets_fade_in(self):
        cf = HannCrossfader(overlap_samples=10)
        chunk = np.ones(100, dtype=np.float32)
        result = cf.process(chunk, is_first=True)
        # First samples should be faded in (less than 1.0)
        assert result[0] < 1.0

    def test_last_chunk_gets_fade_out(self):
        cf = HannCrossfader(overlap_samples=10)
        # Process a first chunk to set up state
        chunk1 = np.ones(100, dtype=np.float32)
        cf.process(chunk1, is_first=True)

        # Last chunk
        chunk2 = np.ones(100, dtype=np.float32)
        result = cf.process(chunk2, is_last=True)
        # Last samples should be faded out
        assert result[-1] < 1.0

    def test_zero_overlap_passes_through(self):
        cf = HannCrossfader(overlap_samples=0)
        chunk = np.ones(100, dtype=np.float32) * 0.5
        result = cf.process(chunk)
        np.testing.assert_array_equal(result, chunk)

    def test_two_chunks_blend_smoothly(self):
        ov = 20
        cf = HannCrossfader(overlap_samples=ov)

        chunk1 = np.ones(100, dtype=np.float32) * 1.0
        chunk2 = np.ones(100, dtype=np.float32) * 1.0

        result1 = cf.process(chunk1, is_first=True)
        result2 = cf.process(chunk2, is_last=True)

        # No discontinuities — both chunks are constant 1.0
        # After crossfade, blended region should also be ~1.0
        # (since fade_in + fade_out = 1.0 and both inputs are 1.0)
        all_audio = np.concatenate([result1, result2])
        # Should be smooth (after initial fade-in and final fade-out)
        mid = all_audio[ov:-ov]
        if len(mid) > 0:
            np.testing.assert_allclose(mid, 1.0, atol=0.1)

    def test_reset_clears_state(self):
        cf = HannCrossfader(overlap_samples=10)
        chunk = np.ones(100, dtype=np.float32)
        cf.process(chunk, is_first=True)
        assert cf._prev_tail is not None

        cf.reset()
        assert cf._prev_tail is None

    def test_empty_chunk_returns_empty(self):
        cf = HannCrossfader(overlap_samples=10)
        result = cf.process(np.array([], dtype=np.float32))
        assert len(result) == 0

    def test_energy_preservation_with_constant_signal(self):
        """Crossfading a constant signal should preserve amplitude."""
        ov = 50
        cf = HannCrossfader(overlap_samples=ov)

        # Three chunks of constant amplitude
        c1 = np.ones(200, dtype=np.float32) * 0.8
        c2 = np.ones(200, dtype=np.float32) * 0.8
        c3 = np.ones(200, dtype=np.float32) * 0.8

        r1 = cf.process(c1, is_first=True)
        r2 = cf.process(c2)
        r3 = cf.process(c3, is_last=True)

        # Middle of each chunk should be ~0.8
        # (excluding fade-in/fade-out regions)
        if len(r2) > 2 * ov:
            mid = r2[ov:-ov]
            np.testing.assert_allclose(mid, 0.8, atol=0.05)
