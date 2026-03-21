"""Tests for sampling module — 100% CPU, no GPU needed."""

import torch
import pytest

from macaw_tts.sampling import sample_logits, CircularRepetitionPenalty


class TestSampleLogits:
    """Tests for sample_logits function."""

    def test_greedy_returns_argmax(self):
        logits = torch.tensor([1.0, 3.0, 2.0])
        result = sample_logits(logits, do_sample=False)
        assert result.item() == 1  # index of max value

    def test_greedy_with_suppress_mask(self):
        logits = torch.tensor([1.0, 3.0, 2.0])
        suppress = torch.tensor([False, True, False])
        result = sample_logits(logits, do_sample=False, suppress_mask=suppress)
        assert result.item() == 2  # max after suppressing index 1

    def test_greedy_with_suppress_tokens(self):
        logits = torch.tensor([1.0, 3.0, 2.0])
        result = sample_logits(logits, do_sample=False, suppress_tokens=[1])
        assert result.item() == 2

    def test_sampling_returns_valid_index(self):
        logits = torch.tensor([0.0, 0.0, 10.0])
        # High logit at index 2 should be sampled most often
        results = [
            sample_logits(logits, temperature=0.1, top_k=0, top_p=1.0, do_sample=True).item()
            for _ in range(100)
        ]
        assert 2 in results
        # With very low temperature, should almost always pick 2
        assert results.count(2) > 90

    def test_top_k_filters_low_probability(self):
        logits = torch.tensor([10.0, 9.0, 1.0, 0.0, 0.0])
        results = set()
        for _ in range(200):
            r = sample_logits(logits, temperature=0.5, top_k=2, top_p=1.0, do_sample=True).item()
            results.add(r)
        # Should only pick from top-2 (indices 0, 1)
        assert results.issubset({0, 1})

    def test_top_p_nucleus_filtering(self):
        # One dominant logit
        logits = torch.tensor([10.0, 0.0, 0.0, 0.0])
        results = set()
        for _ in range(100):
            r = sample_logits(logits, temperature=1.0, top_k=0, top_p=0.5, do_sample=True).item()
            results.add(r)
        # With very dominant first token and top_p=0.5, should mostly pick 0
        assert 0 in results

    def test_temperature_affects_distribution(self):
        logits = torch.tensor([2.0, 1.0, 0.5])
        # Low temperature → more deterministic
        low_temp = [
            sample_logits(logits, temperature=0.01, top_k=0, top_p=1.0, do_sample=True).item()
            for _ in range(50)
        ]
        assert low_temp.count(0) > 45  # Should almost always pick highest


class TestCircularRepetitionPenalty:
    """Tests for CircularRepetitionPenalty."""

    def test_no_penalty_when_disabled(self):
        rp = CircularRepetitionPenalty(penalty=1.0, vocab_size=10, device="cpu")
        logits = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]])
        result = rp.apply(logits)
        assert torch.equal(result, logits)

    def test_penalty_reduces_repeated_tokens(self):
        rp = CircularRepetitionPenalty(window=4, penalty=2.0, vocab_size=5, device="cpu")
        rp.update(torch.tensor(2))  # Token 2 seen

        logits = torch.tensor([[1.0, 1.0, 5.0, 1.0, 1.0]])
        result = rp.apply(logits.clone())

        # Token 2 (positive logit=5.0) should be divided by penalty
        assert result[0, 2].item() == pytest.approx(2.5)
        # Other tokens unaffected
        assert result[0, 0].item() == pytest.approx(1.0)

    def test_penalty_on_negative_logits(self):
        rp = CircularRepetitionPenalty(window=4, penalty=2.0, vocab_size=3, device="cpu")
        rp.update(torch.tensor(1))

        logits = torch.tensor([[1.0, -2.0, 1.0]])
        result = rp.apply(logits.clone())

        # Negative logit multiplied by penalty (makes more negative)
        assert result[0, 1].item() == pytest.approx(-4.0)

    def test_circular_buffer_wraps(self):
        rp = CircularRepetitionPenalty(window=3, penalty=2.0, vocab_size=5, device="cpu")
        # Fill buffer: [0, 1, 2]
        rp.update(torch.tensor(0))
        rp.update(torch.tensor(1))
        rp.update(torch.tensor(2))

        # Overwrite oldest: buffer becomes [3, 1, 2]
        rp.update(torch.tensor(3))

        logits = torch.ones(1, 5) * 4.0
        result = rp.apply(logits.clone())

        # Token 0 no longer in buffer → no penalty
        assert result[0, 0].item() == pytest.approx(4.0)
        # Tokens 1, 2, 3 in buffer → penalized
        assert result[0, 1].item() == pytest.approx(2.0)
        assert result[0, 2].item() == pytest.approx(2.0)
        assert result[0, 3].item() == pytest.approx(2.0)

    def test_reset_clears_history(self):
        rp = CircularRepetitionPenalty(window=4, penalty=2.0, vocab_size=5, device="cpu")
        rp.update(torch.tensor(2))
        rp.reset()

        logits = torch.ones(1, 5) * 4.0
        result = rp.apply(logits.clone())

        # After reset, no penalty on any token
        assert torch.allclose(result, logits)

    def test_update_with_scalar_tensor(self):
        """update() with a 0-dim tensor should work without .item() CPU sync."""
        rp = CircularRepetitionPenalty(window=4, penalty=2.0, vocab_size=5, device="cpu")
        rp.update(torch.tensor(2))
        logits = torch.ones(1, 5) * 4.0
        result = rp.apply(logits.clone())
        assert result[0, 2].item() == pytest.approx(2.0)

    def test_update_with_1d_tensor(self):
        """update() with a 1-dim tensor [token_id] should work."""
        rp = CircularRepetitionPenalty(window=4, penalty=2.0, vocab_size=5, device="cpu")
        rp.update(torch.tensor([3]))
        logits = torch.ones(1, 5) * 4.0
        result = rp.apply(logits.clone())
        assert result[0, 3].item() == pytest.approx(2.0)
