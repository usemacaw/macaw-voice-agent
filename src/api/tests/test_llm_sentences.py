"""Tests for LLM sentence segmentation and splitting logic."""

import pytest

from providers.llm import split_long_sentence


class TestSplitLongSentence:
    """Test split_long_sentence pure logic."""

    def test_empty_string(self):
        assert split_long_sentence("", 50) == []

    def test_whitespace_only(self):
        assert split_long_sentence("   ", 50) == []

    def test_short_sentence_unchanged(self):
        assert split_long_sentence("Hello world.", 50) == ["Hello world."]

    def test_exact_max_chars(self):
        s = "A" * 50
        assert split_long_sentence(s, 50) == [s]

    def test_split_at_comma(self):
        s = "Hello world, this is a test sentence that is quite long."
        result = split_long_sentence(s, 30)
        assert len(result) >= 2
        rejoined = " ".join(result)
        # All words should be preserved
        for word in ["Hello", "world,", "test", "sentence"]:
            assert word in rejoined

    def test_split_at_semicolon(self):
        s = "First part; second part that continues on"
        result = split_long_sentence(s, 20)
        assert len(result) >= 2
        assert result[0] == "First part;"

    def test_split_at_space_when_no_punctuation(self):
        s = "abcdefghij klmnopqrs tuvwxyz"
        result = split_long_sentence(s, 15)
        assert len(result) >= 2
        # Should split at space boundaries
        for part in result:
            assert len(part) <= 15

    def test_split_hard_break_no_spaces(self):
        s = "A" * 100
        result = split_long_sentence(s, 30)
        assert len(result) >= 3
        for part in result:
            assert len(part) <= 30

    def test_preserves_all_text(self):
        s = "Olá, como vai? Tudo bem, espero que sim; muito obrigado pela ajuda."
        result = split_long_sentence(s, 25)
        rejoined = " ".join(result)
        # All significant words preserved
        for word in s.split():
            assert word in rejoined

    def test_unicode_em_dash_break(self):
        s = "Primeira parte\u2014segunda parte bem longa aqui"
        result = split_long_sentence(s, 25)
        assert len(result) >= 2

    def test_ellipsis_break(self):
        s = "Wait for it... and then something happened"
        result = split_long_sentence(s, 20)
        assert len(result) >= 2


class TestSplitEdgeCases:

    def test_single_char(self):
        assert split_long_sentence("A", 50) == ["A"]

    def test_max_chars_1(self):
        result = split_long_sentence("ABC", 1)
        assert len(result) == 3
        assert all(len(p) <= 1 for p in result)
