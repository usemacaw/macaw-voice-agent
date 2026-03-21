"""Tests for IncrementalSplitter — incremental sentence splitting for streaming TTS."""

from pipeline.sentence_splitter import IncrementalSplitter


class TestIncrementalSplitter:
    def test_complete_sentence(self):
        s = IncrementalSplitter()
        result = s.feed("Olá mundo.")
        assert result == ["Olá mundo."]

    def test_incomplete_sentence_buffered(self):
        s = IncrementalSplitter()
        assert s.feed("Olá mun") == []
        assert s.feed("do.") == ["Olá mundo."]

    def test_multiple_sentences_incremental(self):
        """Sentences are split incrementally, not from a single buffer."""
        s = IncrementalSplitter()
        # First feed ends with period (sentence end at $)
        result = s.feed("Primeira.")
        assert result == ["Primeira."]
        result = s.feed(" Segunda.")
        assert result == ["Segunda."]

    def test_eager_first_sentence_at_clause_break(self):
        s = IncrementalSplitter(min_eager_chars=10)
        # Clause break at end of buffer triggers eager first sentence
        result = s.feed("Isto é um texto longo,")
        assert len(result) == 1
        assert result[0].endswith(",")

    def test_no_eager_if_short(self):
        s = IncrementalSplitter(min_eager_chars=100)
        # Buffer too short for eager clause break
        result = s.feed("Curto,")
        assert result == []

    def test_flush_remaining(self):
        s = IncrementalSplitter()
        s.feed("Texto sem pontuação final")
        remaining = s.flush()
        assert remaining == "Texto sem pontuação final"

    def test_flush_empty(self):
        s = IncrementalSplitter()
        assert s.flush() is None

    def test_flush_after_complete_sentence(self):
        s = IncrementalSplitter()
        s.feed("Completa.")
        assert s.flush() is None

    def test_question_mark(self):
        s = IncrementalSplitter()
        result = s.feed("Como você está?")
        assert result == ["Como você está?"]

    def test_exclamation_mark(self):
        s = IncrementalSplitter()
        result = s.feed("Que legal!")
        assert result == ["Que legal!"]

    def test_incremental_streaming_simulation(self):
        """Simulate token-by-token LLM streaming."""
        s = IncrementalSplitter()
        tokens = ["Eu ", "gosto ", "de ", "café. ", "E ", "de ", "chá."]
        sentences = []
        for token in tokens:
            sentences.extend(s.feed(token))
        remaining = s.flush()
        if remaining:
            sentences.append(remaining)
        assert sentences == ["Eu gosto de café.", "E de chá."]

    def test_only_clause_break_no_eager_on_second(self):
        """After first sentence, only split at sentence-end punctuation."""
        s = IncrementalSplitter(min_eager_chars=5)
        s.feed("Primeira.")  # First sentence done
        result = s.feed("Segunda parte, com vírgula mas sem ponto")
        assert result == []  # Clause break NOT used after first sentence
