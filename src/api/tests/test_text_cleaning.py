"""Tests for audio.text_cleaning — single source of truth for voice text cleaning."""

from audio.text_cleaning import clean_for_voice, strip_emojis


class TestCleanForVoice:
    def test_strips_emojis(self):
        assert clean_for_voice("Olá! 😊 Tudo bem?") == "Olá!  Tudo bem?"

    def test_strips_thinking_blocks(self):
        text = "<think>let me think about this</think>A resposta é 42."
        assert clean_for_voice(text) == "A resposta é 42."

    def test_strips_multiline_thinking(self):
        text = "<think>\nstep 1\nstep 2\n</think>\nResultado final."
        assert clean_for_voice(text) == "Resultado final."

    def test_strips_both_emoji_and_thinking(self):
        text = "<think>hmm</think>Olá! 🎉 Mundo!"
        result = clean_for_voice(text)
        assert "think" not in result
        assert "\U0001f389" not in result
        assert "Olá" in result

    def test_preserves_plain_text(self):
        text = "Texto simples sem emojis."
        assert clean_for_voice(text) == text

    def test_empty_string(self):
        assert clean_for_voice("") == ""

    def test_only_emojis(self):
        assert clean_for_voice("😊🎉🔥") == ""

    def test_strips_whitespace(self):
        assert clean_for_voice("  texto  ") == "texto"


class TestStripEmojis:
    def test_strips_emojis_only(self):
        assert strip_emojis("Hello 🌍 World") == "Hello  World"

    def test_preserves_thinking_blocks(self):
        text = "<think>x</think>Hello"
        assert "<think>" in strip_emojis(text)

    def test_empty_string(self):
        assert strip_emojis("") == ""
