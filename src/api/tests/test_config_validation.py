"""Tests for SessionConfig input validation."""

import pytest

from protocol.models import SessionConfig, SessionConfigValidationError


class TestSessionConfigValidation:
    """Test that SessionConfig.update() validates input from WebSocket clients."""

    def _config(self) -> SessionConfig:
        return SessionConfig()

    def test_valid_update(self):
        cfg = self._config()
        cfg.update({"temperature": 0.5, "instructions": "Be helpful"})
        assert cfg.temperature == 0.5
        assert cfg.instructions == "Be helpful"

    def test_temperature_not_a_number(self):
        cfg = self._config()
        with pytest.raises(SessionConfigValidationError, match="temperature must be a number"):
            cfg.update({"temperature": "banana"})

    def test_temperature_too_high(self):
        cfg = self._config()
        with pytest.raises(SessionConfigValidationError, match="between 0.0 and 2.0"):
            cfg.update({"temperature": 5.0})

    def test_temperature_negative(self):
        cfg = self._config()
        with pytest.raises(SessionConfigValidationError, match="between 0.0 and 2.0"):
            cfg.update({"temperature": -0.1})

    def test_modalities_not_a_list(self):
        cfg = self._config()
        with pytest.raises(SessionConfigValidationError, match="list of strings"):
            cfg.update({"modalities": "text"})

    def test_modalities_invalid_value(self):
        cfg = self._config()
        with pytest.raises(SessionConfigValidationError, match="Invalid modalities"):
            cfg.update({"modalities": ["text", "video"]})

    def test_modalities_valid(self):
        cfg = self._config()
        cfg.update({"modalities": ["text"]})
        assert cfg.modalities == ["text"]

    def test_instructions_not_string(self):
        cfg = self._config()
        with pytest.raises(SessionConfigValidationError, match="instructions must be a string"):
            cfg.update({"instructions": 42})

    def test_voice_not_string(self):
        cfg = self._config()
        with pytest.raises(SessionConfigValidationError, match="voice must be a string"):
            cfg.update({"voice": 123})

    def test_input_audio_format_invalid(self):
        cfg = self._config()
        with pytest.raises(SessionConfigValidationError, match="Invalid input_audio_format"):
            cfg.update({"input_audio_format": "mp3"})

    def test_output_audio_format_invalid(self):
        cfg = self._config()
        with pytest.raises(SessionConfigValidationError, match="Invalid output_audio_format"):
            cfg.update({"output_audio_format": "wav"})

    def test_audio_format_valid(self):
        cfg = self._config()
        cfg.update({
            "input_audio_format": "g711_ulaw",
            "output_audio_format": "g711_alaw",
        })
        assert cfg.input_audio_format == "g711_ulaw"
        assert cfg.output_audio_format == "g711_alaw"

    def test_tools_not_a_list(self):
        cfg = self._config()
        with pytest.raises(SessionConfigValidationError, match="tools must be a list"):
            cfg.update({"tools": "not a list"})

    def test_max_tokens_not_int(self):
        cfg = self._config()
        with pytest.raises(SessionConfigValidationError, match="integer or 'inf'"):
            cfg.update({"max_response_output_tokens": "banana"})

    def test_max_tokens_negative(self):
        cfg = self._config()
        with pytest.raises(SessionConfigValidationError, match="positive"):
            cfg.update({"max_response_output_tokens": -10})

    def test_max_tokens_inf_valid(self):
        cfg = self._config()
        cfg.update({"max_response_output_tokens": "inf"})
        assert cfg.max_response_output_tokens == "inf"

    def test_max_tokens_int_valid(self):
        cfg = self._config()
        cfg.update({"max_response_output_tokens": 2048})
        assert cfg.max_response_output_tokens == 2048

    def test_turn_detection_none_valid(self):
        cfg = self._config()
        cfg.update({"turn_detection": None})
        assert cfg.turn_detection is None

    def test_turn_detection_dict_valid(self):
        cfg = self._config()
        cfg.update({"turn_detection": {"type": "server_vad", "threshold": 0.7}})
        assert cfg.turn_detection is not None
        assert cfg.turn_detection.threshold == 0.7

    def test_partial_update_preserves_other_fields(self):
        cfg = self._config()
        original_voice = cfg.voice
        cfg.update({"temperature": 1.0})
        assert cfg.voice == original_voice
        assert cfg.temperature == 1.0

    def test_invalid_field_does_not_modify_config(self):
        cfg = self._config()
        original_temp = cfg.temperature
        with pytest.raises(SessionConfigValidationError):
            cfg.update({"temperature": "invalid"})
        # temperature should be unchanged since validation failed before assignment
        assert cfg.temperature == original_temp
