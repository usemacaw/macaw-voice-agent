"""
Testes para o ParakeetSTT provider.

Testa o pipeline audio sem NeMo real (mock do modelo).
"""

import asyncio
import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock

import numpy as np
import pytest

# Paths para imports (src/ como raiz)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))


# Mock config
_mock_audio_config = {
    "sample_rate": 8000, "channels": 1, "sample_width": 2,
}
_mock_stt_config = {"provider": "parakeet", "language": "pt"}


@pytest.fixture(autouse=True)
def mock_configs():
    with patch.dict("common.config.STT_CONFIG", _mock_stt_config, clear=False):
        with patch.dict("common.config.AUDIO_CONFIG", _mock_audio_config, clear=False):
            yield


def _make_pcm_audio(duration_sec: float = 0.5, sample_rate: int = 8000) -> bytes:
    """Gera audio PCM16 sintetico (sine wave 440Hz)."""
    t = np.linspace(0, duration_sec, int(sample_rate * duration_sec), dtype=np.float32)
    samples = (np.sin(2 * np.pi * 440 * t) * 16000).astype(np.int16)
    return samples.tobytes()


class TestParakeetSTTTranscribe:
    """Testes para transcricao batch."""

    @pytest.fixture
    def mock_nemo(self):
        """Mock do nemo.collections.asr."""
        mock_model = MagicMock()

        # Mock Hypothesis object com .text
        mock_hypothesis = MagicMock()
        mock_hypothesis.text = "Texto transcrito pelo Parakeet"
        mock_model.transcribe.return_value = [mock_hypothesis]
        mock_model.to.return_value = mock_model

        mock_asr_module = MagicMock()
        mock_asr_module.models.ASRModel.from_pretrained.return_value = mock_model

        return mock_asr_module, mock_model

    @pytest.mark.asyncio
    async def test_transcribe_returns_text(self, mock_nemo):
        """Transcricao retorna texto do modelo."""
        mock_asr_module, mock_model = mock_nemo

        with patch.dict("sys.modules", {"nemo": MagicMock(), "nemo.collections": MagicMock(), "nemo.collections.asr": mock_asr_module}):
            from stt.providers.parakeet_stt import ParakeetSTT

            provider = ParakeetSTT()
            provider._model = mock_model

            audio = _make_pcm_audio(0.5)
            result = await provider.transcribe(audio)

            assert result == "Texto transcrito pelo Parakeet"
            mock_model.transcribe.assert_called_once()

    @pytest.mark.asyncio
    async def test_transcribe_empty_audio_returns_empty(self, mock_nemo):
        """Audio vazio retorna string vazia."""
        _, mock_model = mock_nemo

        from stt.providers.parakeet_stt import ParakeetSTT

        provider = ParakeetSTT()
        provider._model = mock_model

        result = await provider.transcribe(b"")
        assert result == ""
        mock_model.transcribe.assert_not_called()

    @pytest.mark.asyncio
    async def test_transcribe_not_connected_raises(self):
        """Transcricao sem connect() levanta RuntimeError."""
        from stt.providers.parakeet_stt import ParakeetSTT

        provider = ParakeetSTT()
        audio = _make_pcm_audio(0.5)

        with pytest.raises(RuntimeError, match="nao conectado"):
            await provider.transcribe(audio)

    @pytest.mark.asyncio
    async def test_transcribe_handles_string_result(self, mock_nemo):
        """NeMo pode retornar list[str] ao inves de list[Hypothesis]."""
        _, mock_model = mock_nemo
        mock_model.transcribe.return_value = ["Texto como string"]

        from stt.providers.parakeet_stt import ParakeetSTT

        provider = ParakeetSTT()
        provider._model = mock_model

        audio = _make_pcm_audio(0.5)
        result = await provider.transcribe(audio)

        assert result == "Texto como string"

    @pytest.mark.asyncio
    async def test_transcribe_empty_results(self, mock_nemo):
        """Lista vazia de resultados retorna string vazia."""
        _, mock_model = mock_nemo
        mock_model.transcribe.return_value = []

        from stt.providers.parakeet_stt import ParakeetSTT

        provider = ParakeetSTT()
        provider._model = mock_model

        audio = _make_pcm_audio(0.5)
        result = await provider.transcribe(audio)

        assert result == ""

    @pytest.mark.asyncio
    async def test_transcribe_resamples_to_16khz(self, mock_nemo):
        """Audio e resampled de 8kHz para 16kHz antes da transcricao."""
        _, mock_model = mock_nemo
        mock_hypothesis = MagicMock()
        mock_hypothesis.text = "ok"
        mock_model.transcribe.return_value = [mock_hypothesis]

        from stt.providers.parakeet_stt import ParakeetSTT

        provider = ParakeetSTT()
        provider._model = mock_model

        # 0.5s @ 8kHz = 4000 samples
        audio = _make_pcm_audio(0.5, sample_rate=8000)
        await provider.transcribe(audio)

        # Verifica que o modelo recebeu audio resampled
        call_args = mock_model.transcribe.call_args
        audio_list = call_args[0][0]  # primeiro arg posicional
        resampled = audio_list[0]

        # 0.5s @ 16kHz = 8000 samples (aproximado por resampling)
        assert len(resampled) == pytest.approx(8000, abs=100)


class TestParakeetSTTConnect:
    """Testes para connect/disconnect."""

    @pytest.mark.asyncio
    async def test_connect_loads_model(self):
        """Connect carrega modelo via NeMo e move para device."""
        mock_model = MagicMock()
        mock_model.to.return_value = mock_model

        mock_asr = MagicMock()
        mock_asr.models.ASRModel.from_pretrained.return_value = mock_model

        with patch.dict("sys.modules", {
            "nemo": MagicMock(),
            "nemo.collections": MagicMock(),
            "nemo.collections.asr": mock_asr,
        }):
            from stt.providers.parakeet_stt import ParakeetSTT

            provider = ParakeetSTT()
            provider._device = "cuda:0"
            await provider.connect()

            mock_asr.models.ASRModel.from_pretrained.assert_called_once_with(
                model_name="nvidia/parakeet-tdt-0.6b-v2",
            )
            assert provider._model is not None

    @pytest.mark.asyncio
    async def test_connect_cpu_skips_to(self):
        """Connect em CPU nao chama .to()."""
        mock_model = MagicMock()

        mock_asr = MagicMock()
        mock_asr.models.ASRModel.from_pretrained.return_value = mock_model

        with patch.dict("sys.modules", {
            "nemo": MagicMock(),
            "nemo.collections": MagicMock(),
            "nemo.collections.asr": mock_asr,
        }):
            from stt.providers.parakeet_stt import ParakeetSTT

            provider = ParakeetSTT()
            provider._device = "cpu"
            await provider.connect()

            mock_model.to.assert_not_called()

    @pytest.mark.asyncio
    async def test_disconnect_clears_model(self):
        """Disconnect libera modelo."""
        from stt.providers.parakeet_stt import ParakeetSTT

        provider = ParakeetSTT()
        provider._model = MagicMock()

        with patch("stt.providers.parakeet_stt.run_inference", new_callable=AsyncMock) as mock_run:
            mock_run.side_effect = lambda fn: fn()
            await provider.disconnect()

        assert provider._model is None


class TestParakeetSTTRegistration:
    """Testes para auto-registro do provider."""

    def test_provider_registered(self):
        """Parakeet esta registrado no auto-discovery."""
        from stt.providers.base import _KNOWN_STT_MODULES, _STT_PROVIDERS

        assert "parakeet" in _KNOWN_STT_MODULES
        assert "parakeet" in _STT_PROVIDERS

    def test_provider_name(self):
        """provider_name esta correto."""
        from stt.providers.parakeet_stt import ParakeetSTT

        assert ParakeetSTT.provider_name == "parakeet"
