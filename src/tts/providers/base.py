"""
Text-to-Speech (TTS) - Converte texto em audio

Interface ABC pura para providers de TTS.
Implementadores devem criar suas proprias classes herdando de TTSProvider.

Inclui MockTTS como referencia e para testes sem providers reais.
"""

import importlib
import logging
from abc import ABC, abstractmethod
from typing import AsyncGenerator, Dict, Type

import numpy as np

from common.config import TTS_CONFIG, AUDIO_CONFIG

logger = logging.getLogger("ai-agent.tts")


# ==================== Base TTS Interface ====================

class TTSProvider(ABC):
    """Interface base para provedores de TTS.

    Implementadores devem:
    1. Herdar desta classe
    2. Implementar connect(), disconnect(), synthesize()
    3. Sobrescrever synthesize_stream() para streaming real (opcional)
    4. Registrar com register_tts_provider()

    Exemplo:
        class MeuTTS(TTSProvider):
            provider_name = "meu-tts"

            async def connect(self):
                self._engine = init_engine()

            async def disconnect(self):
                self._engine.close()

            async def synthesize(self, text: str) -> bytes:
                return self._engine.speak(text)

        register_tts_provider("meu-tts", MeuTTS)
    """

    provider_name: str = "base"

    async def connect(self) -> None:
        """Inicializa o provider (carrega modelos, conecta APIs, etc)."""
        logger.info(f"{self.provider_name} connected")

    async def disconnect(self) -> None:
        """Libera recursos do provider."""
        logger.info(f"{self.provider_name} disconnected")

    @abstractmethod
    async def synthesize(self, text: str) -> bytes:
        """Converte texto em audio PCM 8kHz mono 16-bit.

        Args:
            text: Texto para sintetizar.

        Returns:
            Bytes de audio PCM, ou b"" em caso de erro.
        """
        pass

    async def synthesize_stream(self, text: str) -> AsyncGenerator[bytes, None]:
        """Converte texto em audio com streaming.

        Yield chunks de audio conforme sao gerados.
        Implementacao default faz fallback para synthesize() e chunka o resultado.
        """
        audio = await self.synthesize(text)
        if audio:
            chunk_size = int(AUDIO_CONFIG["sample_rate"] * 0.1 * 2)  # 100ms
            for i in range(0, len(audio), chunk_size):
                yield audio[i:i + chunk_size]

    @property
    def supports_streaming(self) -> bool:
        """Indica se o provedor suporta streaming real."""
        return False


# ==================== Mock TTS Provider ====================

class MockTTS(TTSProvider):
    """TTS mock para testes (gera tom).

    Serve como referencia de implementacao e permite rodar o sistema
    sem providers reais instalados.
    """

    provider_name = "mock"

    async def connect(self) -> None:
        logger.info("Mock TTS inicializado (modo teste)")

    async def disconnect(self) -> None:
        logger.info("Mock TTS desconectado")

    async def synthesize(self, text: str) -> bytes:
        """Gera tom de teste proporcional ao tamanho do texto."""
        sample_rate = AUDIO_CONFIG["sample_rate"]
        duration = max(1.0, len(text) * 0.05)
        frequency = 440
        n_samples = int(sample_rate * duration)

        t = np.arange(n_samples, dtype=np.float32) / sample_rate
        envelope = np.minimum(1.0, t * 10) * np.minimum(1.0, (duration - t) * 10)
        samples = (16000 * envelope * np.sin(2 * np.pi * frequency * t)).astype(np.int16)

        pcm_data = samples.tobytes()
        logger.info(f"TTS (mock): {len(pcm_data)} bytes")
        return pcm_data


# ==================== Factory ====================

_TTS_PROVIDERS: Dict[str, Type[TTSProvider]] = {
    "mock": MockTTS,
}

# Modulos conhecidos para auto-discovery (importados sob demanda)
_KNOWN_TTS_MODULES = {
    "qwen": "tts.providers.qwen_tts",
    "kokoro": "tts.providers.kokoro_tts",
    "faster": "tts.providers.faster_tts",
}


def register_tts_provider(name: str, cls: Type[TTSProvider]) -> None:
    """Registra um provider TTS customizado.

    Args:
        name: Nome do provider (usado em TTS_PROVIDER env var).
        cls: Classe que herda de TTSProvider.
    """
    _TTS_PROVIDERS[name] = cls
    logger.info(f"TTS provider registrado: {name}")


async def create_tts_provider(provider_name: str = None) -> TTSProvider:
    """Factory assincrona para criar e conectar provider TTS.

    Args:
        provider_name: Nome do provider. Se None, usa TTS_CONFIG['provider'].

    Raises:
        ValueError: Se provider nao esta registrado.
    """
    name = provider_name or TTS_CONFIG.get("provider", "mock")

    # Auto-discovery: importa modulo conhecido que se auto-registra
    if name not in _TTS_PROVIDERS and name in _KNOWN_TTS_MODULES:
        try:
            importlib.import_module(_KNOWN_TTS_MODULES[name])
        except ImportError as e:
            raise ValueError(
                f"TTS provider '{name}' requer dependencias adicionais: {e}"
            ) from e

    if name not in _TTS_PROVIDERS:
        available = ", ".join(_TTS_PROVIDERS.keys())
        raise ValueError(
            f"TTS provider '{name}' nao registrado. "
            f"Disponiveis: {available}. "
            f"Use register_tts_provider() para registrar novos providers."
        )

    tts = _TTS_PROVIDERS[name]()
    await tts.connect()
    return tts
