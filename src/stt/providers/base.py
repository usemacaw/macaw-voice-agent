"""
Speech-to-Text (STT/ASR) - Converte audio em texto

Interface ABC pura para providers de STT.
Implementadores devem criar suas proprias classes herdando de STTProvider.

Inclui MockSTT como referencia e para testes sem providers reais.
"""

import importlib
import logging
from abc import ABC, abstractmethod
from typing import Dict, Type

from common.config import STT_CONFIG

logger = logging.getLogger("ai-agent.stt")


# ==================== Base STT Interface ====================

class STTProvider(ABC):
    """Interface base para provedores de STT.

    Implementadores devem:
    1. Herdar desta classe
    2. Implementar connect(), disconnect(), transcribe()
    3. Opcionalmente implementar streaming: start_streaming(), process_chunk(), finish_streaming()
    4. Registrar com register_stt_provider()

    Exemplo (batch):
        class MeuSTT(STTProvider):
            provider_name = "meu-stt"

            async def connect(self):
                self._model = load_model()

            async def disconnect(self):
                del self._model

            async def transcribe(self, audio_data: bytes) -> str:
                return self._model.transcribe(audio_data)

        register_stt_provider("meu-stt", MeuSTT)

    Exemplo (streaming):
        class MeuStreamingSTT(STTProvider):
            provider_name = "meu-stt-streaming"

            @property
            def supports_streaming(self) -> bool:
                return True

            async def start_streaming(self, stream_id: str = "") -> None:
                self._state = self._model.init_streaming()

            async def process_chunk(self, audio_chunk: bytes, stream_id: str = "") -> str:
                return self._model.transcribe_chunk(audio_chunk, self._state)

            async def finish_streaming(self, stream_id: str = "") -> str:
                return self._model.finish(self._state)
    """

    provider_name: str = "base"

    async def connect(self) -> None:
        """Inicializa o provider (carrega modelos, conecta APIs, etc)."""
        logger.info(f"{self.provider_name} connected")

    async def disconnect(self) -> None:
        """Libera recursos do provider."""
        logger.info(f"{self.provider_name} disconnected")

    @abstractmethod
    async def transcribe(self, audio_data: bytes) -> str:
        """Transcreve audio PCM para texto (modo batch).

        Args:
            audio_data: Audio em formato PCM (sample_rate, channels, sample_width
                       conforme AUDIO_CONFIG).

        Returns:
            Texto transcrito, ou string vazia se nada foi detectado.
        """
        pass

    # ==================== Streaming STT Interface ====================

    @property
    def supports_streaming(self) -> bool:
        """Indica se o provider suporta transcricao incremental (streaming)."""
        return False

    async def start_streaming(self, stream_id: str = "") -> None:
        """Inicia sessao de streaming STT.

        Deve ser chamado antes de process_chunk(). Pode ser chamado
        multiplas vezes para reiniciar a sessao (ex: apos speech_end).

        Args:
            stream_id: Identificador do stream (ex: session_id). Permite
                       multiplos streams concorrentes no mesmo provider singleton.

        Raises:
            NotImplementedError: Se provider nao suporta streaming.
        """
        raise NotImplementedError(f"{self.provider_name} nao suporta streaming STT")

    async def process_chunk(self, audio_chunk: bytes, stream_id: str = "") -> str:
        """Processa chunk de audio e retorna transcricao parcial acumulada.

        Args:
            audio_chunk: Chunk de audio PCM (mesmo formato de transcribe()).
            stream_id: Identificador do stream (deve corresponder ao start_streaming).

        Returns:
            Transcricao parcial acumulada ate o momento.

        Raises:
            NotImplementedError: Se provider nao suporta streaming.
        """
        raise NotImplementedError(f"{self.provider_name} nao suporta streaming STT")

    async def finish_streaming(self, stream_id: str = "") -> str:
        """Finaliza sessao de streaming e retorna transcricao final.

        Args:
            stream_id: Identificador do stream (deve corresponder ao start_streaming).

        Returns:
            Transcricao final completa.

        Raises:
            NotImplementedError: Se provider nao suporta streaming.
        """
        raise NotImplementedError(f"{self.provider_name} nao suporta streaming STT")



# ==================== Mock STT Provider ====================

class MockSTT(STTProvider):
    """STT mock para testes (retorna texto fixo).

    Serve como referencia de implementacao e permite rodar o sistema
    sem providers reais instalados.
    """

    provider_name = "mock"

    async def connect(self) -> None:
        logger.info("Mock STT inicializado (modo teste)")

    async def disconnect(self) -> None:
        logger.info("Mock STT desconectado")

    async def transcribe(self, audio_data: bytes) -> str:
        """Retorna texto fixo para qualquer audio."""
        text = "Texto de teste do mock STT."
        logger.info(f"STT (mock): '{text}'")
        return text


# ==================== Factory ====================

_STT_PROVIDERS: Dict[str, Type[STTProvider]] = {
    "mock": MockSTT,
}

# Modulos conhecidos para auto-discovery (importados sob demanda)
_KNOWN_STT_MODULES = {
    "qwen": "stt.providers.qwen_stt",
    "qwen-streaming": "stt.providers.qwen_stt",
    "parakeet": "stt.providers.parakeet_stt",
    "parakeet-streaming": "stt.providers.parakeet_stt",
    "qwen-native-streaming": "stt.providers.qwen_streaming_stt",
}


def register_stt_provider(name: str, cls: Type[STTProvider]) -> None:
    """Registra um provider STT customizado.

    Args:
        name: Nome do provider (usado em STT_PROVIDER env var).
        cls: Classe que herda de STTProvider.
    """
    _STT_PROVIDERS[name] = cls
    logger.info(f"STT provider registrado: {name}")


async def create_stt_provider(provider_name: str = None) -> STTProvider:
    """Factory assincrona para criar e conectar provider STT.

    Args:
        provider_name: Nome do provider. Se None, usa STT_CONFIG['provider'].

    Raises:
        ValueError: Se provider nao esta registrado.
    """
    name = provider_name or STT_CONFIG.get("provider", "mock")

    # Auto-discovery: importa modulo conhecido que se auto-registra
    if name not in _STT_PROVIDERS and name in _KNOWN_STT_MODULES:
        try:
            importlib.import_module(_KNOWN_STT_MODULES[name])
        except ImportError as e:
            raise ValueError(
                f"STT provider '{name}' requer dependencias adicionais: {e}"
            ) from e

    if name not in _STT_PROVIDERS:
        available = ", ".join(_STT_PROVIDERS.keys())
        raise ValueError(
            f"STT provider '{name}' nao registrado. "
            f"Disponiveis: {available}. "
            f"Use register_stt_provider() para registrar novos providers."
        )

    stt = _STT_PROVIDERS[name]()
    await stt.connect()
    return stt
