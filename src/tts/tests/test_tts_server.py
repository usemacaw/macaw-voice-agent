"""
Testes para o TTS gRPC Server.

Testa o servicer usando MockTTS como provider (sem deps externas).
"""

import asyncio
import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Paths para imports (src/ como raiz)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))


_mock_tts_config = {"provider": "mock", "language": "pt"}
_mock_audio_config = {
    "sample_rate": 8000, "channels": 1, "sample_width": 2,
    "frame_duration_ms": 20, "vad_aggressiveness": 2,
    "silence_threshold_ms": 500, "min_speech_ms": 250,
    "energy_threshold": 500, "max_buffer_seconds": 60,
    "vad_ring_buffer_size": 5, "vad_speech_ratio_threshold": 0.4,
    "chunk_size_bytes": 2000, "speech_end_debounce_ms": 300,
}


@pytest.fixture(autouse=True)
def mock_configs():
    with patch.dict("common.config.TTS_CONFIG", _mock_tts_config, clear=False):
        with patch.dict("common.config.AUDIO_CONFIG", _mock_audio_config, clear=False):
            yield


# ==================== TTSServicer Tests ====================


class TestTTSServicerSynthesize:
    """Testes para Synthesize (batch)."""

    @pytest.fixture
    def mock_provider(self):
        provider = AsyncMock()
        provider.provider_name = "mock"
        provider.synthesize = AsyncMock(return_value=b"\x00" * 16000)
        provider.supports_streaming = False
        return provider

    @pytest.fixture
    def servicer(self, mock_provider):
        from tts.server import TTSServicer
        return TTSServicer(mock_provider)

    @pytest.fixture
    def mock_context(self):
        ctx = MagicMock()
        ctx.set_code = MagicMock()
        ctx.set_details = MagicMock()
        return ctx

    @pytest.mark.asyncio
    async def test_synthesize_success(self, servicer, mock_provider, mock_context):
        """Synthesize retorna audio do provider."""
        from shared.grpc_gen import tts_service_pb2 as tts_pb

        request = tts_pb.SynthesizeRequest(text="Olá mundo", language="pt")
        response = await servicer.Synthesize(request, mock_context)

        assert len(response.audio_data) == 16000
        assert response.processing_ms > 0
        assert response.duration_ms > 0
        mock_provider.synthesize.assert_awaited_once_with("Olá mundo")

    @pytest.mark.asyncio
    async def test_synthesize_empty_text_invalid(self, servicer, mock_context):
        """Texto vazio retorna INVALID_ARGUMENT."""
        from shared.grpc_gen import tts_service_pb2 as tts_pb
        import grpc

        request = tts_pb.SynthesizeRequest(text="")
        await servicer.Synthesize(request, mock_context)

        mock_context.set_code.assert_called_with(grpc.StatusCode.INVALID_ARGUMENT)

    @pytest.mark.asyncio
    async def test_synthesize_whitespace_only_invalid(self, servicer, mock_context):
        """Texto com apenas espacos retorna INVALID_ARGUMENT."""
        from shared.grpc_gen import tts_service_pb2 as tts_pb
        import grpc

        request = tts_pb.SynthesizeRequest(text="   ")
        await servicer.Synthesize(request, mock_context)

        mock_context.set_code.assert_called_with(grpc.StatusCode.INVALID_ARGUMENT)

    @pytest.mark.asyncio
    async def test_synthesize_provider_error(self, servicer, mock_provider, mock_context):
        """Erro no provider retorna INTERNAL."""
        from shared.grpc_gen import tts_service_pb2 as tts_pb
        import grpc

        mock_provider.synthesize = AsyncMock(side_effect=RuntimeError("OOM"))

        request = tts_pb.SynthesizeRequest(text="Teste")
        await servicer.Synthesize(request, mock_context)

        mock_context.set_code.assert_called_with(grpc.StatusCode.INTERNAL)


class TestTTSServicerStream:
    """Testes para SynthesizeStream."""

    @pytest.fixture
    def mock_provider(self):
        provider = AsyncMock()
        provider.provider_name = "mock"

        async def fake_stream(text):
            yield b"\x00" * 1600
            yield b"\x00" * 1600

        provider.synthesize_stream = fake_stream
        provider.supports_streaming = True
        return provider

    @pytest.fixture
    def servicer(self, mock_provider):
        from tts.server import TTSServicer
        return TTSServicer(mock_provider)

    @pytest.fixture
    def mock_context(self):
        ctx = MagicMock()
        ctx.set_code = MagicMock()
        ctx.set_details = MagicMock()
        return ctx

    @pytest.mark.asyncio
    async def test_stream_yields_chunks(self, servicer, mock_context):
        """SynthesizeStream yield chunks e finaliza com is_last."""
        from shared.grpc_gen import tts_service_pb2 as tts_pb

        request = tts_pb.SynthesizeRequest(text="Olá", language="pt")

        chunks = []
        async for chunk in servicer.SynthesizeStream(request, mock_context):
            chunks.append(chunk)

        # 2 chunks de audio + 1 chunk final (is_last)
        assert len(chunks) == 3
        assert chunks[-1].is_last is True
        assert chunks[0].audio_payload == b"\x00" * 1600

    @pytest.mark.asyncio
    async def test_stream_empty_text_invalid(self, servicer, mock_context):
        """Texto vazio retorna INVALID_ARGUMENT."""
        from shared.grpc_gen import tts_service_pb2 as tts_pb
        import grpc

        request = tts_pb.SynthesizeRequest(text="")

        chunks = []
        async for chunk in servicer.SynthesizeStream(request, mock_context):
            chunks.append(chunk)

        mock_context.set_code.assert_called_with(grpc.StatusCode.INVALID_ARGUMENT)
        assert len(chunks) == 0


# ==================== TTSServer Lifecycle Tests ====================


class TestTTSServerLifecycle:

    @pytest.mark.asyncio
    async def test_server_starts_and_stops(self):
        """Server inicia e para gracefully."""
        from tts.server import TTSServer

        with patch("tts.server._create_provider") as mock_create:
            mock_provider = AsyncMock()
            mock_provider.provider_name = "mock"
            mock_provider.disconnect = AsyncMock()
            mock_create.return_value = mock_provider

            server = TTSServer()
            await server.start(host="127.0.0.1", port=0)

            assert server._server is not None
            assert server._provider is mock_provider

            await server.stop()
            mock_provider.disconnect.assert_awaited_once()
