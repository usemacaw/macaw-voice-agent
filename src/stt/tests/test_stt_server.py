"""
Testes para o STT gRPC Server.

Testa o servicer usando MockSTT como provider (sem deps externas).
"""

import asyncio
import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Paths para imports (src/ como raiz)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))


# Mock minimo de config para evitar carregar ai-agent/config.py inteiro
_mock_stt_config = {"provider": "mock", "language": "pt"}
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
    with patch.dict("common.config.STT_CONFIG", _mock_stt_config, clear=False):
        with patch.dict("common.config.AUDIO_CONFIG", _mock_audio_config, clear=False):
            yield


# ==================== STTServicer Tests ====================


class TestSTTServicerTranscribe:
    """Testes para Transcribe (batch)."""

    @pytest.fixture
    def mock_provider(self):
        provider = AsyncMock()
        provider.provider_name = "mock"
        provider.transcribe = AsyncMock(return_value="Texto transcrito")
        provider.supports_streaming = False
        return provider

    @pytest.fixture
    def servicer(self, mock_provider):
        from stt.server import STTServicer
        return STTServicer(mock_provider)

    @pytest.fixture
    def mock_context(self):
        ctx = MagicMock()
        ctx.set_code = MagicMock()
        ctx.set_details = MagicMock()
        return ctx

    @pytest.mark.asyncio
    async def test_transcribe_success(self, servicer, mock_provider, mock_context):
        """Transcribe retorna texto do provider."""
        from shared.grpc_gen import stt_service_pb2 as stt_pb

        request = stt_pb.TranscribeRequest(audio_data=b"\x00" * 1000)
        response = await servicer.Transcribe(request, mock_context)

        assert response.text == "Texto transcrito"
        assert response.processing_ms > 0
        mock_provider.transcribe.assert_awaited_once_with(b"\x00" * 1000)

    @pytest.mark.asyncio
    async def test_transcribe_empty_audio_invalid(self, servicer, mock_context):
        """Audio vazio retorna INVALID_ARGUMENT."""
        from shared.grpc_gen import stt_service_pb2 as stt_pb
        import grpc

        request = stt_pb.TranscribeRequest(audio_data=b"")
        await servicer.Transcribe(request, mock_context)

        mock_context.set_code.assert_called_with(grpc.StatusCode.INVALID_ARGUMENT)

    @pytest.mark.asyncio
    async def test_transcribe_provider_error(self, servicer, mock_provider, mock_context):
        """Erro no provider retorna INTERNAL."""
        from shared.grpc_gen import stt_service_pb2 as stt_pb
        import grpc

        mock_provider.transcribe = AsyncMock(side_effect=RuntimeError("Model crash"))

        request = stt_pb.TranscribeRequest(audio_data=b"\x00" * 1000)
        await servicer.Transcribe(request, mock_context)

        mock_context.set_code.assert_called_with(grpc.StatusCode.INTERNAL)

    @pytest.mark.asyncio
    async def test_transcribe_none_returns_empty(self, servicer, mock_provider, mock_context):
        """Provider retorna None -> response.text vazio."""
        from shared.grpc_gen import stt_service_pb2 as stt_pb

        mock_provider.transcribe = AsyncMock(return_value=None)

        request = stt_pb.TranscribeRequest(audio_data=b"\x00" * 1000)
        response = await servicer.Transcribe(request, mock_context)

        assert response.text == ""


class TestSTTServicerStream:
    """Testes para TranscribeStream."""

    @pytest.fixture
    def streaming_provider(self):
        provider = AsyncMock()
        provider.provider_name = "mock-streaming"
        provider.supports_streaming = True
        provider.start_streaming = AsyncMock()
        provider.process_chunk = AsyncMock(return_value="parcial")
        provider.finish_streaming = AsyncMock(return_value="texto final")
        return provider

    @pytest.fixture
    def servicer(self, streaming_provider):
        from stt.server import STTServicer
        return STTServicer(streaming_provider)

    @pytest.fixture
    def mock_context(self):
        ctx = MagicMock()
        ctx.set_code = MagicMock()
        ctx.set_details = MagicMock()
        return ctx

    @pytest.mark.asyncio
    async def test_stream_start_and_chunks(self, servicer, streaming_provider, mock_context):
        """Streaming: envia chunks e recebe parciais."""
        from shared.grpc_gen import stt_service_pb2 as stt_pb

        async def request_iter():
            yield stt_pb.AudioChunk(stream_id="s1", audio_payload=b"\x00" * 100)
            yield stt_pb.AudioChunk(stream_id="s1", audio_payload=b"\x00" * 100)
            yield stt_pb.AudioChunk(stream_id="s1", end_of_stream=True)

        results = []
        async for result in servicer.TranscribeStream(request_iter(), mock_context):
            results.append(result)

        # Deve ter parciais + final
        assert any(r.is_final for r in results)
        streaming_provider.start_streaming.assert_awaited_once()
        streaming_provider.finish_streaming.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_stream_non_streaming_provider_unimplemented(self, mock_context):
        """Provider sem streaming retorna UNIMPLEMENTED."""
        from stt.server import STTServicer
        from shared.grpc_gen import stt_service_pb2 as stt_pb
        import grpc

        provider = AsyncMock()
        provider.supports_streaming = False
        provider.provider_name = "batch-only"

        servicer = STTServicer(provider)

        async def request_iter():
            yield stt_pb.AudioChunk(stream_id="s1", audio_payload=b"\x00")

        results = []
        async for r in servicer.TranscribeStream(request_iter(), mock_context):
            results.append(r)

        mock_context.set_code.assert_called_with(grpc.StatusCode.UNIMPLEMENTED)
        assert len(results) == 0


# ==================== STTServer Lifecycle Tests ====================


class TestSTTServerLifecycle:
    """Testa start/stop do servidor."""

    @pytest.mark.asyncio
    async def test_server_starts_and_stops(self):
        """Server inicia na porta e para gracefully."""
        from stt.server import STTServer

        with patch("stt.server._create_provider") as mock_create:
            mock_provider = AsyncMock()
            mock_provider.provider_name = "mock"
            mock_provider.disconnect = AsyncMock()
            mock_create.return_value = mock_provider

            server = STTServer()
            await server.start(host="127.0.0.1", port=0)  # Porta 0 = aleatorio

            assert server._micro._server is not None
            assert server._provider is mock_provider

            await server.stop()
            mock_provider.disconnect.assert_awaited_once()
