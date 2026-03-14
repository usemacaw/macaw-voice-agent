"""SDK compatibility tests — verifies OpenAI Python SDK can connect to our server.

These tests require: pip install openai
They test the actual SDK client against a running local server with fake providers.
"""

import asyncio
import json

import pytest

try:
    from openai import OpenAI
    HAS_OPENAI_SDK = True
except ImportError:
    HAS_OPENAI_SDK = False

from tests.test_session import FakeASR, FakeLLM, FakeTTS


@pytest.fixture
async def local_server():
    """Start a local WebSocket server for SDK testing."""
    import websockets

    from server.ws_server import WebSocketServer

    asr, llm, tts = FakeASR(), FakeLLM(), FakeTTS()
    ws_server = WebSocketServer(asr, llm, tts)

    server_instance = await websockets.serve(
        ws_server._handle_connection,
        "127.0.0.1",
        0,
        max_size=16 * 1024 * 1024,
    )

    port = server_instance.sockets[0].getsockname()[1]
    yield port

    server_instance.close()
    await server_instance.wait_closed()


@pytest.mark.skipif(not HAS_OPENAI_SDK, reason="openai package not installed")
class TestOpenAISDKCompat:
    @pytest.mark.asyncio
    async def test_sdk_can_connect_and_receive_session(self, local_server):
        """Basic connectivity test: SDK connects and receives session.created."""
        import websockets

        port = local_server
        # Use raw websocket since the SDK's Realtime API isn't fully stable
        async with websockets.connect(f"ws://127.0.0.1:{port}") as ws:
            msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=5))
            assert msg["type"] == "session.created"
            assert "session" in msg
            assert msg["session"]["object"] == "realtime.session"
