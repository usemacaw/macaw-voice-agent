"""Integration tests for WebSocket server."""

import asyncio
import json

import pytest
import websockets

from tests.test_session import FakeASR, FakeLLM, FakeTTS


@pytest.fixture
async def server():
    """Start a WebSocket server on a random port for testing."""
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
    yield f"ws://127.0.0.1:{port}"

    server_instance.close()
    await server_instance.wait_closed()


class TestWebSocketIntegration:
    @pytest.mark.asyncio
    async def test_connect_receives_session_created(self, server):
        async with websockets.connect(server) as ws:
            msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=5))
            assert msg["type"] == "session.created"
            assert "session" in msg

            msg2 = json.loads(await asyncio.wait_for(ws.recv(), timeout=5))
            assert msg2["type"] == "conversation.created"

    @pytest.mark.asyncio
    async def test_session_update_flow(self, server):
        async with websockets.connect(server) as ws:
            await ws.recv()  # session.created
            await ws.recv()  # conversation.created

            await ws.send(json.dumps({
                "type": "session.update",
                "session": {"instructions": "Test instructions"},
            }))

            msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=5))
            assert msg["type"] == "session.updated"
            assert msg["session"]["instructions"] == "Test instructions"

    @pytest.mark.asyncio
    async def test_text_response_flow(self, server):
        async with websockets.connect(server) as ws:
            await ws.recv()  # session.created
            await ws.recv()  # conversation.created

            # Switch to text-only mode
            await ws.send(json.dumps({
                "type": "session.update",
                "session": {"modalities": ["text"]},
            }))
            await ws.recv()  # session.updated

            # Create user message
            await ws.send(json.dumps({
                "type": "conversation.item.create",
                "item": {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "Hello"}],
                },
            }))
            await ws.recv()  # conversation.item.created

            # Request response
            await ws.send(json.dumps({"type": "response.create"}))

            # Collect response events
            response_events = []
            for _ in range(30):
                try:
                    msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=5))
                    response_events.append(msg)
                    if msg["type"] == "response.done":
                        break
                except asyncio.TimeoutError:
                    break

            types = [e["type"] for e in response_events]
            assert "response.created" in types
            assert "response.done" in types
