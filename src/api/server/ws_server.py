"""
WebSocket server for OpenAI Realtime API compatibility.

Accepts connections on /v1/realtime, authenticates via Bearer token
or query parameter, and creates a RealtimeSession per connection.
Also serves a /health endpoint for readiness probes.
"""

from __future__ import annotations

import asyncio
import hmac
import json
import logging
import os
from typing import TYPE_CHECKING
from urllib.parse import urlparse, parse_qs

import websockets
from websockets.asyncio.server import ServerConnection

from config import WS_CONFIG
from server.session import RealtimeSession

if TYPE_CHECKING:
    from providers.asr import ASRProvider
    from providers.llm import LLMProvider
    from providers.tts import TTSProvider
    from tools.registry import ToolRegistry

logger = logging.getLogger("open-voice-api.ws-server")

# Allowed WebSocket origins (comma-separated). Empty = allow all.
_ALLOWED_ORIGINS_RAW = os.getenv("WS_ALLOWED_ORIGINS", "")
_ALLOWED_ORIGINS: frozenset[str] | None = (
    frozenset(o.strip().lower() for o in _ALLOWED_ORIGINS_RAW.split(",") if o.strip())
    if _ALLOWED_ORIGINS_RAW
    else None
)


class WebSocketServer:
    """WebSocket server that creates RealtimeSession per connection."""

    def __init__(
        self,
        asr: ASRProvider,
        llm: LLMProvider,
        tts: TTSProvider,
        tool_registry: ToolRegistry | None = None,
    ):
        self._asr = asr
        self._llm = llm
        self._tts = tts
        self._tool_registry = tool_registry
        self._active_sessions: set[asyncio.Task] = set()
        self._server = None

    async def start(self) -> None:
        host = WS_CONFIG["host"]
        port = WS_CONFIG["port"]

        self._server = await websockets.serve(
            self._handle_connection,
            host,
            port,
            process_request=self._process_request,
            max_size=16 * 1024 * 1024,
        )
        if not WS_CONFIG["api_key"]:
            logger.warning(
                "WARNING: No REALTIME_API_KEY configured. "
                "Server accepting unauthenticated connections on %s:%s",
                host, port,
            )
        if _ALLOWED_ORIGINS:
            logger.info(f"CORS: allowing origins {_ALLOWED_ORIGINS}")
        else:
            logger.warning("CORS: WS_ALLOWED_ORIGINS not set — accepting all origins")
        logger.info(f"WebSocket server listening on ws://{host}:{port}{WS_CONFIG['path']}")

    async def stop(self) -> None:
        if self._server:
            self._server.close()
            await self._server.wait_closed()

        for task in list(self._active_sessions):
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass

        logger.info("WebSocket server stopped")

    @property
    def active_session_count(self) -> int:
        return len(self._active_sessions)

    def _check_provider_health(self) -> dict:
        """Lightweight provider health check — verifies providers are initialized."""
        status = {
            "asr": self._asr is not None,
            "llm": self._llm is not None,
            "tts": self._tts is not None,
        }
        # Check gRPC stub for remote providers
        if hasattr(self._asr, "_stub"):
            status["asr_connected"] = self._asr._stub is not None
        if hasattr(self._tts, "_stub"):
            status["tts_connected"] = self._tts._stub is not None
        return status

    async def _process_request(self, connection: ServerConnection, request):
        """Validate path, origin, auth, and limits before upgrade."""
        path = request.path
        parsed = urlparse(path)

        # Health check endpoint (responds before WebSocket upgrade)
        if parsed.path == "/health":
            provider_health = self._check_provider_health()
            all_healthy = all(provider_health.values())
            return connection.respond(
                200 if all_healthy else 503,
                json.dumps({
                    "status": "ok" if all_healthy else "degraded",
                    "active_sessions": len(self._active_sessions),
                    "max_connections": WS_CONFIG["max_connections"],
                    "providers": provider_health,
                }) + "\n",
            )

        # Check path
        expected = WS_CONFIG["path"]
        if parsed.path != expected:
            return connection.respond(404, f"Not found: {parsed.path}\n")

        # Check origin (CORS-like protection for WebSocket)
        if _ALLOWED_ORIGINS:
            origin = request.headers.get("Origin", "").lower()
            if origin and origin not in _ALLOWED_ORIGINS:
                logger.warning(f"Rejected connection from origin: {origin}")
                return connection.respond(403, "Origin not allowed\n")

        # Check connection limit
        if len(self._active_sessions) >= WS_CONFIG["max_connections"]:
            return connection.respond(503, "Too many connections\n")

        # Check auth (Bearer header or query parameter)
        api_key = WS_CONFIG["api_key"]
        if api_key:
            token = self._extract_token(request, parsed)
            if token is None:
                return connection.respond(401, "Missing API key (Authorization header or api_key query param)\n")
            if not hmac.compare_digest(token, api_key):
                return connection.respond(401, "Invalid API key\n")

    @staticmethod
    def _extract_token(request, parsed) -> str | None:
        """Extract auth token from Authorization header or api_key query param."""
        auth_header = request.headers.get("Authorization", "")
        if auth_header.startswith("Bearer "):
            return auth_header[7:]

        query_params = parse_qs(parsed.query)
        api_keys = query_params.get("api_key", [])
        if api_keys:
            return api_keys[0]

        return None

    async def _handle_connection(self, ws: ServerConnection) -> None:
        session = RealtimeSession(
            ws, self._asr, self._llm, self._tts,
            tool_registry=self._tool_registry,
        )
        task = asyncio.current_task()
        self._active_sessions.add(task)

        logger.info(f"New session: {session.session_id[:12]}")

        # Pre-warm gRPC channels in parallel to eliminate cold-start on first call
        await asyncio.gather(
            self._asr.warmup(),
            self._tts.warmup(),
            return_exceptions=True,
        )

        try:
            await session.run()
        except Exception as e:
            logger.error(f"Session error: {e}", exc_info=True)
        finally:
            self._active_sessions.discard(task)
            logger.info(f"Session ended: {session.session_id[:12]}")
