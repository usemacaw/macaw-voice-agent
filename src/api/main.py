"""
OpenVoiceAPI — Drop-in replacement for OpenAI Realtime API.

Entry point: starts WebSocket server with configured providers.
"""

from __future__ import annotations

import asyncio
import logging
import signal
import sys

from config import ASR_CONFIG, TTS_CONFIG, LLM_CONFIG, TOOL_CONFIG, LOG_CONFIG
from providers.asr import create_asr_provider
from providers.tts import create_tts_provider
from providers.llm import create_llm_provider
from server.ws_server import WebSocketServer
from tools.registry import create_tool_registry

logging.basicConfig(
    level=getattr(logging, LOG_CONFIG["level"]),
    format="%(asctime)s %(name)-40s %(levelname)-5s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("open-voice-api")

_MAX_CONNECT_RETRIES = 5
_INITIAL_BACKOFF_S = 1.0


async def _connect_provider(provider, name: str) -> None:
    """Connect a provider with exponential backoff retry."""
    backoff = _INITIAL_BACKOFF_S
    for attempt in range(1, _MAX_CONNECT_RETRIES + 1):
        try:
            await provider.connect()
            return
        except Exception as e:
            if attempt == _MAX_CONNECT_RETRIES:
                logger.error(f"{name} provider failed to connect after {_MAX_CONNECT_RETRIES} attempts: {e}")
                raise
            logger.warning(
                f"{name} provider connect attempt {attempt}/{_MAX_CONNECT_RETRIES} failed: {e}. "
                f"Retrying in {backoff:.1f}s..."
            )
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 30.0)


async def main() -> None:
    logger.info("Starting OpenVoiceAPI server...")

    # Create providers
    asr = create_asr_provider(ASR_CONFIG["provider"])
    tts = create_tts_provider(TTS_CONFIG["provider"])
    llm = create_llm_provider(LLM_CONFIG["provider"])

    # Create tool registry (registers mock handlers if TOOL_ENABLE_MOCK=true)
    tool_registry = create_tool_registry()

    logger.info(
        f"Providers: ASR={ASR_CONFIG['provider']}, "
        f"TTS={TTS_CONFIG['provider']}, "
        f"LLM={LLM_CONFIG['provider']}, "
        f"Tools={'enabled (' + str(len(tool_registry.get_schemas())) + ' tools)' if tool_registry.has_server_tools else 'disabled'}"
    )

    # Connect providers with retry
    await _connect_provider(asr, "ASR")
    await _connect_provider(tts, "TTS")
    await _connect_provider(llm, "LLM")

    # Start WebSocket server
    server = WebSocketServer(asr, llm, tts, tool_registry=tool_registry)
    await server.start()

    # Wait for shutdown signal
    stop_event = asyncio.Event()

    def _signal_handler():
        logger.info("Shutdown signal received")
        stop_event.set()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, _signal_handler)

    await stop_event.wait()

    # Cleanup
    logger.info("Shutting down...")
    await server.stop()
    for name, provider in [("ASR", asr), ("TTS", tts), ("LLM", llm)]:
        try:
            await provider.disconnect()
        except Exception as e:
            logger.error(f"Error disconnecting {name} provider: {e}", exc_info=True)
    logger.info("Server stopped.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
