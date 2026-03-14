"""
Serial event sender with auto-generated event IDs.

Ensures events are sent in order over a WebSocket connection.
Differentiates droppable events (audio/text deltas) from structural
events (response lifecycle, errors) — structural events cause connection
termination on timeout instead of being silently dropped.
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from websockets.asyncio.server import ServerConnection

logger = logging.getLogger("open-voice-api.event-emitter")


_SEND_TIMEOUT_S = 5.0

# Events that can be dropped on slow clients without causing state divergence.
# All other events are structural and MUST be delivered or the connection is closed.
_DROPPABLE_EVENTS = frozenset({
    "response.audio.delta",
    "response.audio_transcript.delta",
    "response.text.delta",
    "response.function_call_arguments.delta",
})


class SlowClientError(Exception):
    """Raised when a structural event cannot be delivered to a slow client."""


class EventEmitter:
    """Sends server events over WebSocket with auto event_id generation."""

    def __init__(self, ws: ServerConnection, session_id: str = ""):
        self._ws = ws
        self._session_id = session_id

    @staticmethod
    def _next_event_id() -> str:
        return f"evt_{uuid.uuid4().hex[:24]}"

    async def emit(self, event: dict) -> None:
        """Send a server event, auto-generating event_id if missing.

        Raises:
            SlowClientError: If a structural event cannot be delivered within timeout.
        """
        if "event_id" not in event or not event["event_id"]:
            event["event_id"] = self._next_event_id()
        raw = json.dumps(event, separators=(",", ":"))
        event_type = event.get("type", "unknown")
        try:
            await asyncio.wait_for(self._ws.send(raw), timeout=_SEND_TIMEOUT_S)
        except asyncio.TimeoutError:
            if event_type in _DROPPABLE_EVENTS:
                logger.warning(
                    f"[{self._session_id[:8]}] Slow client: dropping {event_type}"
                )
                return
            logger.error(
                f"[{self._session_id[:8]}] Slow client: cannot deliver structural "
                f"event {event_type}, terminating session"
            )
            raise SlowClientError(
                f"Cannot deliver structural event {event_type} within {_SEND_TIMEOUT_S}s"
            )
        if event_type != "response.audio.delta":
            logger.debug(f"[{self._session_id[:8]}] → {event_type}")

    async def emit_many(self, events: list[dict]) -> None:
        """Send multiple events in order."""
        for event in events:
            await self.emit(event)
