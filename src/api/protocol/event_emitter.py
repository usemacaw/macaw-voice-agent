"""
Serial event sender with auto-generated event IDs.

Ensures events are sent in order over a WebSocket connection.

Features:
- Cancellation fences: events from stale response_ids are dropped
- Progressive backpressure: degrades gracefully instead of killing sessions
- Droppable vs structural event classification
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from websockets.asyncio.server import ServerConnection

from protocol.contract import DROPPABLE_EVENTS

logger = logging.getLogger("open-voice-api.event-emitter")


_SEND_TIMEOUT_S = 1.0

# Transcript deltas are first to be throttled under pressure
_THROTTLEABLE_EVENTS = frozenset({
    "response.audio_transcript.delta",
})

# Backpressure level thresholds (consecutive drops)
_LEVEL_LIGHT = 3
_LEVEL_HEAVY = 10
_LEVEL_CRITICAL = 20


class SlowClientError(Exception):
    """Raised when a structural event cannot be delivered to a slow client."""


class EventEmitter:
    """Sends server events over WebSocket with auto event_id generation.

    Supports:
    - Response-level fencing (cancellation safety)
    - Progressive backpressure (4 levels of degradation)
    """

    def __init__(self, ws: ServerConnection, session_id: str = ""):
        self._ws = ws
        self._session_id = session_id
        # Cancellation fence
        self._active_response_id: str | None = None
        # Backpressure tracking
        self._consecutive_drops: int = 0
        self._total_drops: int = 0
        self._throttle_counter: int = 0
        self._pressure_level: int = 0

    def set_active_response(self, response_id: str) -> None:
        """Register the currently active response_id."""
        self._active_response_id = response_id

    def invalidate_response(self, response_id: str) -> None:
        """Invalidate a response_id so its late events are dropped."""
        if self._active_response_id == response_id:
            self._active_response_id = None

    @property
    def pressure_level(self) -> int:
        """Current backpressure level (0=normal, 1=light, 2=heavy, 3=critical)."""
        return self._pressure_level

    @property
    def total_drops(self) -> int:
        """Total events dropped in this session."""
        return self._total_drops

    @staticmethod
    def _next_event_id() -> str:
        return f"evt_{uuid.uuid4().hex[:24]}"

    def _update_pressure_level(self) -> None:
        """Update pressure level based on consecutive drops."""
        if self._consecutive_drops >= _LEVEL_CRITICAL:
            new_level = 3
        elif self._consecutive_drops >= _LEVEL_HEAVY:
            new_level = 2
        elif self._consecutive_drops >= _LEVEL_LIGHT:
            new_level = 1
        else:
            new_level = 0

        if new_level != self._pressure_level:
            logger.warning(
                f"[{self._session_id[:8]}] Backpressure level {self._pressure_level} → {new_level} "
                f"(consecutive_drops={self._consecutive_drops})"
            )
            self._pressure_level = new_level

    async def emit(self, event: dict) -> None:
        """Send a server event, auto-generating event_id if missing.

        Events with a response_id that doesn't match the active response
        are silently dropped (cancellation fence).

        Progressive backpressure:
        - Level 0 (normal): send everything
        - Level 1 (light): throttle transcript deltas (1 in 3)
        - Level 2 (heavy): drop all transcript deltas
        - Level 3 (critical): raise SlowClientError on structural timeout

        Raises:
            SlowClientError: If a structural event cannot be delivered within timeout.
        """
        if "event_id" not in event or not event["event_id"]:
            event["event_id"] = self._next_event_id()

        event_type = event.get("type", "unknown")

        # Cancellation fence: drop events from stale responses
        event_response_id = event.get("response_id")
        if (
            event_response_id
            and self._active_response_id
            and event_response_id != self._active_response_id
        ):
            logger.debug(
                f"[{self._session_id[:8]}] Fence: dropping stale {event_type} "
                f"(response {event_response_id[:12]} != active {self._active_response_id[:12]})"
            )
            return

        # Progressive backpressure: throttle/drop under pressure
        if self._pressure_level >= 2 and event_type in _THROTTLEABLE_EVENTS:
            # Level 2+: drop all transcript deltas
            self._total_drops += 1
            return

        if self._pressure_level >= 1 and event_type in _THROTTLEABLE_EVENTS:
            # Level 1: send 1 in 3 transcript deltas
            self._throttle_counter += 1
            if self._throttle_counter % 3 != 0:
                self._total_drops += 1
                return

        raw = json.dumps(event, separators=(",", ":"))
        try:
            await asyncio.wait_for(self._ws.send(raw), timeout=_SEND_TIMEOUT_S)
            # Successful send: reset consecutive drops
            self._consecutive_drops = 0
            self._update_pressure_level()
        except asyncio.TimeoutError:
            if event_type in DROPPABLE_EVENTS:
                self._consecutive_drops += 1
                self._total_drops += 1
                self._update_pressure_level()
                if event_type != "response.audio.delta" or self._consecutive_drops <= 3:
                    logger.warning(
                        f"[{self._session_id[:8]}] Slow client: dropping {event_type} "
                        f"(level={self._pressure_level})"
                    )
                return

            # Structural event failed
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
