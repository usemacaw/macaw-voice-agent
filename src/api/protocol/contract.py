"""
Protocol contract — formalized invariants of the Realtime API protocol.

Documents and enforces ordering, delivery, and lifecycle guarantees
that clients can rely on. These invariants were previously implicit
across event_emitter.py, response_runner.py, and session.py.

This module serves as both documentation and runtime reference.
"""

from __future__ import annotations


# ============================================================================
# Delivery Guarantees
# ============================================================================

# Events that can be dropped on slow clients without causing state divergence.
# All other events are STRUCTURAL and MUST be delivered or the connection is closed.
DROPPABLE_EVENTS = frozenset({
    "response.audio.delta",
    "response.audio_transcript.delta",
    "response.text.delta",
    "response.function_call_arguments.delta",
})


def is_droppable(event_type: str) -> bool:
    """True if this event can be safely dropped on slow clients."""
    return event_type in DROPPABLE_EVENTS


def is_structural(event_type: str) -> bool:
    """True if this event MUST be delivered or session should terminate."""
    return event_type not in DROPPABLE_EVENTS


# ============================================================================
# Lifecycle Ordering Invariants
# ============================================================================

# These are the guarantees the server makes about event ordering.
# Documented here for reference — enforced by the emission sequence
# in ResponseRunner/ResponseOrchestrator.

ORDERING_INVARIANTS = """
Protocol Ordering Guarantees:

1. SESSION LIFECYCLE
   - session.created always precedes any other event
   - session.updated follows session.created
   - conversation.created follows session.created

2. RESPONSE LIFECYCLE (monotonic)
   - response.created → output_item.added → content → output_item.done → response.done
   - response.done is NEVER dropped (structural)
   - Only ONE active response per session at a time

3. AUDIO OUTPUT LIFECYCLE
   - response.audio.delta events are ordered but droppable
   - response.audio.done always follows all deltas (structural)
   - response.audio_transcript.done follows transcript deltas

4. TOOL CALLING LIFECYCLE
   - function_call_arguments.done precedes tool execution
   - conversation.item.created (output) follows tool execution
   - Tool rounds complete before final text/audio response

5. INPUT AUDIO
   - speech_started precedes speech_stopped
   - transcription_completed follows speech_stopped
   - Barge-in cancels active response BEFORE starting new one

6. METRICS
   - macaw.metrics always emitted AFTER response.done
   - Contains timing for all completed stages
"""


# ============================================================================
# Limits
# ============================================================================

# Rate limiting
MAX_EVENTS_PER_SECOND = 200

# Session timeouts
SESSION_IDLE_TIMEOUT_S = 600.0  # 10 minutes

# Audio buffer
MAX_AUDIO_BUFFER_SECONDS = 600  # 10 minutes

# Conversation
MAX_CONVERSATION_ITEMS = 200

# Validation
MAX_TOOLS = 128
MAX_TOOL_SCHEMA_CHARS = 200_000
MAX_INSTRUCTIONS_CHARS = 50_000
MAX_ITEM_CONTENT_PARTS = 50
MAX_ITEM_TEXT_CHARS = 100_000
