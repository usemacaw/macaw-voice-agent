"""
OpenAI Realtime API server event definitions.

Server events implement to_dict() as plain dicts.

Reference: https://platform.openai.com/docs/api-reference/realtime
"""

from __future__ import annotations

from typing import Any

from protocol.models import ConversationItem, ContentPart, SessionConfig


# ---------------------------------------------------------------------------
# Server Events (28+)
# ---------------------------------------------------------------------------

def _server_event(type_: str, event_id: str = "", **kwargs: Any) -> dict:
    """Build a server event dict."""
    event: dict[str, Any] = {"type": type_}
    if event_id:
        event["event_id"] = event_id
    event.update(kwargs)
    return event


def session_created(event_id: str, session_id: str, config: SessionConfig) -> dict:
    return _server_event(
        "session.created",
        event_id=event_id,
        session={
            "id": session_id,
            "object": "realtime.session",
            **config.to_dict(),
        },
    )


def session_updated(event_id: str, session_id: str, config: SessionConfig) -> dict:
    return _server_event(
        "session.updated",
        event_id=event_id,
        session={
            "id": session_id,
            "object": "realtime.session",
            **config.to_dict(),
        },
    )


def conversation_created(event_id: str, conversation_id: str) -> dict:
    return _server_event(
        "conversation.created",
        event_id=event_id,
        conversation={
            "id": conversation_id,
            "object": "realtime.conversation",
        },
    )


def error_event(event_id: str, message: str, error_type: str = "invalid_request_error", code: str = "", param: str = "") -> dict:
    err: dict[str, Any] = {"type": error_type, "message": message}
    if code:
        err["code"] = code
    if param:
        err["param"] = param
    return _server_event("error", event_id=event_id, error=err)


def input_audio_buffer_committed(event_id: str, previous_item_id: str, item_id: str) -> dict:
    return _server_event(
        "input_audio_buffer.committed",
        event_id=event_id,
        previous_item_id=previous_item_id,
        item_id=item_id,
    )


def input_audio_buffer_cleared(event_id: str) -> dict:
    return _server_event("input_audio_buffer.cleared", event_id=event_id)


def input_audio_buffer_speech_started(event_id: str, audio_start_ms: int, item_id: str) -> dict:
    return _server_event(
        "input_audio_buffer.speech_started",
        event_id=event_id,
        audio_start_ms=audio_start_ms,
        item_id=item_id,
    )


def input_audio_buffer_speech_stopped(event_id: str, audio_end_ms: int, item_id: str) -> dict:
    return _server_event(
        "input_audio_buffer.speech_stopped",
        event_id=event_id,
        audio_end_ms=audio_end_ms,
        item_id=item_id,
    )


def conversation_item_created(event_id: str, previous_item_id: str, item: ConversationItem) -> dict:
    return _server_event(
        "conversation.item.created",
        event_id=event_id,
        previous_item_id=previous_item_id,
        item=item.to_dict(),
    )


def conversation_item_deleted(event_id: str, item_id: str) -> dict:
    return _server_event(
        "conversation.item.deleted",
        event_id=event_id,
        item_id=item_id,
    )


def conversation_item_retrieved(event_id: str, item: ConversationItem) -> dict:
    return _server_event(
        "conversation.item.retrieved",
        event_id=event_id,
        item=item.to_dict(),
    )


def conversation_item_truncated(event_id: str, item_id: str, content_index: int, audio_end_ms: int) -> dict:
    return _server_event(
        "conversation.item.truncated",
        event_id=event_id,
        item_id=item_id,
        content_index=content_index,
        audio_end_ms=audio_end_ms,
    )


def input_audio_transcription_completed(event_id: str, item_id: str, content_index: int, transcript: str) -> dict:
    return _server_event(
        "conversation.item.input_audio_transcription.completed",
        event_id=event_id,
        item_id=item_id,
        content_index=content_index,
        transcript=transcript,
    )


def input_audio_transcription_failed(event_id: str, item_id: str, content_index: int, error_message: str) -> dict:
    return _server_event(
        "conversation.item.input_audio_transcription.failed",
        event_id=event_id,
        item_id=item_id,
        content_index=content_index,
        error={"type": "transcription_error", "message": error_message},
    )


def response_created(event_id: str, response_id: str, status: str = "in_progress") -> dict:
    return _server_event(
        "response.created",
        event_id=event_id,
        response={
            "id": response_id,
            "object": "realtime.response",
            "status": status,
            "output": [],
        },
    )


def response_done(
    event_id: str,
    response_id: str,
    status: str = "completed",
    output: list[dict] | None = None,
    usage: dict | None = None,
) -> dict:
    resp: dict[str, Any] = {
        "id": response_id,
        "object": "realtime.response",
        "status": status,
        "output": output or [],
    }
    if usage:
        resp["usage"] = usage
    return _server_event("response.done", event_id=event_id, response=resp)


def response_output_item_added(event_id: str, response_id: str, output_index: int, item: ConversationItem) -> dict:
    return _server_event(
        "response.output_item.added",
        event_id=event_id,
        response_id=response_id,
        output_index=output_index,
        item=item.to_dict(),
    )


def response_output_item_done(event_id: str, response_id: str, output_index: int, item: ConversationItem) -> dict:
    return _server_event(
        "response.output_item.done",
        event_id=event_id,
        response_id=response_id,
        output_index=output_index,
        item=item.to_dict(),
    )


def response_content_part_added(event_id: str, response_id: str, item_id: str, output_index: int, content_index: int, part: ContentPart) -> dict:
    return _server_event(
        "response.content_part.added",
        event_id=event_id,
        response_id=response_id,
        item_id=item_id,
        output_index=output_index,
        content_index=content_index,
        part=part.to_dict(),
    )


def response_content_part_done(event_id: str, response_id: str, item_id: str, output_index: int, content_index: int, part: ContentPart) -> dict:
    return _server_event(
        "response.content_part.done",
        event_id=event_id,
        response_id=response_id,
        item_id=item_id,
        output_index=output_index,
        content_index=content_index,
        part=part.to_dict(),
    )


def response_audio_delta(event_id: str, response_id: str, item_id: str, output_index: int, content_index: int, delta: str) -> dict:
    return _server_event(
        "response.audio.delta",
        event_id=event_id,
        response_id=response_id,
        item_id=item_id,
        output_index=output_index,
        content_index=content_index,
        delta=delta,
    )


def response_audio_done(event_id: str, response_id: str, item_id: str, output_index: int, content_index: int) -> dict:
    return _server_event(
        "response.audio.done",
        event_id=event_id,
        response_id=response_id,
        item_id=item_id,
        output_index=output_index,
        content_index=content_index,
    )


def response_audio_transcript_delta(event_id: str, response_id: str, item_id: str, output_index: int, content_index: int, delta: str) -> dict:
    return _server_event(
        "response.audio_transcript.delta",
        event_id=event_id,
        response_id=response_id,
        item_id=item_id,
        output_index=output_index,
        content_index=content_index,
        delta=delta,
    )


def response_audio_transcript_done(event_id: str, response_id: str, item_id: str, output_index: int, content_index: int, transcript: str) -> dict:
    return _server_event(
        "response.audio_transcript.done",
        event_id=event_id,
        response_id=response_id,
        item_id=item_id,
        output_index=output_index,
        content_index=content_index,
        transcript=transcript,
    )


def response_text_delta(event_id: str, response_id: str, item_id: str, output_index: int, content_index: int, delta: str) -> dict:
    return _server_event(
        "response.text.delta",
        event_id=event_id,
        response_id=response_id,
        item_id=item_id,
        output_index=output_index,
        content_index=content_index,
        delta=delta,
    )


def response_text_done(event_id: str, response_id: str, item_id: str, output_index: int, content_index: int, text: str) -> dict:
    return _server_event(
        "response.text.done",
        event_id=event_id,
        response_id=response_id,
        item_id=item_id,
        output_index=output_index,
        content_index=content_index,
        text=text,
    )


def response_function_call_arguments_delta(event_id: str, response_id: str, item_id: str, output_index: int, call_id: str, delta: str) -> dict:
    return _server_event(
        "response.function_call_arguments.delta",
        event_id=event_id,
        response_id=response_id,
        item_id=item_id,
        output_index=output_index,
        call_id=call_id,
        delta=delta,
    )


def response_function_call_arguments_done(event_id: str, response_id: str, item_id: str, output_index: int, call_id: str, arguments: str) -> dict:
    return _server_event(
        "response.function_call_arguments.done",
        event_id=event_id,
        response_id=response_id,
        item_id=item_id,
        output_index=output_index,
        call_id=call_id,
        arguments=arguments,
    )


def rate_limits_updated(event_id: str, rate_limits: list[dict]) -> dict:
    return _server_event(
        "rate_limits.updated",
        event_id=event_id,
        rate_limits=rate_limits,
    )


# ---------------------------------------------------------------------------
# Custom extension events (macaw.*)
# ---------------------------------------------------------------------------

def macaw_metrics(response_id: str, metrics: dict) -> dict:
    """Per-response observability metrics."""
    return _server_event(
        "macaw.metrics",
        response_id=response_id,
        metrics=metrics,
    )
