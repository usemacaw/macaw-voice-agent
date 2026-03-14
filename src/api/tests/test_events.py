"""Tests for protocol server events serialization."""

import json

from protocol.events import (
    session_created,
    session_updated,
    conversation_created,
    error_event,
    input_audio_buffer_committed,
    input_audio_buffer_cleared,
    input_audio_buffer_speech_started,
    input_audio_buffer_speech_stopped,
    conversation_item_created,
    conversation_item_deleted,
    response_created,
    response_done,
    response_output_item_added,
    response_content_part_added,
    response_audio_delta,
    response_audio_done,
    response_audio_transcript_delta,
    response_audio_transcript_done,
    response_text_delta,
    response_text_done,
    response_function_call_arguments_delta,
    response_function_call_arguments_done,
    rate_limits_updated,
)
from protocol.models import SessionConfig, ConversationItem, ContentPart


class TestServerEvents:
    """Test server events generate valid JSON with correct structure."""

    def test_session_created(self):
        config = SessionConfig()
        event = session_created("evt_1", "sess_abc", config)
        assert event["type"] == "session.created"
        assert event["session"]["id"] == "sess_abc"
        assert event["session"]["object"] == "realtime.session"
        assert "modalities" in event["session"]
        # Ensure JSON-serializable
        json.dumps(event)

    def test_session_updated(self):
        config = SessionConfig(temperature=0.5)
        event = session_updated("", "sess_abc", config)
        assert event["type"] == "session.updated"
        assert event["session"]["temperature"] == 0.5

    def test_conversation_created(self):
        event = conversation_created("", "conv_xyz")
        assert event["type"] == "conversation.created"
        assert event["conversation"]["id"] == "conv_xyz"

    def test_error_event(self):
        event = error_event("", "Something went wrong", code="bad_request")
        assert event["type"] == "error"
        assert event["error"]["message"] == "Something went wrong"
        assert event["error"]["code"] == "bad_request"

    def test_input_audio_buffer_committed(self):
        event = input_audio_buffer_committed("", "prev_1", "item_2")
        assert event["type"] == "input_audio_buffer.committed"
        assert event["item_id"] == "item_2"

    def test_input_audio_buffer_cleared(self):
        event = input_audio_buffer_cleared("")
        assert event["type"] == "input_audio_buffer.cleared"

    def test_speech_started_stopped(self):
        started = input_audio_buffer_speech_started("", 1000, "item_3")
        assert started["type"] == "input_audio_buffer.speech_started"
        assert started["audio_start_ms"] == 1000

        stopped = input_audio_buffer_speech_stopped("", 2000, "item_3")
        assert stopped["type"] == "input_audio_buffer.speech_stopped"
        assert stopped["audio_end_ms"] == 2000

    def test_conversation_item_created(self):
        item = ConversationItem(id="item_5", type="message", role="user")
        event = conversation_item_created("", "", item)
        assert event["type"] == "conversation.item.created"
        assert event["item"]["id"] == "item_5"

    def test_response_lifecycle_events(self):
        event = response_created("", "resp_1")
        assert event["type"] == "response.created"
        assert event["response"]["status"] == "in_progress"

        item = ConversationItem(id="item_6", type="message", role="assistant")
        added = response_output_item_added("", "resp_1", 0, item)
        assert added["type"] == "response.output_item.added"

        part = ContentPart(type="audio")
        part_added = response_content_part_added("", "resp_1", "item_6", 0, 0, part)
        assert part_added["type"] == "response.content_part.added"

        done = response_done("", "resp_1", status="completed")
        assert done["type"] == "response.done"
        assert done["response"]["status"] == "completed"

    def test_audio_delta_events(self):
        delta = response_audio_delta("", "resp_1", "item_6", 0, 0, "AQID")
        assert delta["type"] == "response.audio.delta"
        assert delta["delta"] == "AQID"

        done = response_audio_done("", "resp_1", "item_6", 0, 0)
        assert done["type"] == "response.audio.done"

    def test_transcript_delta_events(self):
        delta = response_audio_transcript_delta("", "resp_1", "item_6", 0, 0, "Hello")
        assert delta["type"] == "response.audio_transcript.delta"
        assert delta["delta"] == "Hello"

        done = response_audio_transcript_done("", "resp_1", "item_6", 0, 0, "Hello world")
        assert done["type"] == "response.audio_transcript.done"
        assert done["transcript"] == "Hello world"

    def test_text_delta_events(self):
        delta = response_text_delta("", "resp_1", "item_6", 0, 0, "Hi")
        assert delta["type"] == "response.text.delta"

        done = response_text_done("", "resp_1", "item_6", 0, 0, "Hi there")
        assert done["type"] == "response.text.done"
        assert done["text"] == "Hi there"

    def test_function_call_events(self):
        delta = response_function_call_arguments_delta("", "resp_1", "item_7", 0, "call_1", '{"loc')
        assert delta["type"] == "response.function_call_arguments.delta"

        done = response_function_call_arguments_done("", "resp_1", "item_7", 0, "call_1", '{"location":"NYC"}')
        assert done["type"] == "response.function_call_arguments.done"
        assert done["arguments"] == '{"location":"NYC"}'

    def test_rate_limits_updated(self):
        event = rate_limits_updated("", [{"name": "requests", "limit": 100, "remaining": 99}])
        assert event["type"] == "rate_limits.updated"
        assert len(event["rate_limits"]) == 1

    def test_all_server_events_json_serializable(self):
        """All server events must be JSON-serializable."""
        config = SessionConfig()
        item = ConversationItem(id="i", role="user")
        part = ContentPart(type="text", text="t")

        all_events = [
            session_created("", "s", config),
            session_updated("", "s", config),
            conversation_created("", "c"),
            error_event("", "e"),
            input_audio_buffer_committed("", "", "i"),
            input_audio_buffer_cleared(""),
            input_audio_buffer_speech_started("", 0, "i"),
            input_audio_buffer_speech_stopped("", 0, "i"),
            conversation_item_created("", "", item),
            conversation_item_deleted("", "i"),
            response_created("", "r"),
            response_done("", "r"),
            response_output_item_added("", "r", 0, item),
            response_content_part_added("", "r", "i", 0, 0, part),
            response_audio_delta("", "r", "i", 0, 0, "d"),
            response_audio_done("", "r", "i", 0, 0),
            response_audio_transcript_delta("", "r", "i", 0, 0, "t"),
            response_audio_transcript_done("", "r", "i", 0, 0, "t"),
            response_text_delta("", "r", "i", 0, 0, "t"),
            response_text_done("", "r", "i", 0, 0, "t"),
            response_function_call_arguments_delta("", "r", "i", 0, "c", "d"),
            response_function_call_arguments_done("", "r", "i", 0, "c", "a"),
            rate_limits_updated("", []),
        ]

        for event in all_events:
            serialized = json.dumps(event)
            parsed = json.loads(serialized)
            assert "type" in parsed, f"Missing 'type' in {event}"
