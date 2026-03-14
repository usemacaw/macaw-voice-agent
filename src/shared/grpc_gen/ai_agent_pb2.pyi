import asp_common_pb2 as _asp_common_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class ClientMessage(_message.Message):
    __slots__ = ("session_start", "session_end", "session_update", "audio", "speech_end")
    SESSION_START_FIELD_NUMBER: _ClassVar[int]
    SESSION_END_FIELD_NUMBER: _ClassVar[int]
    SESSION_UPDATE_FIELD_NUMBER: _ClassVar[int]
    AUDIO_FIELD_NUMBER: _ClassVar[int]
    SPEECH_END_FIELD_NUMBER: _ClassVar[int]
    session_start: SessionStart
    session_end: SessionEnd
    session_update: SessionUpdate
    audio: _asp_common_pb2.AudioData
    speech_end: AudioSpeechEnd
    def __init__(self, session_start: _Optional[_Union[SessionStart, _Mapping]] = ..., session_end: _Optional[_Union[SessionEnd, _Mapping]] = ..., session_update: _Optional[_Union[SessionUpdate, _Mapping]] = ..., audio: _Optional[_Union[_asp_common_pb2.AudioData, _Mapping]] = ..., speech_end: _Optional[_Union[AudioSpeechEnd, _Mapping]] = ...) -> None: ...

class SessionStart(_message.Message):
    __slots__ = ("session_id", "audio", "vad", "call_id", "metadata")
    class MetadataEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
    SESSION_ID_FIELD_NUMBER: _ClassVar[int]
    AUDIO_FIELD_NUMBER: _ClassVar[int]
    VAD_FIELD_NUMBER: _ClassVar[int]
    CALL_ID_FIELD_NUMBER: _ClassVar[int]
    METADATA_FIELD_NUMBER: _ClassVar[int]
    session_id: str
    audio: _asp_common_pb2.AudioConfig
    vad: _asp_common_pb2.VADConfig
    call_id: str
    metadata: _containers.ScalarMap[str, str]
    def __init__(self, session_id: _Optional[str] = ..., audio: _Optional[_Union[_asp_common_pb2.AudioConfig, _Mapping]] = ..., vad: _Optional[_Union[_asp_common_pb2.VADConfig, _Mapping]] = ..., call_id: _Optional[str] = ..., metadata: _Optional[_Mapping[str, str]] = ...) -> None: ...

class SessionEnd(_message.Message):
    __slots__ = ("session_id", "reason")
    SESSION_ID_FIELD_NUMBER: _ClassVar[int]
    REASON_FIELD_NUMBER: _ClassVar[int]
    session_id: str
    reason: str
    def __init__(self, session_id: _Optional[str] = ..., reason: _Optional[str] = ...) -> None: ...

class SessionUpdate(_message.Message):
    __slots__ = ("session_id", "vad")
    SESSION_ID_FIELD_NUMBER: _ClassVar[int]
    VAD_FIELD_NUMBER: _ClassVar[int]
    session_id: str
    vad: _asp_common_pb2.VADConfig
    def __init__(self, session_id: _Optional[str] = ..., vad: _Optional[_Union[_asp_common_pb2.VADConfig, _Mapping]] = ...) -> None: ...

class AudioSpeechEnd(_message.Message):
    __slots__ = ("session_id", "duration_ms")
    SESSION_ID_FIELD_NUMBER: _ClassVar[int]
    DURATION_MS_FIELD_NUMBER: _ClassVar[int]
    session_id: str
    duration_ms: int
    def __init__(self, session_id: _Optional[str] = ..., duration_ms: _Optional[int] = ...) -> None: ...

class ServerMessage(_message.Message):
    __slots__ = ("capabilities", "session_started", "session_ended", "session_updated", "audio", "speech_start", "speech_end", "response_start", "response_end", "call_action", "error")
    CAPABILITIES_FIELD_NUMBER: _ClassVar[int]
    SESSION_STARTED_FIELD_NUMBER: _ClassVar[int]
    SESSION_ENDED_FIELD_NUMBER: _ClassVar[int]
    SESSION_UPDATED_FIELD_NUMBER: _ClassVar[int]
    AUDIO_FIELD_NUMBER: _ClassVar[int]
    SPEECH_START_FIELD_NUMBER: _ClassVar[int]
    SPEECH_END_FIELD_NUMBER: _ClassVar[int]
    RESPONSE_START_FIELD_NUMBER: _ClassVar[int]
    RESPONSE_END_FIELD_NUMBER: _ClassVar[int]
    CALL_ACTION_FIELD_NUMBER: _ClassVar[int]
    ERROR_FIELD_NUMBER: _ClassVar[int]
    capabilities: _asp_common_pb2.ProtocolCapabilities
    session_started: SessionStarted
    session_ended: SessionEnded
    session_updated: SessionUpdated
    audio: _asp_common_pb2.AudioData
    speech_start: SpeechStartEvent
    speech_end: SpeechEndEvent
    response_start: ResponseStart
    response_end: ResponseEnd
    call_action: CallAction
    error: _asp_common_pb2.ProtocolError
    def __init__(self, capabilities: _Optional[_Union[_asp_common_pb2.ProtocolCapabilities, _Mapping]] = ..., session_started: _Optional[_Union[SessionStarted, _Mapping]] = ..., session_ended: _Optional[_Union[SessionEnded, _Mapping]] = ..., session_updated: _Optional[_Union[SessionUpdated, _Mapping]] = ..., audio: _Optional[_Union[_asp_common_pb2.AudioData, _Mapping]] = ..., speech_start: _Optional[_Union[SpeechStartEvent, _Mapping]] = ..., speech_end: _Optional[_Union[SpeechEndEvent, _Mapping]] = ..., response_start: _Optional[_Union[ResponseStart, _Mapping]] = ..., response_end: _Optional[_Union[ResponseEnd, _Mapping]] = ..., call_action: _Optional[_Union[CallAction, _Mapping]] = ..., error: _Optional[_Union[_asp_common_pb2.ProtocolError, _Mapping]] = ...) -> None: ...

class SessionStarted(_message.Message):
    __slots__ = ("session_id", "status", "negotiated", "errors")
    SESSION_ID_FIELD_NUMBER: _ClassVar[int]
    STATUS_FIELD_NUMBER: _ClassVar[int]
    NEGOTIATED_FIELD_NUMBER: _ClassVar[int]
    ERRORS_FIELD_NUMBER: _ClassVar[int]
    session_id: str
    status: _asp_common_pb2.SessionStatus
    negotiated: _asp_common_pb2.NegotiatedConfig
    errors: _containers.RepeatedCompositeFieldContainer[_asp_common_pb2.ProtocolError]
    def __init__(self, session_id: _Optional[str] = ..., status: _Optional[_Union[_asp_common_pb2.SessionStatus, str]] = ..., negotiated: _Optional[_Union[_asp_common_pb2.NegotiatedConfig, _Mapping]] = ..., errors: _Optional[_Iterable[_Union[_asp_common_pb2.ProtocolError, _Mapping]]] = ...) -> None: ...

class SessionEnded(_message.Message):
    __slots__ = ("session_id", "duration_seconds", "statistics")
    SESSION_ID_FIELD_NUMBER: _ClassVar[int]
    DURATION_SECONDS_FIELD_NUMBER: _ClassVar[int]
    STATISTICS_FIELD_NUMBER: _ClassVar[int]
    session_id: str
    duration_seconds: float
    statistics: _asp_common_pb2.SessionStatistics
    def __init__(self, session_id: _Optional[str] = ..., duration_seconds: _Optional[float] = ..., statistics: _Optional[_Union[_asp_common_pb2.SessionStatistics, _Mapping]] = ...) -> None: ...

class SessionUpdated(_message.Message):
    __slots__ = ("session_id", "status", "negotiated", "errors")
    SESSION_ID_FIELD_NUMBER: _ClassVar[int]
    STATUS_FIELD_NUMBER: _ClassVar[int]
    NEGOTIATED_FIELD_NUMBER: _ClassVar[int]
    ERRORS_FIELD_NUMBER: _ClassVar[int]
    session_id: str
    status: _asp_common_pb2.SessionStatus
    negotiated: _asp_common_pb2.NegotiatedConfig
    errors: _containers.RepeatedCompositeFieldContainer[_asp_common_pb2.ProtocolError]
    def __init__(self, session_id: _Optional[str] = ..., status: _Optional[_Union[_asp_common_pb2.SessionStatus, str]] = ..., negotiated: _Optional[_Union[_asp_common_pb2.NegotiatedConfig, _Mapping]] = ..., errors: _Optional[_Iterable[_Union[_asp_common_pb2.ProtocolError, _Mapping]]] = ...) -> None: ...

class SpeechStartEvent(_message.Message):
    __slots__ = ("session_id",)
    SESSION_ID_FIELD_NUMBER: _ClassVar[int]
    session_id: str
    def __init__(self, session_id: _Optional[str] = ...) -> None: ...

class SpeechEndEvent(_message.Message):
    __slots__ = ("session_id", "duration_ms")
    SESSION_ID_FIELD_NUMBER: _ClassVar[int]
    DURATION_MS_FIELD_NUMBER: _ClassVar[int]
    session_id: str
    duration_ms: int
    def __init__(self, session_id: _Optional[str] = ..., duration_ms: _Optional[int] = ...) -> None: ...

class ResponseStart(_message.Message):
    __slots__ = ("session_id", "response_id", "text")
    SESSION_ID_FIELD_NUMBER: _ClassVar[int]
    RESPONSE_ID_FIELD_NUMBER: _ClassVar[int]
    TEXT_FIELD_NUMBER: _ClassVar[int]
    session_id: str
    response_id: str
    text: str
    def __init__(self, session_id: _Optional[str] = ..., response_id: _Optional[str] = ..., text: _Optional[str] = ...) -> None: ...

class ResponseEnd(_message.Message):
    __slots__ = ("session_id", "response_id", "interrupted")
    SESSION_ID_FIELD_NUMBER: _ClassVar[int]
    RESPONSE_ID_FIELD_NUMBER: _ClassVar[int]
    INTERRUPTED_FIELD_NUMBER: _ClassVar[int]
    session_id: str
    response_id: str
    interrupted: bool
    def __init__(self, session_id: _Optional[str] = ..., response_id: _Optional[str] = ..., interrupted: bool = ...) -> None: ...

class CallAction(_message.Message):
    __slots__ = ("session_id", "action", "target", "reason")
    SESSION_ID_FIELD_NUMBER: _ClassVar[int]
    ACTION_FIELD_NUMBER: _ClassVar[int]
    TARGET_FIELD_NUMBER: _ClassVar[int]
    REASON_FIELD_NUMBER: _ClassVar[int]
    session_id: str
    action: _asp_common_pb2.CallActionType
    target: str
    reason: str
    def __init__(self, session_id: _Optional[str] = ..., action: _Optional[_Union[_asp_common_pb2.CallActionType, str]] = ..., target: _Optional[str] = ..., reason: _Optional[str] = ...) -> None: ...
