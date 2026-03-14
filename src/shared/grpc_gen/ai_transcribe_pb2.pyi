import asp_common_pb2 as _asp_common_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class TranscribeClientMessage(_message.Message):
    __slots__ = ("session_start", "session_end", "audio", "speech_end")
    SESSION_START_FIELD_NUMBER: _ClassVar[int]
    SESSION_END_FIELD_NUMBER: _ClassVar[int]
    AUDIO_FIELD_NUMBER: _ClassVar[int]
    SPEECH_END_FIELD_NUMBER: _ClassVar[int]
    session_start: TranscribeSessionStart
    session_end: TranscribeSessionEnd
    audio: _asp_common_pb2.AudioData
    speech_end: TranscribeAudioSpeechEnd
    def __init__(self, session_start: _Optional[_Union[TranscribeSessionStart, _Mapping]] = ..., session_end: _Optional[_Union[TranscribeSessionEnd, _Mapping]] = ..., audio: _Optional[_Union[_asp_common_pb2.AudioData, _Mapping]] = ..., speech_end: _Optional[_Union[TranscribeAudioSpeechEnd, _Mapping]] = ...) -> None: ...

class TranscribeSessionStart(_message.Message):
    __slots__ = ("session_id", "audio", "vad", "call_id", "caller_id", "metadata")
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
    CALLER_ID_FIELD_NUMBER: _ClassVar[int]
    METADATA_FIELD_NUMBER: _ClassVar[int]
    session_id: str
    audio: _asp_common_pb2.AudioConfig
    vad: _asp_common_pb2.VADConfig
    call_id: str
    caller_id: str
    metadata: _containers.ScalarMap[str, str]
    def __init__(self, session_id: _Optional[str] = ..., audio: _Optional[_Union[_asp_common_pb2.AudioConfig, _Mapping]] = ..., vad: _Optional[_Union[_asp_common_pb2.VADConfig, _Mapping]] = ..., call_id: _Optional[str] = ..., caller_id: _Optional[str] = ..., metadata: _Optional[_Mapping[str, str]] = ...) -> None: ...

class TranscribeSessionEnd(_message.Message):
    __slots__ = ("session_id", "reason")
    SESSION_ID_FIELD_NUMBER: _ClassVar[int]
    REASON_FIELD_NUMBER: _ClassVar[int]
    session_id: str
    reason: str
    def __init__(self, session_id: _Optional[str] = ..., reason: _Optional[str] = ...) -> None: ...

class TranscribeAudioSpeechEnd(_message.Message):
    __slots__ = ("session_id",)
    SESSION_ID_FIELD_NUMBER: _ClassVar[int]
    session_id: str
    def __init__(self, session_id: _Optional[str] = ...) -> None: ...

class TranscribeServerMessage(_message.Message):
    __slots__ = ("capabilities", "session_started", "session_ended", "error")
    CAPABILITIES_FIELD_NUMBER: _ClassVar[int]
    SESSION_STARTED_FIELD_NUMBER: _ClassVar[int]
    SESSION_ENDED_FIELD_NUMBER: _ClassVar[int]
    ERROR_FIELD_NUMBER: _ClassVar[int]
    capabilities: _asp_common_pb2.ProtocolCapabilities
    session_started: TranscribeSessionStarted
    session_ended: TranscribeSessionEnded
    error: _asp_common_pb2.ProtocolError
    def __init__(self, capabilities: _Optional[_Union[_asp_common_pb2.ProtocolCapabilities, _Mapping]] = ..., session_started: _Optional[_Union[TranscribeSessionStarted, _Mapping]] = ..., session_ended: _Optional[_Union[TranscribeSessionEnded, _Mapping]] = ..., error: _Optional[_Union[_asp_common_pb2.ProtocolError, _Mapping]] = ...) -> None: ...

class TranscribeSessionStarted(_message.Message):
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

class TranscribeSessionEnded(_message.Message):
    __slots__ = ("session_id", "duration_seconds")
    SESSION_ID_FIELD_NUMBER: _ClassVar[int]
    DURATION_SECONDS_FIELD_NUMBER: _ClassVar[int]
    session_id: str
    duration_seconds: float
    def __init__(self, session_id: _Optional[str] = ..., duration_seconds: _Optional[float] = ...) -> None: ...
