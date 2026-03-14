from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class AudioEncoding(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    AUDIO_ENCODING_UNSPECIFIED: _ClassVar[AudioEncoding]
    PCM_S16LE: _ClassVar[AudioEncoding]
    MULAW: _ClassVar[AudioEncoding]
    ALAW: _ClassVar[AudioEncoding]

class SessionStatus(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    SESSION_STATUS_UNSPECIFIED: _ClassVar[SessionStatus]
    ACCEPTED: _ClassVar[SessionStatus]
    ACCEPTED_WITH_CHANGES: _ClassVar[SessionStatus]
    REJECTED: _ClassVar[SessionStatus]

class CallActionType(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    CALL_ACTION_TYPE_UNSPECIFIED: _ClassVar[CallActionType]
    TRANSFER: _ClassVar[CallActionType]
    HANGUP: _ClassVar[CallActionType]

class AudioDirection(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    AUDIO_DIRECTION_UNSPECIFIED: _ClassVar[AudioDirection]
    INBOUND: _ClassVar[AudioDirection]
    OUTBOUND: _ClassVar[AudioDirection]

class ErrorCategory(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    ERROR_CATEGORY_UNSPECIFIED: _ClassVar[ErrorCategory]
    PROTOCOL: _ClassVar[ErrorCategory]
    AUDIO: _ClassVar[ErrorCategory]
    VAD: _ClassVar[ErrorCategory]
    SESSION: _ClassVar[ErrorCategory]
AUDIO_ENCODING_UNSPECIFIED: AudioEncoding
PCM_S16LE: AudioEncoding
MULAW: AudioEncoding
ALAW: AudioEncoding
SESSION_STATUS_UNSPECIFIED: SessionStatus
ACCEPTED: SessionStatus
ACCEPTED_WITH_CHANGES: SessionStatus
REJECTED: SessionStatus
CALL_ACTION_TYPE_UNSPECIFIED: CallActionType
TRANSFER: CallActionType
HANGUP: CallActionType
AUDIO_DIRECTION_UNSPECIFIED: AudioDirection
INBOUND: AudioDirection
OUTBOUND: AudioDirection
ERROR_CATEGORY_UNSPECIFIED: ErrorCategory
PROTOCOL: ErrorCategory
AUDIO: ErrorCategory
VAD: ErrorCategory
SESSION: ErrorCategory

class AudioConfig(_message.Message):
    __slots__ = ("sample_rate", "encoding", "channels", "frame_duration_ms")
    SAMPLE_RATE_FIELD_NUMBER: _ClassVar[int]
    ENCODING_FIELD_NUMBER: _ClassVar[int]
    CHANNELS_FIELD_NUMBER: _ClassVar[int]
    FRAME_DURATION_MS_FIELD_NUMBER: _ClassVar[int]
    sample_rate: int
    encoding: AudioEncoding
    channels: int
    frame_duration_ms: int
    def __init__(self, sample_rate: _Optional[int] = ..., encoding: _Optional[_Union[AudioEncoding, str]] = ..., channels: _Optional[int] = ..., frame_duration_ms: _Optional[int] = ...) -> None: ...

class VADConfig(_message.Message):
    __slots__ = ("enabled", "silence_threshold_ms", "min_speech_ms", "threshold", "ring_buffer_frames", "speech_ratio", "prefix_padding_ms")
    ENABLED_FIELD_NUMBER: _ClassVar[int]
    SILENCE_THRESHOLD_MS_FIELD_NUMBER: _ClassVar[int]
    MIN_SPEECH_MS_FIELD_NUMBER: _ClassVar[int]
    THRESHOLD_FIELD_NUMBER: _ClassVar[int]
    RING_BUFFER_FRAMES_FIELD_NUMBER: _ClassVar[int]
    SPEECH_RATIO_FIELD_NUMBER: _ClassVar[int]
    PREFIX_PADDING_MS_FIELD_NUMBER: _ClassVar[int]
    enabled: bool
    silence_threshold_ms: int
    min_speech_ms: int
    threshold: float
    ring_buffer_frames: int
    speech_ratio: float
    prefix_padding_ms: int
    def __init__(self, enabled: bool = ..., silence_threshold_ms: _Optional[int] = ..., min_speech_ms: _Optional[int] = ..., threshold: _Optional[float] = ..., ring_buffer_frames: _Optional[int] = ..., speech_ratio: _Optional[float] = ..., prefix_padding_ms: _Optional[int] = ...) -> None: ...

class ProtocolCapabilities(_message.Message):
    __slots__ = ("version", "supported_sample_rates", "supported_encodings", "supported_frame_durations", "vad_configurable", "vad_parameters", "max_session_duration_seconds", "features", "server_id")
    VERSION_FIELD_NUMBER: _ClassVar[int]
    SUPPORTED_SAMPLE_RATES_FIELD_NUMBER: _ClassVar[int]
    SUPPORTED_ENCODINGS_FIELD_NUMBER: _ClassVar[int]
    SUPPORTED_FRAME_DURATIONS_FIELD_NUMBER: _ClassVar[int]
    VAD_CONFIGURABLE_FIELD_NUMBER: _ClassVar[int]
    VAD_PARAMETERS_FIELD_NUMBER: _ClassVar[int]
    MAX_SESSION_DURATION_SECONDS_FIELD_NUMBER: _ClassVar[int]
    FEATURES_FIELD_NUMBER: _ClassVar[int]
    SERVER_ID_FIELD_NUMBER: _ClassVar[int]
    version: str
    supported_sample_rates: _containers.RepeatedScalarFieldContainer[int]
    supported_encodings: _containers.RepeatedScalarFieldContainer[str]
    supported_frame_durations: _containers.RepeatedScalarFieldContainer[int]
    vad_configurable: bool
    vad_parameters: _containers.RepeatedScalarFieldContainer[str]
    max_session_duration_seconds: int
    features: _containers.RepeatedScalarFieldContainer[str]
    server_id: str
    def __init__(self, version: _Optional[str] = ..., supported_sample_rates: _Optional[_Iterable[int]] = ..., supported_encodings: _Optional[_Iterable[str]] = ..., supported_frame_durations: _Optional[_Iterable[int]] = ..., vad_configurable: bool = ..., vad_parameters: _Optional[_Iterable[str]] = ..., max_session_duration_seconds: _Optional[int] = ..., features: _Optional[_Iterable[str]] = ..., server_id: _Optional[str] = ...) -> None: ...

class ConfigAdjustment(_message.Message):
    __slots__ = ("field", "requested", "applied", "reason")
    FIELD_FIELD_NUMBER: _ClassVar[int]
    REQUESTED_FIELD_NUMBER: _ClassVar[int]
    APPLIED_FIELD_NUMBER: _ClassVar[int]
    REASON_FIELD_NUMBER: _ClassVar[int]
    field: str
    requested: str
    applied: str
    reason: str
    def __init__(self, field: _Optional[str] = ..., requested: _Optional[str] = ..., applied: _Optional[str] = ..., reason: _Optional[str] = ...) -> None: ...

class NegotiatedConfig(_message.Message):
    __slots__ = ("audio", "vad", "adjustments")
    AUDIO_FIELD_NUMBER: _ClassVar[int]
    VAD_FIELD_NUMBER: _ClassVar[int]
    ADJUSTMENTS_FIELD_NUMBER: _ClassVar[int]
    audio: AudioConfig
    vad: VADConfig
    adjustments: _containers.RepeatedCompositeFieldContainer[ConfigAdjustment]
    def __init__(self, audio: _Optional[_Union[AudioConfig, _Mapping]] = ..., vad: _Optional[_Union[VADConfig, _Mapping]] = ..., adjustments: _Optional[_Iterable[_Union[ConfigAdjustment, _Mapping]]] = ...) -> None: ...

class ProtocolError(_message.Message):
    __slots__ = ("code", "category", "message", "details", "recoverable")
    class DetailsEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
    CODE_FIELD_NUMBER: _ClassVar[int]
    CATEGORY_FIELD_NUMBER: _ClassVar[int]
    MESSAGE_FIELD_NUMBER: _ClassVar[int]
    DETAILS_FIELD_NUMBER: _ClassVar[int]
    RECOVERABLE_FIELD_NUMBER: _ClassVar[int]
    code: int
    category: str
    message: str
    details: _containers.ScalarMap[str, str]
    recoverable: bool
    def __init__(self, code: _Optional[int] = ..., category: _Optional[str] = ..., message: _Optional[str] = ..., details: _Optional[_Mapping[str, str]] = ..., recoverable: bool = ...) -> None: ...

class SessionStatistics(_message.Message):
    __slots__ = ("audio_frames_received", "audio_frames_sent", "vad_speech_events", "barge_in_count", "average_response_latency_ms")
    AUDIO_FRAMES_RECEIVED_FIELD_NUMBER: _ClassVar[int]
    AUDIO_FRAMES_SENT_FIELD_NUMBER: _ClassVar[int]
    VAD_SPEECH_EVENTS_FIELD_NUMBER: _ClassVar[int]
    BARGE_IN_COUNT_FIELD_NUMBER: _ClassVar[int]
    AVERAGE_RESPONSE_LATENCY_MS_FIELD_NUMBER: _ClassVar[int]
    audio_frames_received: int
    audio_frames_sent: int
    vad_speech_events: int
    barge_in_count: int
    average_response_latency_ms: float
    def __init__(self, audio_frames_received: _Optional[int] = ..., audio_frames_sent: _Optional[int] = ..., vad_speech_events: _Optional[int] = ..., barge_in_count: _Optional[int] = ..., average_response_latency_ms: _Optional[float] = ...) -> None: ...

class AudioData(_message.Message):
    __slots__ = ("audio_payload", "direction")
    AUDIO_PAYLOAD_FIELD_NUMBER: _ClassVar[int]
    DIRECTION_FIELD_NUMBER: _ClassVar[int]
    audio_payload: bytes
    direction: AudioDirection
    def __init__(self, audio_payload: _Optional[bytes] = ..., direction: _Optional[_Union[AudioDirection, str]] = ...) -> None: ...
