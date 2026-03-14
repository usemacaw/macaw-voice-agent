import asp_common_pb2 as _asp_common_pb2
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class SynthesizeRequest(_message.Message):
    __slots__ = ("text", "language", "output_config", "voice_id")
    TEXT_FIELD_NUMBER: _ClassVar[int]
    LANGUAGE_FIELD_NUMBER: _ClassVar[int]
    OUTPUT_CONFIG_FIELD_NUMBER: _ClassVar[int]
    VOICE_ID_FIELD_NUMBER: _ClassVar[int]
    text: str
    language: str
    output_config: _asp_common_pb2.AudioConfig
    voice_id: str
    def __init__(self, text: _Optional[str] = ..., language: _Optional[str] = ..., output_config: _Optional[_Union[_asp_common_pb2.AudioConfig, _Mapping]] = ..., voice_id: _Optional[str] = ...) -> None: ...

class SynthesizeResponse(_message.Message):
    __slots__ = ("audio_data", "duration_ms", "processing_ms")
    AUDIO_DATA_FIELD_NUMBER: _ClassVar[int]
    DURATION_MS_FIELD_NUMBER: _ClassVar[int]
    PROCESSING_MS_FIELD_NUMBER: _ClassVar[int]
    audio_data: bytes
    duration_ms: float
    processing_ms: float
    def __init__(self, audio_data: _Optional[bytes] = ..., duration_ms: _Optional[float] = ..., processing_ms: _Optional[float] = ...) -> None: ...

class AudioChunk(_message.Message):
    __slots__ = ("audio_payload", "is_last", "sequence")
    AUDIO_PAYLOAD_FIELD_NUMBER: _ClassVar[int]
    IS_LAST_FIELD_NUMBER: _ClassVar[int]
    SEQUENCE_FIELD_NUMBER: _ClassVar[int]
    audio_payload: bytes
    is_last: bool
    sequence: int
    def __init__(self, audio_payload: _Optional[bytes] = ..., is_last: bool = ..., sequence: _Optional[int] = ...) -> None: ...
