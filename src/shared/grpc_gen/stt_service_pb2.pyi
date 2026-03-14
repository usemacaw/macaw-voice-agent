import asp_common_pb2 as _asp_common_pb2
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class AudioChunk(_message.Message):
    __slots__ = ("stream_id", "audio_payload", "audio_config", "end_of_stream")
    STREAM_ID_FIELD_NUMBER: _ClassVar[int]
    AUDIO_PAYLOAD_FIELD_NUMBER: _ClassVar[int]
    AUDIO_CONFIG_FIELD_NUMBER: _ClassVar[int]
    END_OF_STREAM_FIELD_NUMBER: _ClassVar[int]
    stream_id: str
    audio_payload: bytes
    audio_config: _asp_common_pb2.AudioConfig
    end_of_stream: bool
    def __init__(self, stream_id: _Optional[str] = ..., audio_payload: _Optional[bytes] = ..., audio_config: _Optional[_Union[_asp_common_pb2.AudioConfig, _Mapping]] = ..., end_of_stream: bool = ...) -> None: ...

class TranscribeResult(_message.Message):
    __slots__ = ("stream_id", "text", "is_final", "confidence")
    STREAM_ID_FIELD_NUMBER: _ClassVar[int]
    TEXT_FIELD_NUMBER: _ClassVar[int]
    IS_FINAL_FIELD_NUMBER: _ClassVar[int]
    CONFIDENCE_FIELD_NUMBER: _ClassVar[int]
    stream_id: str
    text: str
    is_final: bool
    confidence: float
    def __init__(self, stream_id: _Optional[str] = ..., text: _Optional[str] = ..., is_final: bool = ..., confidence: _Optional[float] = ...) -> None: ...

class TranscribeRequest(_message.Message):
    __slots__ = ("audio_data", "audio_config", "language")
    AUDIO_DATA_FIELD_NUMBER: _ClassVar[int]
    AUDIO_CONFIG_FIELD_NUMBER: _ClassVar[int]
    LANGUAGE_FIELD_NUMBER: _ClassVar[int]
    audio_data: bytes
    audio_config: _asp_common_pb2.AudioConfig
    language: str
    def __init__(self, audio_data: _Optional[bytes] = ..., audio_config: _Optional[_Union[_asp_common_pb2.AudioConfig, _Mapping]] = ..., language: _Optional[str] = ...) -> None: ...

class TranscribeResponse(_message.Message):
    __slots__ = ("text", "confidence", "processing_ms")
    TEXT_FIELD_NUMBER: _ClassVar[int]
    CONFIDENCE_FIELD_NUMBER: _ClassVar[int]
    PROCESSING_MS_FIELD_NUMBER: _ClassVar[int]
    text: str
    confidence: float
    processing_ms: float
    def __init__(self, text: _Optional[str] = ..., confidence: _Optional[float] = ..., processing_ms: _Optional[float] = ...) -> None: ...
