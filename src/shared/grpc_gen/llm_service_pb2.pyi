from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class ChatMessage(_message.Message):
    __slots__ = ("role", "content", "tool_calls", "tool_call_id")
    ROLE_FIELD_NUMBER: _ClassVar[int]
    CONTENT_FIELD_NUMBER: _ClassVar[int]
    TOOL_CALLS_FIELD_NUMBER: _ClassVar[int]
    TOOL_CALL_ID_FIELD_NUMBER: _ClassVar[int]
    role: str
    content: str
    tool_calls: _containers.RepeatedCompositeFieldContainer[ToolCall]
    tool_call_id: str
    def __init__(self, role: _Optional[str] = ..., content: _Optional[str] = ..., tool_calls: _Optional[_Iterable[_Union[ToolCall, _Mapping]]] = ..., tool_call_id: _Optional[str] = ...) -> None: ...

class ToolCall(_message.Message):
    __slots__ = ("id", "name", "arguments")
    ID_FIELD_NUMBER: _ClassVar[int]
    NAME_FIELD_NUMBER: _ClassVar[int]
    ARGUMENTS_FIELD_NUMBER: _ClassVar[int]
    id: str
    name: str
    arguments: str
    def __init__(self, id: _Optional[str] = ..., name: _Optional[str] = ..., arguments: _Optional[str] = ...) -> None: ...

class ToolDefinition(_message.Message):
    __slots__ = ("name", "description", "parameters_json")
    NAME_FIELD_NUMBER: _ClassVar[int]
    DESCRIPTION_FIELD_NUMBER: _ClassVar[int]
    PARAMETERS_JSON_FIELD_NUMBER: _ClassVar[int]
    name: str
    description: str
    parameters_json: str
    def __init__(self, name: _Optional[str] = ..., description: _Optional[str] = ..., parameters_json: _Optional[str] = ...) -> None: ...

class GenerateRequest(_message.Message):
    __slots__ = ("messages", "system_prompt", "tools", "temperature", "max_tokens", "model")
    MESSAGES_FIELD_NUMBER: _ClassVar[int]
    SYSTEM_PROMPT_FIELD_NUMBER: _ClassVar[int]
    TOOLS_FIELD_NUMBER: _ClassVar[int]
    TEMPERATURE_FIELD_NUMBER: _ClassVar[int]
    MAX_TOKENS_FIELD_NUMBER: _ClassVar[int]
    MODEL_FIELD_NUMBER: _ClassVar[int]
    messages: _containers.RepeatedCompositeFieldContainer[ChatMessage]
    system_prompt: str
    tools: _containers.RepeatedCompositeFieldContainer[ToolDefinition]
    temperature: float
    max_tokens: int
    model: str
    def __init__(self, messages: _Optional[_Iterable[_Union[ChatMessage, _Mapping]]] = ..., system_prompt: _Optional[str] = ..., tools: _Optional[_Iterable[_Union[ToolDefinition, _Mapping]]] = ..., temperature: _Optional[float] = ..., max_tokens: _Optional[int] = ..., model: _Optional[str] = ...) -> None: ...

class GenerateResponse(_message.Message):
    __slots__ = ("text", "tool_calls", "processing_ms")
    TEXT_FIELD_NUMBER: _ClassVar[int]
    TOOL_CALLS_FIELD_NUMBER: _ClassVar[int]
    PROCESSING_MS_FIELD_NUMBER: _ClassVar[int]
    text: str
    tool_calls: _containers.RepeatedCompositeFieldContainer[ToolCall]
    processing_ms: float
    def __init__(self, text: _Optional[str] = ..., tool_calls: _Optional[_Iterable[_Union[ToolCall, _Mapping]]] = ..., processing_ms: _Optional[float] = ...) -> None: ...

class StreamEvent(_message.Message):
    __slots__ = ("event_type", "text", "tool_call_id", "tool_name", "tool_arguments_delta")
    EVENT_TYPE_FIELD_NUMBER: _ClassVar[int]
    TEXT_FIELD_NUMBER: _ClassVar[int]
    TOOL_CALL_ID_FIELD_NUMBER: _ClassVar[int]
    TOOL_NAME_FIELD_NUMBER: _ClassVar[int]
    TOOL_ARGUMENTS_DELTA_FIELD_NUMBER: _ClassVar[int]
    event_type: str
    text: str
    tool_call_id: str
    tool_name: str
    tool_arguments_delta: str
    def __init__(self, event_type: _Optional[str] = ..., text: _Optional[str] = ..., tool_call_id: _Optional[str] = ..., tool_name: _Optional[str] = ..., tool_arguments_delta: _Optional[str] = ...) -> None: ...
