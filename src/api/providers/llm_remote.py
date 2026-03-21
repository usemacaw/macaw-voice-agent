"""
Remote LLM Provider — delegates to LLM Service via gRPC.

Reuses the LLM gRPC service (proto/llm_service.proto).

Config:
    LLM_PROVIDER=remote
    LLM_REMOTE_TARGET=localhost:50080
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import AsyncGenerator

import grpc

from grpc_gen import llm_service_pb2 as llm_pb
from grpc_gen import llm_service_pb2_grpc

from config import LLM
from providers.llm import LLMProvider, LLMStreamEvent, register_llm_provider

logger = logging.getLogger("open-voice-api.llm.remote")


class RemoteLLM(LLMProvider):
    """LLM provider that delegates to external LLM Service via gRPC."""

    provider_name = "remote"

    def __init__(self):
        self._target = LLM.remote_target
        self._timeout = LLM.timeout
        self._channel: grpc.aio.Channel | None = None
        self._stub: llm_service_pb2_grpc.LLMServiceStub | None = None

    async def connect(self) -> None:
        max_msg = 10 * 1024 * 1024
        self._channel = grpc.aio.insecure_channel(
            self._target,
            options=[
                ("grpc.max_receive_message_length", max_msg),
                ("grpc.max_send_message_length", max_msg),
                ("grpc.keepalive_time_ms", 30000),
                ("grpc.keepalive_timeout_ms", 10000),
            ],
        )
        self._stub = llm_service_pb2_grpc.LLMServiceStub(self._channel)
        logger.info(f"Remote LLM channel created: {self._target} (lazy connect)")

    async def warmup(self) -> None:
        if not self._channel:
            return
        try:
            await asyncio.wait_for(
                self._channel.channel_ready(), timeout=5.0
            )
            logger.info(f"LLM gRPC channel warmed up: {self._target}")
        except Exception as e:
            logger.warning(f"LLM warmup failed (will retry on first call): {e}")

    async def disconnect(self) -> None:
        if self._channel:
            await self._channel.close()
            self._channel = None
            self._stub = None

    async def generate_stream(
        self,
        messages: list[dict],
        system: str = "",
        tools: list[dict] | None = None,
        temperature: float = 0.8,
        max_tokens: int = 1024,
    ) -> AsyncGenerator[str, None]:
        if not self._stub:
            raise RuntimeError("Remote LLM not connected")

        request = _build_request(messages, system, tools, temperature, max_tokens)
        t0 = time.perf_counter()
        first_token = False

        try:
            async for event in self._stub.GenerateStream(
                request, timeout=self._timeout
            ):
                if not first_token:
                    first_token = True
                    self.last_ttft_ms = (time.perf_counter() - t0) * 1000
                    logger.info(f"LLM first token in {self.last_ttft_ms:.0f}ms")
                if event.event_type == "text_delta" and event.text:
                    yield event.text
        except grpc.aio.AioRpcError as e:
            elapsed_ms = (time.perf_counter() - t0) * 1000
            logger.error(f"LLM stream error after {elapsed_ms:.0f}ms: {e}")
            raise RuntimeError(f"LLM service unavailable: {e}") from e
        finally:
            self.last_stream_total_ms = (time.perf_counter() - t0) * 1000

    async def generate_stream_with_tools(
        self,
        messages: list[dict],
        system: str = "",
        tools: list[dict] | None = None,
        temperature: float = 0.8,
        max_tokens: int = 1024,
    ) -> AsyncGenerator[LLMStreamEvent, None]:
        if not self._stub:
            raise RuntimeError("Remote LLM not connected")

        request = _build_request(messages, system, tools, temperature, max_tokens)
        t0 = time.perf_counter()
        first_token = False

        try:
            async for event in self._stub.GenerateStream(
                request, timeout=self._timeout
            ):
                if not first_token:
                    first_token = True
                    self.last_ttft_ms = (time.perf_counter() - t0) * 1000
                    logger.info(f"LLM first event in {self.last_ttft_ms:.0f}ms")

                yield LLMStreamEvent(
                    type=event.event_type,
                    text=event.text,
                    tool_call_id=event.tool_call_id,
                    tool_name=event.tool_name,
                    tool_arguments_delta=event.tool_arguments_delta,
                )
        except grpc.aio.AioRpcError as e:
            elapsed_ms = (time.perf_counter() - t0) * 1000
            logger.error(f"LLM stream error after {elapsed_ms:.0f}ms: {e}")
            raise RuntimeError(f"LLM service unavailable: {e}") from e
        finally:
            self.last_stream_total_ms = (time.perf_counter() - t0) * 1000


# ---------------------------------------------------------------------------
# Message conversion: OpenAI-format dicts → proto messages
# ---------------------------------------------------------------------------

def _build_request(
    messages: list[dict],
    system: str,
    tools: list[dict] | None,
    temperature: float,
    max_tokens: int,
) -> llm_pb.GenerateRequest:
    proto_messages = []
    for msg in messages:
        proto_msg = llm_pb.ChatMessage(
            role=msg.get("role", ""),
            content=msg.get("content", "") or "",
            tool_call_id=msg.get("tool_call_id", ""),
        )
        # Convert tool_calls list (assistant messages)
        for tc in msg.get("tool_calls", []):
            func = tc.get("function", {})
            args = func.get("arguments", "")
            if not isinstance(args, str):
                args = json.dumps(args)
            proto_msg.tool_calls.append(
                llm_pb.ToolCall(
                    id=tc.get("id", ""),
                    name=func.get("name", ""),
                    arguments=args,
                )
            )
        proto_messages.append(proto_msg)

    proto_tools = []
    if tools:
        for tool in tools:
            func = tool.get("function", tool)
            params = func.get("parameters", {})
            proto_tools.append(
                llm_pb.ToolDefinition(
                    name=func.get("name", ""),
                    description=func.get("description", ""),
                    parameters_json=json.dumps(params) if params else "",
                )
            )

    return llm_pb.GenerateRequest(
        messages=proto_messages,
        system_prompt=system,
        tools=proto_tools,
        temperature=temperature,
        max_tokens=max_tokens,
    )


register_llm_provider("remote", RemoteLLM)
