"""
LLM gRPC Server — Language Model inference como servico independente.

Carrega o provider LLM (vLLM, Anthropic, etc.) e expoe via gRPC.
Suporta modo batch (Generate) e streaming (GenerateStream) com tool calling.
"""

import asyncio
import json
import logging
import time
from typing import AsyncIterator

import grpc

from shared.grpc_gen import llm_service_pb2 as llm_pb
from shared.grpc_gen import llm_service_pb2_grpc

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("llm-server")


# =============================================================================
# LLM Provider loading
# =============================================================================

async def _create_provider():
    from llm.providers.base import create_llm_provider
    provider = await create_llm_provider()
    logger.info(f"LLM provider carregado: {provider.provider_name}")
    return provider


# =============================================================================
# Proto ↔ dict conversion
# =============================================================================

def _proto_messages_to_dicts(proto_messages) -> list[dict]:
    """Convert proto ChatMessage list to OpenAI-format dicts."""
    messages = []
    for pm in proto_messages:
        msg: dict = {"role": pm.role, "content": pm.content}

        if pm.role == "tool" or pm.tool_call_id:
            msg["tool_call_id"] = pm.tool_call_id

        if pm.tool_calls:
            msg["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.name,
                        "arguments": tc.arguments,
                    },
                }
                for tc in pm.tool_calls
            ]

        messages.append(msg)
    return messages


def _proto_tools_to_dicts(proto_tools) -> list[dict] | None:
    """Convert proto ToolDefinition list to OpenAI-format tool dicts."""
    if not proto_tools:
        return None
    tools = []
    for pt in proto_tools:
        tool = {
            "type": "function",
            "function": {
                "name": pt.name,
                "description": pt.description,
            },
        }
        if pt.parameters_json:
            tool["function"]["parameters"] = json.loads(pt.parameters_json)
        tools.append(tool)
    return tools


# =============================================================================
# gRPC Servicer
# =============================================================================

class LLMServicer(llm_service_pb2_grpc.LLMServiceServicer):
    """Implementacao do LLMService gRPC."""

    def __init__(self, provider, health_servicer=None):
        self._provider = provider
        from common.grpc_server import HealthTracker
        self._health_tracker = HealthTracker(
            "theo.llm.LLMService", health_servicer,
        )

    async def Generate(
        self,
        request: llm_pb.GenerateRequest,
        context: grpc.aio.ServicerContext,
    ) -> llm_pb.GenerateResponse:
        """Batch: messages -> complete response."""
        start = time.perf_counter()

        if not request.messages:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details("messages vazio")
            return llm_pb.GenerateResponse()

        messages = _proto_messages_to_dicts(request.messages)
        tools = _proto_tools_to_dicts(request.tools)

        try:
            text_parts = []
            async for chunk in self._provider.generate_stream(
                messages,
                system=request.system_prompt,
                tools=tools,
                temperature=request.temperature or 0.8,
                max_tokens=request.max_tokens or 1024,
            ):
                text_parts.append(chunk)
            self._health_tracker.record_success()
        except Exception as e:
            logger.error(f"Generate error: {e}")
            self._health_tracker.record_error()
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return llm_pb.GenerateResponse()

        processing_ms = (time.perf_counter() - start) * 1000
        full_text = "".join(text_parts)
        logger.info(
            f"Generate: {len(full_text)} chars ({processing_ms:.0f}ms proc)"
        )

        return llm_pb.GenerateResponse(
            text=full_text,
            processing_ms=processing_ms,
        )

    async def GenerateStream(
        self,
        request: llm_pb.GenerateRequest,
        context: grpc.aio.ServicerContext,
    ) -> AsyncIterator[llm_pb.StreamEvent]:
        """Streaming: messages -> stream of events."""
        if not request.messages:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details("messages vazio")
            return

        messages = _proto_messages_to_dicts(request.messages)
        tools = _proto_tools_to_dicts(request.tools)

        try:
            event_count = 0
            async for event in self._provider.generate_stream_with_tools(
                messages,
                system=request.system_prompt,
                tools=tools,
                temperature=request.temperature or 0.8,
                max_tokens=request.max_tokens or 1024,
            ):
                yield llm_pb.StreamEvent(
                    event_type=event.type,
                    text=event.text,
                    tool_call_id=event.tool_call_id,
                    tool_name=event.tool_name,
                    tool_arguments_delta=event.tool_arguments_delta,
                )
                event_count += 1

            self._health_tracker.record_success()
            logger.info(f"GenerateStream: {event_count} events")

        except asyncio.CancelledError:
            logger.info("GenerateStream cancelado")
            raise
        except Exception as e:
            logger.error(f"GenerateStream error: {e}")
            self._health_tracker.record_error()
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))


# =============================================================================
# Server lifecycle
# =============================================================================

class LLMServer:
    """Wrapper do servidor gRPC para LLM."""

    def __init__(self):
        import os
        from common.grpc_server import GrpcMicroservice
        self._micro = GrpcMicroservice(
            "theo.llm.LLMService",
            port=int(os.getenv("GRPC_PORT", "50080")),
        )
        self._provider = None

    async def start(self, host: str = None, port: int = None):
        self._provider = await _create_provider()

        if host:
            self._micro._host = host
        if port:
            self._micro._port = port

        def add_servicers(server, health_servicer):
            servicer = LLMServicer(self._provider, health_servicer)
            llm_service_pb2_grpc.add_LLMServiceServicer_to_server(servicer, server)

        await self._micro.start(
            add_servicers=add_servicers,
            service_names=(
                llm_pb.DESCRIPTOR.services_by_name["LLMService"].full_name,
            ),
            provider=self._provider,
        )

    async def stop(self):
        await self._micro.stop()

    async def wait(self):
        await self._micro.wait()


async def main():
    from common.grpc_server import configure_logging
    configure_logging()
    server = LLMServer()
    await server.start()
    await server._micro.run_until_signal()


if __name__ == "__main__":
    asyncio.run(main())
