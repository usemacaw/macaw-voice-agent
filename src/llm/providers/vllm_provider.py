"""
vLLM provider (server-side) — connects to vLLM via OpenAI-compatible API.

Config (env vars):
    LLM_BACKEND_PROVIDER=vllm
    VLLM_BASE_URL=http://localhost:8000/v1
    LLM_MODEL=Qwen/Qwen2.5-7B-Instruct-AWQ
"""

from __future__ import annotations

import logging
import os
import time
from typing import AsyncGenerator

from llm.providers.base import LLMProvider, LLMStreamEvent, register_llm_provider

logger = logging.getLogger("llm-server.vllm")


class VLLMProvider(LLMProvider):
    """LLM provider backed by a vLLM server (OpenAI-compatible API)."""

    provider_name = "vllm"

    async def connect(self) -> None:
        from openai import AsyncOpenAI

        base_url = os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1")
        self._model = os.getenv("LLM_MODEL", "Qwen/Qwen2.5-7B-Instruct-AWQ")
        timeout = float(os.getenv("LLM_TIMEOUT", "30.0"))

        self._client = AsyncOpenAI(
            base_url=base_url,
            api_key="not-needed",
            timeout=timeout,
        )
        logger.info(f"vLLM provider connected: model={self._model}, base_url={base_url}")

    async def disconnect(self) -> None:
        if hasattr(self, "_client"):
            await self._client.close()

    async def generate_stream(
        self,
        messages: list[dict],
        system: str = "",
        tools: list[dict] | None = None,
        temperature: float = 0.8,
        max_tokens: int = 1024,
    ) -> AsyncGenerator[str, None]:
        full_messages = []
        if system:
            full_messages.append({"role": "system", "content": system})
        full_messages.extend(messages)

        kwargs = {
            "model": self._model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": full_messages,
            "stream": True,
            "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
        }
        if tools:
            kwargs["tools"] = tools

        stream = await self._client.chat.completions.create(**kwargs)
        async for chunk in stream:
            delta = chunk.choices[0].delta if chunk.choices else None
            if delta and delta.content:
                yield delta.content

    async def generate_stream_with_tools(
        self,
        messages: list[dict],
        system: str = "",
        tools: list[dict] | None = None,
        temperature: float = 0.8,
        max_tokens: int = 1024,
    ) -> AsyncGenerator[LLMStreamEvent, None]:
        full_messages = []
        if system:
            full_messages.append({"role": "system", "content": system})
        full_messages.extend(messages)

        kwargs = {
            "model": self._model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": full_messages,
            "stream": True,
            "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
        }
        if tools:
            kwargs["tools"] = tools

        stream = await self._client.chat.completions.create(**kwargs)

        active_tool_calls: dict[int, dict] = {}

        async for chunk in stream:
            delta = chunk.choices[0].delta if chunk.choices else None
            if not delta:
                continue

            if delta.content:
                yield LLMStreamEvent(type="text_delta", text=delta.content)

            if delta.tool_calls:
                for tc in delta.tool_calls:
                    idx = tc.index
                    if idx not in active_tool_calls:
                        tc_id = tc.id or ""
                        tc_name = tc.function.name if tc.function and tc.function.name else ""
                        active_tool_calls[idx] = {"id": tc_id, "name": tc_name}
                        yield LLMStreamEvent(
                            type="tool_call_start",
                            tool_call_id=tc_id,
                            tool_name=tc_name,
                        )
                    if tc.function and tc.function.arguments:
                        info = active_tool_calls[idx]
                        yield LLMStreamEvent(
                            type="tool_call_delta",
                            tool_call_id=info["id"],
                            tool_name=info["name"],
                            tool_arguments_delta=tc.function.arguments,
                        )

        for info in active_tool_calls.values():
            yield LLMStreamEvent(
                type="tool_call_end",
                tool_call_id=info["id"],
                tool_name=info["name"],
            )


register_llm_provider("vllm", VLLMProvider)
