"""
vLLM LLM provider for OpenVoiceAPI.

Connects to a vLLM server via its OpenAI-compatible API.
Supports streaming text generation and function calling.

Config:
    LLM_PROVIDER=vllm
    VLLM_BASE_URL=http://localhost:8000/v1
    LLM_MODEL=Qwen/Qwen2.5-7B-Instruct-AWQ
"""

from __future__ import annotations

import logging
import os
import time
from typing import AsyncGenerator

from config import LLM_CONFIG
from providers.llm import LLMProvider, LLMStreamEvent, register_llm_provider
from providers._openai_stream import parse_openai_tool_stream

logger = logging.getLogger("open-voice-api.llm.vllm")


class VLLMProvider(LLMProvider):
    """LLM provider that connects to a vLLM server via OpenAI-compatible API."""

    provider_name = "vllm"

    def __init__(self):
        from openai import AsyncOpenAI

        base_url = os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1")

        self._model = LLM_CONFIG["model"]
        self._client = AsyncOpenAI(
            base_url=base_url,
            api_key="not-needed",
            timeout=LLM_CONFIG["timeout"],
        )
        logger.info(f"vLLM provider initialized: model={self._model}, base_url={base_url}")

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

        t0 = time.perf_counter()
        first_token = False

        stream = await self._client.chat.completions.create(**kwargs)
        async for chunk in stream:
            delta = chunk.choices[0].delta if chunk.choices else None
            if delta and delta.content:
                if not first_token:
                    first_token = True
                    self.last_ttft_ms = (time.perf_counter() - t0) * 1000
                    logger.info(f"LLM first token in {self.last_ttft_ms:.0f}ms")
                yield delta.content

        self.last_stream_total_ms = (time.perf_counter() - t0) * 1000
        logger.info(f"LLM stream complete: {self.last_stream_total_ms:.0f}ms")

    async def generate_stream_with_tools(
        self,
        messages: list[dict],
        system: str = "",
        tools: list[dict] | None = None,
        temperature: float = 0.8,
        max_tokens: int = 1024,
    ) -> AsyncGenerator[LLMStreamEvent, None]:
        """Stream with function calling support via vLLM's OpenAI-compatible API."""
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

        t0 = time.perf_counter()

        stream = await self._client.chat.completions.create(**kwargs)

        def _on_first_token():
            self.last_ttft_ms = (time.perf_counter() - t0) * 1000
            logger.info(f"LLM first event in {self.last_ttft_ms:.0f}ms")

        async for event in parse_openai_tool_stream(stream, on_first_token=_on_first_token):
            yield event

        self.last_stream_total_ms = (time.perf_counter() - t0) * 1000
        logger.info(f"LLM stream (with tools) complete: {self.last_stream_total_ms:.0f}ms")

    async def connect(self) -> None:
        pass

    async def disconnect(self) -> None:
        await self._client.close()


register_llm_provider("vllm", VLLMProvider)
