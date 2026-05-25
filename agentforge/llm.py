"""Thin LLM client wrapper. Two responsibilities:
1. Hide the differences between Anthropic and OpenAI message formats
2. Track tokens and cost per call so runs have an accurate cost field

If we needed a third provider we'd consider litellm, but for two providers
the direct integration is simpler and avoids a dependency.
"""
from __future__ import annotations

import json
import logging
from typing import Any

from anthropic import AsyncAnthropic
from openai import AsyncOpenAI
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from .settings import get_settings
from .types import Message, Role, ToolCall

log = logging.getLogger(__name__)


# Rough cost per million tokens. Update when pricing changes.
# Source: provider pricing pages, last checked 2025-Q1.
COSTS_PER_MTOK: dict[str, tuple[float, float]] = {
    # model_name: (input_per_mtok, output_per_mtok) in USD
    "claude-opus-4-5": (15.0, 75.0),
    "claude-sonnet-4-5": (3.0, 15.0),
    "claude-haiku-4-5": (0.80, 4.0),
    "gpt-4o": (2.50, 10.0),
    "gpt-4o-mini": (0.15, 0.60),
}


def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    if model not in COSTS_PER_MTOK:
        log.warning("No cost data for model %s, returning 0.", model)
        return 0.0
    in_cost, out_cost = COSTS_PER_MTOK[model]
    return (input_tokens * in_cost + output_tokens * out_cost) / 1_000_000


class LLMResponse:
    """Provider-agnostic response. Returned by LLMClient.complete()."""

    def __init__(
        self,
        text: str,
        tool_calls: list[ToolCall],
        input_tokens: int,
        output_tokens: int,
        model: str,
        stop_reason: str,
    ):
        self.text = text
        self.tool_calls = tool_calls
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.model = model
        self.stop_reason = stop_reason
        self.cost_usd = estimate_cost(model, input_tokens, output_tokens)


class LLMClient:
    """Routes to Anthropic or OpenAI based on model name prefix.

    We intentionally don't support streaming here — the server layer does
    streaming via SSE by composing multiple non-streaming calls. Streaming
    inside the agent loop made the trace viewer logic painful; not worth it.
    """

    def __init__(self) -> None:
        s = get_settings()
        self._anthropic = AsyncAnthropic(api_key=s.anthropic_api_key) if s.anthropic_api_key else None
        self._openai = AsyncOpenAI(api_key=s.openai_api_key) if s.openai_api_key else None

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=20),
        retry=retry_if_exception_type(Exception),  # in practice narrow this
        reraise=True,
    )
    async def complete(
        self,
        messages: list[Message],
        model: str,
        system: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        response_format: dict[str, Any] | None = None,
    ) -> LLMResponse:
        if model.startswith("claude"):
            return await self._anthropic_complete(
                messages, model, system, tools, temperature, max_tokens
            )
        elif model.startswith("gpt"):
            return await self._openai_complete(
                messages, model, system, tools, temperature, max_tokens, response_format
            )
        else:
            raise ValueError(f"Unknown model prefix: {model}")

    async def _anthropic_complete(
        self,
        messages: list[Message],
        model: str,
        system: str | None,
        tools: list[dict[str, Any]] | None,
        temperature: float,
        max_tokens: int,
    ) -> LLMResponse:
        if not self._anthropic:
            raise RuntimeError("ANTHROPIC_API_KEY not configured")

        anthro_messages = _to_anthropic_messages(messages)
        kwargs: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": anthro_messages,
        }
        if system:
            kwargs["system"] = system
        if tools:
            kwargs["tools"] = tools

        resp = await self._anthropic.messages.create(**kwargs)

        # Extract text and tool calls from content blocks
        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        for block in resp.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                tool_calls.append(
                    ToolCall(id=block.id, name=block.name, arguments=block.input)
                )

        return LLMResponse(
            text="".join(text_parts),
            tool_calls=tool_calls,
            input_tokens=resp.usage.input_tokens,
            output_tokens=resp.usage.output_tokens,
            model=model,
            stop_reason=resp.stop_reason or "end_turn",
        )

    async def _openai_complete(
        self,
        messages: list[Message],
        model: str,
        system: str | None,
        tools: list[dict[str, Any]] | None,
        temperature: float,
        max_tokens: int,
        response_format: dict[str, Any] | None,
    ) -> LLMResponse:
        if not self._openai:
            raise RuntimeError("OPENAI_API_KEY not configured")

        oai_messages = _to_openai_messages(messages, system=system)
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": oai_messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if tools:
            # OpenAI uses {"type": "function", "function": {...}} wrapping
            kwargs["tools"] = [{"type": "function", "function": t} for t in tools]
        if response_format:
            kwargs["response_format"] = response_format

        resp = await self._openai.chat.completions.create(**kwargs)
        choice = resp.choices[0]

        tool_calls: list[ToolCall] = []
        for tc in choice.message.tool_calls or []:
            try:
                args = json.loads(tc.function.arguments)
            except json.JSONDecodeError:
                log.warning("Failed to parse tool args: %s", tc.function.arguments)
                args = {}
            tool_calls.append(ToolCall(id=tc.id, name=tc.function.name, arguments=args))

        return LLMResponse(
            text=choice.message.content or "",
            tool_calls=tool_calls,
            input_tokens=resp.usage.prompt_tokens if resp.usage else 0,
            output_tokens=resp.usage.completion_tokens if resp.usage else 0,
            model=model,
            stop_reason=choice.finish_reason or "stop",
        )


def _to_anthropic_messages(messages: list[Message]) -> list[dict[str, Any]]:
    """Convert our Message format to Anthropic's. Anthropic doesn't have a
    top-level system role — it goes as a `system` parameter, handled by caller."""
    out: list[dict[str, Any]] = []
    for m in messages:
        if m.role == Role.SYSTEM:
            continue  # handled separately
        if m.role == Role.TOOL:
            # tool result message
            out.append({
                "role": "user",
                "content": [{
                    "type": "tool_result",
                    "tool_use_id": m.tool_call_id,
                    "content": m.content,
                }],
            })
            continue

        content_blocks: list[dict[str, Any]] = []
        if m.content:
            content_blocks.append({"type": "text", "text": m.content})
        for tc in m.tool_calls:
            content_blocks.append({
                "type": "tool_use",
                "id": tc.id,
                "name": tc.name,
                "input": tc.arguments,
            })

        out.append({
            "role": "assistant" if m.role == Role.ASSISTANT else "user",
            "content": content_blocks if content_blocks else m.content,
        })
    return out


def _to_openai_messages(
    messages: list[Message], system: str | None = None
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if system:
        out.append({"role": "system", "content": system})
    for m in messages:
        if m.role == Role.TOOL:
            out.append({
                "role": "tool",
                "tool_call_id": m.tool_call_id,
                "content": m.content,
            })
            continue
        msg: dict[str, Any] = {"role": m.role.value, "content": m.content or None}
        if m.tool_calls:
            msg["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)},
                }
                for tc in m.tool_calls
            ]
        out.append(msg)
    return out
