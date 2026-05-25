"""Test fixtures. We mock the LLM client because hitting real APIs in unit
tests is slow, flaky, and expensive. Integration tests in tests/integration/
use real keys."""
from __future__ import annotations

import pytest

from agentforge.llm import LLMResponse
from agentforge.types import ToolCall


class FakeLLMClient:
    """A scriptable LLM client. Push responses with `enqueue`."""

    def __init__(self):
        self._responses: list[LLMResponse] = []
        self.calls: list[dict] = []

    def enqueue(
        self,
        text: str = "",
        tool_calls: list[ToolCall] | None = None,
        input_tokens: int = 100,
        output_tokens: int = 100,
        model: str = "fake",
        stop_reason: str = "end_turn",
    ) -> None:
        self._responses.append(LLMResponse(
            text=text,
            tool_calls=tool_calls or [],
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            model=model,
            stop_reason=stop_reason,
        ))

    async def complete(self, **kwargs) -> LLMResponse:
        self.calls.append(kwargs)
        if not self._responses:
            raise AssertionError("FakeLLMClient ran out of scripted responses")
        return self._responses.pop(0)


@pytest.fixture
def fake_llm():
    return FakeLLMClient()
