"""Tests for the tool registry."""
from __future__ import annotations

import pytest

from agentforge.tools.base import ToolSpec
from agentforge.tools.registry import ToolRegistry
from agentforge.types import ToolResult


class _AddTool:
    spec = ToolSpec(
        name="add",
        description="Add two numbers.",
        parameters={
            "type": "object",
            "properties": {
                "a": {"type": "number"},
                "b": {"type": "number"},
            },
            "required": ["a", "b"],
        },
    )

    async def run(self, a: float, b: float) -> ToolResult:
        return ToolResult(tool_call_id="", output=str(a + b))


class _CrashTool:
    spec = ToolSpec(
        name="crash",
        description="Always raises.",
        parameters={"type": "object", "properties": {}},
    )

    async def run(self) -> ToolResult:
        raise RuntimeError("boom")


async def test_register_and_invoke():
    r = ToolRegistry()
    r.register(_AddTool())
    result = await r.invoke("add", {"a": 2, "b": 3})
    assert result.output == "5"
    assert not result.is_error


async def test_unknown_tool_returns_error():
    r = ToolRegistry()
    result = await r.invoke("nope", {})
    assert result.is_error
    assert "unknown tool" in result.output.lower()


async def test_crashing_tool_is_caught():
    r = ToolRegistry()
    r.register(_CrashTool())
    result = await r.invoke("crash", {})
    assert result.is_error
    assert "boom" in result.output


async def test_provider_spec_conversion():
    r = ToolRegistry()
    r.register(_AddTool())
    anthropic_specs = r.specs_for_provider("anthropic")
    assert anthropic_specs[0]["name"] == "add"
    assert "input_schema" in anthropic_specs[0]

    openai_specs = r.specs_for_provider("openai")
    assert openai_specs[0]["name"] == "add"
    assert "parameters" in openai_specs[0]
