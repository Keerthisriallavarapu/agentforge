"""Tool protocol. A tool is anything with a name, a JSON schema for inputs,
and an async `run` method. We don't use ABCs because the duck-typed protocol
plays nicer with dynamically-registered tools (e.g. MCP servers in the future).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from ..types import ToolResult


@dataclass(frozen=True)
class ToolSpec:
    """Provider-agnostic tool spec. We convert this to Anthropic or OpenAI
    schemas in the registry. Keeping our own intermediate form lets us add
    metadata (timeouts, sandboxing flags) without polluting the LLM-facing
    schema."""

    name: str
    description: str
    parameters: dict[str, Any]  # JSON Schema
    requires_sandbox: bool = False
    timeout_seconds: int = 30


@runtime_checkable
class Tool(Protocol):
    spec: ToolSpec

    async def run(self, **kwargs: Any) -> ToolResult: ...
