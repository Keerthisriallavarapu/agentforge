"""Tool registry. Tools register themselves here; the registry handles:
- Conversion to Anthropic/OpenAI tool specs
- Invocation with timeout enforcement
- Error wrapping so a crashing tool can't take down the agent loop
"""
from __future__ import annotations

import asyncio
import logging
import traceback
from typing import Any

from ..settings import get_settings
from ..types import ToolResult
from .base import Tool, ToolSpec

log = logging.getLogger(__name__)


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        if tool.spec.name in self._tools:
            log.warning("Tool %s already registered; overwriting.", tool.spec.name)
        self._tools[tool.spec.name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def list_names(self) -> list[str]:
        return sorted(self._tools.keys())

    def describe_for_prompt(self) -> str:
        """Compact text representation for prompts (planner uses this)."""
        if not self._tools:
            return "(none)"
        lines = []
        for name in sorted(self._tools):
            t = self._tools[name]
            lines.append(f"- {name}: {t.spec.description}")
        return "\n".join(lines)

    def specs_for_provider(self, provider: str) -> list[dict[str, Any]]:
        if provider == "anthropic":
            return [self._to_anthropic_spec(t.spec) for t in self._tools.values()]
        elif provider == "openai":
            return [self._to_openai_spec(t.spec) for t in self._tools.values()]
        else:
            raise ValueError(f"Unknown provider: {provider}")

    async def invoke(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        tool = self._tools.get(name)
        if tool is None:
            return ToolResult(
                tool_call_id="",
                output=f"Error: unknown tool '{name}'. Available: {self.list_names()}",
                is_error=True,
            )

        timeout = tool.spec.timeout_seconds or get_settings().tool_timeout_seconds
        try:
            result = await asyncio.wait_for(tool.run(**arguments), timeout=timeout)
            return result
        except asyncio.TimeoutError:
            log.error("Tool %s timed out after %ds", name, timeout)
            return ToolResult(
                tool_call_id="",
                output=f"Error: tool '{name}' timed out after {timeout}s",
                is_error=True,
            )
        except Exception as e:
            log.error("Tool %s crashed: %s", name, e)
            log.debug(traceback.format_exc())
            return ToolResult(
                tool_call_id="",
                output=f"Error executing '{name}': {e}",
                is_error=True,
            )

    @staticmethod
    def _to_anthropic_spec(spec: ToolSpec) -> dict[str, Any]:
        return {
            "name": spec.name,
            "description": spec.description,
            "input_schema": spec.parameters,
        }

    @staticmethod
    def _to_openai_spec(spec: ToolSpec) -> dict[str, Any]:
        return {
            "name": spec.name,
            "description": spec.description,
            "parameters": spec.parameters,
        }


_default_registry: ToolRegistry | None = None


def get_default_registry() -> ToolRegistry:
    """Lazy singleton. Builtin tools are registered here on first access."""
    global _default_registry
    if _default_registry is not None:
        return _default_registry

    registry = ToolRegistry()
    # Lazy imports to avoid circular dependencies
    from .builtins import PythonReplTool, WebSearchTool, ReadFileTool, WriteFileTool

    registry.register(PythonReplTool())
    registry.register(WebSearchTool())
    registry.register(ReadFileTool())
    registry.register(WriteFileTool())
    _default_registry = registry
    return registry
