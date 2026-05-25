"""Executor: works through plan steps, calling tools as needed.

This is where the bulk of LLM calls happen, so we use the cheaper/faster
model (sonnet vs opus). The executor loop continues until the model returns
no tool calls — that's our signal it's done with the step.
"""
from __future__ import annotations

import logging
from typing import AsyncIterator

from ..llm import LLMClient, LLMResponse
from ..settings import get_settings
from ..tools.registry import ToolRegistry
from ..types import Message, PlanStep, Role, ToolResult

log = logging.getLogger(__name__)


SYSTEM_PROMPT = """You are the Executor in a multi-agent system. You are given a single subtask and must complete it, using tools when helpful.

Rules:
- Take small, verifiable steps.
- Call tools to gather information or perform actions. Don't guess.
- When the subtask is complete, summarize the result clearly in a final message with no tool calls.
- If you encounter an error you can't resolve in 3 attempts, explain what went wrong and stop."""


class ExecutorEvent:
    """Yielded by execute_step so the trace viewer can render in real time."""

    def __init__(self, kind: str, payload: dict):
        self.kind = kind  # "thought" | "tool_call" | "tool_result" | "done"
        self.payload = payload


class Executor:
    def __init__(self, llm: LLMClient, tools: ToolRegistry, max_iterations: int = 10):
        self._llm = llm
        self._tools = tools
        self._max_iters = max_iterations
        self._model = get_settings().executor_model

    async def execute_step(
        self,
        step: PlanStep,
        prior_messages: list[Message],
    ) -> AsyncIterator[ExecutorEvent]:
        """Execute one plan step. Yields events for streaming to a trace viewer.

        prior_messages contains the conversation history including previous
        step results, so the executor has context for what's already been done.
        """
        tool_specs = self._tools.specs_for_provider("anthropic")
        # Prepend step framing to the conversation
        step_intro = Message(
            role=Role.USER,
            content=(
                f"Subtask: {step.description}\n"
                f"Expected output: {step.expected_output}\n"
                f"Suggested tools: {', '.join(step.suggested_tools) or 'none'}"
            ),
        )
        messages = [*prior_messages, step_intro]

        for iteration in range(self._max_iters):
            log.debug("Executor iter %d for step %s", iteration, step.id)
            resp: LLMResponse = await self._llm.complete(
                messages=messages,
                model=self._model,
                system=SYSTEM_PROMPT,
                tools=tool_specs,
                temperature=0.5,
            )

            # Record the assistant message
            assistant_msg = Message(
                role=Role.ASSISTANT,
                content=resp.text,
                tool_calls=resp.tool_calls,
            )
            messages.append(assistant_msg)

            if resp.text:
                yield ExecutorEvent("thought", {"text": resp.text, "model": resp.model})

            # No tool calls -> step is done
            if not resp.tool_calls:
                yield ExecutorEvent("done", {
                    "final_message": resp.text,
                    "iterations": iteration + 1,
                    "cost_usd": resp.cost_usd,
                })
                return

            # Execute each tool call and append results
            for tc in resp.tool_calls:
                yield ExecutorEvent("tool_call", {
                    "id": tc.id,
                    "name": tc.name,
                    "arguments": tc.arguments,
                })
                result = await self._tools.invoke(tc.name, tc.arguments)
                yield ExecutorEvent("tool_result", {
                    "id": tc.id,
                    "output": result.output,
                    "is_error": result.is_error,
                })
                messages.append(Message(
                    role=Role.TOOL,
                    content=result.output,
                    tool_call_id=tc.id,
                ))

        # Hit max iterations without finishing
        log.warning("Executor hit max iterations on step %s", step.id)
        yield ExecutorEvent("done", {
            "final_message": "Step exceeded max iterations.",
            "iterations": self._max_iters,
            "incomplete": True,
        })

    def messages_from_events(self, events: list[ExecutorEvent]) -> list[Message]:
        """Reconstruct messages from a list of events. Useful for replay
        and for feeding step N+1 the history from step N."""
        messages: list[Message] = []
        pending_tool_calls: list = []
        last_thought = ""
        for ev in events:
            if ev.kind == "thought":
                last_thought = ev.payload["text"]
            elif ev.kind == "tool_call":
                pending_tool_calls.append(ev.payload)
            elif ev.kind == "tool_result":
                # finalize the assistant message with tool calls, then the result
                # (simplified — production would batch these properly)
                pass
        if last_thought:
            messages.append(Message(role=Role.ASSISTANT, content=last_thought))
        return messages
