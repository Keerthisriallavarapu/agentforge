"""Planner: decomposes a goal into a flat list of subtasks.

Design notes:
- We force structured JSON output. Free-form plans look impressive but break
  downstream parsing constantly. The structured contract is non-negotiable.
- The planner sees the available tool list so it can suggest tools per step,
  but it does NOT call tools. Strict separation of "what to do" vs "do it".
"""
from __future__ import annotations

import json
import logging

from ..llm import LLMClient, LLMResponse
from ..settings import get_settings
from ..tools.registry import ToolRegistry
from ..types import Message, Plan, PlanStep, Role

log = logging.getLogger(__name__)


SYSTEM_PROMPT = """You are the Planner in a multi-agent system. Your job is to decompose the user's goal into a clear, ordered list of executable subtasks.

Rules:
- 3-8 steps is the right size for most goals. Don't pad; don't underspecify.
- Each step must have a concrete expected output (a deliverable), not a vague verb.
- Suggest tools from the provided list when a step needs them; leave empty if pure reasoning.
- Do not perform the work yourself. Just plan.

You MUST respond with a JSON object matching this schema:
{
  "reasoning": "Brief explanation of your decomposition strategy.",
  "steps": [
    {
      "description": "What to do in this step.",
      "expected_output": "What this step produces.",
      "suggested_tools": ["tool_name", ...]
    }
  ]
}
Do not include any prose outside the JSON."""


class Planner:
    def __init__(self, llm: LLMClient, tools: ToolRegistry):
        self._llm = llm
        self._tools = tools
        self._model = get_settings().planner_model

    async def plan(self, goal: str, context: str = "") -> tuple[Plan, LLMResponse]:
        tool_descriptions = self._tools.describe_for_prompt()
        user_content = f"Goal: {goal}\n\nAvailable tools:\n{tool_descriptions}"
        if context:
            user_content += f"\n\nRelevant context from memory:\n{context}"

        messages = [Message(role=Role.USER, content=user_content)]

        resp = await self._llm.complete(
            messages=messages,
            model=self._model,
            system=SYSTEM_PROMPT,
            temperature=0.3,  # planning benefits from determinism
            max_tokens=2048,
        )

        plan_dict = _parse_json(resp.text)
        steps = [
            PlanStep(
                description=s["description"],
                expected_output=s["expected_output"],
                suggested_tools=s.get("suggested_tools", []),
            )
            for s in plan_dict["steps"]
        ]
        plan = Plan(goal=goal, steps=steps, reasoning=plan_dict.get("reasoning", ""))
        log.info("Planner generated %d steps for goal: %s", len(steps), goal[:80])
        return plan, resp


def _parse_json(text: str) -> dict:
    """LLMs sometimes wrap JSON in ```json fences. Strip them."""
    text = text.strip()
    if text.startswith("```"):
        # remove opening fence
        first_newline = text.find("\n")
        if first_newline != -1:
            text = text[first_newline + 1 :]
        # remove closing fence
        if text.endswith("```"):
            text = text[:-3]
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        log.error("Failed to parse plan JSON: %s\nText was: %s", e, text[:500])
        raise
