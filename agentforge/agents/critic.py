"""Critic: validates whether the executor's output actually addresses the goal.

The critic exists because executors hallucinate completion. The structured
output here is the key — without it, the critic becomes a yes-man.
"""
from __future__ import annotations

import json
import logging

from ..llm import LLMClient, LLMResponse
from ..settings import get_settings
from ..types import Critique, Message, Role

log = logging.getLogger(__name__)


SYSTEM_PROMPT = """You are the Critic in a multi-agent system. Your job is to rigorously evaluate whether the Executor's output actually satisfies the original goal.

Be skeptical. Common failure modes to check for:
- Hallucinated facts presented as data
- Output that "sounds right" but doesn't answer the actual question
- Missed parts of multi-part goals
- Errors silently swallowed in tool results

You MUST respond with a JSON object matching this schema:
{
  "approved": true | false,
  "confidence": 0.0 to 1.0,
  "issues": ["specific issue 1", "specific issue 2"],
  "suggested_revisions": ["concrete fix 1", "concrete fix 2"],
  "reasoning": "Your evaluation in 2-4 sentences."
}

If approved is true, issues and suggested_revisions should be empty.
Do not include any prose outside the JSON."""


class Critic:
    def __init__(self, llm: LLMClient):
        self._llm = llm
        self._model = get_settings().critic_model

    async def critique(
        self,
        goal: str,
        executor_output: str,
        tool_trace: str = "",
    ) -> tuple[Critique, LLMResponse]:
        user_content = (
            f"Original goal:\n{goal}\n\n"
            f"Executor's final output:\n{executor_output}"
        )
        if tool_trace:
            user_content += f"\n\nTool calls and results:\n{tool_trace}"

        resp = await self._llm.complete(
            messages=[Message(role=Role.USER, content=user_content)],
            model=self._model,
            system=SYSTEM_PROMPT,
            temperature=0.2,  # critic should be consistent
            max_tokens=1024,
        )

        data = _parse_json(resp.text)
        critique = Critique(
            approved=data["approved"],
            confidence=float(data.get("confidence", 0.5)),
            issues=data.get("issues", []),
            suggested_revisions=data.get("suggested_revisions", []),
            reasoning=data.get("reasoning", ""),
        )
        log.info(
            "Critic: approved=%s confidence=%.2f issues=%d",
            critique.approved, critique.confidence, len(critique.issues),
        )
        return critique, resp


def _parse_json(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        first_newline = text.find("\n")
        if first_newline != -1:
            text = text[first_newline + 1 :]
        if text.endswith("```"):
            text = text[:-3]
    return json.loads(text)
