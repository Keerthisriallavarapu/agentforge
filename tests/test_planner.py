"""Tests for the planner agent."""
from __future__ import annotations

import json

import pytest

from agentforge.agents.planner import Planner
from agentforge.tools.registry import ToolRegistry


async def test_planner_parses_valid_json(fake_llm):
    plan_data = {
        "reasoning": "Decompose into research and write.",
        "steps": [
            {
                "description": "Research the topic.",
                "expected_output": "A summary of key points.",
                "suggested_tools": ["web_search"],
            },
            {
                "description": "Write the article.",
                "expected_output": "Final article text.",
                "suggested_tools": [],
            },
        ],
    }
    fake_llm.enqueue(text=json.dumps(plan_data))

    planner = Planner(fake_llm, ToolRegistry())
    plan, _ = await planner.plan("Write an article about RAG.")

    assert plan.goal == "Write an article about RAG."
    assert len(plan.steps) == 2
    assert plan.steps[0].suggested_tools == ["web_search"]


async def test_planner_strips_code_fences(fake_llm):
    plan_data = {"reasoning": "x", "steps": [
        {"description": "do thing", "expected_output": "thing done"}
    ]}
    fenced = f"```json\n{json.dumps(plan_data)}\n```"
    fake_llm.enqueue(text=fenced)

    planner = Planner(fake_llm, ToolRegistry())
    plan, _ = await planner.plan("anything")
    assert len(plan.steps) == 1


async def test_planner_raises_on_invalid_json(fake_llm):
    fake_llm.enqueue(text="this is not json")
    planner = Planner(fake_llm, ToolRegistry())
    with pytest.raises(json.JSONDecodeError):
        await planner.plan("anything")
