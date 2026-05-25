"""Tests for the critic agent."""
from __future__ import annotations

import json

from agentforge.agents.critic import Critic


async def test_critic_approves(fake_llm):
    response = {
        "approved": True,
        "confidence": 0.9,
        "issues": [],
        "suggested_revisions": [],
        "reasoning": "Output addresses the goal completely.",
    }
    fake_llm.enqueue(text=json.dumps(response))

    critic = Critic(fake_llm)
    crit, _ = await critic.critique(goal="x", executor_output="y")
    assert crit.approved is True
    assert crit.confidence == 0.9


async def test_critic_rejects_with_issues(fake_llm):
    response = {
        "approved": False,
        "confidence": 0.6,
        "issues": ["missing citation", "claim is unsupported"],
        "suggested_revisions": ["add a citation for X", "verify Y"],
        "reasoning": "Two specific issues block approval.",
    }
    fake_llm.enqueue(text=json.dumps(response))

    critic = Critic(fake_llm)
    crit, _ = await critic.critique(goal="x", executor_output="y")
    assert crit.approved is False
    assert len(crit.issues) == 2
    assert "citation" in crit.issues[0]
