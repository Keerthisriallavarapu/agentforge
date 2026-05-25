"""Core types shared across agents, tools, runtime, and tracing.

Keeping these in one module so import cycles never happen. Models here are
intentionally minimal — runtime adds behavior, this is just data.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


class Role(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    TOOL = "tool"


class Message(BaseModel):
    """A single message in a conversation. Mirrors the Anthropic/OpenAI shape
    closely so we can hand it to either provider without translation."""

    role: Role
    content: str
    tool_calls: list[ToolCall] = Field(default_factory=list)
    tool_call_id: str | None = None
    name: str | None = None  # used for tool result messages


class ToolCall(BaseModel):
    id: str = Field(default_factory=lambda: _new_id("tc"))
    name: str
    arguments: dict[str, Any]


class ToolResult(BaseModel):
    tool_call_id: str
    output: str
    is_error: bool = False
    # Tools can attach structured metadata for the trace viewer
    metadata: dict[str, Any] = Field(default_factory=dict)


class AgentRole(str, Enum):
    PLANNER = "planner"
    EXECUTOR = "executor"
    CRITIC = "critic"


class RunStatus(str, Enum):
    PENDING = "pending"
    PLANNING = "planning"
    EXECUTING = "executing"
    CRITIQUING = "critiquing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class Plan(BaseModel):
    """A plan is a list of subtasks the executor will work through.
    We keep it flat for now; nested plans were considered but in practice
    the critic loop handles refinement better than recursion.
    See docs/DECISIONS.md D-003."""

    goal: str
    steps: list[PlanStep]
    reasoning: str = ""


class PlanStep(BaseModel):
    id: str = Field(default_factory=lambda: _new_id("step"))
    description: str
    expected_output: str
    suggested_tools: list[str] = Field(default_factory=list)
    status: Literal["pending", "in_progress", "done", "failed"] = "pending"


class Critique(BaseModel):
    """Critic output. Structured to prevent 'looks good to me' hallucinations."""

    approved: bool
    confidence: float = Field(ge=0.0, le=1.0)
    issues: list[str] = Field(default_factory=list)
    suggested_revisions: list[str] = Field(default_factory=list)
    reasoning: str


class RunState(BaseModel):
    """The complete state of an agent run. Persisted between transitions so
    runs can be resumed after crashes. See runtime/state_machine.py for the
    transition rules."""

    id: str = Field(default_factory=lambda: _new_id("run"))
    goal: str
    status: RunStatus = RunStatus.PENDING
    plan: Plan | None = None
    messages: list[Message] = Field(default_factory=list)
    tool_results: list[ToolResult] = Field(default_factory=list)
    critiques: list[Critique] = Field(default_factory=list)
    revision_count: int = 0
    max_revisions: int = 3
    created_at: datetime = Field(default_factory=_utcnow)
    updated_at: datetime = Field(default_factory=_utcnow)
    final_output: str | None = None
    error: str | None = None
    # Cost tracking — populated by the LLM client wrapper
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    estimated_cost_usd: float = 0.0


# Forward refs
Message.model_rebuild()
Plan.model_rebuild()
