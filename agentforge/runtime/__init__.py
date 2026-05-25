"""The agent runtime. State machine that drives a run from PENDING -> COMPLETED.

We deliberately keep this synchronous-looking (sequential awaits) rather than
emitting events from inside the orchestrator. The trace viewer subscribes to
the tracing module; the runtime itself stays linear and debuggable.
"""
from __future__ import annotations

import logging
from typing import AsyncIterator

from ..agents import Critic, Executor, Planner
from ..llm import LLMClient
from ..tools.registry import ToolRegistry, get_default_registry
from ..tracing import Span, get_tracer
from ..types import (
    Message,
    Plan,
    PlanStep,
    Role,
    RunState,
    RunStatus,
    ToolResult,
)

log = logging.getLogger(__name__)


class RunResult:
    def __init__(self, state: RunState):
        self.state = state

    @property
    def final_output(self) -> str | None:
        return self.state.final_output

    @property
    def cost_usd(self) -> float:
        return self.state.estimated_cost_usd

    @property
    def status(self) -> RunStatus:
        return self.state.status


class Runtime:
    """Coordinates planner -> executor (loop) -> critic -> (revise or finish).

    A single Runtime instance is reusable across runs. State is kept inside
    the RunState object passed through the pipeline."""

    def __init__(
        self,
        llm: LLMClient | None = None,
        tools: ToolRegistry | None = None,
        max_revisions: int = 3,
    ):
        self._llm = llm or LLMClient()
        self._tools = tools or get_default_registry()
        self._planner = Planner(self._llm, self._tools)
        self._executor = Executor(self._llm, self._tools)
        self._critic = Critic(self._llm)
        self._max_revisions = max_revisions
        self._tracer = get_tracer()

    async def run(self, goal: str, state: RunState | None = None) -> RunResult:
        state = state or RunState(goal=goal, max_revisions=self._max_revisions)
        log.info("Starting run %s for goal: %s", state.id, goal[:120])

        async with self._tracer.span("run", run_id=state.id, goal=goal) as run_span:
            try:
                # --- Plan ---
                state.status = RunStatus.PLANNING
                run_span.event("status_change", status=state.status.value)
                async with self._tracer.span("plan") as plan_span:
                    plan, plan_resp = await self._planner.plan(goal)
                    state.plan = plan
                    self._account(state, plan_resp.input_tokens, plan_resp.output_tokens, plan_resp.cost_usd)
                    plan_span.event("plan_generated", steps=len(plan.steps))

                # --- Execute (with critic loop) ---
                state.status = RunStatus.EXECUTING
                run_span.event("status_change", status=state.status.value)

                final_output = ""
                while state.revision_count <= state.max_revisions:
                    final_output = await self._execute_plan(plan, state)

                    state.status = RunStatus.CRITIQUING
                    async with self._tracer.span("critique") as crit_span:
                        tool_trace = self._format_tool_trace(state.tool_results)
                        critique, crit_resp = await self._critic.critique(
                            goal=goal,
                            executor_output=final_output,
                            tool_trace=tool_trace,
                        )
                        state.critiques.append(critique)
                        self._account(state, crit_resp.input_tokens, crit_resp.output_tokens, crit_resp.cost_usd)
                        crit_span.event(
                            "critique_done",
                            approved=critique.approved,
                            confidence=critique.confidence,
                        )

                    if critique.approved:
                        break

                    state.revision_count += 1
                    if state.revision_count > state.max_revisions:
                        log.warning("Run %s: max revisions exhausted.", state.id)
                        break

                    # Feed the critique back as guidance for the next iteration
                    revision_msg = Message(
                        role=Role.USER,
                        content=(
                            "The output was not approved. Issues identified:\n"
                            + "\n".join(f"- {i}" for i in critique.issues)
                            + "\n\nSuggested revisions:\n"
                            + "\n".join(f"- {r}" for r in critique.suggested_revisions)
                            + "\n\nPlease address these and produce a revised output."
                        ),
                    )
                    state.messages.append(revision_msg)

                state.final_output = final_output
                state.status = RunStatus.COMPLETED
                run_span.event("status_change", status=state.status.value)
                return RunResult(state)

            except Exception as e:
                log.exception("Run %s failed", state.id)
                state.status = RunStatus.FAILED
                state.error = str(e)
                run_span.event("error", message=str(e))
                return RunResult(state)

    async def _execute_plan(self, plan: Plan, state: RunState) -> str:
        """Run through plan steps sequentially, accumulating context."""
        accumulated_msgs: list[Message] = list(state.messages)
        final_step_output = ""

        for step in plan.steps:
            step.status = "in_progress"
            async with self._tracer.span("step", step_id=step.id, description=step.description) as step_span:
                events = []
                async for ev in self._executor.execute_step(step, accumulated_msgs):
                    events.append(ev)
                    step_span.event(ev.kind, **ev.payload)

                # Find the final "done" event
                done_events = [e for e in events if e.kind == "done"]
                if done_events:
                    final_step_output = done_events[-1].payload.get("final_message", "")
                    cost = done_events[-1].payload.get("cost_usd", 0)
                    state.estimated_cost_usd += cost

                # Append a single summary message representing what this step did
                accumulated_msgs.append(Message(
                    role=Role.ASSISTANT,
                    content=f"[Step {step.id}] {final_step_output}",
                ))
                # Record tool results for the critic
                for ev in events:
                    if ev.kind == "tool_result":
                        state.tool_results.append(ToolResult(
                            tool_call_id=ev.payload.get("id", ""),
                            output=ev.payload.get("output", ""),
                            is_error=ev.payload.get("is_error", False),
                        ))

                step.status = "done"

        state.messages = accumulated_msgs
        return final_step_output

    @staticmethod
    def _format_tool_trace(results: list[ToolResult]) -> str:
        if not results:
            return ""
        lines = []
        for r in results[:20]:  # cap to avoid blowing critic context
            status = "ERROR" if r.is_error else "OK"
            lines.append(f"[{status}] {r.output[:300]}")
        return "\n".join(lines)

    @staticmethod
    def _account(state: RunState, in_tok: int, out_tok: int, cost: float) -> None:
        state.total_input_tokens += in_tok
        state.total_output_tokens += out_tok
        state.estimated_cost_usd += cost
