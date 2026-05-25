"""FastAPI application with REST endpoints and SSE stream for trace events."""
from __future__ import annotations

import asyncio
import json
import logging
from typing import AsyncIterator

from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from ..runtime import Runtime
from ..settings import get_settings
from ..tracing import get_tracer
from ..types import RunState

log = logging.getLogger(__name__)


class CreateRunRequest(BaseModel):
    goal: str
    max_revisions: int | None = None


class RunResponse(BaseModel):
    id: str
    status: str
    final_output: str | None
    cost_usd: float
    revision_count: int


# In-memory run store. Production would use Postgres + Redis for state.
_runs: dict[str, RunState] = {}
_runtime: Runtime | None = None


def get_runtime() -> Runtime:
    global _runtime
    if _runtime is None:
        _runtime = Runtime()
    return _runtime


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(title="AgentForge", version="0.1.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/health")
    async def health():
        return {"ok": True}

    @app.post("/runs", response_model=RunResponse)
    async def create_run(req: CreateRunRequest, bg: BackgroundTasks):
        state = RunState(goal=req.goal)
        if req.max_revisions is not None:
            state.max_revisions = req.max_revisions
        _runs[state.id] = state

        async def execute():
            result = await get_runtime().run(req.goal, state=state)
            _runs[state.id] = result.state

        bg.add_task(execute)
        return RunResponse(
            id=state.id,
            status=state.status.value,
            final_output=None,
            cost_usd=0.0,
            revision_count=0,
        )

    @app.get("/runs/{run_id}", response_model=RunResponse)
    async def get_run(run_id: str):
        state = _runs.get(run_id)
        if not state:
            raise HTTPException(404, f"Run {run_id} not found")
        return RunResponse(
            id=state.id,
            status=state.status.value,
            final_output=state.final_output,
            cost_usd=state.estimated_cost_usd,
            revision_count=state.revision_count,
        )

    @app.get("/runs/{run_id}/stream")
    async def stream_run(run_id: str):
        """SSE endpoint streaming trace events for a specific run.
        Note: we subscribe to *all* events and filter client-side by run_id
        in the trace context. For high-concurrency use, partition by run_id."""
        state = _runs.get(run_id)
        if not state:
            raise HTTPException(404, f"Run {run_id} not found")

        async def event_gen() -> AsyncIterator[dict]:
            tracer = get_tracer()
            q = tracer.subscribe()
            try:
                # Send an initial state snapshot
                yield {
                    "event": "snapshot",
                    "data": json.dumps({"status": state.status.value, "run_id": run_id}),
                }
                while True:
                    try:
                        ev = await asyncio.wait_for(q.get(), timeout=30)
                    except asyncio.TimeoutError:
                        # heartbeat to keep connection alive
                        yield {"event": "ping", "data": ""}
                        continue
                    yield {
                        "event": ev.kind,
                        "data": json.dumps({
                            "span_id": ev.span_id,
                            "timestamp": ev.timestamp,
                            "data": ev.data,
                        }),
                    }
            finally:
                tracer.unsubscribe(q)

        return EventSourceResponse(event_gen())

    return app


app = create_app()
