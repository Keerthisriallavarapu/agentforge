# Architecture

## The 60-second mental model

```
User goal
   │
   ▼
┌─────────┐     ┌────────────┐     ┌────────┐
│ Planner │ ──▶ │  Executor  │ ──▶ │ Critic │
└─────────┘     │   (loop)   │     └───┬────┘
                └──────┬─────┘         │
                       │               │
                       ▼          approved?
                   Tool calls    ┌─────┴─────┐
                       │         │           │
                       │       yes           no
                       │         │           │
                       ▼         ▼           ▼
                  Tool results  Done   Revise & rerun
```

The Runtime owns this state machine. Each agent is a thin class that wraps an LLM call with a specific prompt and output contract.

## Components

### `agents/`

- **Planner**: one-shot LLM call, JSON-only output, returns a `Plan` with ordered `PlanStep`s.
- **Executor**: looped LLM calls. Each iteration: model returns either (text + tool calls) or (text only — done). On tool calls, we invoke the tool and feed the result back.
- **Critic**: one-shot LLM call, JSON-only output, returns a `Critique`. If `approved=False`, runtime triggers a revision.

### `tools/`

- **`base.py`**: `ToolSpec` (provider-agnostic) and the `Tool` protocol.
- **`registry.py`**: per-process tool index, handles provider conversion (Anthropic vs OpenAI schemas) and timeout-wrapped invocation.
- **`builtins.py`**: `python_repl`, `web_search`, `read_file`, `write_file`. These are intentionally small — the registry is designed for users to add their own.

### `runtime/`

The orchestrator. Runs `Plan → Execute → Critique → (Revise | Done)`. Holds onto the `RunState` and accumulates cost/token data as it goes.

### `tracing/`

A custom span/event model with an in-process pub/sub. Spans are nested (parent/child via stack), and emit events as they go. The SSE endpoint subscribes to all events and filters them client-side.

This module is *not* OpenTelemetry. See [DECISIONS.md D-001](DECISIONS.md) for why.

### `memory/`

Qdrant-backed long-term memory + recursive summarization for working memory compression. The embedder is a sentence-transformers model loaded at startup.

### `server/`

FastAPI app exposing:

- `POST /runs` — start a run (returns immediately, executes in background)
- `GET /runs/{id}` — poll run state
- `GET /runs/{id}/stream` — SSE stream of trace events for the run

### `llm.py`

The provider abstraction. Handles Anthropic and OpenAI message-format conversion, retries (tenacity), and cost tracking via a per-model price table.

## Request flow: `POST /runs`

1. Client posts `{goal: "..."}`
2. Server creates a `RunState`, stores it, schedules a background task
3. Background task calls `Runtime.run(goal, state)`
4. Runtime enters `PLANNING` → calls `Planner.plan()` → updates state
5. Runtime enters `EXECUTING` → for each step in plan, calls `Executor.execute_step()`, accumulating messages
6. Runtime enters `CRITIQUING` → calls `Critic.critique()` on the final output
7. If critic rejects and `revision_count < max_revisions`, loop back to step 5 with critique feedback prepended
8. Runtime sets `state.final_output`, status to `COMPLETED`, returns

Throughout, the `tracing` module is recording spans and emitting events to any subscribers. The SSE endpoint streams these to the UI live.

## State persistence

In-memory only as of v0.1. Runs are lost on process restart. See [DECISIONS.md D-006](DECISIONS.md).

## Concurrency

Multiple concurrent runs work; they share the LLM client (which is thread-safe via httpx) and the tool registry (read-only after init). The tracing pub/sub is process-global, which means all subscribers see all events — fine for low concurrency, but for high QPS you'd want per-run channels.
