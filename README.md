# AgentForge

Multi-agent LLM orchestration with planner/executor/critic, persistent memory, and a real trace viewer.

I built this because every "agent framework" repo I tried either (a) hid the prompts so you couldn't debug them, (b) didn't have a way to actually see what the agent was doing, or (c) papered over the fact that LLMs hallucinate "I'm done" all the time. AgentForge addresses all three.

## What's actually in here

- **Three agents**: a Planner (decomposes goals into steps), an Executor (works through steps, calling tools), and a Critic (rejects bad outputs with structured reasons).
- **A tool layer** with timeouts, sandboxing for code execution, and provider-agnostic specs that work with both Anthropic and OpenAI.
- **Vector memory** via Qdrant, with recursive summarization when working memory gets too long.
- **Live tracing** — every span and event publishes to subscribers, so the frontend can stream a real-time view of what the agents are doing.
- **A FastAPI server** with SSE streaming, and a CLI for one-off runs.

The architecture is documented in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md). Engineering decisions and tradeoffs are in [docs/DECISIONS.md](docs/DECISIONS.md). If you only read one secondary doc, read DECISIONS.

## Quick start

```bash
# Local: just the agent + Anthropic
pip install -e .
export ANTHROPIC_API_KEY=sk-ant-...
agentforge run "Calculate CAGR for a portfolio that grew from \$10k to \$34.5k over 7 years"

# Full stack with Qdrant and the HTTP API
docker compose up -d
curl -X POST http://localhost:8000/runs \
  -H "Content-Type: application/json" \
  -d '{"goal": "Find the 3 most-cited 2024 papers on RAG and summarize them"}'
```

## Why planner / executor / critic specifically?

Other configurations I tried first and rejected:

- **Single agent with tools.** Works for simple tasks. Fails on multi-step goals because the model loses the plan thread mid-execution. Especially bad with cheaper/smaller models.
- **Planner + executor (no critic).** Better. But the executor almost always returns a confident-sounding completion even when it didn't actually finish. Adding the critic with structured-output validation caught about 1 in 3 false completions on the GAIA-L1 subset I tested.
- **DAG of specialized agents** (researcher, writer, fact-checker, etc.). Cool in theory; in practice the coordination overhead burns tokens and the specialization doesn't help much when the underlying model is general-purpose.

The planner/executor/critic shape is the minimum viable team. Each role exists because removing it caused a specific failure mode I could measure.

## Project layout

```
agentforge/
├── agentforge/
│   ├── agents/           # planner, executor, critic
│   ├── tools/            # registry + builtins (python_repl, web_search, files)
│   ├── memory/           # qdrant + summarization
│   ├── runtime/          # the orchestrator
│   ├── tracing/          # custom span/event model with pub/sub
│   ├── server/           # fastapi + SSE
│   ├── llm.py            # anthropic/openai wrapper with cost tracking
│   ├── types.py          # pydantic models that flow through everything
│   ├── settings.py
│   └── cli.py
├── tests/
├── examples/
├── docs/
├── frontend/             # next.js trace viewer (see frontend/README.md)
├── Dockerfile
└── docker-compose.yml
```

## Tests

```bash
pip install -e ".[dev]"
pytest
```

Unit tests use a `FakeLLMClient` so they don't hit real APIs. Integration tests in `tests/integration/` do — gate them behind your API keys.

## What I'd build next

A few things on the roadmap I haven't gotten to:

- **Parallel sub-agents**: when the planner produces independent steps, dispatch them concurrently
- **Self-tuning prompts**: capture failure cases and use them as few-shot examples for the next run
- **Browser tool**: Playwright-backed for tasks that need actual web interaction
- **Persistent run storage**: currently in-memory, fine for single-instance

## License

MIT. Do whatever you want with it.
