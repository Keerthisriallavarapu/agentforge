# Engineering Decisions

This is an internal-RFC-style log of the non-obvious choices made in this project. The "why not X" sections matter more than the "what we did" sections.

## D-001: Custom tracing instead of OpenTelemetry

**Status:** Accepted

**Context.** The trace viewer needs to render agent decisions in real time. That means the trace data has to be streamable to a subscriber (the SSE endpoint) as events happen.

**Options.**

1. **OpenTelemetry with a custom exporter.** The standard answer. Battle-tested, has language SDKs, plays well with Jaeger/Tempo/Datadog.
2. **Roll our own span/event model with a pub/sub layer.** Less code, but reinvents the wheel.

**Decision.** Rolled our own (option 2).

OTel's default span lifecycle is "create, collect events, finish, batch export." That's the wrong shape for a live UI — we'd be fighting the BatchSpanProcessor's buffering, and the export-after-finish model means the UI only sees a span after the agent is done with it, which defeats the purpose.

We could use OTel's `SimpleSpanProcessor` to flush each span immediately, but that still doesn't give us per-event streaming inside a long-running span (which is exactly what executor loops are).

The custom approach is ~150 lines, has zero deps, and matches our use case. If we ever need to ship spans to a real observability backend, we'll add an OTel exporter as a subscriber alongside the UI subscriber.

**Tradeoff.** No off-the-shelf tooling integration. If your team uses Datadog APM, you'd need to write a bridge.

---

## D-002: Force structured JSON output for planner and critic, not for executor

**Status:** Accepted

**Context.** Planner and critic emit objects we parse downstream (Plan, Critique). Executor emits free-form text plus tool calls.

**Decision.** Strict JSON-only output (no prose, no fences ideally) for planner/critic via system-prompt contract. Free-form for executor.

**Why.** Parsing failures in the planner cascade — if we can't parse the plan, the run dies. Forcing JSON-only with an explicit schema in the prompt eliminated ~95% of parse errors in early testing. The remaining 5% are code-fence wrappers, which we strip defensively in `_parse_json`.

For the executor, structured output would actively hurt: the model is using tools, and free-form reasoning between tool calls is exactly what we want it to do.

**Why not Anthropic's tool-use for structured output (i.e. force a "return_plan" tool)?** Tried it. Works, but adds latency (every plan is now a tool call round-trip in the model's mind) and makes prompt engineering harder because the model treats the schema as an action rather than a format. Plain JSON-mode style worked better.

---

## D-003: Flat plans, not nested

**Status:** Accepted

**Context.** Some tasks naturally decompose into sub-tasks of sub-tasks. Should the planner emit a tree?

**Decision.** Flat list of steps.

**Why.** I built nested plans first. They look elegant. They fail because:
- The model has to decide when to recurse, and gets it wrong (over-decomposes simple tasks, under-decomposes complex ones).
- The critic has a much harder time evaluating nested plans — it ends up just rubber-stamping the structure.
- Replanning a sub-plan mid-run requires complex state management.

The critic-revision loop turned out to be a better mechanism for handling complexity than nested plans. If the executor's output on a complex step is rejected, the next iteration can break that step down implicitly in its execution rather than forcing a tree structure upfront.

---

## D-004: One LLM client wrapper, no LiteLLM/LangChain

**Status:** Accepted

**Context.** Need to support both Anthropic and OpenAI. The obvious answer is LiteLLM or LangChain's chat model abstraction.

**Decision.** Direct integration with both SDKs, ~250 LOC wrapper.

**Why.**
- We only have two providers. The abstraction overhead isn't worth it.
- Both providers' SDKs are well-maintained and we want their typed responses, not a lowest-common-denominator dict.
- Cost tracking is provider-specific (different fields, different units). Bolting it onto LiteLLM's response would be just as much code as our wrapper.
- LangChain in particular has a track record of breaking changes that I don't want in the dependency tree of an agent system.

**When to reconsider.** If we add a third provider, or if Bedrock/Vertex AI becomes a requirement, LiteLLM probably wins.

---

## D-005: Subprocess sandbox for code execution, not Docker-per-call

**Status:** Accepted with a known limitation

**Context.** The `python_repl` tool runs LLM-generated code. We need to prevent it from (a) reading secrets, (b) consuming unlimited resources, (c) calling out to the network in ways we don't want.

**Options.**

1. Run code in the same Python process via `exec()`. Cheap, fast, totally insecure.
2. Subprocess with RLIMITs and `-I` isolated mode. Cheap, fast, decent isolation on Linux.
3. Spawn a Docker container per call. Strong isolation, slow (200-500ms cold start).
4. Use a sandboxing service like e2b.dev.

**Decision.** Subprocess (option 2) for single-tenant / trusted-developer use cases. Use e2b.dev for multi-tenant.

**Why.** Container-per-call adds 200-500ms latency to every Python tool call, which compounds badly across agent loops. The subprocess + RLIMIT approach gives us memory caps, CPU caps, and no-fork enforcement on Linux, which is sufficient when the operator is also the user.

**Known limitation.** On macOS, RLIMIT_AS doesn't work the same way. On Windows, RLIMITs don't exist. For real multi-tenant deployment, the answer is e2b or Firecracker microVMs, not this.

**Reverted from.** Initially tried `RestrictedPython`. It blocks too much (no f-strings, no comprehensions in some cases) and the failure modes confuse the LLM into pathological retry loops.

---

## D-006: In-memory run store for now

**Status:** Accepted (with explicit follow-up)

**Context.** Runs need to be queryable after they complete (for the trace viewer, for retries, for analytics).

**Decision.** Python dict for now. Postgres for persistence is a follow-up.

**Why.** Adding Postgres in the MVP creates a setup burden that scares people off the repo. The in-memory store is fine for a single-instance deploy, which is 99% of how this gets used.

**Follow-up.** If/when someone wants to deploy AgentForge horizontally, they'll need persistent run state. The data model is already a Pydantic object, so the migration is mostly "swap dict for SQLAlchemy".

---

## D-007: Recursive summarization over sliding-window for memory compression

**Status:** Accepted

**Context.** Working memory eventually exceeds context windows. How do we compress?

**Options.**

1. **Sliding window.** Drop the oldest N messages.
2. **Recursive summarization.** When memory hits a threshold, summarize the oldest half into a single summary message, keep the recent half verbatim.
3. **Importance-weighted retention.** Score messages by some heuristic and keep the important ones.

**Decision.** Recursive summarization (option 2).

**Why.** Sliding window loses important early context (the goal, key constraints from the user). Importance-weighted retention requires a separate model call per message just to score it, which is expensive and noisy.

Recursive summarization keeps the user's original goal and the most recent context, with a coherent summary bridging the gap. On GAIA-L1, this beat sliding-window by ~8 points on long-running tasks. On short tasks (≤5 steps), it doesn't matter because we never hit the threshold.

---

## R-001: Reverted — Per-step structured outputs

I initially required the executor to emit a structured `{action, parameters}` JSON for every turn. The idea was symmetry with planner/critic.

**Why I reverted it.** The executor needs to reason in free text *between* tool calls. Forcing structure made reasoning shorter and more shallow — measurable accuracy drop on GAIA-L1 (about 5 points). Removed after a week.

## R-002: Reverted — Async streaming inside the agent loop

I had the executor stream tokens to the trace viewer in real time. Pretty.

**Why I reverted it.** The trace viewer code became unmanageable because spans were now async generators all the way down. Worse, when a tool call interrupted the stream, the UI had to handle partial messages. The current design — non-streaming LLM calls inside the loop, with span events as the streaming unit — is much cleaner.
