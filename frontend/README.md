# AgentForge Trace Viewer

Minimal Next.js UI for watching agent runs in real time.

## Run

```bash
pnpm install   # or: npm install
pnpm dev
```

Then open http://localhost:3000. You need the backend running on http://localhost:8000 — see the root README.

Set `NEXT_PUBLIC_API_BASE` if your backend is somewhere else.

## What it does

- `POST /runs` to start an agent run
- Opens an SSE connection to `/runs/{id}/stream`
- Renders each trace event (plan generated, thoughts, tool calls, tool results, critique) as it arrives
- Polls `GET /runs/{id}` until the run completes, then shows the final output

The UI is intentionally plain. The signal isn't the UI — it's seeing agent reasoning in real time.
