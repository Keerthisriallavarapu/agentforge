"use client";

import { useState, useRef, useEffect } from "react";
import { Play, Loader2, CheckCircle, AlertCircle, Wrench, Brain, Eye } from "lucide-react";

type TraceEvent = {
  kind: string;
  span_id: string;
  timestamp: number;
  data: Record<string, any>;
};

const API_BASE = process.env.NEXT_PUBLIC_API_BASE ?? "http://localhost:8000";

export default function Home() {
  const [goal, setGoal] = useState("");
  const [runId, setRunId] = useState<string | null>(null);
  const [events, setEvents] = useState<TraceEvent[]>([]);
  const [status, setStatus] = useState<string>("idle");
  const [output, setOutput] = useState<string | null>(null);
  const esRef = useRef<EventSource | null>(null);

  useEffect(() => {
    return () => esRef.current?.close();
  }, []);

  async function startRun() {
    if (!goal.trim()) return;
    setEvents([]);
    setOutput(null);
    setStatus("starting");

    const r = await fetch(`${API_BASE}/runs`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ goal }),
    });
    const data = await r.json();
    setRunId(data.id);
    setStatus("running");

    // Open SSE
    esRef.current?.close();
    const es = new EventSource(`${API_BASE}/runs/${data.id}/stream`);
    esRef.current = es;

    // Listen to all named events
    const handler = (e: MessageEvent) => {
      try {
        const parsed = JSON.parse(e.data);
        setEvents((prev) => [...prev, { kind: e.type, ...parsed }]);
      } catch {}
    };

    [
      "span_start", "span_end", "thought", "tool_call", "tool_result",
      "plan_generated", "critique_done", "status_change", "error",
    ].forEach((evt) => es.addEventListener(evt, handler));

    // Poll for final state
    const poll = setInterval(async () => {
      const r2 = await fetch(`${API_BASE}/runs/${data.id}`);
      const s = await r2.json();
      if (s.status === "completed" || s.status === "failed") {
        setStatus(s.status);
        setOutput(s.final_output);
        clearInterval(poll);
        es.close();
      }
    }, 1500);
  }

  return (
    <main className="min-h-screen bg-slate-50 text-slate-900">
      <div className="max-w-5xl mx-auto p-8">
        <header className="mb-8">
          <h1 className="text-3xl font-semibold tracking-tight">AgentForge</h1>
          <p className="text-slate-600 mt-1">Multi-agent trace viewer</p>
        </header>

        <div className="bg-white rounded-lg shadow-sm border border-slate-200 p-6 mb-6">
          <label className="block text-sm font-medium text-slate-700 mb-2">
            Goal
          </label>
          <textarea
            value={goal}
            onChange={(e) => setGoal(e.target.value)}
            className="w-full border border-slate-300 rounded px-3 py-2 text-sm font-mono focus:ring-2 focus:ring-slate-400 focus:outline-none"
            rows={3}
            placeholder='e.g. "Calculate the CAGR for $10k -> $34.5k over 7 years"'
          />
          <button
            onClick={startRun}
            disabled={status === "running" || status === "starting" || !goal.trim()}
            className="mt-3 inline-flex items-center gap-2 bg-slate-900 text-white px-4 py-2 rounded text-sm font-medium hover:bg-slate-800 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {status === "running" || status === "starting" ? (
              <Loader2 className="w-4 h-4 animate-spin" />
            ) : (
              <Play className="w-4 h-4" />
            )}
            {status === "running" ? "Running..." : "Run"}
          </button>
          {runId && (
            <span className="ml-3 text-xs text-slate-500 font-mono">
              run: {runId}
            </span>
          )}
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <section className="bg-white rounded-lg shadow-sm border border-slate-200 p-6">
            <h2 className="font-semibold text-slate-800 mb-4 flex items-center gap-2">
              <Eye className="w-4 h-4" /> Trace ({events.length})
            </h2>
            <div className="space-y-2 max-h-[60vh] overflow-y-auto pr-2">
              {events.length === 0 && (
                <p className="text-slate-400 text-sm">No events yet.</p>
              )}
              {events.map((e, i) => (
                <EventRow key={i} event={e} />
              ))}
            </div>
          </section>

          <section className="bg-white rounded-lg shadow-sm border border-slate-200 p-6">
            <h2 className="font-semibold text-slate-800 mb-4 flex items-center gap-2">
              {status === "completed" && <CheckCircle className="w-4 h-4 text-emerald-600" />}
              {status === "failed" && <AlertCircle className="w-4 h-4 text-red-600" />}
              Output
            </h2>
            {output ? (
              <pre className="whitespace-pre-wrap font-mono text-sm text-slate-800 leading-relaxed">
                {output}
              </pre>
            ) : (
              <p className="text-slate-400 text-sm">
                {status === "running" ? "Agents are working..." : "Output will appear here."}
              </p>
            )}
          </section>
        </div>
      </div>
    </main>
  );
}

function EventRow({ event }: { event: TraceEvent }) {
  const { kind, data } = event;

  const icon = (() => {
    if (kind === "thought") return <Brain className="w-3.5 h-3.5 text-violet-500" />;
    if (kind === "tool_call") return <Wrench className="w-3.5 h-3.5 text-amber-500" />;
    if (kind === "tool_result") return <CheckCircle className="w-3.5 h-3.5 text-emerald-500" />;
    return <span className="w-3.5 h-3.5 inline-block rounded-full bg-slate-300" />;
  })();

  const summary = (() => {
    if (kind === "thought") return truncate(data.data?.text ?? "", 200);
    if (kind === "tool_call") return `${data.data?.name}(${JSON.stringify(data.data?.arguments).slice(0, 80)}...)`;
    if (kind === "tool_result") return truncate(data.data?.output ?? "", 200);
    if (kind === "span_start") return `→ ${data.data?.name}`;
    if (kind === "span_end") return `← (${(data.data?.duration_ms ?? 0).toFixed(0)}ms)`;
    if (kind === "plan_generated") return `Plan: ${data.data?.steps} steps`;
    if (kind === "critique_done")
      return `Critique: ${data.data?.approved ? "approved" : "rejected"} (conf ${(data.data?.confidence ?? 0).toFixed(2)})`;
    return JSON.stringify(data.data ?? {}).slice(0, 200);
  })();

  return (
    <div className="flex gap-2 items-start text-sm border-l-2 border-slate-200 pl-3 py-1">
      <div className="mt-1">{icon}</div>
      <div className="flex-1 min-w-0">
        <div className="text-xs uppercase tracking-wide text-slate-500">{kind}</div>
        <div className="text-slate-700 font-mono break-words">{summary}</div>
      </div>
    </div>
  );
}

function truncate(s: string, n: number) {
  return s.length > n ? s.slice(0, n) + "…" : s;
}
