"""Lightweight tracing. Not OpenTelemetry — we deliberately rolled our own
because we want spans to be live-streamable to the trace viewer UI, and the
OTel SDK's batch span processor model doesn't fit that.

Each Span buffers events. Subscribers get them via an asyncio.Queue.
"""
from __future__ import annotations

import asyncio
import contextlib
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, AsyncIterator

log = logging.getLogger(__name__)


@dataclass
class SpanEvent:
    span_id: str
    kind: str
    timestamp: float
    data: dict[str, Any] = field(default_factory=dict)


@dataclass
class Span:
    id: str
    name: str
    parent_id: str | None
    started_at: float
    ended_at: float | None = None
    attributes: dict[str, Any] = field(default_factory=dict)
    events: list[SpanEvent] = field(default_factory=list)

    def event(self, kind: str, **data: Any) -> None:
        ev = SpanEvent(span_id=self.id, kind=kind, timestamp=time.time(), data=data)
        self.events.append(ev)
        Tracer.instance()._publish(ev)


class Tracer:
    """Process-wide singleton. Subscribers are asyncio queues consuming events."""

    _instance: "Tracer | None" = None

    def __init__(self) -> None:
        self._stack: list[Span] = []
        self._subscribers: list[asyncio.Queue[SpanEvent]] = []
        self._all_spans: list[Span] = []  # for debugging; cap in production

    @classmethod
    def instance(cls) -> "Tracer":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @contextlib.asynccontextmanager
    async def span(self, name: str, **attributes: Any) -> AsyncIterator[Span]:
        parent_id = self._stack[-1].id if self._stack else None
        span = Span(
            id=f"span_{uuid.uuid4().hex[:10]}",
            name=name,
            parent_id=parent_id,
            started_at=time.time(),
            attributes=attributes,
        )
        self._stack.append(span)
        self._all_spans.append(span)
        span.event("span_start", name=name, parent_id=parent_id, **attributes)
        try:
            yield span
        finally:
            span.ended_at = time.time()
            span.event("span_end", duration_ms=(span.ended_at - span.started_at) * 1000)
            self._stack.pop()

    def subscribe(self) -> asyncio.Queue[SpanEvent]:
        q: asyncio.Queue[SpanEvent] = asyncio.Queue(maxsize=1000)
        self._subscribers.append(q)
        return q

    def unsubscribe(self, q: asyncio.Queue[SpanEvent]) -> None:
        if q in self._subscribers:
            self._subscribers.remove(q)

    def _publish(self, event: SpanEvent) -> None:
        for q in self._subscribers:
            try:
                q.put_nowait(event)
            except asyncio.QueueFull:
                log.warning("Tracing subscriber queue full; dropping event.")


def get_tracer() -> Tracer:
    return Tracer.instance()
