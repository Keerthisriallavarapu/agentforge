"""Long-term memory via Qdrant vector store with recursive summarization.

The summarizer is intentionally simple — when working memory exceeds the
threshold, we summarize the oldest half into a single message. We tried
sliding-window and importance-weighted summarization; the simple recursive
approach worked best on GAIA. See docs/DECISIONS.md D-007.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from sentence_transformers import SentenceTransformer

from .settings import get_settings
from .types import Message

log = logging.getLogger(__name__)


@dataclass
class MemoryEntry:
    id: str
    content: str
    metadata: dict[str, Any]
    score: float = 0.0


class Memory:
    """Persistent memory across runs. Embeds messages and stores in Qdrant."""

    def __init__(self, embedder: SentenceTransformer | None = None):
        s = get_settings()
        self._client = AsyncQdrantClient(url=s.qdrant_url, api_key=s.qdrant_api_key)
        self._collection = s.memory_collection
        self._embedder = embedder or SentenceTransformer(s.embedding_model)
        self._dim = self._embedder.get_sentence_embedding_dimension()
        self._initialized = False

    async def _ensure_collection(self) -> None:
        if self._initialized:
            return
        existing = await self._client.get_collections()
        names = {c.name for c in existing.collections}
        if self._collection not in names:
            await self._client.create_collection(
                collection_name=self._collection,
                vectors_config=VectorParams(size=self._dim, distance=Distance.COSINE),
            )
            log.info("Created memory collection %s (dim=%d)", self._collection, self._dim)
        self._initialized = True

    async def remember(self, content: str, metadata: dict[str, Any]) -> str:
        await self._ensure_collection()
        vector = self._embedder.encode(content).tolist()
        entry_id = metadata.get("id") or f"mem_{abs(hash(content)) % 10**12}"
        await self._client.upsert(
            collection_name=self._collection,
            points=[PointStruct(
                id=entry_id,
                vector=vector,
                payload={"content": content, **metadata},
            )],
        )
        return entry_id

    async def recall(self, query: str, k: int = 5) -> list[MemoryEntry]:
        await self._ensure_collection()
        vector = self._embedder.encode(query).tolist()
        results = await self._client.search(
            collection_name=self._collection,
            query_vector=vector,
            limit=k,
        )
        return [
            MemoryEntry(
                id=str(r.id),
                content=r.payload.get("content", ""),
                metadata={k: v for k, v in r.payload.items() if k != "content"},
                score=r.score,
            )
            for r in results
        ]


def estimate_tokens(messages: list[Message]) -> int:
    """Quick char-based token estimate. Tiktoken would be more accurate but
    adds a heavy dep for marginal gain at compression-threshold scale."""
    total_chars = sum(len(m.content or "") for m in messages)
    return total_chars // 4  # rough rule of thumb


async def maybe_compress(
    messages: list[Message],
    threshold: int,
    summarizer_fn,
) -> list[Message]:
    """If messages exceed token threshold, summarize the oldest half.
    summarizer_fn takes list[Message] and returns a summary string.

    Returns a new message list with [summary, ...recent_messages].
    """
    if estimate_tokens(messages) < threshold:
        return messages

    half = len(messages) // 2
    older, recent = messages[:half], messages[half:]
    summary = await summarizer_fn(older)
    from .types import Role
    summary_msg = Message(
        role=Role.SYSTEM,
        content=f"[Summary of earlier conversation]\n{summary}",
    )
    log.info("Compressed %d messages -> 1 summary + %d recent", len(older), len(recent))
    return [summary_msg, *recent]
