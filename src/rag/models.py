"""
Core data models for the RAG system.

These are intentionally lightweight and framework-agnostic so they can be
used by chunkers, retrievers, evaluators, and any future agentic layers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field


# -----------------------------
# Core content models
# -----------------------------

ChunkType = Literal["text", "table", "code", "list", "figure", "heading", "other"]


class Chunk(BaseModel):
    """
    A single atomic chunk of content from the source document.

    Rich metadata is the foundation of good retrieval for illustrated technical books.
    """

    id: str = Field(default_factory=lambda: str(uuid4()))
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)

    # Optional pre-computed embedding (populated during ingestion)
    embedding: list[float] | None = None

    # Convenience accessors (derived from metadata in most cases)
    @property
    def page(self) -> int | None:
        return self.metadata.get("page")

    @property
    def section_path(self) -> list[str]:
        return self.metadata.get("section_path", [])

    @property
    def chunk_type(self) -> ChunkType:
        return self.metadata.get("chunk_type", "text")

    @property
    def parent_id(self) -> str | None:
        return self.metadata.get("parent_id")

    def short_repr(self, max_chars: int = 120) -> str:
        text_preview = self.text.replace("\n", " ")[:max_chars]
        if len(self.text) > max_chars:
            text_preview += "..."
        page = self.page or "?"
        return f"[p.{page} | {self.chunk_type}] {text_preview}"


class Document(BaseModel):
    """A full source document (currently one PDF, but extensible)."""

    id: str
    title: str
    source_path: str
    total_pages: int
    metadata: dict[str, Any] = Field(default_factory=dict)


# -----------------------------
# Retrieval models
# -----------------------------

@dataclass
class RetrievedChunk:
    """A chunk returned by a retriever, with score and provenance."""

    chunk: Chunk
    score: float
    retrieval_method: str  # e.g. "dense", "bm25", "hybrid", "reranked"
    rank: int | None = None

    def __post_init__(self):
        if self.rank is None:
            # rank will be set by fusion / reranker stages
            pass


@dataclass
class RetrievalResult:
    """Complete result of a retrieval operation (before or after generation)."""

    query: str
    chunks: list[RetrievedChunk]
    total_candidates: int
    retrieval_time_ms: float
    strategy: str  # e.g. "hybrid+rerank+parent"
    metadata: dict[str, Any] = field(default_factory=dict)

    def top_k(self, k: int) -> list[RetrievedChunk]:
        return sorted(self.chunks, key=lambda x: x.score, reverse=True)[:k]

    def context_text(self, max_chunks: int | None = None) -> str:
        """Concatenated context suitable for stuffing into an LLM prompt."""
        items = self.chunks
        if max_chunks:
            items = items[:max_chunks]
        parts = []
        for rc in items:
            header = f"[Page {rc.chunk.page} | {rc.chunk.chunk_type}]"
            if rc.chunk.section_path:
                header += f" | {' > '.join(rc.chunk.section_path)}"
            parts.append(f"{header}\n{rc.chunk.text}")
        return "\n\n---\n\n".join(parts)


# -----------------------------
# Evaluation models
# -----------------------------

class GoldenExample(BaseModel):
    """
    One item from the golden evaluation set.

    The extra fields (expected_pages, must_retrieve_terms, etc.) enable
    retrieval-specific metrics that go far beyond naive "does the answer contain X".
    """

    question: str
    ground_truth: str
    expected_pages: list[int] = Field(default_factory=list)
    expected_sections: list[str] = Field(default_factory=list)
    must_retrieve_terms: list[str] = Field(default_factory=list)


@dataclass
class EvaluationRecord:
    """One row in an evaluation run."""

    question: str
    ground_truth: str
    generated_answer: str
    retrieved_chunks: list[RetrievedChunk]
    metrics: dict[str, float | list[float]]
    latency_ms: float
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())


@dataclass
class EvaluationRun:
    """Full results of running the golden set."""

    run_id: str
    config_snapshot: dict[str, Any]
    records: list[EvaluationRecord]
    aggregate_metrics: dict[str, float]
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def to_summary(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "created_at": self.created_at,
            "num_examples": len(self.records),
            "aggregate_metrics": self.aggregate_metrics,
        }


# -----------------------------
# Observability
# -----------------------------

@dataclass
class Trace:
    """Lightweight execution trace for a single query or eval item."""

    trace_id: str
    query: str
    events: list[dict[str, Any]] = field(default_factory=list)
    final_retrieval: RetrievalResult | None = None
    final_answer: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def log(self, event: str, **data: Any) -> None:
        self.events.append({"event": event, "ts": datetime.utcnow().isoformat(), **data})
