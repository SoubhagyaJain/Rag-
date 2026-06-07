"""
Base classes and data structures for all retrievers.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from rag.models import Chunk, RetrievalResult, RetrievedChunk


@dataclass
class RetrievalRequest:
    query: str
    top_k: int = 8
    filters: dict[str, Any] | None = None  # e.g. {"page": {"$gte": 30}}
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievalResponse:
    chunks: list[RetrievedChunk]
    total_candidates: int
    retrieval_time_ms: float
    strategy_name: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_result(self, query: str) -> RetrievalResult:
        return RetrievalResult(
            query=query,
            chunks=self.chunks,
            total_candidates=self.total_candidates,
            retrieval_time_ms=self.retrieval_time_ms,
            strategy=self.strategy_name,
            metadata=self.metadata,
        )

    def context_text(self, max_chunks: int | None = None) -> str:
        """Convenience: concatenated context for LLM prompts."""
        items = self.chunks
        if max_chunks is not None:
            items = items[:max_chunks]
        parts = []
        for rc in items:
            header = f"[Page {rc.chunk.page} | {rc.chunk.chunk_type}]"
            if rc.chunk.section_path:
                header += f" | {' > '.join(rc.chunk.section_path)}"
            parts.append(f"{header}\n{rc.chunk.text}")
        return "\n\n---\n\n".join(parts)


class BaseRetriever(ABC):
    """All retrievers (dense, sparse, hybrid, etc.) implement this."""

    @abstractmethod
    def retrieve(self, request: RetrievalRequest) -> RetrievalResponse:
        raise NotImplementedError

    @property
    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError
