"""
Sparse retriever using BM25 (rank-bm25).

In-memory for v1. For very large corpora you would persist the tokenized corpus + index.
"""

from __future__ import annotations

import time
from typing import Any

from rank_bm25 import BM25Okapi

from rag.models import Chunk, RetrievedChunk
from rag.retrieval.base import BaseRetriever, RetrievalRequest, RetrievalResponse
from rag.utils.text import clean_text


class BM25Retriever(BaseRetriever):
    def __init__(self, chunks: list[Chunk] | None = None):
        self.chunks: list[Chunk] = chunks or []
        self.bm25: BM25Okapi | None = None
        self._tokenized: list[list[str]] = []
        if self.chunks:
            self._build_index()

    @property
    def name(self) -> str:
        return "bm25"

    def _tokenize(self, text: str) -> list[str]:
        # Simple but effective tokenization for BM25
        return clean_text(text).lower().split()

    def _build_index(self) -> None:
        self._tokenized = [self._tokenize(c.text) for c in self.chunks]
        self.bm25 = BM25Okapi(self._tokenized)

    def add_chunks(self, new_chunks: list[Chunk]) -> None:
        self.chunks.extend(new_chunks)
        self._build_index()

    def retrieve(self, request: RetrievalRequest) -> RetrievalResponse:
        if not self.bm25 or not self.chunks:
            return RetrievalResponse(
                chunks=[],
                total_candidates=0,
                retrieval_time_ms=0.0,
                strategy_name=self.name,
            )

        t0 = time.perf_counter()

        q_tokens = self._tokenize(request.query)
        scores = self.bm25.get_scores(q_tokens)

        # Get top candidates
        scored = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        top = scored[: request.top_k * 2]

        retrieved: list[RetrievedChunk] = []
        for rank, (idx, score) in enumerate(top, start=1):
            chunk = self.chunks[idx]
            rc = RetrievedChunk(
                chunk=chunk,
                score=float(score),
                retrieval_method="bm25",
                rank=rank,
            )
            retrieved.append(rc)

        retrieved = retrieved[: request.top_k]
        elapsed = (time.perf_counter() - t0) * 1000

        return RetrievalResponse(
            chunks=retrieved,
            total_candidates=len(scored),
            retrieval_time_ms=elapsed,
            strategy_name=self.name,
        )
