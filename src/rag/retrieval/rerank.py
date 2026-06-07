"""
Cross-encoder reranker (sentence-transformers).

Rerankers are extremely effective for the last 20-50 candidates and dramatically
improve precision for technical / nuanced questions.
"""

from __future__ import annotations

import time

from sentence_transformers import CrossEncoder

from rag.config import CONFIG, AppConfig
from rag.models import RetrievedChunk
from rag.retrieval.base import RetrievalRequest, RetrievalResponse


class Reranker:
    def __init__(self, model_name: str | None = None, config: AppConfig | None = None):
        self.cfg = config or CONFIG
        name = model_name or self.cfg.retrieval.reranker.model_name
        self.model = CrossEncoder(name)
        self.name = f"rerank:{name.split('/')[-1]}"

    def rerank(
        self, query: str, candidates: list[RetrievedChunk], top_k: int
    ) -> list[RetrievedChunk]:
        if not candidates:
            return []

        t0 = time.perf_counter()

        pairs = [(query, c.chunk.text) for c in candidates]
        scores = self.model.predict(pairs)

        reranked: list[RetrievedChunk] = []
        for rc, score in zip(candidates, scores):
            new_rc = RetrievedChunk(
                chunk=rc.chunk,
                score=float(score),
                retrieval_method=f"{rc.retrieval_method}+rerank",
                rank=None,
            )
            reranked.append(new_rc)

        reranked.sort(key=lambda x: x.score, reverse=True)
        result = reranked[:top_k]

        for rank, rc in enumerate(result, start=1):
            rc.rank = rank

        return result


def apply_reranker(
    response: RetrievalResponse,
    query: str,
    reranker: Reranker,
    final_top_k: int,
    cfg: AppConfig | None = None,
) -> RetrievalResponse:
    """Convenience wrapper used by higher-level retrievers."""
    cfg = cfg or CONFIG
    before = len(response.chunks)

    reranked_chunks = reranker.rerank(query, response.chunks, final_top_k)

    return RetrievalResponse(
        chunks=reranked_chunks,
        total_candidates=response.total_candidates,
        retrieval_time_ms=response.retrieval_time_ms + (time.perf_counter() * 0) ,  # caller will measure total
        strategy_name=response.strategy_name + "+rerank",
        metadata={
            **response.metadata,
            "reranker": reranker.name,
            "candidates_before_rerank": before,
        },
    )
