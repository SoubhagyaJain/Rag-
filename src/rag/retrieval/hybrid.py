"""
Hybrid retriever = Dense + BM25 fused with Reciprocal Rank Fusion (RRF).

This is one of the most reliable ways to improve retrieval quality on technical content.
"""

from __future__ import annotations

import time
from collections import defaultdict

from chromadb.api.models.Collection import Collection
from sentence_transformers import SentenceTransformer

from rag.config import CONFIG, AppConfig
from rag.models import RetrievedChunk
from rag.retrieval.base import BaseRetriever, RetrievalRequest, RetrievalResponse
from rag.retrieval.dense import DenseRetriever
from rag.retrieval.sparse import BM25Retriever


def reciprocal_rank_fusion(
    results: list[list[RetrievedChunk]], k: int = 60
) -> list[RetrievedChunk]:
    """
    Classic RRF. Works surprisingly well even with very different score distributions.
    """
    scores: dict[str, float] = defaultdict(float)
    chunk_map: dict[str, RetrievedChunk] = {}

    for res_list in results:
        for rank, rc in enumerate(res_list, start=1):
            chunk_id = rc.chunk.id
            scores[chunk_id] += 1.0 / (k + rank)
            if chunk_id not in chunk_map:
                # clone to avoid mutating original method
                chunk_map[chunk_id] = RetrievedChunk(
                    chunk=rc.chunk,
                    score=0.0,
                    retrieval_method="hybrid",
                )

    # Assign fused scores and sort
    fused: list[RetrievedChunk] = []
    for cid, fused_score in scores.items():
        rc = chunk_map[cid]
        rc.score = fused_score
        fused.append(rc)

    fused.sort(key=lambda x: x.score, reverse=True)
    return fused


class HybridRetriever(BaseRetriever):
    def __init__(
        self,
        dense: DenseRetriever,
        sparse: BM25Retriever,
        config: AppConfig | None = None,
    ):
        self.dense = dense
        self.sparse = sparse
        self.cfg = config or CONFIG

    @property
    def name(self) -> str:
        return "hybrid"

    def retrieve(self, request: RetrievalRequest) -> RetrievalResponse:
        t0 = time.perf_counter()

        alpha = self.cfg.retrieval.hybrid.alpha
        rrf_k = self.cfg.retrieval.hybrid.rrf_k

        # Fetch more from each so fusion has room to work
        fetch_k = max(request.top_k, 20)

        dense_req = RetrievalRequest(
            query=request.query, top_k=fetch_k, filters=request.filters
        )
        sparse_req = RetrievalRequest(
            query=request.query, top_k=fetch_k, filters=request.filters
        )

        dense_resp = self.dense.retrieve(dense_req)
        sparse_resp = self.sparse.retrieve(sparse_req)

        # Weighting via alpha: we can bias the lists before fusion or just use RRF (RRF is robust)
        # For explicit alpha we can duplicate lists or adjust ranks, but RRF + alpha post-mix is simpler.
        combined_lists = []
        if alpha > 0.05:
            combined_lists.append(dense_resp.chunks)
        if alpha < 0.95:
            combined_lists.append(sparse_resp.chunks)

        fused = reciprocal_rank_fusion(combined_lists, k=rrf_k)

        # Apply light alpha bias to final scores (optional but nice for experiments)
        if 0.0 < alpha < 1.0:
            for rc in fused:
                # Small multiplicative nudge based on original method
                if "dense" in rc.retrieval_method or rc in dense_resp.chunks:
                    rc.score *= (0.5 + alpha)
                else:
                    rc.score *= (1.5 - alpha)

        fused = fused[: request.top_k]

        elapsed = (time.perf_counter() - t0) * 1000

        return RetrievalResponse(
            chunks=fused,
            total_candidates=len(dense_resp.chunks) + len(sparse_resp.chunks),
            retrieval_time_ms=elapsed,
            strategy_name="hybrid",
            metadata={
                "alpha": alpha,
                "rrf_k": rrf_k,
                "dense_count": len(dense_resp.chunks),
                "sparse_count": len(sparse_resp.chunks),
            },
        )
