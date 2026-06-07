"""
Factory that wires a full retrieval stack from configuration.

This is the main entry point for experiments and scripts.
You can easily create different stacks for A/B testing.
"""

from __future__ import annotations

from chromadb.api.models.Collection import Collection
from sentence_transformers import SentenceTransformer

from rag.config import CONFIG, AppConfig
from rag.retrieval.base import BaseRetriever, RetrievalRequest, RetrievalResponse
from rag.retrieval.dense import DenseRetriever
from rag.retrieval.hybrid import HybridRetriever
from rag.retrieval.parent_child import expand_with_parent_context
from rag.retrieval.rerank import Reranker, apply_reranker
from rag.retrieval.sparse import BM25Retriever


def build_retriever(
    collection: Collection,
    embed_model: SentenceTransformer,
    all_chunks_for_bm25: list | None = None,
    config: AppConfig | None = None,
) -> "ComposedRetriever":
    """
    Build a ComposedRetriever according to current config.

    Returns an object with a clean `.retrieve(query, top_k=...)` API.
    """
    cfg = config or CONFIG

    dense = DenseRetriever(collection, embed_model, cfg)

    # For BM25 we need the actual chunk objects. In practice the caller
    # should pass them (they can be fetched from the collection metadata if needed).
    sparse = BM25Retriever(all_chunks_for_bm25 or [])

    if cfg.retrieval.hybrid.enabled:
        base: BaseRetriever = HybridRetriever(dense, sparse, cfg)
    else:
        base = dense

    reranker = None
    if cfg.retrieval.reranker.enabled:
        reranker = Reranker(config=cfg)

    return ComposedRetriever(base, reranker, cfg)


class ComposedRetriever:
    """
    High-level retriever that applies the full pipeline:
    (hybrid or dense) -> optional rerank -> optional parent expansion
    """

    def __init__(
        self,
        base_retriever: BaseRetriever,
        reranker: Reranker | None,
        config: AppConfig | None = None,
    ):
        self.base = base_retriever
        self.reranker = reranker
        self.cfg = config or CONFIG

    def retrieve(self, query: str, top_k: int | None = None) -> RetrievalResponse:
        top_k = top_k or self.cfg.retrieval.top_k

        req = RetrievalRequest(query=query, top_k=top_k)

        # 1. Base retrieval (dense / hybrid)
        resp = self.base.retrieve(req)

        # Remember desired k for later trimming
        resp.metadata["original_top_k"] = top_k

        # 2. Reranker (fetch more internally if configured)
        if self.reranker and self.cfg.retrieval.reranker.enabled:
            fetch_before = self.cfg.retrieval.reranker.top_k_before_rerank
            # Re-fetch a larger set from base if we didn't already
            if len(resp.chunks) < fetch_before:
                big_req = RetrievalRequest(query=query, top_k=fetch_before)
                resp = self.base.retrieve(big_req)
            resp = apply_reranker(resp, query, self.reranker, top_k, self.cfg)

        # 3. Parent / section context expansion
        if self.cfg.retrieval.parent_child.enabled:
            resp = expand_with_parent_context(resp, config=self.cfg)

        # Final trim
        resp.chunks = resp.chunks[:top_k]
        return resp
