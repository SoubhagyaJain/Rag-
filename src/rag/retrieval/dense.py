"""
Dense (vector) retriever backed by Chroma.
"""

from __future__ import annotations

import time

from chromadb.api.models.Collection import Collection
from sentence_transformers import SentenceTransformer

from rag.config import CONFIG, AppConfig
from rag.models import Chunk, RetrievedChunk
from rag.retrieval.base import BaseRetriever, RetrievalRequest, RetrievalResponse


class DenseRetriever(BaseRetriever):
    def __init__(
        self,
        collection: Collection,
        embed_model: SentenceTransformer,
        config: AppConfig | None = None,
    ):
        self.collection = collection
        self.embed_model = embed_model
        self.cfg = config or CONFIG

    @property
    def name(self) -> str:
        return "dense"

    def retrieve(self, request: RetrievalRequest) -> RetrievalResponse:
        t0 = time.perf_counter()

        qvec = self.embed_model.encode(
            [request.query],
            normalize_embeddings=self.cfg.embedding.normalize_embeddings,
        )[0].tolist()

        results = self.collection.query(
            query_embeddings=[qvec],
            n_results=request.top_k * 2,  # fetch a bit more for post-processing
            where=request.filters,
            include=["documents", "metadatas", "distances", "embeddings"],
        )

        retrieved: list[RetrievedChunk] = []

        ids = results["ids"][0]
        docs = results["documents"][0]
        metas = results["metadatas"][0]
        dists = results["distances"][0]

        for rank, (cid, doc, meta, dist) in enumerate(zip(ids, docs, metas, dists), start=1):
            # Chroma distance for cosine is 1 - similarity (lower is better)
            # Convert to a similarity-ish score in [0, 1] range for consistency
            score = 1.0 - float(dist) if self.cfg.vectorstore.distance == "cosine" else float(dist)

            chunk = Chunk(id=cid, text=doc, metadata=meta or {})
            rc = RetrievedChunk(
                chunk=chunk,
                score=score,
                retrieval_method="dense",
                rank=rank,
            )
            retrieved.append(rc)

        # Trim to requested top_k after possible parent expansion later
        retrieved = sorted(retrieved, key=lambda x: x.score, reverse=True)[: request.top_k]

        elapsed = (time.perf_counter() - t0) * 1000

        return RetrievalResponse(
            chunks=retrieved,
            total_candidates=len(ids),
            retrieval_time_ms=elapsed,
            strategy_name=self.name,
        )
