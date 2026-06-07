"""
Ingestion pipeline: PDF -> StructureAwareChunker -> Embeddings -> Vector Store.

Designed to be:
- Idempotent (can re-run safely)
- Observable (logs + traces)
- Config-driven
- Easy to extend (add a new embedding provider or vector store backend)
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import chromadb
from chromadb.utils import embedding_functions
from loguru import logger
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from rag.chunking.structure_aware import StructureAwareChunker
from rag.config import CONFIG, AppConfig
from rag.models import Chunk
from rag.observability.tracer import setup_logging
from rag.utils.io import ensure_dir


@dataclass
class IngestionResult:
    num_chunks: int
    num_pages: int
    vectorstore_path: str
    collection_name: str
    stats: dict[str, Any]


class LocalEmbeddingFunction:
    """Wrapper so Chroma can use our sentence-transformers model directly."""

    def __init__(self, model: SentenceTransformer, normalize: bool = True):
        self.model = model
        self.normalize = normalize

    def __call__(self, texts: list[str]) -> list[list[float]]:
        embs = self.model.encode(
            texts,
            normalize_embeddings=self.normalize,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return embs.tolist()


class IngestionPipeline:
    def __init__(self, config: AppConfig | None = None):
        self.cfg = config or CONFIG
        setup_logging(self.cfg)

        self.chunker = StructureAwareChunker(self.cfg)

        # Embedding model (loaded once)
        logger.info(f"Loading embedding model: {self.cfg.embedding.model_name}")
        device = self.cfg.embedding.device
        self.embed_model = SentenceTransformer(
            self.cfg.embedding.model_name, device=device
        )
        self.embed_model.max_seq_length = 512  # sensible cap

        # Chroma client (persistent)
        vs_dir = ensure_dir(self.cfg.paths.vectorstore_dir)
        self.chroma_client = chromadb.PersistentClient(path=str(vs_dir))

        # We will (re)create or get collection on demand
        self.collection = None

    def run(self, force_reingest: bool = False) -> IngestionResult:
        pdf_path = self.cfg.paths.pdf
        if not Path(pdf_path).exists():
            raise FileNotFoundError(f"PDF not found at {pdf_path}")

        logger.info(f"Starting ingestion of {pdf_path}")

        # 1. Chunk
        logger.info("Running structure-aware chunker...")
        t0 = time.time()
        chunk_result = self.chunker.chunk(pdf_path)
        chunks: list[Chunk] = chunk_result.chunks
        logger.info(
            f"Chunked into {len(chunks)} chunks across {chunk_result.document.total_pages} pages "
            f"in {time.time() - t0:.1f}s"
        )

        # 2. Embed + upsert into Chroma
        collection_name = self.cfg.vectorstore.collection_name

        if force_reingest:
            try:
                self.chroma_client.delete_collection(collection_name)
                logger.warning(f"Deleted existing collection: {collection_name}")
            except Exception:
                pass

        # Create or get
        try:
            self.collection = self.chroma_client.get_collection(collection_name)
            logger.info(f"Using existing collection: {collection_name}")
        except Exception:
            self.collection = self.chroma_client.create_collection(
                name=collection_name,
                metadata={"hnsw:space": self.cfg.vectorstore.distance},
            )
            logger.info(f"Created new collection: {collection_name}")

        # Prepare for upsert
        texts = [c.text for c in chunks]
        ids = [c.id for c in chunks]
        metadatas = [c.metadata for c in chunks]

        # Embed in batches
        logger.info("Computing embeddings...")
        batch_size = self.cfg.embedding.batch_size
        embeddings: list[list[float]] = []

        for i in tqdm(range(0, len(texts), batch_size), desc="Embedding"):
            batch = texts[i : i + batch_size]
            embs = self.embed_model.encode(
                batch,
                normalize_embeddings=self.cfg.embedding.normalize_embeddings,
                show_progress_bar=False,
            )
            embeddings.extend(embs.tolist())

        # Upsert
        logger.info(f"Upserting {len(ids)} chunks into Chroma...")
        # Chroma has a limit on how many we can send at once in some versions; batch it
        upsert_batch = 500
        for i in range(0, len(ids), upsert_batch):
            self.collection.upsert(
                ids=ids[i : i + upsert_batch],
                documents=texts[i : i + upsert_batch],
                embeddings=embeddings[i : i + upsert_batch],
                metadatas=metadatas[i : i + upsert_batch],
            )

        logger.success("Ingestion complete.")

        return IngestionResult(
            num_chunks=len(chunks),
            num_pages=chunk_result.document.total_pages,
            vectorstore_path=str(self.cfg.paths.vectorstore_dir),
            collection_name=collection_name,
            stats={
                "chunking_stats": chunk_result.stats,
                "embedding_model": self.cfg.embedding.model_name,
            },
        )

    def get_collection(self):
        if self.collection is None:
            name = self.cfg.vectorstore.collection_name
            self.collection = self.chroma_client.get_collection(name)
        return self.collection
