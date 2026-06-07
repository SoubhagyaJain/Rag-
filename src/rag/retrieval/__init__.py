"""
Modular retrieval stack.

The design goal: make it trivial to try new combinations in experiments
(hybrid + rerank + parent expansion, dense only, BM25 only, etc.)
without touching ingestion or evaluation code.
"""

from .base import BaseRetriever, RetrievalRequest, RetrievalResponse
from .dense import DenseRetriever
from .sparse import BM25Retriever
from .hybrid import HybridRetriever
from .rerank import Reranker

__all__ = [
    "BaseRetriever",
    "RetrievalRequest",
    "RetrievalResponse",
    "DenseRetriever",
    "BM25Retriever",
    "HybridRetriever",
    "Reranker",
]
