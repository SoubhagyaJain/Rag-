"""
Base interfaces for chunkers.

This abstraction lets us swap in better chunkers later (e.g. semantic chunker,
layout-aware using vision models, or LLM-based sectioning) with zero changes
to the ingestion pipeline or retrieval code.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from rag.models import Chunk, Document


@dataclass
class ChunkingResult:
    document: Document
    chunks: list[Chunk]
    stats: dict[str, Any]


class Chunker(ABC):
    """Abstract chunker interface."""

    @abstractmethod
    def chunk(self, pdf_path: str, doc_id: str | None = None) -> ChunkingResult:
        """Parse the PDF and return richly annotated chunks."""
        raise NotImplementedError
