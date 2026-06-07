"""Basic smoke tests for the structure-aware chunker."""

from pathlib import Path

import pytest

from rag.chunking.structure_aware import StructureAwareChunker
from rag.config import load_config


@pytest.mark.skipif(not Path("data/raw/ai_agents_guidebook.pdf").exists(), reason="PDF not present")
def test_chunker_produces_chunks_with_rich_metadata():
    cfg = load_config()
    chunker = StructureAwareChunker(cfg)
    result = chunker.chunk(cfg.paths.pdf)

    assert len(result.chunks) > 50
    assert result.document.total_pages >= 100

    # At least some chunks should have rich signals
    has_section = any(c.section_path for c in result.chunks)
    has_code_or_table = any(c.metadata.get("has_code") or c.metadata.get("has_table") for c in result.chunks)

    assert has_section, "Expected at least some chunks to carry section_path"
    assert has_code_or_table, "Expected to detect code or tables in an illustrated agents book"
