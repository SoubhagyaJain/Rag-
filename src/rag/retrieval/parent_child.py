"""
Parent-child (hierarchical) context expansion.

When enabled, we retrieve fine-grained chunks but return additional context
from the same section / parent. This is extremely powerful for technical books
where a small paragraph only makes sense with its surrounding diagram or section.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from rag.config import CONFIG, AppConfig
from rag.models import RetrievedChunk
from rag.retrieval.base import RetrievalResponse


def expand_with_parent_context(
    response: RetrievalResponse,
    all_chunks_by_id: dict[str, RetrievedChunk] | None = None,
    config: AppConfig | None = None,
) -> RetrievalResponse:
    """
    Expand each retrieved chunk with sibling/parent chunks from the same section.

    Note: This implementation works best when the chunker has populated
    `parent_id` or `section_id` in metadata (done by StructureAwareChunker).
    """
    cfg = config or CONFIG
    if not cfg.retrieval.parent_child.enabled:
        return response

    expansion = cfg.retrieval.parent_child.parent_expansion
    if expansion <= 0:
        return response

    # Group chunks by their section (or parent_id)
    section_to_chunks: dict[str, list[RetrievedChunk]] = defaultdict(list)
    id_to_rc: dict[str, RetrievedChunk] = {rc.chunk.id: rc for rc in response.chunks}

    # First pass: index everything we have in the response + the provided map
    for rc in response.chunks:
        sec = rc.chunk.metadata.get("section_id") or rc.chunk.metadata.get("parent_id") or str(rc.chunk.page)
        section_to_chunks[sec].append(rc)

    if all_chunks_by_id:
        for rc in all_chunks_by_id.values():
            sec = rc.chunk.metadata.get("section_id") or rc.chunk.metadata.get("parent_id") or str(rc.chunk.page)
            section_to_chunks[sec].append(rc)

    expanded: list[RetrievedChunk] = []
    seen: set[str] = set()

    for rc in response.chunks:
        if rc.chunk.id in seen:
            continue
        seen.add(rc.chunk.id)
        expanded.append(rc)

        sec = rc.chunk.metadata.get("section_id") or rc.chunk.metadata.get("parent_id")
        if not sec:
            continue

        siblings = sorted(
            section_to_chunks.get(sec, []),
            key=lambda x: (x.chunk.page or 0, x.chunk.id),
        )

        # Add a few neighbors (left and right in section order)
        try:
            idx = siblings.index(rc)
        except ValueError:
            idx = 0

        for offset in range(1, expansion + 1):
            for direction in (-1, 1):
                j = idx + direction * offset
                if 0 <= j < len(siblings):
                    sib = siblings[j]
                    if sib.chunk.id not in seen:
                        seen.add(sib.chunk.id)
                        # Slightly lower the score of expanded context
                        sib = RetrievedChunk(
                            chunk=sib.chunk,
                            score=sib.score * 0.85,
                            retrieval_method=sib.retrieval_method + "+parent",
                            rank=sib.rank,
                        )
                        expanded.append(sib)

    # Re-sort by original score (expanded context should not dominate ordering)
    expanded.sort(key=lambda x: x.score, reverse=True)

    # Dedup while preserving order
    final: list[RetrievedChunk] = []
    seen.clear()
    for rc in expanded:
        if rc.chunk.id not in seen:
            seen.add(rc.chunk.id)
            final.append(rc)

    return RetrievalResponse(
        chunks=final[: response.metadata.get("original_top_k", len(final))],
        total_candidates=response.total_candidates,
        retrieval_time_ms=response.retrieval_time_ms,
        strategy_name=response.strategy_name + "+parent",
        metadata={**response.metadata, "parent_expansion": expansion},
    )
