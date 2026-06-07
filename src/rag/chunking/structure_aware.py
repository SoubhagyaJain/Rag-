"""
Structure-aware chunker for heavily illustrated technical books.

Key capabilities (why this matters for "AI Agents: The Illustrated Guidebook"):
- Detects and preserves tables as atomic units (pdfplumber shines here)
- Detects code blocks using font + density heuristics and keeps them whole
- Builds section_path breadcrumbs using heading font-size / boldness signals
- Attaches rich metadata per chunk: page, section_path, chunk_type, has_*, etc.
- Produces parent links suitable for parent-child retrieval strategies
- Respects token budgets while strongly preferring not to split semantic units

This is one of the highest-leverage parts of the entire system.
"""

from __future__ import annotations

import re
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Any

import pdfplumber
from pdfplumber.page import Page

from rag.chunking.base import Chunker, ChunkingResult
from rag.config import CONFIG, AppConfig
from rag.models import Chunk, ChunkType, Document
from rag.utils.text import approx_token_count, clean_text


class StructureAwareChunker(Chunker):
    def __init__(self, config: AppConfig | None = None):
        self.cfg = config or CONFIG
        self.ccfg = self.cfg.chunking

    def chunk(self, pdf_path: str, doc_id: str | None = None) -> ChunkingResult:
        pdf_path = str(Path(pdf_path).resolve())
        if doc_id is None:
            doc_id = Path(pdf_path).stem

        all_chunks: list[Chunk] = []
        section_stack: list[str] = []
        page_stats: dict[int, dict] = {}

        with pdfplumber.open(pdf_path) as pdf:
            total_pages = len(pdf.pages)
            doc = Document(
                id=doc_id,
                title="AI Agents: The Illustrated Guidebook (2025 Edition)",
                source_path=pdf_path,
                total_pages=total_pages,
            )

            for page_num, page in enumerate(pdf.pages, start=1):
                page_chunks = self._process_page(
                    page, page_num, section_stack, doc_id
                )
                all_chunks.extend(page_chunks)

                # crude stats
                page_stats[page_num] = {
                    "num_chunks": len(page_chunks),
                    "has_tables": any(c.chunk_type == "table" for c in page_chunks),
                }

        # Post-process: create lightweight parent chunks for sections if enabled
        if self.ccfg.preserve_sections:
            all_chunks = self._attach_parent_links(all_chunks)

        stats = {
            "total_chunks": len(all_chunks),
            "pages_processed": total_pages,
            "chunk_type_counts": self._count_types(all_chunks),
            "avg_chunk_tokens": sum(approx_token_count(c.text) for c in all_chunks) / max(1, len(all_chunks)),
            "page_stats": page_stats,
        }

        return ChunkingResult(document=doc, chunks=all_chunks, stats=stats)

    # ---------------- internal ----------------

    def _process_page(
        self, page: Page, page_num: int, section_stack: list[str], doc_id: str
    ) -> list[Chunk]:
        chunks: list[Chunk] = []
        words = page.extract_words(extra_attrs=["fontname", "size", "non_stroking_color"]) or []

        # --- Tables (highest priority - never split) ---
        tables = page.extract_tables() or []
        table_bboxes = []
        for t_idx, table in enumerate(tables):
            # Compute rough bbox from the table content
            # pdfplumber tables don't always give bbox; we approximate
            table_text = self._table_to_markdown(table)
            if not table_text.strip():
                continue

            chunk = Chunk(
                id=f"{doc_id}_p{page_num}_table{t_idx}",
                text=table_text,
                metadata={
                    "page": page_num,
                    "chunk_type": "table",
                    "section_path": list(section_stack),
                    "has_table": True,
                    "source": "pdfplumber_table",
                    "doc_id": doc_id,
                },
            )
            chunks.append(chunk)
            # We will later filter text that overlaps these table regions if needed

        # --- Detect headings and update section stack ---
        avg_font = self._avg_font_size(words) or 11.0
        heading_threshold = avg_font * self.ccfg.heading_font_size_threshold

        # Simple heading detection
        current_section_title = None
        for line in self._group_words_into_lines(words):
            text = clean_text(" ".join(w["text"] for w in line))
            if not text or len(text) > 120:
                continue
            max_size = max(w.get("size", 0) for w in line)
            is_bold = any("bold" in (w.get("fontname") or "").lower() for w in line)
            is_heading = (max_size >= heading_threshold) or (is_bold and max_size > avg_font * 1.1)

            if is_heading and len(text.split()) <= 12:
                # Update section stack (very simple heuristic)
                section_stack = self._update_section_stack(section_stack, text)
                current_section_title = text

        # --- Extract plain text, avoiding table regions (crude but effective) ---
        # For v1 we take page.extract_text() and then split heuristically.
        # A more advanced version would use char-level or word-level bboxes to mask table areas.
        raw_text = page.extract_text() or ""
        if not raw_text.strip():
            return chunks

        # Split into blocks trying to respect code / lists / paragraphs
        blocks = self._split_into_structural_blocks(raw_text)

        buffer: list[str] = []
        buffer_tokens = 0
        block_idx = 0

        for block in blocks:
            block = clean_text(block)
            if not block:
                continue

            btype: ChunkType = self._classify_block(block)
            b_tokens = approx_token_count(block)

            # Code and tables (already extracted) should preferably stay atomic
            if btype in ("code", "table") and self.ccfg.code_as_single_chunk:
                # Flush buffer first
                if buffer:
                    chunks.append(
                        self._make_chunk(
                            "\n".join(buffer),
                            page_num,
                            list(section_stack),
                            "text",
                            doc_id,
                            block_idx,
                        )
                    )
                    buffer, buffer_tokens = [], 0

                if b_tokens > self.ccfg.max_tokens * 1.8:
                    # Very long code block - split but mark as code
                    for sub in self._split_long_block(block, self.ccfg.max_tokens):
                        chunks.append(
                            self._make_chunk(
                                sub, page_num, list(section_stack), "code", doc_id, block_idx
                            )
                        )
                else:
                    chunks.append(
                        self._make_chunk(
                            block, page_num, list(section_stack), btype, doc_id, block_idx
                        )
                    )
                block_idx += 1
                continue

            # Normal accumulation with soft boundary respect
            if buffer_tokens + b_tokens > self.ccfg.max_tokens and buffer:
                chunks.append(
                    self._make_chunk(
                        "\n".join(buffer), page_num, list(section_stack), "text", doc_id, block_idx
                    )
                )
                buffer, buffer_tokens = [], 0
                block_idx += 1

            buffer.append(block)
            buffer_tokens += b_tokens

            # Overlap is handled at a higher level or by sliding a small window later.
            # For simplicity in first version we rely on the overlap_tokens config
            # during embedding-time or by duplicating tail of previous chunk.

        if buffer:
            chunks.append(
                self._make_chunk(
                    "\n".join(buffer), page_num, list(section_stack), "text", doc_id, block_idx
                )
            )

        # Add small overlap from previous chunk on same page (cheap and effective)
        chunks = self._add_page_overlap(chunks, self.ccfg.overlap_tokens)

        return chunks

    def _make_chunk(
        self,
        text: str,
        page: int,
        section_path: list[str],
        chunk_type: ChunkType,
        doc_id: str,
        local_idx: int,
    ) -> Chunk:
        cid = f"{doc_id}_p{page}_{chunk_type}{local_idx}_{uuid.uuid4().hex[:6]}"
        return Chunk(
            id=cid,
            text=text.strip(),
            metadata={
                "page": page,
                "section_path": section_path,
                "chunk_type": chunk_type,
                "doc_id": doc_id,
                "has_code": "```" in text or self._looks_like_code(text),
                "has_table": " | " in text and "---" in text,  # markdown table-ish
                "has_diagram": any(
                    kw in text.lower() for kw in ["diagram", "figure", "workflow", "architecture"]
                ),
            },
        )

    def _add_page_overlap(self, chunks: list[Chunk], overlap_tokens: int) -> list[Chunk]:
        if overlap_tokens <= 0 or len(chunks) < 2:
            return chunks

        overlapped: list[Chunk] = []
        prev_text = ""
        for ch in chunks:
            if prev_text and ch.chunk_type == "text":
                # prepend a small tail of previous chunk
                tail = prev_text.split()[-int(overlap_tokens * 0.7) :]
                prefix = " ".join(tail)
                if prefix:
                    ch = ch.model_copy(update={"text": prefix + "\n" + ch.text})
            overlapped.append(ch)
            prev_text = ch.text
        return overlapped

    # --- heuristics ---

    def _avg_font_size(self, words: list[dict]) -> float | None:
        sizes = [w.get("size", 0) for w in words if w.get("size")]
        if not sizes:
            return None
        return sum(sizes) / len(sizes)

    def _group_words_into_lines(self, words: list[dict]) -> list[list[dict]]:
        """Group words that are on roughly the same y-line."""
        if not words:
            return []
        lines: dict[int, list[dict]] = defaultdict(list)
        for w in words:
            # Round y to group into lines (top or bottom; pdfplumber uses 'top')
            y = int(w.get("top", 0) / 4) * 4   # 4pt tolerance
            lines[y].append(w)
        # Sort lines top-to-bottom, words left-to-right
        sorted_lines = []
        for y in sorted(lines.keys()):
            line = sorted(lines[y], key=lambda ww: ww.get("x0", 0))
            sorted_lines.append(line)
        return sorted_lines

    def _update_section_stack(self, stack: list[str], heading: str) -> list[str]:
        """Very simple section stack management. Good enough for illustrated guidebooks."""
        h = heading.strip()
        if not h:
            return stack

        # If it looks like a chapter or major section, reset deeper levels
        if re.match(r"^(chapter|part|section)\s+\d+", h, re.I) or len(h.split()) <= 3:
            # Major reset
            return [h]

        # Otherwise append or replace last
        if stack and len(stack[-1].split()) <= 3:
            return stack[:-1] + [h]
        if len(stack) >= 3:
            return stack[-2:] + [h]
        return stack + [h]

    def _split_into_structural_blocks(self, text: str) -> list[str]:
        """
        Split page text into blocks that are likely coherent (paragraphs, code, lists).
        This is intentionally heuristic.
        """
        # Split on double newlines first (paragraph-ish)
        blocks = re.split(r"\n\s*\n", text)
        refined: list[str] = []
        for b in blocks:
            b = b.strip()
            if not b:
                continue
            # Further split very long blocks on numbered list starts or obvious code fences
            if "```" in b:
                parts = re.split(r"(```[\s\S]*?```)", b)
                refined.extend([p for p in parts if p.strip()])
            else:
                refined.append(b)
        return refined

    def _classify_block(self, block: str) -> ChunkType:
        lowered = block.lower()
        if block.strip().startswith("```") or self._looks_like_code(block):
            return "code"
        if re.search(r"^\s*(\d+[\.\)]\s+|[-*+]\s+)", block, re.M):
            return "list"
        if " | " in block and re.search(r"---\s*\|", block):
            return "table"
        if any(kw in lowered for kw in ["figure", "diagram", "workflow", "architecture diagram"]):
            return "figure"
        return "text"

    def _looks_like_code(self, text: str) -> bool:
        indicators = [
            r"\bdef\s+\w+\s*\(",
            r"\bclass\s+\w+",
            r"^\s{2,}\w+\s*=",          # indented assignment
            r"import\s+[\w.]+",
            r"from\s+[\w.]+\s+import",
            r"\{[\s\S]{10,}\}",         # json-ish
            r"```",
        ]
        score = sum(1 for pat in indicators if re.search(pat, text))
        return score >= 1 or (len(re.findall(r"[{}();]", text)) > 6 and len(text.splitlines()) > 2)

    def _table_to_markdown(self, table: list[list[Any]]) -> str:
        if not table:
            return ""
        # Filter empty rows
        rows = [[str(cell).strip() if cell is not None else "" for cell in row] for row in table if any(cell for cell in row)]
        if not rows:
            return ""
        header = rows[0]
        body = rows[1:]
        md = "| " + " | ".join(header) + " |\n"
        md += "| " + " | ".join(["---"] * len(header)) + " |\n"
        for r in body:
            md += "| " + " | ".join(r) + " |\n"
        return md.strip()

    def _split_long_block(self, block: str, max_tokens: int) -> list[str]:
        """Split a long atomic-ish block (e.g. huge code example) into token-bounded pieces."""
        lines = block.splitlines(keepends=True)
        out: list[str] = []
        current: list[str] = []
        cur_tok = 0
        for line in lines:
            t = approx_token_count(line)
            if cur_tok + t > max_tokens and current:
                out.append("".join(current).strip())
                current = [line]
                cur_tok = t
            else:
                current.append(line)
                cur_tok += t
        if current:
            out.append("".join(current).strip())
        return out

    def _attach_parent_links(self, chunks: list[Chunk]) -> list[Chunk]:
        """
        Group chunks by (page, top-level section) and create lightweight parent references.
        We don't create separate parent Chunk objects here (to keep things simple);
        instead we set parent_id and a 'section_id' so the retrieval layer can expand context.
        """
        from collections import defaultdict

        section_groups: dict[str, list[Chunk]] = defaultdict(list)

        for ch in chunks:
            sp = ch.section_path or ["_root"]
            key = f"p{ch.page}::{sp[0] if sp else '_root'}"
            section_groups[key].append(ch)

        for sec_key, group in section_groups.items():
            parent_id = f"parent_{sec_key}_{uuid.uuid4().hex[:6]}"
            for ch in group:
                ch.metadata["parent_id"] = parent_id
                ch.metadata["section_id"] = sec_key
                # Store a compact parent hint (first 300 chars of the first chunk in group)
                if "parent_text_preview" not in ch.metadata:
                    ch.metadata["parent_text_preview"] = group[0].text[:300]

        return chunks

    def _count_types(self, chunks: list[Chunk]) -> dict[str, int]:
        counts: dict[str, int] = {}
        for c in chunks:
            t = c.chunk_type
            counts[t] = counts.get(t, 0) + 1
        return counts
