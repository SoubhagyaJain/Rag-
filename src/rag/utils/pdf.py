"""PDF-related utilities (future: page images, figure extraction, etc.)."""

from __future__ import annotations

from pathlib import Path

import pdfplumber


def get_page_count(pdf_path: str) -> int:
    with pdfplumber.open(pdf_path) as pdf:
        return len(pdf.pages)


def extract_page_text(pdf_path: str, page: int) -> str:
    with pdfplumber.open(pdf_path) as pdf:
        return pdf.pages[page - 1].extract_text() or ""
