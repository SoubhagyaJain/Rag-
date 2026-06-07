"""
Text and token utilities.

We keep token counting cheap and dependency-light by default.
For production token-accurate chunking, install tiktoken and we will use it.
"""

from __future__ import annotations

import re
from typing import Any


def approx_token_count(text: str) -> int:
    """
    Fast approximate token count.

    - ~4 chars per token for English is a common rule of thumb.
    - Slightly more accurate than len(text)/4 for code-heavy text.
    """
    if not text:
        return 0
    # Rough but good enough: words + punctuation
    words = len(re.findall(r"\b\w+\b", text))
    # Add a bit for punctuation and subword splits common in code/LLM tokenizers
    extra = len(text) // 5
    return max(1, int(words * 1.3 + extra))


def clean_text(text: str) -> str:
    """Normalize whitespace and common PDF artifacts."""
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = text.strip()
    return text


def split_into_sentences(text: str) -> list[str]:
    """Very lightweight sentence splitter. Good enough for overlap logic."""
    # Split on . ! ? followed by space or newline, keep delimiter
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [p.strip() for p in parts if p.strip()]


def truncate_to_tokens(text: str, max_tokens: int) -> str:
    """Cheap truncation by char count (use for safety)."""
    # Conservative: assume 5 chars/token worst case for safety
    max_chars = max_tokens * 5
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rsplit(" ", 1)[0] + " ..."
