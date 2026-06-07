"""
Lightweight but production-grade observability.

- Structured logging via loguru
- Per-query / per-eval Trace objects that can be serialized to JSONL
- Easy to extend later with Phoenix, LangSmith, or Prometheus
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from loguru import logger
from rich.logging import RichHandler

from rag.config import CONFIG, AppConfig
from rag.models import RetrievalResult, Trace


def setup_logging(config: AppConfig | None = None) -> None:
    """Configure loguru once at startup. Call early in scripts and notebooks."""
    cfg = config or CONFIG
    logger.remove()  # remove default

    level = cfg.observability.log_level.upper()

    # Human-friendly console (rich)
    logger.add(
        RichHandler(rich_tracebacks=True, markup=True),
        level=level,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <7}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    )

    # Also always write a plain text log for CI / postmortems
    log_dir = Path(cfg.paths.artifacts_dir) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    logger.add(
        log_dir / "rag_system.log",
        level=level,
        rotation="10 MB",
        retention="14 days",
        encoding="utf-8",
    )


def get_logger(name: str | None = None):
    """Return a contextual logger."""
    return logger.bind(name=name or "rag")


class Tracer:
    """
    Context manager + collector for execution traces.

    Usage:
        with Tracer("my_query_123", query=question) as trace:
            trace.log("retrieval_start", strategy="hybrid")
            ...
            trace.final_retrieval = result
    """

    def __init__(self, trace_id: str, query: str, metadata: dict[str, Any] | None = None):
        self.trace = Trace(trace_id=trace_id, query=query, metadata=metadata or {})
        self._closed = False

    def __enter__(self) -> Trace:
        self.trace.log("trace_start")
        return self.trace

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type:
            self.trace.log("error", error=str(exc_val), type=str(exc_type))
        self.trace.log("trace_end")
        self._closed = True

    def save(self, directory: str | Path | None = None) -> Path:
        """Persist this trace as JSONL line (append mode)."""
        if not self.trace.events:
            return Path()

        cfg = CONFIG
        out_dir = Path(directory or cfg.observability.trace_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        path = out_dir / f"{self.trace.trace_id}.jsonl"
        record = {
            "trace_id": self.trace.trace_id,
            "query": self.trace.query,
            "events": self.trace.events,
            "final_answer": self.trace.final_answer,
            "metadata": self.trace.metadata,
        }
        if self.trace.final_retrieval:
            record["retrieval"] = {
                "strategy": self.trace.final_retrieval.strategy,
                "num_chunks": len(self.trace.final_retrieval.chunks),
                "time_ms": self.trace.final_retrieval.retrieval_time_ms,
                "top_chunks": [
                    {
                        "id": rc.chunk.id,
                        "page": rc.chunk.page,
                        "score": rc.score,
                        "method": rc.retrieval_method,
                        "preview": rc.chunk.text[:200],
                    }
                    for rc in self.trace.final_retrieval.chunks[:5]
                ],
            }

        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        return path
