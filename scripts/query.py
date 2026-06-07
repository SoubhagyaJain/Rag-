"""
Interactive / one-shot query CLI.

Useful for manual testing of different retrieval strategies.
"""

import typer
from rich import print as rprint
from rich.panel import Panel

from rag.config import load_config
from rag.ingestion.pipeline import IngestionPipeline
from rag.llm.client import get_llm_client
from rag.observability.tracer import setup_logging
from rag.retrieval.factory import build_retriever

app = typer.Typer(help="Query the RAG system (great for quick experiments).")


@app.command()
def main(
    question: str = typer.Argument(..., help="The question to ask the book"),
    config: str = typer.Option("config/default.yaml", "--config", "-c"),
    k: int = typer.Option(6, "--top-k", "-k"),
    show_context: bool = typer.Option(True, "--context/--no-context"),
):
    cfg = load_config(config)
    setup_logging(cfg)

    pipeline = IngestionPipeline(cfg)
    collection = pipeline.get_collection()
    embed_model = pipeline.embed_model

    all_data = collection.get(include=["documents", "metadatas"])
    from rag.models import Chunk

    all_chunks = [
        Chunk(id=cid, text=doc, metadata=meta or {})
        for cid, doc, meta in zip(all_data["ids"], all_data["documents"], all_data["metadatas"])
    ]

    retriever = build_retriever(collection, embed_model, all_chunks_for_bm25=all_chunks, config=cfg)
    llm = get_llm_client(cfg)

    rprint(f"[bold]Question:[/bold] {question}\n")

    resp = retriever.retrieve(question, top_k=k)

    rprint(f"[bold green]Strategy:[/bold green] {resp.strategy_name} | {len(resp.chunks)} chunks | {resp.retrieval_time_ms:.0f}ms")

    if show_context:
        for i, rc in enumerate(resp.chunks, 1):
            page = rc.chunk.page or "?"
            sec = " > ".join(rc.chunk.section_path) if rc.chunk.section_path else ""
            preview = rc.chunk.text[:280].replace("\n", " ")
            rprint(f"\n[cyan]{i}.[/cyan] p.{page} | {rc.retrieval_method} | score={rc.score:.3f}")
            if sec:
                rprint(f"   [dim]{sec}[/dim]")
            rprint(f"   {preview}...")

    # Optional generation
    context = "\n\n".join([rc.chunk.text for rc in resp.chunks])
    answer = llm.generate(f"Answer the question using only the context.\n\nQuestion: {question}\n\nContext:\n{context}\n\nAnswer:")
    rprint(Panel(answer or "(no generation)", title="Generated Answer", border_style="green"))


if __name__ == "__main__":
    app()
