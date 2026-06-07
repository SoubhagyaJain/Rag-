"""
CLI: rag-evaluate

Runs the full golden set evaluation and writes rich artifacts.
"""

import typer
from rich import print as rprint
from rich.table import Table

from rag.config import load_config
from rag.evaluation.harness import EvaluationHarness, load_golden_dataset
from rag.ingestion.pipeline import IngestionPipeline
from rag.llm.client import get_llm_client
from rag.observability.tracer import setup_logging
from rag.retrieval.factory import build_retriever
from rag.utils.io import read_json

app = typer.Typer(help="Run evaluation over the golden dataset.")


@app.command()
def main(
    config: str = typer.Option("config/default.yaml", "--config", "-c"),
    k: int = typer.Option(8, "--top-k", "-k"),
    limit: int | None = typer.Option(None, "--limit", help="Run only first N examples (for quick iteration)"),
):
    cfg = load_config(config)
    setup_logging(cfg)

    rprint("[bold blue]Loading pipeline and retriever...[/bold blue]")

    # Ingestion pipeline gives us the collection + embedder
    pipeline = IngestionPipeline(cfg)
    collection = pipeline.get_collection()
    embed_model = pipeline.embed_model

    # For BM25 we need the chunks. In a real system we would store them or re-derive.
    # For now we do a simple trick: query all (or we can load from golden only).
    # Better: we can ask Chroma for all documents (slow for huge collections, fine here).
    all_data = collection.get(include=["documents", "metadatas"])
    from rag.models import Chunk

    all_chunks = []
    for cid, doc, meta in zip(all_data["ids"], all_data["documents"], all_data["metadatas"]):
        all_chunks.append(Chunk(id=cid, text=doc, metadata=meta or {}))

    retriever = build_retriever(collection, embed_model, all_chunks_for_bm25=all_chunks, config=cfg)

    llm = get_llm_client(cfg) if cfg.evaluation.use_llm_as_judge else None

    harness = EvaluationHarness(retriever, embed_model, llm_client=llm, config=cfg)

    examples = load_golden_dataset(cfg.paths.golden_dataset)
    if limit:
        examples = examples[:limit]

    rprint(f"Running evaluation on [bold]{len(examples)}[/bold] examples...")

    run = harness.run_full_evaluation(k=k)

    # Pretty summary
    table = Table(title=f"Evaluation Run {run.run_id}")
    table.add_column("Metric", style="cyan")
    table.add_column("Score", style="green")

    for name, val in sorted(run.aggregate_metrics.items()):
        table.add_row(name, f"{val:.4f}")

    rprint(table)
    rprint(f"\nDetailed results: artifacts/evaluation_results/{run.run_id}.json")


if __name__ == "__main__":
    app()
