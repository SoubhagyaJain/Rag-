"""
CLI: rag-ingest

Usage:
    python -m scripts.ingest
    # or after install: rag-ingest
"""

import typer
from rich import print as rprint

from rag.config import load_config
from rag.ingestion.pipeline import IngestionPipeline
from rag.observability.tracer import setup_logging

app = typer.Typer(help="Ingest the AI Agents Guidebook into the vector store.")


@app.command()
def main(
    config: str = typer.Option("config/default.yaml", "--config", "-c"),
    force: bool = typer.Option(False, "--force", "-f", help="Delete existing collection and re-ingest"),
):
    cfg = load_config(config)
    setup_logging(cfg)

    rprint(f"[bold]Ingesting[/bold] {cfg.paths.pdf}")
    pipeline = IngestionPipeline(cfg)
    result = pipeline.run(force_reingest=force)

    rprint("[green]Ingestion successful[/green]")
    rprint(result)


if __name__ == "__main__":
    app()
