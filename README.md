# RAG for "AI Agents: The Illustrated Guidebook (2025 Edition)"

**A clean, modular, evaluation-first RAG system** built for richly illustrated technical documentation.

This project treats the book by Avi Chawla & Akshay Pachaar as a first-class citizen: diagrams, code examples, tables, workflows, and hierarchical sections are all respected during chunking and retrieval.

## Design Goals (as a senior staff AI/ML engineer)

- **Structure-aware chunking first** — The single highest-leverage lever for illustrated books.
- **Experimentation-friendly retrieval** — Swap or compose dense, BM25, hybrid (RRF), cross-encoder reranking, and parent-child expansion with almost zero code changes.
- **Evaluation is a first-class citizen** — 18+ high-quality golden questions with `expected_pages`, `expected_sections`, and `must_retrieve_terms`. Easy to grow to 100+.
- **Observability from day one** — Every retrieval and evaluation decision is traceable.
- **Modularity without over-engineering** — Improvements in month 2 and 3 should not require rewriting month 1 code.

## Precise Project Structure

```
Rag-/
├── config/default.yaml
├── data/
│   ├── raw/ai_agents_guidebook.pdf
│   ├── evaluation/golden_dataset.json
│   └── vectorstore/               # runtime, gitignored
├── src/rag/
│   ├── chunking/                  # StructureAwareChunker (the star)
│   ├── ingestion/pipeline.py
│   ├── retrieval/                 # base, dense, sparse, hybrid, rerank, parent_child, factory
│   ├── evaluation/harness.py
│   ├── llm/client.py
│   ├── observability/tracer.py
│   ├── models.py
│   └── config.py
├── scripts/
│   ├── ingest.py
│   ├── evaluate.py
│   └── query.py
├── notebooks/
├── pyproject.toml
└── README.md
```

## Quick Start

### 1. Environment

```powershell
cd "C:\Users\jains\OneDrive\Desktop\Rag-"

# Create venv (uv recommended)
uv venv
.venv\Scripts\activate

# Install
uv pip install -e ".[dev,llm]"
# or
pip install -e ".[dev,llm]"
```

### 2. (Recommended) Pull a local LLM

```bash
ollama pull gemma2:9b          # or 27b, qwen2.5:14b, llama3.1:8b etc.
ollama pull nomic-embed-text   # optional - we default to sentence-transformers
```

### 3. Ensure the source PDF is in place

The project expects the book at:

```
data/raw/ai_agents_guidebook.pdf
```

If the automated copy didn't fully hydrate the file (common with OneDrive), copy it manually:

```powershell
Copy-Item "..\RAG-SYSTEM\data\raw\ai_agents_guidebook.pdf" "data\raw\ai_agents_guidebook.pdf"
```

### 4. Ingest the book (structure-aware chunking + embeddings + Chroma)

```powershell
python -m scripts.ingest
# or after pip install: rag-ingest
```

This runs `StructureAwareChunker` (preserves tables, code, sections, adds rich metadata) then embeds with sentence-transformers and stores in Chroma.

### 4. Run the golden evaluation

```powershell
python -m scripts.evaluate --top-k 8
```

Results (with per-question metrics + aggregates) are written to `artifacts/evaluation_results/`.

### 5. Interactive queries

```powershell
python -m scripts.query "What is the difference between ReAct and Plan-and-Execute?"
```

## Key Configuration (config/default.yaml + .env)

All important levers live in one place:

- `chunking.*` — max_tokens, overlap, preserve tables/code/sections
- `retrieval.*` — hybrid alpha, reranker toggle, parent-child expansion
- `evaluation.use_llm_as_judge`
- `llm.*` — provider (ollama / openai / none)

Override anything via environment variables (see `.env.example`).

## How to Experiment (the whole point)

### Different retrieval strategies

```python
from rag.ingestion.pipeline import IngestionPipeline
from rag.retrieval.factory import build_retriever

pipeline = IngestionPipeline()
retriever = build_retriever(...)   # automatically follows config

# Or manually compose for an experiment
from rag.retrieval.dense import DenseRetriever
from rag.retrieval.hybrid import HybridRetriever
from rag.retrieval.rerank import Reranker

# ... build your own stack and pass to EvaluationHarness
```

### Parent-child expansion

Enabled by default. Retrieved leaf chunks are augmented with sibling chunks from the same section (the chunker writes `section_id` / `parent_id`).

### Adding a new golden question

Just append to `data/evaluation/golden_dataset.json`:

```json
{
  "question": "...",
  "ground_truth": "...",
  "expected_pages": [42, 43],
  "expected_sections": ["ReAct Pattern"],
  "must_retrieve_terms": ["ReAct", "Thought", "Action", "Observation"]
}
```

The harness will automatically use the rich signals for retrieval metrics.

## Architecture Highlights

See [ARCHITECTURE.md](ARCHITECTURE.md) for deep details.

Core ideas:
- `Chunk` is the central currency (rich metadata > raw text)
- Chunking is **not** a black box — `StructureAwareChunker` is explicit and tunable
- Retrieval is a **pipeline of composable stages**
- Evaluation measures **what actually matters** for technical RAG (term recall + page alignment + LLM judge)

## Next Steps / Roadmap Ideas

- Add vision-based figure captioning / description for diagrams
- Persistent BM25 index + hybrid index
- Agentic layer on top (ReAct-style retrieval agent)
- RAGAS integration (optional)
- FastAPI service + simple Streamlit explorer
- Systematic chunking ablations (report in artifacts/)

## Contributing

This is intended as a living, high-signal research codebase. When you add a new retrieval idea:

1. Add it as a new file under `retrieval/`
2. Wire it through `factory.py` or expose a clean constructor
3. Add the corresponding config section
4. Run `python -m scripts.evaluate --limit 5` before and after
5. Commit the before/after aggregate numbers in your PR description

---

Built with care for the illustrated nature of the source material.
