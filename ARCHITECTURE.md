# Architecture & Design Decisions

This document explains *why* the system is built the way it is.

## 1. Why Structure-Aware Chunking Is The Foundation

Most RAG systems treat PDFs as streams of text and apply `SentenceSplitter` or `RecursiveCharacterTextSplitter`.

For a heavily illustrated book like *AI Agents: The Illustrated Guidebook*, this is disastrous:

- Tables get split mid-row
- Code examples get fragmented across chunks
- Diagram captions become detached from their visuals (even if we don't embed images yet)
- Section hierarchy is lost

**Our approach** (`StructureAwareChunker`):

- Uses `pdfplumber` because it gives us tables as first-class objects + word-level font/size/bold info.
- Detects headings via relative font size + boldness heuristics (surprisingly effective on well-designed books).
- Maintains a `section_path` (e.g. `["Agent Architectures", "ReAct Pattern", "Thought → Action → Observation"]`).
- Treats tables and code blocks as mostly-atomic when config says so.
- Emits `Chunk` objects that carry `chunk_type`, `has_table`, `has_code`, `parent_id`, `section_id`, etc.

This metadata is then used by:
- Retrieval (filtering, parent expansion)
- Evaluation (page overlap + term matching)
- Future features (only retrieve "code" chunks when user asks for implementation details)

## 2. Retrieval as a Composable Pipeline

We deliberately avoided a single "magic" retriever class.

Instead we have:

```
BaseRetriever (protocol)
├── DenseRetriever (Chroma + sentence-transformers)
├── BM25Retriever (rank-bm25)
├── HybridRetriever (RRF fusion + optional alpha bias)
└── (post-processors)
    ├── Reranker (cross-encoder)
    └── expand_with_parent_context
```

The `ComposedRetriever` + `build_retriever()` factory in `retrieval/factory.py` wires the current `config/default.yaml` choices.

**Why this wins for experimentation**:
- You can run an ablation in a notebook in < 10 lines.
- You can have multiple named retriever configs for A/B.
- Adding a new idea (e.g. HyDE, Colbert, graph retrieval) is a new file + small wiring.

## 3. Evaluation That Actually Measures Retrieval Quality

The golden dataset is special because every example carries:

- `expected_pages`
- `expected_sections`
- `must_retrieve_terms`

This lets us compute **retrieval-specific** metrics that pure answer correctness misses:

- Did we surface chunks from the right pages?
- Did we surface chunks containing the critical technical terms the author used?

We still compute answer similarity (embedding cosine to ground truth) and optional LLM-as-judge scores.

The harness writes **full per-example records** (retrieved chunks + scores + generated answer) so you can do error analysis instead of just staring at aggregate numbers.

## 4. Observability

Every important operation goes through:

- `loguru` structured + rich console logs
- `Tracer` context manager that produces JSONL traces in `artifacts/traces/`

This makes it trivial later to feed traces into an observability platform or do offline analysis of "which queries are failing at retrieval vs generation".

## 5. Technology Choices (and Trade-offs)

| Concern              | Choice                        | Why                                                                 |
|----------------------|-------------------------------|---------------------------------------------------------------------|
| PDF parsing          | pdfplumber                    | Best-in-class table + layout + font info for Python                 |
| Embeddings (default) | sentence-transformers         | Reliable, no server needed, great local quality                     |
| Vector store         | Chroma (persistent)           | Simple, good enough, local, easy to inspect                         |
| Sparse               | rank-bm25                     | Zero dependency pain, excellent for hybrid                          |
| Reranker             | CrossEncoder (sbert)          | Local, strong gains on technical text                               |
| LLM (gen + judge)    | Ollama first-class            | Privacy, cost, iteration speed. OpenAI path also supported          |
| Config               | YAML + Pydantic + env         | Humans edit YAML, machines override with env in prod/CI             |
| Chunk model          | Pydantic `Chunk`              | Typed, serializable, rich metadata as first-class citizen           |

We deliberately avoided a heavy framework lock-in (LlamaIndex / LangChain) for the *core* retrieval and chunking paths. You can still add them later for agents/tools/memory.

## 6. Parent-Child / Hierarchical Retrieval

When `retrieval.parent_child.enabled`:

1. The chunker assigns `section_id` / `parent_id` to chunks that belong together.
2. At retrieval time we pull a few neighboring chunks from the same section.
3. We slightly down-weight the expanded context so the originally retrieved precise chunk still ranks highest.

This pattern is one of the highest-ROI techniques for long-form technical content.

## 7. Future Directions (without breaking changes)

- Store the actual parent section text as a separate "document" level in Chroma (or a second collection) for even better expansion.
- Add figure description generation (using a vision model or caption + surrounding text).
- Make BM25 persistent (we currently rebuild it from the Chroma dump at query time — fine for a 100-page book).
- Add a `RetrievalStrategy` dataclass that can be serialized so you can say "run eval on config v3 vs v7".
- Multi-vector / late interaction (ColBERT-style) retriever.

---

The system is intentionally **boring and explicit** in the right places. That is the Anthropic-style engineering approach to RAG.
