"""
First-class evaluation harness.

Designed around the golden dataset that ships with rich retrieval signals
(expected_pages, must_retrieve_terms, expected_sections).

Supports:
- Pure retrieval metrics (no LLM needed)
- Answer quality via embedding similarity
- Optional LLM-as-judge (faithfulness, relevancy, correctness)
- Full artifacting of every decision for analysis
"""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Any

from loguru import logger

from rag.config import CONFIG, AppConfig
from rag.models import (
    EvaluationRecord,
    EvaluationRun,
    GoldenExample,
    RetrievedChunk,
)
from rag.observability.tracer import Tracer, setup_logging
from rag.retrieval.factory import ComposedRetriever
from rag.utils.io import ensure_dir, read_json, write_json

# Heavy deps are imported lazily inside methods that need them
# (allows using retrieval-only parts of the system without sentence-transformers installed)


def load_golden_dataset(path: str | Path) -> list[GoldenExample]:
    raw = read_json(path)
    examples = []
    for item in raw:
        examples.append(GoldenExample(**item))
    logger.info(f"Loaded {len(examples)} golden examples from {path}")
    return examples


class EvaluationHarness:
    def __init__(
        self,
        retriever: ComposedRetriever,
        embed_model: "SentenceTransformer",
        llm_client: Any | None = None,   # pluggable
        config: AppConfig | None = None,
    ):
        self.retriever = retriever
        self.embed_model = embed_model
        self.llm = llm_client
        self.cfg = config or CONFIG
        setup_logging(self.cfg)

    def evaluate_example(
        self, ex: GoldenExample, k: int | None = None
    ) -> EvaluationRecord:
        k = k or self.cfg.retrieval.top_k
        t0 = time.perf_counter()

        with Tracer(f"eval_{uuid.uuid4().hex[:8]}", query=ex.question) as trace:
            # 1. Retrieve
            resp = self.retriever.retrieve(ex.question, top_k=k)
            trace.final_retrieval = resp.to_result(ex.question) if hasattr(resp, "to_result") else None

            retrieved = resp.chunks

            # 2. (Optional) Generate an answer. For now we do a simple context-stuffed generation
            # if an LLM client is provided. Otherwise we use the top context as "answer" proxy.
            context = resp.to_result(ex.question).context_text(max_chunks=k) if hasattr(resp, "to_result") else ""
            generated = self._maybe_generate_answer(ex.question, context)

            trace.final_answer = generated

            # 3. Compute metrics
            metrics = self._compute_metrics(ex, retrieved, generated, k)

            latency = (time.perf_counter() - t0) * 1000

            record = EvaluationRecord(
                question=ex.question,
                ground_truth=ex.ground_truth,
                generated_answer=generated,
                retrieved_chunks=retrieved,
                metrics=metrics,
                latency_ms=latency,
            )

        return record

    def _maybe_generate_answer(self, question: str, context: str) -> str:
        if not self.llm:
            # No LLM: return a pseudo-answer that is just the top context (for retrieval-focused evals)
            return context[:2000]

        # Very simple RAG prompt. Production systems would use better templates + structured output.
        prompt = f"""You are an expert assistant on AI Agents.

Use ONLY the provided context to answer the question. Be precise and quote relevant terms.

Context:
{context}

Question: {question}

Answer:"""

        try:
            return self.llm.generate(prompt)
        except Exception as e:
            logger.warning(f"LLM generation failed: {e}")
            return context[:1500]

    def _compute_metrics(
        self,
        ex: GoldenExample,
        retrieved: list[RetrievedChunk],
        generated: str,
        k: int,
    ) -> dict[str, Any]:
        metrics: dict[str, Any] = {}

        retrieved_pages = {rc.chunk.page for rc in retrieved if rc.chunk.page}
        expected_pages = set(ex.expected_pages)

        # Retrieval page overlap (proxy for good chunk selection)
        page_recall = len(retrieved_pages & expected_pages) / max(1, len(expected_pages))
        metrics["page_recall"] = round(page_recall, 4)

        # Term recall (very strong signal for technical content)
        retrieved_text = " ".join(rc.chunk.text.lower() for rc in retrieved)
        term_hits = sum(1 for term in ex.must_retrieve_terms if term.lower() in retrieved_text)
        term_recall = term_hits / max(1, len(ex.must_retrieve_terms))
        metrics["term_recall"] = round(term_recall, 4)

        # Standard retrieval@K proxies (we treat "any hit in golden pages" as relevant for now)
        for kk in self.cfg.evaluation.k_values:
            top_k = retrieved[:kk]
            top_pages = {rc.chunk.page for rc in top_k if rc.chunk.page}
            hit = 1.0 if top_pages & expected_pages else 0.0
            metrics[f"hit@{kk}"] = hit

        # Answer similarity (embedding)
        if generated and ex.ground_truth:
            from sentence_transformers import util as _util
            emb_gen = self.embed_model.encode([generated], normalize_embeddings=True)
            emb_gt = self.embed_model.encode([ex.ground_truth], normalize_embeddings=True)
            sim = float(_util.cos_sim(emb_gen, emb_gt)[0][0])
            metrics["answer_similarity"] = round(sim, 4)

        # LLM-as-judge (optional, powerful)
        if self.cfg.evaluation.use_llm_as_judge and self.llm:
            judge_scores = self._llm_judge(ex.question, ex.ground_truth, generated, retrieved)
            metrics.update(judge_scores)

        return metrics

    def _llm_judge(
        self, question: str, ground_truth: str, answer: str, retrieved: list[RetrievedChunk]
    ) -> dict[str, float]:
        """Simple but effective LLM judge using the configured model."""
        context_preview = "\n".join([rc.chunk.text[:300] for rc in retrieved[:3]])

        prompt = f"""You are a strict evaluator for RAG systems.

Rate the following on a scale of 1-5 (1=terrible, 5=excellent).

Question: {question}

Ground Truth: {ground_truth}

Retrieved Context (first 3 chunks):
{context_preview}

Generated Answer:
{answer}

Return ONLY a compact JSON object:
{{"faithfulness": <1-5>, "answer_relevancy": <1-5>, "correctness": <1-5>}}"""

        try:
            raw = self.llm.generate(prompt, max_tokens=120, temperature=0.0)
            # Try to extract JSON
            import re, json as _json
            match = re.search(r"\{[\s\S]*?\}", raw)
            if match:
                scores = _json.loads(match.group(0))
                # Normalize to 0-1
                return {k: round(v / 5.0, 3) for k, v in scores.items() if isinstance(v, (int, float))}
        except Exception as e:
            logger.debug(f"Judge failed: {e}")

        return {}

    def run_full_evaluation(self, k: int | None = None) -> EvaluationRun:
        examples = load_golden_dataset(self.cfg.paths.golden_dataset)
        records: list[EvaluationRecord] = []

        for i, ex in enumerate(examples, 1):
            logger.info(f"Evaluating [{i}/{len(examples)}]: {ex.question[:70]}...")
            rec = self.evaluate_example(ex, k=k)
            records.append(rec)

        # Aggregate
        aggregates: dict[str, float] = {}
        numeric_keys = set()
        for r in records:
            for m, v in r.metrics.items():
                if isinstance(v, (int, float)):
                    numeric_keys.add(m)

        for key in numeric_keys:
            vals = [r.metrics[key] for r in records if key in r.metrics and isinstance(r.metrics[key], (int, float))]
            if vals:
                aggregates[key] = round(sum(vals) / len(vals), 4)

        run = EvaluationRun(
            run_id=f"eval_{uuid.uuid4().hex[:10]}",
            config_snapshot=self.cfg.model_dump(),
            records=records,
            aggregate_metrics=aggregates,
        )

        # Persist
        out_dir = ensure_dir(self.cfg.paths.artifacts_dir + "/evaluation_results")
        path = out_dir / f"{run.run_id}.json"
        write_json(path, run.model_dump() if hasattr(run, "model_dump") else run.__dict__)
        logger.success(f"Evaluation run saved to {path}")

        return run
