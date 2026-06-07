"""
Central configuration for the RAG system.

Loads from:
1. config/default.yaml (or CONFIG_PATH)
2. Environment variables (highest precedence)
3. Direct overrides (for experiments)

All downstream modules should import from here rather than hard-coding.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class ChunkingConfig(BaseModel):
    max_tokens: int = 450
    overlap_tokens: int = 60
    preserve_sections: bool = True
    table_as_single_chunk: bool = True
    code_as_single_chunk: bool = True
    list_as_single_chunk: bool = False
    heading_font_size_threshold: float = 1.25


class EmbeddingConfig(BaseModel):
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    normalize_embeddings: bool = True
    batch_size: int = 32
    device: str | None = None  # auto if None


class VectorStoreConfig(BaseModel):
    collection_name: str = "ai_agents_guidebook_v1"
    distance: Literal["cosine", "l2", "ip"] = "cosine"


class HybridConfig(BaseModel):
    enabled: bool = True
    alpha: float = 0.65  # weight dense (1.0 = pure dense)
    rrf_k: int = 60


class RerankerConfig(BaseModel):
    enabled: bool = True
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    top_k_before_rerank: int = 25


class ParentChildConfig(BaseModel):
    enabled: bool = True
    parent_expansion: int = 1


class RetrievalConfig(BaseModel):
    top_k: int = 8
    hybrid: HybridConfig = Field(default_factory=HybridConfig)
    reranker: RerankerConfig = Field(default_factory=RerankerConfig)
    parent_child: ParentChildConfig = Field(default_factory=ParentChildConfig)


class LLMConfig(BaseModel):
    provider: Literal["ollama", "openai", "none"] = "ollama"
    model: str = "gemma2:9b"
    base_url: str = "http://localhost:11434"
    temperature: float = 0.1
    max_tokens: int = 1200
    timeout: int = 120


class EvaluationConfig(BaseModel):
    metrics: list[str] = Field(
        default_factory=lambda: [
            "retrieval_recall_at_k",
            "retrieval_precision_at_k",
            "context_relevance_proxy",
            "answer_similarity",
        ]
    )
    use_llm_as_judge: bool = True
    judge_metrics: list[str] = Field(
        default_factory=lambda: ["faithfulness", "answer_relevancy", "correctness"]
    )
    k_values: list[int] = Field(default_factory=lambda: [3, 5, 8, 10, 15])


class ObservabilityConfig(BaseModel):
    log_level: str = "INFO"
    trace_dir: str = "artifacts/traces"
    save_retrieval_details: bool = True
    save_llm_prompts: bool = False


class PathsConfig(BaseModel):
    pdf: str = "data/raw/ai_agents_guidebook.pdf"
    vectorstore_dir: str = "data/vectorstore"
    golden_dataset: str = "data/evaluation/golden_dataset.json"
    artifacts_dir: str = "artifacts"


class AppConfig(BaseModel):
    paths: PathsConfig = Field(default_factory=PathsConfig)
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    vectorstore: VectorStoreConfig = Field(default_factory=VectorStoreConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    observability: ObservabilityConfig = Field(default_factory=ObservabilityConfig)

    @classmethod
    def from_yaml(cls, path: str | Path | None = None) -> AppConfig:
        if path is None:
            path = os.getenv("CONFIG_PATH", "config/default.yaml")

        path = Path(path)
        if not path.exists():
            # Fallback to defaults if no yaml present
            return cls()

        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        return cls(**data)

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump()


class Settings(BaseSettings):
    """
    Environment-driven overrides (highest precedence).
    Use this for secrets and deployment-specific values.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    embedding_model: str | None = None
    llm_provider: str | None = None
    llm_model: str | None = None
    ollama_base_url: str | None = None
    vectorstore_dir: str | None = None
    golden_dataset_path: str | None = None
    log_level: str | None = None

    def apply_overrides(self, config: AppConfig) -> AppConfig:
        """Mutate a copy of AppConfig with env overrides (non-destructive)."""
        c = config.model_copy(deep=True)

        if self.embedding_model:
            c.embedding.model_name = self.embedding_model
        if self.llm_provider:
            c.llm.provider = self.llm_provider  # type: ignore
        if self.llm_model:
            c.llm.model = self.llm_model
        if self.ollama_base_url:
            c.llm.base_url = self.ollama_base_url
        if self.vectorstore_dir:
            c.paths.vectorstore_dir = self.vectorstore_dir
        if self.golden_dataset_path:
            c.paths.golden_dataset = self.golden_dataset_path
        if self.log_level:
            c.observability.log_level = self.log_level

        return c


def load_config(yaml_path: str | Path | None = None) -> AppConfig:
    """Convenience loader used everywhere in the project."""
    cfg = AppConfig.from_yaml(yaml_path)
    settings = Settings()
    return settings.apply_overrides(cfg)


# Singleton for convenience in scripts/notebooks (override via load_config() when needed)
CONFIG: AppConfig = load_config()
