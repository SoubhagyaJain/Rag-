"""
RAG System for "AI Agents: The Illustrated Guidebook (2025 Edition)"

A clean, modular, evaluation-driven Retrieval-Augmented Generation system
with first-class support for richly illustrated technical documentation.

Key design principles:
- Structure-aware chunking (tables, code, diagrams, sections preserved)
- Composable retrieval (dense + sparse + rerank + parent-child)
- Evaluation harness using golden questions + rich ground truth metadata
- Observability by default (traces, structured logs)
- Easy to experiment and extend without major rewrites
"""

__version__ = "0.1.0"
