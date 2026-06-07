"""
Pluggable LLM client for answer generation and LLM-as-judge.

Current implementations:
- Ollama (local, recommended)
- OpenAI-compatible
- Dummy (no-op, returns context only)
"""

from __future__ import annotations

import os
from typing import Protocol

import requests
from loguru import logger

from rag.config import CONFIG, AppConfig


class LLMClient(Protocol):
    def generate(self, prompt: str, **kwargs) -> str: ...


class OllamaClient:
    def __init__(self, base_url: str, model: str, timeout: int = 120):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout

    def generate(self, prompt: str, **kwargs) -> str:
        url = f"{self.base_url}/api/generate"
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": kwargs.get("temperature", CONFIG.llm.temperature),
                "num_predict": kwargs.get("max_tokens", CONFIG.llm.max_tokens),
            },
        }
        try:
            r = requests.post(url, json=payload, timeout=self.timeout)
            r.raise_for_status()
            data = r.json()
            return data.get("response", "").strip()
        except Exception as e:
            logger.error(f"Ollama error: {e}")
            return ""


class OpenAICompatibleClient:
    def __init__(self, api_key: str, base_url: str, model: str):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        try:
            import openai  # type: ignore
            self.client = openai.OpenAI(api_key=api_key, base_url=base_url)
        except ImportError:
            self.client = None

    def generate(self, prompt: str, **kwargs) -> str:
        if not self.client:
            return ""
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=kwargs.get("temperature", 0.1),
                max_tokens=kwargs.get("max_tokens", 800),
            )
            return resp.choices[0].message.content or ""
        except Exception as e:
            logger.warning(f"OpenAI-compatible error: {e}")
            return ""


class DummyClient:
    """Returns the prompt tail or empty. Useful for pure retrieval evals."""

    def generate(self, prompt: str, **kwargs) -> str:
        # Crude extraction of "context" part if present
        if "Context:" in prompt:
            return prompt.split("Context:")[-1][:1500]
        return prompt[-800:]


def get_llm_client(config: AppConfig | None = None) -> LLMClient:
    cfg = config or CONFIG
    provider = cfg.llm.provider.lower()

    if provider == "ollama":
        return OllamaClient(
            base_url=cfg.llm.base_url,
            model=cfg.llm.model,
            timeout=cfg.llm.timeout,
        )
    elif provider == "openai":
        key = os.getenv("OPENAI_API_KEY", "")
        return OpenAICompatibleClient(
            api_key=key,
            base_url=cfg.llm.base_url or "https://api.openai.com/v1",
            model=cfg.llm.model,
        )
    else:
        logger.warning("No LLM provider configured. Using DummyClient.")
        return DummyClient()
