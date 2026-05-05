"""End-to-end Retrieval-Augmented Generation pipeline."""

from __future__ import annotations

import os
from abc import ABC, abstractmethod

from config import (
    DEFAULT_GENERATION_MODEL,
    DEFAULT_GENERATION_PROVIDER,
    DEFAULT_PROMPT_TEMPLATE,
    DEFAULT_TOP_K,
)
from embeddings import EmbeddingModel
from models import RagResult
from prompts import render_prompt
from vector_store import FaissVectorStore


class GenerationModel(ABC):
    """Interface for text generation providers."""

    model_name: str

    @abstractmethod
    def generate(self, prompt: str) -> str:
        """Generate an answer for a fully rendered prompt."""


class APIChatGenerationModel(GenerationModel):
    """Chat generation through an API-compatible chat-completions endpoint."""

    def __init__(
        self,
        model_name: str,
        temperature: float = 0.0,
        base_url: str | None = None,
        api_key: str | None = None,
        api_key_env: str = "MODEL_API_KEY",
    ) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError("Install openai to use API-compatible generation.") from exc

        if not model_name:
            raise ValueError("model_name is required for API-compatible generation.")

        resolved_api_key = _resolve_api_key(api_key=api_key, api_key_env=api_key_env)
        self.model_name = model_name
        self.temperature = temperature
        self.base_url = base_url
        self.api_key_env = api_key_env
        self._client = OpenAI(api_key=resolved_api_key, base_url=base_url)

    def generate(self, prompt: str) -> str:
        response = self._client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
        )
        answer = response.choices[0].message.content
        return (answer or "").strip()


class LocalTransformersGenerationModel(GenerationModel):
    """Local Hugging Face text-generation model."""

    def __init__(
        self,
        model_name: str,
        max_new_tokens: int = 512,
        temperature: float = 0.0,
        device_map: str | None = None,
    ) -> None:
        try:
            from transformers import pipeline
        except ImportError as exc:
            raise ImportError("Install transformers and torch to use local generation.") from exc

        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self._pipeline = pipeline("text-generation", model=model_name, device_map=device_map)

    def generate(self, prompt: str) -> str:
        output = self._pipeline(
            prompt,
            max_new_tokens=self.max_new_tokens,
            do_sample=self.temperature > 0,
            temperature=max(self.temperature, 1e-6),
            return_full_text=False,
        )
        return str(output[0]["generated_text"]).strip()


class RagPipeline:
    """Coordinates query embedding, retrieval, prompting, and generation."""

    def __init__(
        self,
        vector_store: FaissVectorStore,
        embedding_model: EmbeddingModel,
        generation_model: GenerationModel,
        prompt_template: str = DEFAULT_PROMPT_TEMPLATE,
        top_k: int = DEFAULT_TOP_K,
    ) -> None:
        if top_k <= 0:
            raise ValueError("top_k must be positive.")

        self.vector_store = vector_store
        self.embedding_model = embedding_model
        self.generation_model = generation_model
        self.prompt_template = prompt_template
        self.top_k = top_k

    def answer(self, query: str) -> RagResult:
        """Run a complete RAG answer pass for a user query."""

        if not query.strip():
            raise ValueError("query cannot be empty.")

        sources = self.vector_store.search(query=query, embedding_model=self.embedding_model, top_k=self.top_k)
        prompt = render_prompt(query, sources, template_name=self.prompt_template)
        answer = self.generation_model.generate(prompt)
        return RagResult(answer=answer, sources=sources, prompt=prompt)


def build_generation_model(
    provider: str = DEFAULT_GENERATION_PROVIDER,
    model_name: str | None = None,
    temperature: float = 0.0,
    base_url: str | None = None,
    api_key: str | None = None,
    api_key_env: str = "MODEL_API_KEY",
) -> GenerationModel:
    """Create a generation provider from CLI/config values."""

    provider_key = provider.strip().lower()
    if provider_key in {"api", "api-compatible", "api_compatible", "openai-compatible", "openai_compatible", "openai"}:
        resolved_model_name = model_name or DEFAULT_GENERATION_MODEL
        if not resolved_model_name:
            raise ValueError("A --generation-model is required for API-compatible generation.")
        return APIChatGenerationModel(
            model_name=resolved_model_name,
            temperature=temperature,
            base_url=base_url,
            api_key=api_key,
            api_key_env=api_key_env,
        )
    if provider_key in {"transformers", "huggingface", "hf", "local"}:
        if not model_name:
            raise ValueError("A local transformers model_name is required for local generation.")
        return LocalTransformersGenerationModel(model_name=model_name, temperature=temperature)

    raise ValueError(f"Unsupported generation provider: {provider}")


def _resolve_api_key(api_key: str | None, api_key_env: str) -> str:
    return api_key or os.getenv(api_key_env) or os.getenv("OPENAI_API_KEY") or "unused"
