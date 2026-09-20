"""LLM client abstraction.

A single tiny interface (:class:`LLMClient`) so targets and judges don't care
whether they're talking to OpenAI, a local vLLM/Ollama server, or the built-in
offline mock.
"""

from __future__ import annotations

from typing import Optional

from ..config import Config, default_config
from ..schema import ModelSpec
from .base import LLMClient, LLMResponse
from .mock import MockClient
from .openai_client import OpenAIClient

__all__ = [
    "LLMClient",
    "LLMResponse",
    "MockClient",
    "OpenAIClient",
    "build_client",
]


def build_client(spec: ModelSpec, config: Optional[Config] = None) -> LLMClient:
    """Instantiate a client for the given model spec.

    Provider ``mock`` (the default) needs no network access, which keeps the
    whole pipeline runnable offline and in CI.
    """
    config = config or default_config
    provider = spec.provider.lower()

    if provider in ("mock", "offline", "unknown"):
        return MockClient(spec)
    if provider in ("openai", "vllm", "ollama", "openai-compatible", "local"):
        return OpenAIClient(spec, config=config)
    raise ValueError(
        f"Unknown provider '{spec.provider}'. Use 'mock', 'openai', or an "
        f"OpenAI-compatible provider (vllm/ollama/local)."
    )
