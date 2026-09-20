"""LLM client abstraction.

A single tiny interface (:class:`LLMClient`) so targets and judges don't care
*which* provider they talk to. The target and the judge are fully independent:
you can benchmark Claude Sonnet output and judge it with GPT-4o (or vice versa,
or a local model), each with its own model + API key, chosen at runtime.

The provider is resolved from ``ModelSpec.provider``. Adding a new backend is
just implementing :class:`LLMClient` and registering it in :func:`build_client`.
"""

from __future__ import annotations

from typing import Optional

from ..config import Config, default_config
from ..schema import ModelSpec
from .anthropic_client import AnthropicClient
from .base import LLMClient, LLMResponse
from .mock import MockClient
from .openai_client import OpenAIClient

__all__ = [
    "LLMClient",
    "LLMResponse",
    "MockClient",
    "OpenAIClient",
    "AnthropicClient",
    "build_client",
]

# Provider aliases -> backend. "openai" is really "OpenAI-compatible" and covers
# any server that speaks that API (vLLM, Ollama, LM Studio, TGI, OpenRouter...).
_OPENAI_COMPATIBLE = {"openai", "vllm", "ollama", "openai-compatible", "local", "lmstudio", "together", "groq", "openrouter"}
_ANTHROPIC = {"anthropic", "claude", "sonnet"}
_MOCK = {"mock", "offline", "unknown"}


def build_client(spec: ModelSpec, config: Optional[Config] = None) -> LLMClient:
    """Instantiate a client for the given model spec.

    Provider ``mock`` (the default) needs no network access, which keeps the
    whole pipeline runnable offline and in CI.
    """
    config = config or default_config
    provider = spec.provider.lower()

    if provider in _MOCK:
        return MockClient(spec)
    if provider in _ANTHROPIC:
        return AnthropicClient(spec, config=config)
    if provider in _OPENAI_COMPATIBLE:
        return OpenAIClient(spec, config=config)
    raise ValueError(
        f"Unknown provider '{spec.provider}'. Use 'mock', 'anthropic', "
        f"'openai', or any OpenAI-compatible provider (vllm/ollama/local/...)."
    )
