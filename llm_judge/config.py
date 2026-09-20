"""Runtime configuration, resolved from environment variables with sane
defaults so the whole pipeline runs offline out of the box.

Nothing here requires network access unless you explicitly opt into a real
provider (e.g. by setting ``LLM_JUDGE_JUDGE_PROVIDER=openai``).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional


def _env(*names: str, default: Optional[str] = None) -> Optional[str]:
    for n in names:
        v = os.environ.get(n)
        if v:
            return v
    return default


@dataclass
class Config:
    """Global defaults. Override via env vars or pass explicit specs in code."""

    # Judge (the model doing the grading).
    judge_provider: str = field(
        default_factory=lambda: _env("LLM_JUDGE_JUDGE_PROVIDER", default="mock")
    )
    judge_model: str = field(
        default_factory=lambda: _env("LLM_JUDGE_JUDGE_MODEL", default="mock-judge")
    )

    # Target (the model being tested), used when a target doesn't specify its own.
    target_provider: str = field(
        default_factory=lambda: _env("LLM_JUDGE_TARGET_PROVIDER", default="mock")
    )
    target_model: str = field(
        default_factory=lambda: _env("LLM_JUDGE_TARGET_MODEL", default="mock-target")
    )

    # OpenAI-compatible endpoint config (works for OpenAI, vLLM, Ollama, etc.).
    openai_api_key: Optional[str] = field(
        default_factory=lambda: _env("OPENAI_API_KEY", "LLM_JUDGE_API_KEY")
    )
    openai_base_url: Optional[str] = field(
        default_factory=lambda: _env(
            "LLM_JUDGE_BASE_URL", "OPENAI_BASE_URL"
        )
    )

    # Anthropic (Claude / Sonnet) API key.
    anthropic_api_key: Optional[str] = field(
        default_factory=lambda: _env("ANTHROPIC_API_KEY", "LLM_JUDGE_ANTHROPIC_API_KEY")
    )

    request_timeout_s: float = field(
        default_factory=lambda: float(_env("LLM_JUDGE_TIMEOUT", default="60"))
    )

    def summary(self) -> str:
        return (
            f"judge={self.judge_provider}/{self.judge_model} "
            f"target-default={self.target_provider}/{self.target_model}"
        )


# A module-level default others can import; construct fresh Config() when you
# need isolation (e.g. in tests).
default_config = Config()
