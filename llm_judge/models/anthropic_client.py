"""Anthropic (Claude / Sonnet) client.

Anthropic's API isn't OpenAI-shaped: the system prompt is a top-level argument
(not a message), ``max_tokens`` is required, and the response content is a list
of blocks. This adapter normalizes all that to the same :class:`LLMClient`
interface used everywhere else, so Claude can be a *target*, a *judge*, or both.

The ``anthropic`` package is imported lazily - it's only needed if you actually
call Claude.
"""

from __future__ import annotations

import time
from typing import Optional

from ..config import Config, default_config
from ..schema import Message, ModelSpec, Role, Usage
from .base import LLMClient, LLMResponse

# Anthropic requires an explicit max_tokens; use a sane default if unset.
_DEFAULT_MAX_TOKENS = 1024


class AnthropicClient(LLMClient):
    def __init__(self, spec: ModelSpec, config: Optional[Config] = None) -> None:
        super().__init__(spec)
        self.config = config or default_config
        self._client = None  # lazy

    def _ensure_client(self):
        if self._client is not None:
            return self._client
        try:
            from anthropic import Anthropic
        except ImportError as e:  # pragma: no cover - exercised only w/ real use
            raise ImportError(
                "The 'anthropic' package is required for the Anthropic client. "
                "Install it with: pip install 'llm-judge[anthropic]'"
            ) from e

        self._client = Anthropic(
            api_key=self.config.anthropic_api_key,
            timeout=self.config.request_timeout_s,
        )
        return self._client

    @staticmethod
    def _split_messages(messages: list[Message]) -> tuple[str, list[dict]]:
        """Pull system messages out into a single system string; map the rest to
        Anthropic's user/assistant turns (tool messages are folded into user)."""
        system_parts: list[str] = []
        turns: list[dict] = []
        for m in messages:
            if m.role == Role.SYSTEM:
                system_parts.append(m.content)
            elif m.role == Role.ASSISTANT:
                turns.append({"role": "assistant", "content": m.content})
            else:  # USER or TOOL
                content = m.content
                if m.role == Role.TOOL:
                    content = f"[tool:{m.name or 'result'}] {content}"
                turns.append({"role": "user", "content": content})
        return "\n\n".join(system_parts), turns

    def complete(
        self,
        messages: list[Message],
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        json_mode: bool = False,
    ) -> LLMResponse:
        client = self._ensure_client()
        system, turns = self._split_messages(messages)

        # Anthropic has no strict JSON mode; nudge via the system prompt instead.
        if json_mode:
            system = (system + "\n\n" if system else "") + (
                "Respond with a single valid JSON object and nothing else."
            )

        kwargs: dict = {
            "model": self.spec.name,
            "messages": turns,
            "max_tokens": max_tokens
            or self.spec.max_tokens
            or _DEFAULT_MAX_TOKENS,
        }
        if system:
            kwargs["system"] = system
        temp = temperature if temperature is not None else self.spec.temperature
        if temp is not None:
            kwargs["temperature"] = temp
        if self.spec.top_p is not None:
            kwargs["top_p"] = self.spec.top_p

        t0 = time.time()
        resp = client.messages.create(**kwargs)
        latency = time.time() - t0

        # content is a list of blocks; concatenate text blocks.
        text = "".join(
            getattr(block, "text", "") for block in resp.content
        )
        usage = Usage(latency_s=latency)
        if getattr(resp, "usage", None) is not None:
            usage.prompt_tokens = resp.usage.input_tokens
            usage.completion_tokens = resp.usage.output_tokens
            usage.total_tokens = resp.usage.input_tokens + resp.usage.output_tokens

        return LLMResponse(
            text=text,
            usage=usage,
            raw=resp.model_dump() if hasattr(resp, "model_dump") else {},
        )
