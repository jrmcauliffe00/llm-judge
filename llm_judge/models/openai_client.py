"""OpenAI-compatible client.

Works against OpenAI proper and any OpenAI-compatible server (vLLM, Ollama's
``/v1`` endpoint, LM Studio, TGI, ...). The ``openai`` package is imported
lazily so it's an optional dependency: you only need it if you actually call a
real model.
"""

from __future__ import annotations

import time
from typing import Optional

from ..config import Config, default_config
from ..schema import Message, ModelSpec, Usage
from .base import LLMClient, LLMResponse


class OpenAIClient(LLMClient):
    def __init__(self, spec: ModelSpec, config: Optional[Config] = None) -> None:
        super().__init__(spec)
        self.config = config or default_config
        self._client = None  # lazy

    def _ensure_client(self):
        if self._client is not None:
            return self._client
        try:
            from openai import OpenAI
        except ImportError as e:  # pragma: no cover - exercised only w/ real use
            raise ImportError(
                "The 'openai' package is required for the OpenAI-compatible "
                "client. Install it with: pip install 'llm-judge[openai]'"
            ) from e

        self._client = OpenAI(
            api_key=self.config.openai_api_key or "not-needed-for-local",
            base_url=self.config.openai_base_url,
            timeout=self.config.request_timeout_s,
        )
        return self._client

    def complete(
        self,
        messages: list[Message],
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        json_mode: bool = False,
    ) -> LLMResponse:
        client = self._ensure_client()
        kwargs: dict = {
            "model": self.spec.name,
            "messages": [m.as_dict() for m in messages],
        }
        temp = temperature if temperature is not None else self.spec.temperature
        if temp is not None:
            kwargs["temperature"] = temp
        mt = max_tokens if max_tokens is not None else self.spec.max_tokens
        if mt is not None:
            kwargs["max_tokens"] = mt
        if self.spec.top_p is not None:
            kwargs["top_p"] = self.spec.top_p
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        t0 = time.time()
        resp = client.chat.completions.create(**kwargs)
        latency = time.time() - t0

        text = resp.choices[0].message.content or ""
        usage = Usage(latency_s=latency)
        if getattr(resp, "usage", None) is not None:
            usage.prompt_tokens = resp.usage.prompt_tokens
            usage.completion_tokens = resp.usage.completion_tokens
            usage.total_tokens = resp.usage.total_tokens

        return LLMResponse(
            text=text, usage=usage, raw=resp.model_dump() if hasattr(resp, "model_dump") else {}
        )
