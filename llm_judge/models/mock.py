"""A deterministic, offline mock client.

Lets the entire pipeline (targets, judges, recommender, CLI) run with zero
network access or API keys. Tests inject a custom ``responder`` to simulate a
real judge model returning structured JSON.
"""

from __future__ import annotations

from typing import Callable, Optional

from ..schema import Message, ModelSpec, Role, Usage
from .base import LLMClient, LLMResponse

# A responder maps (messages, json_mode) -> reply text.
Responder = Callable[[list[Message], bool], str]


def _default_responder(messages: list[Message], json_mode: bool) -> str:
    """Deterministic, dependency-free default behavior."""
    if json_mode:
        # No opinion offline; callers should fall back to neutral handling.
        return "{}"
    last_user = ""
    for m in reversed(messages):
        if m.role == Role.USER:
            last_user = m.content
            break
    # A boring but stable echo so downstream code has something to chew on.
    return f"[mock:{last_user.strip()[:120]}]" if last_user else "[mock]"


class MockClient(LLMClient):
    def __init__(
        self, spec: ModelSpec, responder: Optional[Responder] = None
    ) -> None:
        super().__init__(spec)
        self._responder = responder or _default_responder

    def complete(
        self,
        messages: list[Message],
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        json_mode: bool = False,
    ) -> LLMResponse:
        text = self._responder(messages, json_mode)
        # Rough token estimate: ~4 chars/token.
        prompt_chars = sum(len(m.content) for m in messages)
        usage = Usage(
            prompt_tokens=prompt_chars // 4,
            completion_tokens=len(text) // 4,
            total_tokens=(prompt_chars + len(text)) // 4,
            latency_s=0.0,
        )
        return LLMResponse(text=text, usage=usage, raw={"mock": True})
