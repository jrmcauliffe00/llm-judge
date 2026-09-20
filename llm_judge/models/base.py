"""The minimal LLM client interface used throughout the codebase."""

from __future__ import annotations

import abc
from typing import Any, Optional

from pydantic import BaseModel

from ..schema import Message, ModelSpec, Usage


class LLMResponse(BaseModel):
    """A normalized chat-completion response."""

    text: str
    usage: Usage = Usage()
    raw: dict[str, Any] = {}


class LLMClient(abc.ABC):
    """Talk to a chat model. Implementations must be side-effect free besides
    the network/compute call itself."""

    def __init__(self, spec: ModelSpec) -> None:
        self.spec = spec

    @abc.abstractmethod
    def complete(
        self,
        messages: list[Message],
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        json_mode: bool = False,
    ) -> LLMResponse:
        """Return a completion for the given conversation.

        ``json_mode`` is a hint that the caller expects strict JSON output
        (used by the LLM-as-a-judge). Implementations should honor it when the
        backend supports it.
        """
        raise NotImplementedError
