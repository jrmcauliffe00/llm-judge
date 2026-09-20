"""Core data model for capturing *everything* about an LLM/agent run.

The philosophy here is simple: before you can judge or improve an LLM system,
you have to be able to *see* it completely. This module defines the record
types that capture a full picture of a run:

- ``ModelSpec``     - which model (and settings) produced an output.
- ``Message``       - a single chat message (system/user/assistant/tool).
- ``RetrievedChunk``- a piece of dynamically-retrieved context (RAG).
- ``Step``          - one node/iteration of an agent, with the state that
                       flowed *in* and *out* (LangGraph-style), the static
                       prompt used, the retrieved context, and the model I/O.
- ``Trace``         - the full record of one run over one test case: the model,
                       the static prompts, the initial context window, every
                       step, and the final output.

These objects are the "ground truth" that judges score and that the
recommender mines for improvements to retrieval vs. static prompts.
"""

from __future__ import annotations

import time
import uuid
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


def _new_id() -> str:
    return uuid.uuid4().hex[:12]


class Role(str, Enum):
    """Standard chat roles."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class Message(BaseModel):
    """A single message in a conversation / context window."""

    role: Role
    content: str
    # Optional name (e.g. tool name, or a named participant).
    name: Optional[str] = None
    # Free-form metadata (token counts, timestamps, provenance, ...).
    metadata: dict[str, Any] = Field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        """Return an OpenAI-style ``{"role", "content"}`` dict."""
        d: dict[str, Any] = {"role": self.role.value, "content": self.content}
        if self.name:
            d["name"] = self.name
        return d


class ModelSpec(BaseModel):
    """Identifies a model and the settings used to run it.

    The judge is *model-aware*: recommendations differ depending on whether the
    output came from a small local model or a frontier model, so we always
    carry this alongside outputs.
    """

    name: str
    provider: str = "unknown"  # openai, anthropic, local, hf, ...
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    max_tokens: Optional[int] = None
    # Anything else worth recording (quantization, context length, revision).
    extra: dict[str, Any] = Field(default_factory=dict)

    def label(self) -> str:
        return f"{self.provider}/{self.name}"


class RetrievedChunk(BaseModel):
    """A single piece of dynamically retrieved context (RAG / tool result).

    Tracking retrieval separately from static prompts is what lets the
    recommender attribute problems to *retrieval* vs. *prompt engineering*.
    """

    content: str
    source: Optional[str] = None  # doc id, url, tool name, ...
    score: Optional[float] = None  # retriever similarity / rerank score
    metadata: dict[str, Any] = Field(default_factory=dict)


class ToolCall(BaseModel):
    """A tool/function invocation made during a step."""

    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    result: Optional[str] = None
    error: Optional[str] = None


class Usage(BaseModel):
    """Token / latency accounting for a step or trace."""

    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    latency_s: Optional[float] = None
    cost_usd: Optional[float] = None


class Step(BaseModel):
    """One node / iteration of an agent.

    For a looping or multi-node agent (think LangGraph), each pass through a
    node is a ``Step``. We capture the *input* the node saw, what it *produced*,
    the *static prompt* it used, the *retrieved context* it pulled at runtime,
    and the pipeline *state before and after* the node ran.
    """

    index: int
    name: str = "step"  # node name: planner, retriever, tool, responder, ...
    model: Optional[ModelSpec] = None

    # The static/templated portion of the prompt for this step (no runtime data).
    static_prompt: Optional[str] = None
    # Dynamically retrieved context injected at runtime.
    retrieved_context: list[RetrievedChunk] = Field(default_factory=list)

    # The exact messages fed into the model for this step (the full sub-window).
    input_messages: list[Message] = Field(default_factory=list)
    # What the step produced.
    output_messages: list[Message] = Field(default_factory=list)

    tool_calls: list[ToolCall] = Field(default_factory=list)

    # LangGraph-style state observation: the shared state going in and out.
    state_before: dict[str, Any] = Field(default_factory=dict)
    state_after: dict[str, Any] = Field(default_factory=dict)

    usage: Usage = Field(default_factory=Usage)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def output_text(self) -> str:
        """Concatenated assistant output text for this step."""
        return "\n".join(
            m.content for m in self.output_messages if m.role == Role.ASSISTANT
        )


class Trace(BaseModel):
    """The complete record of a single run of the system-under-test.

    This is the unit that a :class:`~llm_judge.judges.base.Judge` scores.
    """

    id: str = Field(default_factory=_new_id)
    case_id: Optional[str] = None  # the TestCase this run corresponds to

    # Primary model for the run (individual steps may override with their own).
    model: ModelSpec

    # Static prompts, keyed by name (e.g. {"system": "...", "planner": "..."}).
    # These are the things that DON'T change at runtime - prime targets for
    # prompt-engineering recommendations.
    static_prompts: dict[str, str] = Field(default_factory=dict)

    # The full context window at the very start of the run.
    initial_context: list[Message] = Field(default_factory=list)

    # Every step the agent took, in order.
    steps: list[Step] = Field(default_factory=list)

    # The final user-facing answer.
    final_output: str = ""

    usage: Usage = Field(default_factory=Usage)
    created_at: float = Field(default_factory=time.time)
    metadata: dict[str, Any] = Field(default_factory=dict)

    # --- convenience accessors -------------------------------------------------
    @property
    def first_user_input(self) -> str:
        for m in self.initial_context:
            if m.role == Role.USER:
                return m.content
        return ""

    @property
    def system_prompt(self) -> Optional[str]:
        if "system" in self.static_prompts:
            return self.static_prompts["system"]
        for m in self.initial_context:
            if m.role == Role.SYSTEM:
                return m.content
        return None

    @property
    def all_retrieved_context(self) -> list[RetrievedChunk]:
        chunks: list[RetrievedChunk] = []
        for step in self.steps:
            chunks.extend(step.retrieved_context)
        return chunks

    def add_step(self, step: Step) -> Step:
        self.steps.append(step)
        return step
