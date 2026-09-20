"""A small helper to instrument your own agent code and produce a Trace.

Drop this into an agentic/LangGraph-style loop to capture, for every node:
the static prompt used, the runtime-retrieved context, the exact model input,
the model output, and the shared state *before* and *after* the node ran.

Example
-------
    rec = TraceRecorder(model=ModelSpec(name="gpt-4o", provider="openai"),
                        case_id=case.id)
    rec.set_static_prompt("system", SYSTEM_PROMPT)
    rec.set_initial_context(messages)

    state = {"question": case.input}
    for node in graph:
        with rec.step(node.name, state_before=state) as step:
            step.static_prompt = node.prompt_template
            step.retrieved_context = node.retrieve(state)
            step.input_messages = node.build_messages(state)
            out = node.call_model(step.input_messages)
            step.add_output(out)
            state = node.update_state(state, out)
            step.state_after = state

    trace = rec.finish(final_output=state["answer"])
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator, Optional

from ..schema import Message, ModelSpec, RetrievedChunk, Role, Step, Trace


class StepBuilder:
    """Mutable view of a :class:`~llm_judge.schema.Step` while it's being built."""

    def __init__(self, step: Step) -> None:
        self._step = step

    @property
    def step(self) -> Step:
        return self._step

    # Ergonomic setters -------------------------------------------------------
    @property
    def static_prompt(self) -> Optional[str]:
        return self._step.static_prompt

    @static_prompt.setter
    def static_prompt(self, value: str) -> None:
        self._step.static_prompt = value

    @property
    def retrieved_context(self) -> list[RetrievedChunk]:
        return self._step.retrieved_context

    @retrieved_context.setter
    def retrieved_context(self, chunks: list[RetrievedChunk]) -> None:
        self._step.retrieved_context = chunks

    @property
    def input_messages(self) -> list[Message]:
        return self._step.input_messages

    @input_messages.setter
    def input_messages(self, msgs: list[Message]) -> None:
        self._step.input_messages = msgs

    @property
    def state_after(self) -> dict[str, Any]:
        return self._step.state_after

    @state_after.setter
    def state_after(self, state: dict[str, Any]) -> None:
        self._step.state_after = dict(state)

    def add_output(self, content: str, role: Role = Role.ASSISTANT) -> None:
        self._step.output_messages.append(Message(role=role, content=content))


class TraceRecorder:
    def __init__(
        self,
        model: ModelSpec,
        case_id: Optional[str] = None,
        static_prompts: Optional[dict[str, str]] = None,
    ) -> None:
        self._trace = Trace(
            model=model,
            case_id=case_id,
            static_prompts=dict(static_prompts or {}),
        )
        self._counter = 0

    # Setup -------------------------------------------------------------------
    def set_static_prompt(self, name: str, prompt: str) -> None:
        self._trace.static_prompts[name] = prompt

    def set_initial_context(self, messages: list[Message]) -> None:
        self._trace.initial_context = list(messages)

    # Per-step capture --------------------------------------------------------
    @contextmanager
    def step(
        self,
        name: str = "step",
        state_before: Optional[dict[str, Any]] = None,
        model: Optional[ModelSpec] = None,
    ) -> Iterator[StepBuilder]:
        step = Step(
            index=self._counter,
            name=name,
            model=model,
            state_before=dict(state_before or {}),
        )
        self._counter += 1
        builder = StepBuilder(step)
        try:
            yield builder
        finally:
            # Default state_after to state_before if the caller didn't set it.
            if not step.state_after:
                step.state_after = dict(step.state_before)
            self._trace.add_step(step)

    # Finish ------------------------------------------------------------------
    def finish(self, final_output: str, **metadata: Any) -> Trace:
        self._trace.final_output = final_output
        if metadata:
            self._trace.metadata.update(metadata)
        return self._trace

    @property
    def trace(self) -> Trace:
        return self._trace
