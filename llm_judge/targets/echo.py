"""A trivial target used for tests and offline demos."""

from __future__ import annotations

from ..dataset import TestCase
from ..schema import Message, ModelSpec, Role, Step, Trace
from .base import Target


class EchoTarget(Target):
    """Echoes the input back, optionally with a fixed prefix.

    Useful to validate the pipeline plumbing without any model.
    """

    name = "echo"

    def __init__(self, prefix: str = "") -> None:
        self.prefix = prefix
        self.model = ModelSpec(name="echo", provider="mock")

    def run(self, case: TestCase) -> Trace:
        question = case.input if isinstance(case.input, str) else str(case.input)
        answer = f"{self.prefix}{question}"
        input_messages = case.input_messages()
        step = Step(
            index=0,
            name="echo",
            model=self.model,
            input_messages=input_messages,
            output_messages=[Message(role=Role.ASSISTANT, content=answer)],
            retrieved_context=list(case.context),
        )
        return Trace(
            model=self.model,
            case_id=case.id,
            initial_context=input_messages,
            steps=[step],
            final_output=answer,
        )
