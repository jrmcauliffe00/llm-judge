"""Wrap any Python callable as a Target."""

from __future__ import annotations

from typing import Callable, Union

from ..dataset import TestCase
from ..schema import ModelSpec, Role, Trace
from .base import Target

# The wrapped fn may return a full Trace (preferred, via TraceRecorder) or just
# a final answer string (we'll build a minimal single-step trace for you).
RunFn = Callable[[TestCase], Union[Trace, str]]


class FunctionTarget(Target):
    def __init__(
        self,
        fn: RunFn,
        name: str = "function",
        model: ModelSpec | None = None,
    ) -> None:
        self.fn = fn
        self.name = name
        # Used only when the fn returns a bare string.
        self.model = model or ModelSpec(name="unknown", provider="unknown")

    def run(self, case: TestCase) -> Trace:
        result = self.fn(case)
        if isinstance(result, Trace):
            if result.case_id is None:
                result.case_id = case.id
            return result

        # Bare string -> minimal trace.
        trace = Trace(
            model=self.model,
            case_id=case.id,
            initial_context=case.input_messages(),
            final_output=str(result),
        )
        return trace
