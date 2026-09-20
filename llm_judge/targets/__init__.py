"""Targets: the system-under-test.

A :class:`Target` takes a :class:`~llm_judge.dataset.TestCase` and produces a
:class:`~llm_judge.schema.Trace` - the full record of what your model/agent did.

Three ways to plug in your system:

1. :class:`SimpleTarget`  - a single LLM call with a static system prompt and
   (optional) retrieved context. Great for benchmarking RAG generation.
2. :class:`FunctionTarget`- wrap any Python callable. Use the provided
   :class:`TraceRecorder` inside agentic/LangGraph loops to capture every step
   and every state transition.
3. :class:`EchoTarget`    - trivial, dependency-free target for tests/demos.
"""

from __future__ import annotations

from .base import Target
from .echo import EchoTarget
from .function import FunctionTarget
from .recorder import TraceRecorder
from .simple import SimpleTarget

__all__ = [
    "Target",
    "EchoTarget",
    "FunctionTarget",
    "TraceRecorder",
    "SimpleTarget",
]
