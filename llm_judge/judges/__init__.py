"""Judges: evaluate a trace and produce a :class:`~llm_judge.results.JudgeResult`.

Two judges ship out of the box:

- :class:`HeuristicJudge` - fully offline, no LLM. Uses lexical-overlap and
  rule-based signals to score faithfulness/relevance/completeness and to guess
  attribution. Good default for fast local iteration and CI.
- :class:`RubricJudge`    - the classic *LLM-as-a-judge*. Model-aware: it's told
  which model produced the answer and asks the judge model to score against a
  rubric AND attribute failures to retrieval vs. static prompt vs. model.
"""

from __future__ import annotations

from typing import Optional

from ..config import Config, default_config
from .base import Criterion, Judge
from .heuristic import HeuristicJudge
from .rubric import DEFAULT_CRITERIA, RubricJudge

__all__ = [
    "Judge",
    "Criterion",
    "HeuristicJudge",
    "RubricJudge",
    "DEFAULT_CRITERIA",
    "build_judge",
]


def build_judge(name: str, config: Optional[Config] = None) -> Judge:
    """Factory used by the CLI. ``name`` is 'heuristic' or 'rubric'."""
    config = config or default_config
    name = name.lower()
    if name in ("heuristic", "offline"):
        return HeuristicJudge()
    if name in ("rubric", "llm", "llm-as-a-judge"):
        return RubricJudge(config=config)
    raise ValueError(f"Unknown judge '{name}'. Use 'heuristic' or 'rubric'.")
