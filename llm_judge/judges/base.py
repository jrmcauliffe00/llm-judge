"""Judge interface and criterion definition."""

from __future__ import annotations

import abc
from typing import Optional

from pydantic import BaseModel, Field

from ..dataset import TestCase
from ..results import CriterionScore, JudgeResult
from ..schema import Trace


class Criterion(BaseModel):
    """One dimension the judge grades on."""

    name: str
    description: str
    weight: float = 1.0
    # Pass threshold on the normalized 0..1 score.
    pass_threshold: float = 0.5


class Judge(abc.ABC):
    name: str = "judge"

    def __init__(self, criteria: Optional[list[Criterion]] = None) -> None:
        self.criteria = criteria or []

    @abc.abstractmethod
    def judge(self, trace: Trace, case: Optional[TestCase] = None) -> JudgeResult:
        """Evaluate a single trace."""
        raise NotImplementedError

    # Shared helper to fold criterion scores into an overall result.
    def _finalize(
        self,
        trace: Trace,
        scores: list[CriterionScore],
        summary: str = "",
        judge_model=None,
        raw: Optional[dict] = None,
    ) -> JudgeResult:
        total_w = sum(s.weight for s in scores) or 1.0
        overall = sum(s.score * s.weight for s in scores) / total_w
        # A trace passes if every weighted criterion clears its own bar.
        passed = all((s.passed is None) or s.passed for s in scores) and bool(scores)
        return JudgeResult(
            trace_id=trace.id,
            case_id=trace.case_id,
            judge_name=self.name,
            judge_model=judge_model,
            target_model=trace.model,
            scores=scores,
            overall=overall,
            passed=passed,
            summary=summary,
            raw=raw or {},
        )
