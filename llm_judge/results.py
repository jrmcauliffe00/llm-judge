"""Result types produced by judges and the recommender.

Kept separate from :mod:`llm_judge.schema` (which describes *what happened*)
because these describe *our evaluation of* what happened.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field

from .schema import ModelSpec


class Attribution(str, Enum):
    """Where a problem most likely originates.

    This is the crux of the whole project: separate failures caused by the
    *runtime retrieval / context* from failures caused by the *static prompt*
    (or the model itself).
    """

    RETRIEVAL = "retrieval"  # wrong/missing/noisy runtime context
    STATIC_PROMPT = "static_prompt"  # the fixed instructions / templates
    MODEL = "model"  # model capability / settings
    INPUT = "input"  # the user's request itself was the issue
    OTHER = "other"


class CriterionScore(BaseModel):
    """Score for a single evaluation criterion."""

    name: str  # e.g. "faithfulness", "relevance", "completeness"
    score: float  # normalized 0..1 (1 = best)
    passed: Optional[bool] = None
    weight: float = 1.0
    rationale: str = ""
    # If this criterion failed, where does the judge think the fault lies?
    attribution: Optional[Attribution] = None


class JudgeResult(BaseModel):
    """A single judge's evaluation of a single trace."""

    trace_id: str
    case_id: Optional[str] = None
    judge_name: str
    # The model that did the judging (None for heuristic/offline judges).
    judge_model: Optional[ModelSpec] = None
    # The model that was judged (copied from the trace for convenience).
    target_model: Optional[ModelSpec] = None

    scores: list[CriterionScore] = Field(default_factory=list)
    overall: float = 0.0  # weighted mean of criteria, 0..1
    passed: bool = False
    summary: str = ""
    # Raw judge output for debugging / auditing.
    raw: dict[str, Any] = Field(default_factory=dict)

    def score_for(self, name: str) -> Optional[CriterionScore]:
        for s in self.scores:
            if s.name == name:
                return s
        return None


class Recommendation(BaseModel):
    """An actionable improvement, attributed to a part of the pipeline."""

    attribution: Attribution
    severity: float  # 0..1, higher = more impactful
    finding: str  # what's wrong
    suggestion: str  # what to do about it
    # Supporting evidence: criteria / traces that motivated this.
    evidence: list[str] = Field(default_factory=list)
    affected_cases: list[str] = Field(default_factory=list)


class BenchmarkReport(BaseModel):
    """Aggregated results across an entire dataset run."""

    name: str = "benchmark"
    target_model: Optional[ModelSpec] = None
    judge_name: str = ""
    judge_model: Optional[ModelSpec] = None

    n_cases: int = 0
    n_passed: int = 0
    mean_overall: float = 0.0
    # Mean score per criterion across all cases.
    criteria_means: dict[str, float] = Field(default_factory=dict)

    results: list[JudgeResult] = Field(default_factory=list)
    recommendations: list[Recommendation] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def pass_rate(self) -> float:
        return self.n_passed / self.n_cases if self.n_cases else 0.0
