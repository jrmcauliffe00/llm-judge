"""The benchmark orchestrator: dataset x target x judge -> report.

This is the glue that runs your system over every test case, judges each run,
aggregates the scores, and asks the recommender what to fix.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Callable, Optional

from .dataset import Dataset
from .judges.base import Judge
from .recommend import recommend
from .results import BenchmarkReport, JudgeResult
from .schema import Trace
from .targets.base import Target

# Optional hook: called after each case with (case_id, trace, result).
Callback = Callable[[str, Trace, JudgeResult], None]


def run_benchmark(
    target: Target,
    judge: Judge,
    dataset: Dataset,
    name: Optional[str] = None,
    on_result: Optional[Callback] = None,
    max_recommendations: int = 8,
) -> BenchmarkReport:
    results: list[JudgeResult] = []

    for case in dataset:
        trace = target.run(case)
        if trace.case_id is None:
            trace.case_id = case.id
        result = judge.judge(trace, case)
        results.append(result)
        if on_result is not None:
            on_result(case.id, trace, result)

    return build_report(
        results,
        name=name or dataset.name,
        judge=judge,
        target=target,
        max_recommendations=max_recommendations,
    )


def build_report(
    results: list[JudgeResult],
    name: str,
    judge: Optional[Judge] = None,
    target: Optional[Target] = None,
    max_recommendations: int = 8,
) -> BenchmarkReport:
    n = len(results)
    n_passed = sum(1 for r in results if r.passed)
    mean_overall = sum(r.overall for r in results) / n if n else 0.0

    # Per-criterion means across all cases.
    crit_sum: dict[str, float] = defaultdict(float)
    crit_count: dict[str, int] = defaultdict(int)
    for r in results:
        for s in r.scores:
            crit_sum[s.name] += s.score
            crit_count[s.name] += 1
    criteria_means = {
        k: round(crit_sum[k] / crit_count[k], 3) for k in crit_sum
    }

    target_model = results[0].target_model if results else None
    judge_model = results[0].judge_model if results else None
    judge_name = results[0].judge_name if results else (judge.name if judge else "")

    return BenchmarkReport(
        name=name,
        target_model=target_model,
        judge_name=judge_name,
        judge_model=judge_model,
        n_cases=n,
        n_passed=n_passed,
        mean_overall=round(mean_overall, 3),
        criteria_means=criteria_means,
        results=results,
        recommendations=recommend(results, max_recommendations=max_recommendations),
    )
