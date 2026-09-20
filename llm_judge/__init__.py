"""llm-judge: an easy, CLI-first harness for benchmarking your LLM output.

Capture a full picture of a run (model, initial context, per-step inputs/outputs,
state transitions, static prompts, retrieved context), judge it, and get ranked
recommendations that separate *retrieval* problems from *static-prompt* problems.

Quick start (Python)::

    from llm_judge import Dataset, TestCase, EchoTarget, HeuristicJudge, run_benchmark, render

    ds = Dataset.from_cases([TestCase(input="What is 2+2?", reference="4")])
    report = run_benchmark(EchoTarget(), HeuristicJudge(), ds)
    print(render(report))

Or from the CLI::

    llm-judge run --dataset data/example.jsonl --judge heuristic
"""

from __future__ import annotations

from .benchmark import build_report, run_benchmark
from .config import Config, default_config
from .dataset import Dataset, TestCase
from .judges import Criterion, HeuristicJudge, Judge, RubricJudge, build_judge
from .rubric import Rubric, UseCaseRubric
from .recommend import attribution_breakdown, recommend
from .report import render, render_json, render_markdown, render_text
from .results import (
    Attribution,
    BenchmarkReport,
    CriterionScore,
    JudgeResult,
    Recommendation,
)
from .schema import (
    Message,
    ModelSpec,
    RetrievedChunk,
    Role,
    Step,
    ToolCall,
    Trace,
    Usage,
)
from .targets import (
    EchoTarget,
    FunctionTarget,
    SimpleTarget,
    Target,
    TraceRecorder,
)

__version__ = "0.1.0"

__all__ = [
    # config
    "Config",
    "default_config",
    # schema
    "Message",
    "ModelSpec",
    "RetrievedChunk",
    "Role",
    "Step",
    "ToolCall",
    "Trace",
    "Usage",
    # dataset
    "Dataset",
    "TestCase",
    # targets
    "Target",
    "EchoTarget",
    "FunctionTarget",
    "SimpleTarget",
    "TraceRecorder",
    # judges
    "Judge",
    "Criterion",
    "HeuristicJudge",
    "RubricJudge",
    "Rubric",
    "UseCaseRubric",
    "build_judge",
    # results
    "Attribution",
    "BenchmarkReport",
    "CriterionScore",
    "JudgeResult",
    "Recommendation",
    # recommend
    "recommend",
    "attribution_breakdown",
    # benchmark
    "run_benchmark",
    "build_report",
    # report
    "render",
    "render_text",
    "render_markdown",
    "render_json",
    "__version__",
]
