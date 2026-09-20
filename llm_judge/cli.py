"""Command-line interface: ``llm-judge``.

CLI-first by design - this is the primary way to run benchmarks. No frontend.

Examples
--------
    # Generate a starter dataset
    llm-judge init data/example.jsonl

    # Run fully offline (mock target + heuristic judge)
    llm-judge run --dataset data/example.jsonl

    # Benchmark a real model as the target, judged by an LLM judge
    export OPENAI_API_KEY=...
    llm-judge run --dataset data/example.jsonl \
        --target-model gpt-4o-mini --target-provider openai \
        --judge rubric --judge-model gpt-4o --judge-provider openai \
        --format markdown --out report.md
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

from . import __version__
from .benchmark import run_benchmark
from .config import Config
from .dataset import Dataset, TestCase
from .judges import build_judge
from .report import render
from .schema import ModelSpec
from .targets import EchoTarget, SimpleTarget
from .targets.base import Target

_EXAMPLE_CASES = [
    {
        "input": "What is the capital of France?",
        "reference": "Paris is the capital of France.",
        "context": [{"content": "France is a country in Europe. Its capital is Paris.", "source": "geo"}],
        "tags": ["rag", "factual"],
    },
    {
        "input": "Summarize the benefits of unit testing in one sentence.",
        "reference": "Unit testing catches bugs early, documents behavior, and enables safe refactoring.",
        "tags": ["summarization"],
    },
    {
        "input": "Who wrote the play Hamlet?",
        "reference": "William Shakespeare wrote Hamlet.",
        "context": [{"content": "Hamlet is a tragedy written by William Shakespeare around 1600.", "source": "lit"}],
        "tags": ["rag", "factual"],
    },
]


def _build_target(args) -> Target:
    if args.target_model:
        model = ModelSpec(name=args.target_model, provider=args.target_provider)
        return SimpleTarget(
            model=model,
            system_prompt=args.system_prompt,
            config=Config(),
        )
    # Offline default: echo target needs no model/keys.
    return EchoTarget(prefix=args.echo_prefix)


def _cmd_init(args) -> int:
    path = Path(args.path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cases = [TestCase(**c) for c in _EXAMPLE_CASES]
    ds = Dataset.from_cases(cases, name=path.stem)
    ds.save(path)
    print(f"Wrote {len(cases)} example cases to {path}")
    return 0


def _cmd_run(args) -> int:
    dataset = Dataset.load(args.dataset)
    config = Config()
    if args.judge_model:
        config.judge_model = args.judge_model
    if args.judge_provider:
        config.judge_provider = args.judge_provider

    target = _build_target(args)
    judge = build_judge(args.judge, config=config)

    traces_dir: Optional[Path] = Path(args.save_traces) if args.save_traces else None
    if traces_dir:
        traces_dir.mkdir(parents=True, exist_ok=True)

    def on_result(case_id, trace, result):
        if traces_dir:
            (traces_dir / f"{case_id}.trace.json").write_text(
                json.dumps(trace.model_dump(mode="json"), indent=2)
            )
            (traces_dir / f"{case_id}.judge.json").write_text(
                json.dumps(result.model_dump(mode="json"), indent=2)
            )
        if args.verbose:
            status = "PASS" if result.passed else "FAIL"
            print(f"[{status}] {case_id}  overall={result.overall:.3f}", file=sys.stderr)

    report = run_benchmark(
        target, judge, dataset, name=args.name or dataset.name, on_result=on_result
    )

    output = render(report, fmt=args.format)
    if args.out:
        Path(args.out).write_text(output)
        print(f"Wrote report to {args.out}")
    else:
        print(output)

    # Non-zero exit if a pass-rate threshold was set and not met (useful in CI).
    if args.fail_under is not None and report.pass_rate < args.fail_under:
        print(
            f"Pass rate {report.pass_rate:.2f} below threshold {args.fail_under:.2f}",
            file=sys.stderr,
        )
        return 1
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="llm-judge",
        description="Benchmark your LLM/agent output with LLM-as-a-judge.",
    )
    parser.add_argument("--version", action="version", version=f"llm-judge {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    # init
    p_init = sub.add_parser("init", help="Write a starter dataset.")
    p_init.add_argument("path", nargs="?", default="data/example.jsonl")
    p_init.set_defaults(func=_cmd_init)

    # run
    p_run = sub.add_parser("run", help="Run a benchmark.")
    p_run.add_argument("--dataset", required=True, help="Path to .jsonl/.json/.yaml dataset.")
    p_run.add_argument("--name", default=None, help="Report name.")
    p_run.add_argument(
        "--judge", default="heuristic", choices=["heuristic", "rubric"],
        help="Which judge to use (default: heuristic, fully offline).",
    )
    p_run.add_argument("--judge-model", default=None)
    p_run.add_argument("--judge-provider", default=None)
    # Target selection.
    p_run.add_argument("--target-model", default=None, help="If set, use a real model as the target.")
    p_run.add_argument("--target-provider", default="openai")
    p_run.add_argument(
        "--system-prompt",
        default="You are a helpful, accurate assistant.",
        help="Static system prompt for the SimpleTarget.",
    )
    p_run.add_argument("--echo-prefix", default="", help="Prefix for the offline echo target.")
    # Output.
    p_run.add_argument("--format", default="text", choices=["text", "markdown", "json"])
    p_run.add_argument("--out", default=None, help="Write report to a file instead of stdout.")
    p_run.add_argument("--save-traces", default=None, help="Directory to dump per-case traces + judgments.")
    p_run.add_argument("--fail-under", type=float, default=None, help="Exit non-zero if pass rate < this (0..1).")
    p_run.add_argument("-v", "--verbose", action="store_true")
    p_run.set_defaults(func=_cmd_run)

    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
