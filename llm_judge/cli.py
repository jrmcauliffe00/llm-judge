"""Command-line interface: ``llm-judge``.

CLI-first by design - this is the primary way to run benchmarks. No frontend.

Examples
--------
    # Generate a starter dataset
    llm-judge init data/example.jsonl

    # Write the project process, then unique (gitignored) use-case rules
    llm-judge rubric init
    llm-judge rubric add --name task --description "Did it do the job?"
    llm-judge rubric add --case hamlet-author --name author --description "Name Shakespeare"

    # Run using that process + {expected} answers in the dataset
    llm-judge run --dataset data/example.jsonl --rubric data/rubric.json
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
from .judges.base import Criterion
from .report import render
from .rubric import (
    DEFAULT_PROJECT_PATH,
    Rubric,
    UseCaseRubric,
    delete_use_case_file,
    load_use_case,
    save_use_case,
    use_cases_dir,
)
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

    rubric = (
        Rubric.load_project(args.rubric, cases_dir=args.use_cases)
        if args.rubric
        else None
    )
    judge_name = args.judge
    if judge_name is None:
        judge_name = "rubric" if rubric is not None else "heuristic"
    if rubric is not None and judge_name in {"heuristic", "offline"}:
        print(
            "A rubric is the process for the rubric judge. "
            "Drop --judge heuristic, or omit --judge.",
            file=sys.stderr,
        )
        return 2

    target = _build_target(args)
    judge = build_judge(judge_name, config=config, rubric=rubric)

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


def _project_path(args) -> Path:
    return Path(args.path)


def _cases_dir(args) -> Path:
    if getattr(args, "use_cases", None):
        return Path(args.use_cases)
    return use_cases_dir(_project_path(args))


def _criterion_from_args(args, existing: Optional[Criterion]) -> Optional[Criterion]:
    if existing is None and not args.description:
        print(
            "--description is required when adding a new criterion.",
            file=sys.stderr,
        )
        return None
    return Criterion(
        name=args.name,
        description=(
            args.description if args.description is not None else existing.description
        ),
        weight=args.weight if args.weight is not None else (existing.weight if existing else 1.0),
        pass_threshold=(
            args.threshold
            if args.threshold is not None
            else (existing.pass_threshold if existing else 0.5)
        ),
    )


def _cmd_rubric_init(args) -> int:
    path = _project_path(args)
    if path.exists() and not args.force:
        print(f"{path} already exists. Pass --force to overwrite.", file=sys.stderr)
        return 1
    Rubric().save(path)
    cases = _cases_dir(args)
    cases.mkdir(parents=True, exist_ok=True)
    print(f"Wrote project process to {path}")
    print(f"Use-case rules go in {cases}/ (gitignored). Same shape as data/use-case.example.json")
    print(f"Add a project rule:  llm-judge rubric add --name task --description '...'")
    print(f"Add a use-case rule: llm-judge rubric add --case CASE_ID --name author --description '...'")
    return 0


def _cmd_rubric_add(args) -> int:
    path = _project_path(args)
    if args.case:
        existing_file = load_use_case(path, args.case, cases_dir=_cases_dir(args))
        use_case = existing_file or UseCaseRubric(id=args.case)
        existing = use_case.get(args.name)
        criterion = _criterion_from_args(args, existing)
        if criterion is None:
            return 2
        action = use_case.add(criterion)
        saved = save_use_case(path, use_case, cases_dir=_cases_dir(args))
        print(f"{action} '{args.name}' on use case {args.case} -> {saved}")
        return 0

    rubric = Rubric.load(path) if path.exists() else Rubric()
    existing = rubric.get(args.name)
    criterion = _criterion_from_args(args, existing)
    if criterion is None:
        return 2
    action = rubric.add(criterion)
    rubric.save(path)
    print(f"{action} '{args.name}' on project process {path}")
    return 0


def _cmd_rubric_remove(args) -> int:
    path = _project_path(args)
    if args.case:
        use_case = load_use_case(path, args.case, cases_dir=_cases_dir(args))
        if use_case is None or not use_case.remove(args.name):
            print(
                f"No criterion named '{args.name}' on use case {args.case}.",
                file=sys.stderr,
            )
            return 1
        if use_case.criteria:
            save_use_case(path, use_case, cases_dir=_cases_dir(args))
        else:
            delete_use_case_file(path, args.case, cases_dir=_cases_dir(args))
        print(f"removed '{args.name}' from use case {args.case}")
        return 0

    if not path.exists():
        print(f"No project process at {path}", file=sys.stderr)
        return 1
    rubric = Rubric.load(path)
    if not rubric.remove(args.name):
        print(f"No criterion named '{args.name}' on the project process.", file=sys.stderr)
        return 1
    rubric.save(path)
    print(f"removed '{args.name}' from project process {path}")
    return 0


def _cmd_rubric_list(args) -> int:
    path = _project_path(args)
    cases = _cases_dir(args)
    if not path.exists() and not cases.exists():
        print(f"No project process at {path}", file=sys.stderr)
        return 1

    loaded = Rubric.load_project(path, cases_dir=cases)
    print(f"Project process ({path})")
    if loaded.criteria:
        for c in loaded.criteria:
            print(f"  {c.name:<16} {c.description}")
    else:
        print("  (none — llm-judge rubric add --name ... --description '...')")

    print(f"\nUse-case rules ({cases}/, gitignored)")
    if loaded.cases:
        for case_id, crits in loaded.cases.items():
            print(f"  {case_id}")
            for c in crits:
                print(f"    {c.name:<14} {c.description}")
    else:
        print("  (none — llm-judge rubric add --case CASE_ID --name ... --description '...')")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="llm-judge",
        description=(
            "Tell the judge what your agent is supposed to do, give it an "
            "expected answer, and get back which input to change."
        ),
    )
    parser.add_argument("--version", action="version", version=f"llm-judge {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    # init
    p_init = sub.add_parser("init", help="Write a starter dataset of {expected} answers.")
    p_init.add_argument("path", nargs="?", default="data/example.jsonl")
    p_init.set_defaults(func=_cmd_init)

    # rubric (the process)
    p_rubric = sub.add_parser(
        "rubric",
        help="Create or update the process the judge looks for (project or use case).",
    )
    rsub = p_rubric.add_subparsers(dest="rubric_command", required=True)

    def _add_rubric_paths(p) -> None:
        p.add_argument(
            "path",
            nargs="?",
            default=str(DEFAULT_PROJECT_PATH),
            help="Project process file (default: data/rubric.json).",
        )
        p.add_argument(
            "--use-cases",
            default=None,
            help="Directory for unique use-case files (default: <project-dir>/use-cases/).",
        )

    p_r_init = rsub.add_parser("init", help="Write an empty project process file.")
    _add_rubric_paths(p_r_init)
    p_r_init.add_argument(
        "--force", action="store_true", help="Overwrite if the file already exists."
    )
    p_r_init.set_defaults(func=_cmd_rubric_init)

    p_r_add = rsub.add_parser(
        "add", help="Add or update a criterion on the project or one use case."
    )
    _add_rubric_paths(p_r_add)
    p_r_add.add_argument("--name", required=True, help="Short id, e.g. task or grounding.")
    p_r_add.add_argument(
        "--description",
        default=None,
        help="What the judge should look for. Required when adding a new name.",
    )
    p_r_add.add_argument("--weight", type=float, default=None)
    p_r_add.add_argument("--threshold", type=float, default=None)
    p_r_add.add_argument(
        "--case",
        default=None,
        help="Test-case id. Writes a gitignored file under use-cases/.",
    )
    p_r_add.set_defaults(func=_cmd_rubric_add)

    p_r_remove = rsub.add_parser("remove", help="Remove a criterion.")
    _add_rubric_paths(p_r_remove)
    p_r_remove.add_argument("--name", required=True)
    p_r_remove.add_argument(
        "--case",
        default=None,
        help="If set, remove the use-case file rule only (project process stays).",
    )
    p_r_remove.set_defaults(func=_cmd_rubric_remove)

    p_r_list = rsub.add_parser("list", help="Print the project process and use-case files.")
    _add_rubric_paths(p_r_list)
    p_r_list.set_defaults(func=_cmd_rubric_list)

    # run
    p_run = sub.add_parser("run", help="Run a benchmark.")
    p_run.add_argument("--dataset", required=True, help="Path to .jsonl/.json/.yaml dataset.")
    p_run.add_argument(
        "--rubric",
        default=None,
        help="Project process file. Use-case files are loaded from <dir>/use-cases/.",
    )
    p_run.add_argument(
        "--use-cases",
        default=None,
        help="Directory of unique use-case rubric files (default: next to --rubric).",
    )
    p_run.add_argument("--name", default=None, help="Report name.")
    p_run.add_argument(
        "--judge",
        default=None,
        choices=["heuristic", "rubric"],
        help="Which judge (default: heuristic, or rubric when --rubric is set).",
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
