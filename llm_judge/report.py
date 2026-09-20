"""Render a :class:`~llm_judge.results.BenchmarkReport` for the terminal.

CLI-first: everything is plain text or markdown/JSON you can pipe around. No
frontend required.
"""

from __future__ import annotations

import json

from .recommend import attribution_breakdown
from .results import BenchmarkReport


def _bar(value: float, width: int = 20) -> str:
    filled = int(round(value * width))
    return "█" * filled + "░" * (width - filled)


def render_text(report: BenchmarkReport) -> str:
    lines: list[str] = []
    lines.append("=" * 64)
    lines.append(f"BENCHMARK: {report.name}")
    lines.append("=" * 64)
    if report.target_model:
        lines.append(f"Target model : {report.target_model.label()}")
    judge_label = report.judge_name
    if report.judge_model:
        judge_label += f" ({report.judge_model.label()})"
    lines.append(f"Judge        : {judge_label}")
    lines.append("")
    lines.append(
        f"Cases        : {report.n_cases}    "
        f"Passed: {report.n_passed}    "
        f"Pass rate: {report.pass_rate * 100:.0f}%"
    )
    lines.append(f"Overall score: {report.mean_overall:.3f}  {_bar(report.mean_overall)}")
    lines.append("")

    if report.criteria_means:
        lines.append("Per-criterion averages")
        lines.append("-" * 64)
        for name, val in sorted(report.criteria_means.items(), key=lambda kv: kv[1]):
            lines.append(f"  {name:<16} {val:.3f}  {_bar(val)}")
        lines.append("")

    breakdown = attribution_breakdown(report.results)
    if breakdown:
        lines.append("Failure attribution (count of failed criteria)")
        lines.append("-" * 64)
        for attr, count in sorted(breakdown.items(), key=lambda kv: -kv[1]):
            lines.append(f"  {attr:<16} {count}")
        lines.append("")

    if report.recommendations:
        lines.append("Recommendations (highest impact first)")
        lines.append("-" * 64)
        for i, rec in enumerate(report.recommendations, 1):
            lines.append(
                f"{i}. [{rec.attribution.value}] severity={rec.severity:.2f}"
            )
            lines.append(f"   Finding : {rec.finding}")
            lines.append(f"   Fix     : {rec.suggestion}")
            lines.append("")
    else:
        lines.append("No recommendations - everything passed. 🎉")
        lines.append("")

    return "\n".join(lines)


def render_markdown(report: BenchmarkReport) -> str:
    lines: list[str] = []
    lines.append(f"# Benchmark: {report.name}\n")
    if report.target_model:
        lines.append(f"- **Target model:** `{report.target_model.label()}`")
    lines.append(f"- **Judge:** `{report.judge_name}`")
    lines.append(
        f"- **Cases:** {report.n_cases} · **Passed:** {report.n_passed} · "
        f"**Pass rate:** {report.pass_rate * 100:.0f}%"
    )
    lines.append(f"- **Overall score:** {report.mean_overall:.3f}\n")

    if report.criteria_means:
        lines.append("## Per-criterion averages\n")
        lines.append("| Criterion | Score |")
        lines.append("| --- | --- |")
        for name, val in sorted(report.criteria_means.items(), key=lambda kv: kv[1]):
            lines.append(f"| {name} | {val:.3f} |")
        lines.append("")

    if report.recommendations:
        lines.append("## Recommendations\n")
        for i, rec in enumerate(report.recommendations, 1):
            lines.append(
                f"{i}. **[{rec.attribution.value}]** (severity {rec.severity:.2f}) — "
                f"{rec.finding}\n   - _Fix:_ {rec.suggestion}"
            )
        lines.append("")

    return "\n".join(lines)


def render_json(report: BenchmarkReport) -> str:
    return json.dumps(report.model_dump(mode="json"), indent=2)


def render(report: BenchmarkReport, fmt: str = "text") -> str:
    fmt = fmt.lower()
    if fmt == "text":
        return render_text(report)
    if fmt in ("md", "markdown"):
        return render_markdown(report)
    if fmt == "json":
        return render_json(report)
    raise ValueError(f"Unknown format '{fmt}'. Use text, markdown, or json.")
