"""LLM-as-a-judge with an explicit rubric and failure attribution.

Design notes
------------
- **Model-aware**: the judge prompt states which model produced the answer and
  its settings. This lets the judge calibrate expectations and, crucially, tell
  the difference between "the model is too weak for this" and "the prompt/context
  is the problem".
- **Attribution-first**: for every criterion the judge fails, it must say where
  the fault lies (retrieval / static_prompt / model / input). That's what powers
  the recommender.
- **Structured output**: we request a strict JSON object and parse defensively.
"""

from __future__ import annotations

import json
import re
from typing import Optional

from ..config import Config, default_config
from ..dataset import TestCase
from ..models import LLMClient, build_client
from ..results import Attribution, CriterionScore, JudgeResult
from ..schema import Message, ModelSpec, Role, Trace
from .base import Criterion, Judge

DEFAULT_CRITERIA: list[Criterion] = [
    Criterion(
        name="relevance",
        description="Does the answer directly address the user's request?",
        weight=1.0,
    ),
    Criterion(
        name="faithfulness",
        description=(
            "Is every claim supported by the retrieved context? Penalize "
            "hallucinations and unsupported claims. If no context was provided, "
            "score based on general factual correctness."
        ),
        weight=1.5,
    ),
    Criterion(
        name="completeness",
        description="Does the answer fully cover what was asked (and the reference, if given)?",
        weight=1.25,
    ),
    Criterion(
        name="coherence",
        description="Is the answer clear, well-structured, and free of contradictions?",
        weight=0.75,
    ),
]

_ALLOWED_ATTRIBUTIONS = [a.value for a in Attribution]

_JUDGE_SYSTEM = (
    "You are a meticulous evaluation judge for LLM systems. You grade an "
    "assistant's answer against a rubric and, for any weakness, you attribute "
    "the root cause to a specific part of the pipeline so engineers know what "
    "to fix. Be strict, specific, and calibrated to the model that produced the "
    "answer. Respond with a single JSON object and nothing else."
)


def _fmt_context(trace: Trace) -> str:
    chunks = trace.all_retrieved_context
    if not chunks:
        return "(no retrieved context was provided)"
    return "\n".join(
        f"[{i}] (source={c.source or 'n/a'}, score={c.score}) {c.content}"
        for i, c in enumerate(chunks)
    )


def _build_user_prompt(
    trace: Trace, case: Optional[TestCase], criteria: list[Criterion]
) -> str:
    model = trace.model
    rubric_lines = "\n".join(f"- {c.name}: {c.description}" for c in criteria)
    reference = (case.reference if case else None) or "(none provided)"
    static_prompts = (
        "\n".join(f"### {k}\n{v}" for k, v in trace.static_prompts.items())
        or "(none recorded)"
    )
    return f"""\
## Model under test
{model.label()} (temperature={model.temperature}, top_p={model.top_p})

## Static prompt(s) used by the system (candidate for prompt-engineering fixes)
{static_prompts}

## User request
{trace.first_user_input}

## Retrieved context injected at runtime (candidate for retrieval fixes)
{_fmt_context(trace)}

## Reference / gold answer
{reference}

## Assistant's final answer (this is what you are grading)
{trace.final_output}

## Rubric (score each 1-5, where 1=terrible and 5=excellent)
{rubric_lines}

## Attribution
For any criterion scoring 3 or below, attribute the primary root cause to ONE of:
{_ALLOWED_ATTRIBUTIONS}
Guidance: use "retrieval" when the needed information was missing/noisy in the
context; "static_prompt" when the fixed instructions failed to steer the model
correctly; "model" when the model lacked the capability; "input" when the
request itself was flawed/ambiguous.

## Output format (respond with ONLY this JSON)
{{
  "scores": [
    {{"name": "<criterion>", "score": <1-5>, "attribution": "<one of the allowed values or null>", "rationale": "<one sentence>"}}
  ],
  "summary": "<one or two sentences on the biggest issue and how to fix it>"
}}
"""


def _extract_json(text: str) -> Optional[dict]:
    text = text.strip()
    # Strip markdown fences if present.
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text).strip()
        text = re.sub(r"```$", "", text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Fall back: grab the first balanced-looking {...} block.
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
    return None


class RubricJudge(Judge):
    name = "rubric"

    def __init__(
        self,
        criteria: Optional[list[Criterion]] = None,
        client: Optional[LLMClient] = None,
        judge_model: Optional[ModelSpec] = None,
        config: Optional[Config] = None,
    ) -> None:
        super().__init__(criteria or list(DEFAULT_CRITERIA))
        self.config = config or default_config
        self.judge_model = judge_model or ModelSpec(
            name=self.config.judge_model, provider=self.config.judge_provider
        )
        self.client = client or build_client(self.judge_model, self.config)

    def judge(self, trace: Trace, case: Optional[TestCase] = None) -> JudgeResult:
        user_prompt = _build_user_prompt(trace, case, self.criteria)
        messages = [
            Message(role=Role.SYSTEM, content=_JUDGE_SYSTEM),
            Message(role=Role.USER, content=user_prompt),
        ]
        response = self.client.complete(messages, temperature=0.0, json_mode=True)
        parsed = _extract_json(response.text)

        scores = self._parse_scores(parsed)
        summary = ""
        if parsed and isinstance(parsed.get("summary"), str):
            summary = parsed["summary"]
        if not scores:
            summary = summary or (
                "Judge returned no parseable scores (offline mock or malformed "
                "output); recorded neutral scores."
            )
            scores = self._neutral_scores()

        return self._finalize(
            trace,
            scores,
            summary=summary,
            judge_model=self.judge_model,
            raw={"judge_text": response.text},
        )

    # --- helpers -------------------------------------------------------------
    def _crit_by_name(self, name: str) -> Optional[Criterion]:
        for c in self.criteria:
            if c.name == name:
                return c
        return None

    def _parse_scores(self, parsed: Optional[dict]) -> list[CriterionScore]:
        if not parsed or not isinstance(parsed.get("scores"), list):
            return []
        out: list[CriterionScore] = []
        for item in parsed["scores"]:
            if not isinstance(item, dict) or "name" not in item:
                continue
            name = str(item["name"])
            crit = self._crit_by_name(name)
            weight = crit.weight if crit else 1.0
            threshold = crit.pass_threshold if crit else 0.5
            raw_score = item.get("score", 3)
            try:
                norm = (float(raw_score) - 1.0) / 4.0  # 1..5 -> 0..1
            except (TypeError, ValueError):
                norm = 0.5
            norm = max(0.0, min(1.0, norm))
            attribution = _coerce_attribution(item.get("attribution"))
            out.append(
                CriterionScore(
                    name=name,
                    score=round(norm, 3),
                    passed=norm >= threshold,
                    weight=weight,
                    rationale=str(item.get("rationale", "")),
                    attribution=attribution if norm < threshold else None,
                )
            )
        return out

    def _neutral_scores(self) -> list[CriterionScore]:
        return [
            CriterionScore(
                name=c.name, score=0.5, passed=None, weight=c.weight,
                rationale="No judge signal (offline).",
            )
            for c in self.criteria
        ]


def _coerce_attribution(value) -> Optional[Attribution]:
    if not value:
        return None
    try:
        return Attribution(str(value).lower())
    except ValueError:
        return Attribution.OTHER
