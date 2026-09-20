"""A fully-offline, dependency-free judge.

It won't match a strong LLM judge, but it's fast, free, deterministic, and
great for tight local loops and CI. It also demonstrates the *attribution*
idea concretely: e.g. if the answer isn't grounded in the retrieved context AND
the context doesn't contain the reference answer, the fault is attributed to
retrieval; if the context is fine but the answer ignores it, the fault is
attributed to the static prompt / model.
"""

from __future__ import annotations

import re
from typing import Optional

from ..dataset import TestCase
from ..results import Attribution, CriterionScore, JudgeResult
from ..schema import Trace

_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "of", "to", "in", "on", "for", "is",
    "are", "was", "were", "be", "been", "with", "as", "at", "by", "this", "that",
    "it", "its", "from", "you", "your", "i", "we", "they", "he", "she", "do",
    "does", "how", "what", "why", "when", "where", "which", "can", "will",
}

_REFUSAL_PATTERNS = [
    r"\bi (?:cannot|can't|can not|won't|will not)\b",
    r"\bi'm (?:sorry|unable)\b",
    r"\bi am (?:sorry|unable)\b",
    r"\bas an ai\b",
    r"\bi'm not able to\b",
]


def _tokens(text: str) -> set[str]:
    words = re.findall(r"[a-z0-9]+", text.lower())
    return {w for w in words if w not in _STOPWORDS and len(w) > 1}


def _recall(reference: set[str], candidate: set[str]) -> float:
    """Fraction of reference tokens present in the candidate."""
    if not reference:
        return 1.0
    return len(reference & candidate) / len(reference)


def _is_refusal(text: str) -> bool:
    low = text.lower()
    return any(re.search(p, low) for p in _REFUSAL_PATTERNS)


class HeuristicJudge:
    name = "heuristic"

    # Presented so callers can introspect (matches Judge's public surface).
    def __init__(self) -> None:
        self.criteria = []

    def judge(self, trace: Trace, case: Optional[TestCase] = None) -> JudgeResult:
        answer = trace.final_output or ""
        answer_toks = _tokens(answer)
        question = trace.first_user_input
        question_toks = _tokens(question)

        context_chunks = trace.all_retrieved_context
        context_text = "\n".join(c.content for c in context_chunks)
        context_toks = _tokens(context_text)

        reference = (case.reference if case else None) or ""
        reference_toks = _tokens(reference)

        scores: list[CriterionScore] = []
        notes: list[str] = []

        # 1) Non-empty / not-a-refusal ----------------------------------------
        if not answer.strip():
            scores.append(
                CriterionScore(
                    name="non_empty", score=0.0, passed=False, weight=1.0,
                    rationale="Empty output.", attribution=Attribution.MODEL,
                )
            )
        elif _is_refusal(answer):
            scores.append(
                CriterionScore(
                    name="non_refusal", score=0.0, passed=False, weight=1.0,
                    rationale="Output looks like a refusal.",
                    attribution=Attribution.STATIC_PROMPT,
                )
            )
            notes.append("model refused")

        # 2) Relevance to the question ----------------------------------------
        relevance = _recall(question_toks, answer_toks) if question_toks else 1.0
        scores.append(
            CriterionScore(
                name="relevance",
                score=round(relevance, 3),
                passed=relevance >= 0.25,
                weight=1.0,
                rationale=f"{int(relevance * 100)}% of question keywords appear in the answer.",
                attribution=Attribution.STATIC_PROMPT if relevance < 0.25 else None,
            )
        )

        # 3) Faithfulness / grounding (only if there was retrieved context) ---
        if context_chunks:
            faithfulness = _recall(answer_toks, context_toks)  # answer supported by ctx
            # Does the context actually contain the answer we needed?
            ctx_has_reference = (
                _recall(reference_toks, context_toks) if reference_toks else None
            )
            attribution: Optional[Attribution] = None
            if faithfulness < 0.5:
                if ctx_has_reference is not None and ctx_has_reference < 0.5:
                    # Context was missing the needed info -> retrieval problem.
                    attribution = Attribution.RETRIEVAL
                else:
                    # Context was fine but the answer didn't use it -> prompt/model.
                    attribution = Attribution.STATIC_PROMPT
            scores.append(
                CriterionScore(
                    name="faithfulness",
                    score=round(faithfulness, 3),
                    passed=faithfulness >= 0.5,
                    weight=1.5,
                    rationale=(
                        f"{int(faithfulness * 100)}% of answer keywords are grounded "
                        f"in retrieved context."
                        + (
                            f" Context covers {int((ctx_has_reference or 0) * 100)}% "
                            f"of the reference."
                            if ctx_has_reference is not None
                            else ""
                        )
                    ),
                    attribution=attribution,
                )
            )

        # 4) Completeness vs. reference (only if a gold answer is provided) ---
        if reference_toks:
            completeness = _recall(reference_toks, answer_toks)
            # If the context had the answer but the response didn't -> prompt/model;
            # if context lacked it -> retrieval.
            attribution = None
            if completeness < 0.5:
                if context_chunks:
                    ctx_cov = _recall(reference_toks, context_toks)
                    attribution = (
                        Attribution.RETRIEVAL if ctx_cov < 0.5 else Attribution.STATIC_PROMPT
                    )
                else:
                    attribution = Attribution.STATIC_PROMPT
            scores.append(
                CriterionScore(
                    name="completeness",
                    score=round(completeness, 3),
                    passed=completeness >= 0.5,
                    weight=1.5,
                    rationale=f"{int(completeness * 100)}% of reference keywords are covered.",
                    attribution=attribution,
                )
            )

        summary = "; ".join(notes) if notes else "heuristic evaluation"
        result = _finalize(trace, scores, summary=summary)
        return result


def _finalize(trace: Trace, scores, summary: str) -> JudgeResult:
    total_w = sum(s.weight for s in scores) or 1.0
    overall = sum(s.score * s.weight for s in scores) / total_w
    passed = all((s.passed is None) or s.passed for s in scores) and bool(scores)
    return JudgeResult(
        trace_id=trace.id,
        case_id=trace.case_id,
        judge_name="heuristic",
        judge_model=None,
        target_model=trace.model,
        scores=scores,
        overall=round(overall, 3),
        passed=passed,
        summary=summary,
    )
