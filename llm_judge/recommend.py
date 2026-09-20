"""Turn judge results into ranked, actionable recommendations.

The recommender aggregates failing criteria across the whole run, groups them by
*attribution* (retrieval vs. static_prompt vs. model vs. input), scores each
group by impact, and emits concrete suggestions. This is the payoff of tracking
static prompts and retrieved context separately in the trace: we can tell you
*which lever to pull*.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Optional

from .results import Attribution, JudgeResult, Recommendation

# Suggestion templates keyed by (attribution, criterion). Falls back to a
# generic per-attribution suggestion when a specific one isn't defined.
_SUGGESTIONS: dict[tuple[Attribution, str], str] = {
    (Attribution.RETRIEVAL, "faithfulness"): (
        "Answers aren't grounded in the retrieved context. Improve retrieval: "
        "increase top-k, add a reranker, chunk documents differently, or verify "
        "the needed documents are actually in your index."
    ),
    (Attribution.RETRIEVAL, "completeness"): (
        "Required information is missing from the context. Broaden retrieval "
        "(higher top-k / hybrid keyword+vector search) or expand the knowledge base."
    ),
    (Attribution.RETRIEVAL, "relevance"): (
        "Retrieved context is off-topic and pulling the answer away from the "
        "question. Tighten retrieval filters or add a relevance reranking step."
    ),
    (Attribution.STATIC_PROMPT, "faithfulness"): (
        "The model has good context but isn't using it. Strengthen the static "
        "prompt: explicitly instruct it to answer ONLY from the provided context "
        "and to cite sources; add a 'say I don't know if unsupported' rule."
    ),
    (Attribution.STATIC_PROMPT, "relevance"): (
        "Answers drift off-topic. Make the static prompt spell out the task, the "
        "expected format, and what to focus on. Add a short worked example."
    ),
    (Attribution.STATIC_PROMPT, "completeness"): (
        "Answers are partial. Update the static prompt to require covering all "
        "sub-parts of the request (a checklist or required-sections instruction)."
    ),
    (Attribution.STATIC_PROMPT, "non_refusal"): (
        "The system is over-refusing. Loosen or clarify safety wording in the "
        "static prompt so legitimate requests aren't declined."
    ),
    (Attribution.STATIC_PROMPT, "coherence"): (
        "Structure is weak. Ask for a specific output format (headings/bullets) "
        "in the static prompt."
    ),
    (Attribution.MODEL, "faithfulness"): (
        "The model may be too weak to stay grounded. Try a stronger model or lower "
        "the temperature for this task."
    ),
    (Attribution.INPUT, "relevance"): (
        "The requests themselves are ambiguous. Add input validation or a "
        "clarification step before generation."
    ),
}

_GENERIC = {
    Attribution.RETRIEVAL: (
        "Retrieval is the leading cause of failures. Focus on what context you "
        "fetch at runtime (top-k, reranking, chunking, index coverage)."
    ),
    Attribution.STATIC_PROMPT: (
        "Your static prompt is the leading cause of failures. Iterate on the "
        "fixed instructions/templates (task spec, format, grounding rules, examples)."
    ),
    Attribution.MODEL: (
        "Model capability/settings appear to be the bottleneck. Consider a stronger "
        "model or tuned decoding parameters for this task."
    ),
    Attribution.INPUT: (
        "The inputs/requests are a leading cause of failures. Consider input "
        "normalization or a clarification step."
    ),
    Attribution.OTHER: "Miscellaneous failures; inspect individual cases.",
}


def recommend(
    results: list[JudgeResult], max_recommendations: int = 8
) -> list[Recommendation]:
    """Produce ranked recommendations from a set of judge results."""
    n = len(results) or 1

    # Bucket failing criteria by (attribution, criterion_name).
    buckets: dict[tuple[Attribution, str], list[JudgeResult]] = defaultdict(list)
    deficits: dict[tuple[Attribution, str], list[float]] = defaultdict(list)

    for res in results:
        for s in res.scores:
            failed = s.passed is False
            if failed and s.attribution is not None:
                key = (s.attribution, s.name)
                buckets[key].append(res)
                deficits[key].append(1.0 - s.score)

    recs: list[Recommendation] = []
    for (attribution, crit), affected in buckets.items():
        frequency = len(affected) / n  # 0..1: how often this failure shows up
        mean_deficit = sum(deficits[(attribution, crit)]) / len(affected)
        severity = round(frequency * mean_deficit, 3)
        suggestion = _SUGGESTIONS.get(
            (attribution, crit), _GENERIC.get(attribution, _GENERIC[Attribution.OTHER])
        )
        finding = (
            f"'{crit}' failed in {len(affected)}/{n} cases "
            f"({int(frequency * 100)}%), attributed to {attribution.value}."
        )
        affected_cases = [r.case_id for r in affected if r.case_id]
        recs.append(
            Recommendation(
                attribution=attribution,
                severity=severity,
                finding=finding,
                suggestion=suggestion,
                evidence=[crit],
                affected_cases=affected_cases[:20],
            )
        )

    recs.sort(key=lambda r: r.severity, reverse=True)
    return recs[:max_recommendations]


def attribution_breakdown(results: list[JudgeResult]) -> dict[str, int]:
    """Count failing criteria by attribution (handy for a summary line)."""
    counts: dict[str, int] = defaultdict(int)
    for res in results:
        for s in res.scores:
            if s.passed is False and s.attribution is not None:
                counts[s.attribution.value] += 1
    return dict(counts)
