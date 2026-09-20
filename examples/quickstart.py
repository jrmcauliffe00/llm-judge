"""Minimal end-to-end example that runs fully offline.

    python examples/quickstart.py
"""

from __future__ import annotations

from llm_judge import (
    Dataset,
    HeuristicJudge,
    Message,
    ModelSpec,
    Role,
    RetrievedChunk,
    Step,
    TestCase,
    Trace,
    FunctionTarget,
    render,
    run_benchmark,
)


def my_system(case: TestCase) -> Trace:
    """Pretend this is your RAG system. It returns a full trace.

    Here we fabricate a grounded answer from the supplied context so the
    heuristic judge has something meaningful to grade.
    """
    model = ModelSpec(name="my-rag-v1", provider="mock", temperature=0.2)
    context_text = " ".join(c.content for c in case.context)
    answer = context_text or f"Answer to: {case.input}"

    step = Step(
        index=0,
        name="generate",
        model=model,
        static_prompt="Answer ONLY from the provided context.",
        retrieved_context=list(case.context),
        input_messages=[Message(role=Role.USER, content=str(case.input))],
        output_messages=[Message(role=Role.ASSISTANT, content=answer)],
    )
    return Trace(
        model=model,
        case_id=case.id,
        static_prompts={"system": "Answer ONLY from the provided context."},
        initial_context=[Message(role=Role.USER, content=str(case.input))],
        steps=[step],
        final_output=answer,
    )


def main() -> None:
    dataset = Dataset.load("data/example.jsonl")
    target = FunctionTarget(my_system, name="my-rag")
    judge = HeuristicJudge()
    report = run_benchmark(target, judge, dataset, name="quickstart")
    print(render(report))


if __name__ == "__main__":
    main()
