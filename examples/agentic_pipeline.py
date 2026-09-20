"""Instrumenting a multi-step / LangGraph-style agent with TraceRecorder.

Shows how to capture, for every node: the static prompt, the runtime-retrieved
context, the exact model input, the output, and the shared state before/after.

    python examples/agentic_pipeline.py
"""

from __future__ import annotations

from llm_judge import (
    Dataset,
    FunctionTarget,
    HeuristicJudge,
    Message,
    ModelSpec,
    RetrievedChunk,
    Role,
    TestCase,
    TraceRecorder,
    render,
    run_benchmark,
)

PLANNER_PROMPT = "Break the user's question into a search query."
RESPONDER_PROMPT = "Answer the question using ONLY the retrieved context. Cite sources."

# A toy knowledge base for the fake retriever.
KB = {
    "capital": "France's capital is Paris.",
    "hamlet": "Hamlet was written by William Shakespeare.",
    "photosynthesis": "Photosynthesis converts sunlight, water, and CO2 into glucose and oxygen.",
    "unit testing": "Unit testing catches bugs early and enables safe refactoring.",
}


def fake_retrieve(query: str) -> list[RetrievedChunk]:
    hits = [
        RetrievedChunk(content=v, source=k, score=0.9)
        for k, v in KB.items()
        if any(word in query.lower() for word in k.split())
    ]
    return hits or [RetrievedChunk(content="(no relevant document found)", source="none", score=0.0)]


def agent(case: TestCase):
    model = ModelSpec(name="toy-agent", provider="mock", temperature=0.0)
    rec = TraceRecorder(model=model, case_id=case.id)
    rec.set_static_prompt("planner", PLANNER_PROMPT)
    rec.set_static_prompt("responder", RESPONDER_PROMPT)

    question = str(case.input)
    rec.set_initial_context([Message(role=Role.USER, content=question)])

    state: dict = {"question": question}

    # --- Node 1: planner -----------------------------------------------------
    with rec.step("planner", state_before=state, model=model) as step:
        step.static_prompt = PLANNER_PROMPT
        step.input_messages = [
            Message(role=Role.SYSTEM, content=PLANNER_PROMPT),
            Message(role=Role.USER, content=question),
        ]
        query = question  # a real planner would call the model here
        step.add_output(query)
        state = {**state, "query": query}
        step.state_after = state

    # --- Node 2: retriever ---------------------------------------------------
    with rec.step("retriever", state_before=state) as step:
        chunks = fake_retrieve(state["query"])
        step.retrieved_context = chunks
        step.add_output(f"retrieved {len(chunks)} chunks", role=Role.TOOL)
        state = {**state, "context": chunks}
        step.state_after = state

    # --- Node 3: responder ---------------------------------------------------
    with rec.step("responder", state_before=state, model=model) as step:
        step.static_prompt = RESPONDER_PROMPT
        context_text = " ".join(c.content for c in state["context"])
        step.retrieved_context = state["context"]
        step.input_messages = [
            Message(role=Role.SYSTEM, content=RESPONDER_PROMPT),
            Message(role=Role.USER, content=f"Context: {context_text}\n\nQ: {question}"),
        ]
        answer = context_text  # grounded answer from context
        step.add_output(answer)
        state = {**state, "answer": answer}
        step.state_after = state

    return rec.finish(final_output=state["answer"])


def main() -> None:
    dataset = Dataset.load("data/example.jsonl")
    target = FunctionTarget(agent, name="toy-agent")
    report = run_benchmark(target, HeuristicJudge(), dataset, name="agentic")
    print(render(report))


if __name__ == "__main__":
    main()
