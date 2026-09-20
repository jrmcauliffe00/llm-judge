"""End-to-end and unit tests. All run fully offline (no API keys)."""

from __future__ import annotations

from llm_judge import (
    Attribution,
    Dataset,
    EchoTarget,
    HeuristicJudge,
    Message,
    ModelSpec,
    RetrievedChunk,
    Role,
    Step,
    TestCase,
    Trace,
    TraceRecorder,
    build_report,
    recommend,
    render,
    render_json,
    render_markdown,
    run_benchmark,
)


def _grounded_trace(case: TestCase) -> Trace:
    model = ModelSpec(name="t", provider="mock")
    answer = " ".join(c.content for c in case.context) or (case.reference or "")
    return Trace(
        model=model,
        case_id=case.id,
        static_prompts={"system": "answer from context"},
        initial_context=[Message(role=Role.USER, content=str(case.input))],
        steps=[
            Step(
                index=0,
                name="gen",
                model=model,
                retrieved_context=list(case.context),
                output_messages=[Message(role=Role.ASSISTANT, content=answer)],
            )
        ],
        final_output=answer,
    )


def test_dataset_roundtrip(tmp_path):
    ds = Dataset.from_cases(
        [TestCase(id="a", input="hi", reference="hello")], name="ds"
    )
    p = tmp_path / "ds.jsonl"
    ds.save(p)
    loaded = Dataset.load(p)
    assert len(loaded) == 1
    assert loaded.cases[0].id == "a"


def test_echo_target_and_heuristic_judge():
    case = TestCase(input="What is the capital of France?", reference="Paris")
    trace = EchoTarget().run(case)
    assert "France" in trace.final_output
    result = HeuristicJudge().judge(trace, case)
    assert 0.0 <= result.overall <= 1.0
    assert result.score_for("relevance") is not None


def test_grounded_answer_scores_high_faithfulness():
    case = TestCase(
        input="Who wrote Hamlet?",
        reference="William Shakespeare wrote Hamlet.",
        context=[RetrievedChunk(content="Hamlet was written by William Shakespeare.")],
    )
    trace = _grounded_trace(case)
    result = HeuristicJudge().judge(trace, case)
    faith = result.score_for("faithfulness")
    assert faith is not None
    assert faith.score >= 0.5


def test_missing_context_attributed_to_retrieval():
    # Context is unrelated to the reference => retrieval should take the blame.
    case = TestCase(
        input="What is the boiling point of water?",
        reference="Water boils at 100 degrees Celsius at sea level.",
        context=[RetrievedChunk(content="Bananas are a yellow fruit rich in potassium.")],
    )
    # Answer parrots the (wrong) context.
    trace = _grounded_trace(case)
    result = HeuristicJudge().judge(trace, case)
    completeness = result.score_for("completeness")
    assert completeness is not None
    assert completeness.passed is False
    assert completeness.attribution == Attribution.RETRIEVAL


def test_recommender_ranks_by_severity():
    cases = [
        TestCase(
            input=f"q{i}",
            reference="alpha beta gamma delta",
            context=[RetrievedChunk(content="totally unrelated content here")],
        )
        for i in range(3)
    ]
    ds = Dataset.from_cases(cases)
    results = [HeuristicJudge().judge(_grounded_trace(c), c) for c in ds]
    recs = recommend(results)
    assert recs, "expected at least one recommendation"
    # Sorted descending by severity.
    severities = [r.severity for r in recs]
    assert severities == sorted(severities, reverse=True)
    assert any(r.attribution == Attribution.RETRIEVAL for r in recs)


def test_run_benchmark_and_render():
    ds = Dataset.load("data/example.jsonl")
    report = run_benchmark(EchoTarget(), HeuristicJudge(), ds, name="t")
    assert report.n_cases == len(ds)
    # All three renderers should produce non-empty output.
    assert render(report, "text").strip()
    assert render_markdown(report).strip()
    assert render_json(report).strip().startswith("{")


def test_trace_recorder_captures_steps_and_state():
    rec = TraceRecorder(model=ModelSpec(name="m", provider="mock"), case_id="c1")
    rec.set_initial_context([Message(role=Role.USER, content="hello")])
    with rec.step("node1", state_before={"x": 1}) as step:
        step.add_output("did stuff")
        step.state_after = {"x": 2}
    trace = rec.finish(final_output="done")
    assert len(trace.steps) == 1
    assert trace.steps[0].state_before == {"x": 1}
    assert trace.steps[0].state_after == {"x": 2}
    assert trace.steps[0].output_text == "did stuff"
    assert trace.final_output == "done"


def test_build_client_routes_by_provider():
    from llm_judge.models import (
        AnthropicClient,
        MockClient,
        OpenAIClient,
        build_client,
    )

    assert isinstance(build_client(ModelSpec(name="x", provider="mock")), MockClient)
    assert isinstance(
        build_client(ModelSpec(name="claude-sonnet", provider="anthropic")),
        AnthropicClient,
    )
    assert isinstance(
        build_client(ModelSpec(name="gpt-4o", provider="openai")), OpenAIClient
    )
    # vLLM / Ollama etc. are just OpenAI-compatible.
    assert isinstance(
        build_client(ModelSpec(name="llama3", provider="ollama")), OpenAIClient
    )


def test_anthropic_message_split_is_provider_shaped():
    # No network: just validate we reshape messages to Anthropic's format
    # (system pulled out, tool folded into user).
    from llm_judge.models.anthropic_client import AnthropicClient

    msgs = [
        Message(role=Role.SYSTEM, content="be terse"),
        Message(role=Role.USER, content="hello"),
        Message(role=Role.ASSISTANT, content="hi"),
        Message(role=Role.TOOL, content="42", name="calc"),
    ]
    system, turns = AnthropicClient._split_messages(msgs)
    assert system == "be terse"
    assert turns[0] == {"role": "user", "content": "hello"}
    assert turns[1] == {"role": "assistant", "content": "hi"}
    assert turns[2]["role"] == "user" and "calc" in turns[2]["content"]


def test_rubric_judge_offline_neutral_and_parsing():
    from llm_judge.judges.rubric import RubricJudge
    from llm_judge.models.mock import MockClient

    case = TestCase(input="q", reference="r")
    trace = _grounded_trace(case)

    # 1) Default offline mock returns "{}" -> neutral scores, still valid result.
    judge = RubricJudge(client=MockClient(ModelSpec(name="mock-judge", provider="mock")))
    res = judge.judge(trace, case)
    assert res.scores  # neutral scores populated
    assert 0.0 <= res.overall <= 1.0

    # 2) Inject a responder that returns proper JSON -> parsed into scores.
    def responder(messages, json_mode):
        return (
            '{"scores": [{"name": "faithfulness", "score": 2, '
            '"attribution": "retrieval", "rationale": "not grounded"}], '
            '"summary": "fix retrieval"}'
        )

    judge2 = RubricJudge(
        client=MockClient(ModelSpec(name="mock-judge", provider="mock"), responder=responder)
    )
    res2 = judge2.judge(trace, case)
    faith = res2.score_for("faithfulness")
    assert faith is not None
    assert faith.attribution == Attribution.RETRIEVAL
    assert res2.summary == "fix retrieval"
