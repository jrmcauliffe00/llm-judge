# llm-judge

An easy, **CLI-first** harness for benchmarking your LLM/agent output — and getting
back concrete recommendations that tell you *what to fix*: your **runtime retrieval /
context**, or your **static prompts**.

Inspired by [JudgeZoo](https://github.com/LLM-QC/judgezoo) (standardized safety judges)
and [NVIDIA NeMo's LLM-as-a-judge](https://docs.nvidia.com/nemo/microservices/latest/evaluator/metrics/llm-as-a-judge.html)
(rubric-based grading), but focused on **local, iterative benchmarking of your own
system** and on **attributing failures to a fixable lever**.

## Why

Before you can improve an LLM system you have to *see* it completely. `llm-judge`
captures a full **trace** of every run and then judges it:

| What you wanted to capture | Where it lives |
| --- | --- |
| The model you're using/testing (and settings) | `Trace.model` / `Step.model` (`ModelSpec`) |
| The complete context window at start | `Trace.initial_context` |
| The input at *every step* of a looping agent | `Step.input_messages` per `Step` |
| Context manipulation / state in↔out (LangGraph-style) | `Step.state_before` / `Step.state_after` |
| What the LLM spits out | `Step.output_messages`, `Trace.final_output` |
| Static prompts (separate from runtime input) | `Trace.static_prompts`, `Step.static_prompt` |
| Runtime-retrieved context (RAG) | `Step.retrieved_context` (`RetrievedChunk`) |

Because static prompts and retrieved context are tracked **separately**, the judge
can attribute each failure to `retrieval`, `static_prompt`, `model`, or `input`, and
the recommender turns those attributions into a ranked to-do list.

The judge is **model-aware**: it's told which model produced the answer, so it can
distinguish "the model is too weak for this" from "the prompt/context is wrong."

## Install

```bash
pip install -e .            # core (offline: mock models + heuristic judge)
pip install -e ".[all]"     # + openai client and YAML dataset support
```

Requires Python 3.10+. The only hard dependency is `pydantic`.

## Quick start (CLI)

```bash
# 1) Create a starter dataset
llm-judge init data/example.jsonl

# 2) Run fully offline (mock/echo target + heuristic judge — no API keys)
llm-judge run --dataset data/example.jsonl

# 3) Benchmark a real model, judged by an LLM-as-a-judge
export OPENAI_API_KEY=...
llm-judge run --dataset data/example.jsonl \
    --target-model gpt-4o-mini --target-provider openai \
    --judge rubric --judge-model gpt-4o --judge-provider openai \
    --format markdown --out report.md

# CI gate: fail the build if pass rate drops
llm-judge run --dataset data/example.jsonl --fail-under 0.8
```

Point `--target-provider`/`--judge-provider` at any OpenAI-compatible server
(vLLM, Ollama, LM Studio) by setting `LLM_JUDGE_BASE_URL`.

Example report:

```
================================================================
BENCHMARK: example
================================================================
Target model : mock/echo
Judge        : heuristic

Cases        : 4    Passed: 1    Pass rate: 25%
Overall score: 0.412  ████████░░░░░░░░░░░░

Per-criterion averages
----------------------------------------------------------------
  faithfulness     0.180  ███░░░░░░░░░░░░░░░░░░
  completeness     0.350  ███████░░░░░░░░░░░░░░
  relevance        0.720  ██████████████░░░░░░

Failure attribution (count of failed criteria)
----------------------------------------------------------------
  retrieval        3
  static_prompt    2

Recommendations (highest impact first)
----------------------------------------------------------------
1. [retrieval] severity=0.61
   Finding : 'faithfulness' failed in 3/4 cases (75%), attributed to retrieval.
   Fix     : Answers aren't grounded in the retrieved context. Improve retrieval...
```

## Quick start (Python)

```python
from llm_judge import Dataset, EchoTarget, HeuristicJudge, run_benchmark, render

ds = Dataset.load("data/example.jsonl")
report = run_benchmark(EchoTarget(), HeuristicJudge(), ds)
print(render(report))
```

### Benchmark your own system

Wrap any callable as a `Target`. For agentic/LangGraph loops, use `TraceRecorder`
to capture every step and state transition — see
[`examples/agentic_pipeline.py`](examples/agentic_pipeline.py):

```python
from llm_judge import TraceRecorder, ModelSpec, Message, Role

rec = TraceRecorder(model=ModelSpec(name="my-agent", provider="openai"), case_id=case.id)
rec.set_static_prompt("responder", RESPONDER_PROMPT)
rec.set_initial_context([Message(role=Role.USER, content=question)])

state = {"question": question}
with rec.step("retriever", state_before=state) as step:
    step.retrieved_context = my_retriever(state["question"])
    state = {**state, "context": step.retrieved_context}
    step.state_after = state
# ... more nodes ...
trace = rec.finish(final_output=answer)
```

## Concepts

- **`TestCase` / `Dataset`** — your inputs, optional gold `reference`, optional
  pre-supplied `context`, plus `tags`/`metadata`. Load from `.jsonl`/`.json`/`.yaml`.
- **`Target`** — the system-under-test. `SimpleTarget` (one LLM call), `FunctionTarget`
  (wrap your code), or `EchoTarget` (offline demo). Produces a `Trace`.
- **`Judge`** — `HeuristicJudge` (offline, lexical-overlap + rules) or `RubricJudge`
  (LLM-as-a-judge, model-aware, attribution-first). Produces a `JudgeResult`.
- **`recommend()`** — aggregates failures by attribution → ranked `Recommendation`s.
- **`run_benchmark()` / `render()`** — orchestrate and print (text/markdown/json).

## Judges

| Judge | Needs a model? | Best for |
| --- | --- | --- |
| `heuristic` | No | Fast local loops, CI, plumbing checks |
| `rubric` | Yes (any OpenAI-compatible) | Nuanced grading + rich attribution |

Customize rubric criteria by passing `Criterion` objects to `RubricJudge`.

## Tests

```bash
pip install -e ".[dev]"
pytest
```

All tests run offline.

## Roadmap

- More built-in judges (safety/toxicity via JudgeZoo-style adapters).
- Reference-free faithfulness via NLI.
- Prompt-optimization loop that proposes and A/B-tests prompt edits.
- Dataset slicing / regression comparison between runs.

## License

MIT
