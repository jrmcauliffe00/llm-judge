# llm-judge

Tell the judge **what your agent is supposed to do**. Give it an **`{expected}`**
answer. It looks at the real output and says **which input to change** so the
next run is closer to `{expected}`.

That is the whole point of this repo.

## The two things you write

| You write | How often it changes | What it is |
| --- | --- | --- |
| **Process** | Once per project / agent | What this LLM is supposed to do. This is what the judge looks for. |
| **`{expected}`** | Per use case, whenever you want | A gold answer for this question. Unstructured prose is fine. Rewrite it anytime. |

Different projects → different process. Different questions → different `{expected}`.

`{expected}` lives in the dataset as `reference`.

## The loop

1. Write the process for this agent (`llm-judge rubric`).
2. Write `{expected}` for a question (messy is OK).
3. Run. The judge compares output to `{expected}` using your process.
4. Read the report: change the **prompt**, the **retrieved context**, the **question**, or the **model**.
5. If `{expected}` was wrong, rewrite it. Run again.

## Install

```bash
pip install -e .            # offline: mock models + heuristic judge
pip install -e ".[all]"     # + OpenAI, Anthropic, YAML datasets
```

Python 3.10+. The only hard dependency is `pydantic`.

## Happy path (CLI)

```bash
# Dataset of questions + {expected} answers
llm-judge init data/example.jsonl

# Project process — what THIS agent is supposed to do
llm-judge rubric init
llm-judge rubric add --name task --description "Did it actually do the job?"
llm-judge rubric add --name grounding --description "Did it only use the given context?"

# Extra rules for one use case (local file, not committed)
llm-judge rubric add --case hamlet-author --name author --description "Must name Shakespeare"

# Run. --rubric also picks up data/use-cases/*.json
llm-judge run --dataset data/example.jsonl --rubric data/rubric.json
```

That is the whole flow. The CLI is the wrapper so every rubric file looks the same.

## Where the rules live

| File | Committed? | What it is |
| --- | --- | --- |
| `data/rubric.json` | Yes, if you want to share the project process | Shared “what this agent does” |
| `data/use-cases/<id>.json` | **No** (gitignored) | Unique rules for one use case |
| `data/rubric.example.json` | Yes | The project file shape |
| `data/use-case.example.json` | Yes | The use-case file shape |

Use-case files are yours. They stay local. The CLI always writes this shape:

```json
{
  "id": "hamlet-author",
  "criteria": [
    {
      "name": "author",
      "description": "The answer must name William Shakespeare.",
      "weight": 1.0,
      "pass_threshold": 0.5
    }
  ]
}
```

Do not invent a format. `llm-judge rubric add --case …` writes that file.

```bash
llm-judge rubric list
llm-judge rubric add --name task --description "Updated wording"
llm-judge rubric remove --name author --case hamlet-author
```

`--system-prompt` on `run` is the **agent’s** prompt (one input you might change).
It is not the process. The process is the rubric.

## `{expected}` is just a field

One line of JSONL:

```json
{"input": "Who wrote Hamlet?", "reference": "William Shakespeare wrote Hamlet."}
```

- `input` — the question you send the agent
- `reference` — `{expected}`. A sentence, a paragraph, a list of facts. Not a regex.
- `context` — optional. The docs you retrieved. Include this if you want the judge to blame retrieval vs. the prompt.

Hate that `{expected}`? Change `reference` and rerun.

## What you get back

```
Recommendations (highest impact first)
----------------------------------------------------------------
1. [retrieval] severity=0.61
   Finding : 'faithfulness' failed in 3/4 cases.
   Fix     : Answers aren't grounded in the retrieved context. Improve retrieval...
```

| It says | Change this |
| --- | --- |
| `retrieval` | the docs / chunks you inject at runtime |
| `static_prompt` | the fixed instructions for this agent |
| `input` | the question itself |
| `model` | which model or settings you used |

## Cheap run (no custom process)

Skip the rubric. The heuristic judge just compares output to `{expected}`:

```bash
llm-judge run --dataset data/example.jsonl
```

Real model as the agent:

```bash
export OPENAI_API_KEY=...
llm-judge run --dataset data/example.jsonl --rubric data/rubric.json \
    --target-model gpt-4o-mini --target-provider openai \
    --system-prompt "Answer in one sentence from the given context." \
    --judge-model gpt-4o --judge-provider openai
```

## Python

Same two objects: a process (`Rubric` / `Criterion`) and a dataset of `{expected}`.

```python
from llm_judge import (
    Criterion,
    Dataset,
    Rubric,
    RubricJudge,
    FunctionTarget,
    run_benchmark,
    render,
)

process = Rubric(criteria=[
    Criterion(name="task", description="Did it actually do the job?"),
])
ds = Dataset.load("data/example.jsonl")

def my_agent(case):
    return "William Shakespeare wrote Hamlet."

print(render(run_benchmark(FunctionTarget(my_agent), RubricJudge(rubric=process), ds)))
```

## Wrap a real agent

If your agent is more than one LLM call, record the inputs the judge can blame.
See [`examples/agentic_pipeline.py`](examples/agentic_pipeline.py).

```python
from llm_judge import TraceRecorder, ModelSpec, Message, Role

rec = TraceRecorder(model=ModelSpec(name="my-agent", provider="openai"), case_id=case.id)
rec.set_static_prompt("responder", RESPONDER_PROMPT)
rec.set_initial_context([Message(role=Role.USER, content=question)])

with rec.step("retriever") as step:
    step.retrieved_context = my_retriever(question)

trace = rec.finish(final_output=answer)
```

Keep static prompts and retrieved context separate. That is how the judge
chooses `static_prompt` vs `retrieval`.

## Judges

| Judge | Needs a model? | Use when |
| --- | --- | --- |
| `heuristic` | No | Fast local loop. Keyword overlap against `{expected}`. Ignores your process. |
| `rubric` | Optional | Happy path. Grades against **your** process. Picked automatically when `--rubric` is set. |

## Providers

The agent (target) and the judge are separate. They do not have to be the same model.

| `provider` | Backend | Extra | Auth |
| --- | --- | --- | --- |
| `mock` (default) | offline | — | — |
| `openai`, `vllm`, `ollama`, `local`, … | OpenAI-compatible | `pip install '.[openai]'` | `OPENAI_API_KEY`, `LLM_JUDGE_BASE_URL` |
| `anthropic`, `claude`, `sonnet` | Anthropic | `pip install '.[anthropic]'` | `ANTHROPIC_API_KEY` |

## Tests

```bash
pip install -e ".[dev]"
pytest
```

All tests run offline.

## License

MIT
