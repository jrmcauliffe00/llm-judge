"""Benchmark inputs: test cases and datasets.

A :class:`TestCase` is one thing you want your system to handle. It can carry:
- the user ``input`` (string or full message list),
- optional ``reference`` (gold/expected answer) for reference-based grading,
- optional ``context`` you already retrieved (so you can benchmark the
  generator independently of the retriever),
- ``tags`` and ``metadata`` for slicing results.

Datasets load from JSONL or YAML/JSON so you can keep them in version control.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Iterator, Optional, Union

from pydantic import BaseModel, Field

from .schema import Message, RetrievedChunk, Role, _new_id


class TestCase(BaseModel):
    # Tell pytest this is not a test class (name starts with "Test").
    __test__ = False

    id: str = Field(default_factory=_new_id)
    # Either a plain user prompt or a full conversation.
    input: Union[str, list[Message]]
    reference: Optional[str] = None  # gold answer, if you have one
    # Pre-supplied retrieval context (optional): benchmark generation in isolation.
    context: list[RetrievedChunk] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def input_messages(self, system_prompt: Optional[str] = None) -> list[Message]:
        """Normalize ``input`` into a message list, optionally prepending a
        system prompt."""
        msgs: list[Message] = []
        if system_prompt:
            msgs.append(Message(role=Role.SYSTEM, content=system_prompt))
        if isinstance(self.input, str):
            msgs.append(Message(role=Role.USER, content=self.input))
        else:
            msgs.extend(self.input)
        return msgs


class Dataset(BaseModel):
    name: str = "dataset"
    cases: list[TestCase] = Field(default_factory=list)

    def __iter__(self) -> Iterator[TestCase]:  # type: ignore[override]
        return iter(self.cases)

    def __len__(self) -> int:
        return len(self.cases)

    @classmethod
    def from_cases(cls, cases: Iterable[TestCase], name: str = "dataset") -> "Dataset":
        return cls(name=name, cases=list(cases))

    @classmethod
    def load(cls, path: Union[str, Path]) -> "Dataset":
        """Load a dataset from ``.jsonl``, ``.json``, or ``.yaml``/``.yml``."""
        path = Path(path)
        suffix = path.suffix.lower()
        name = path.stem

        if suffix == ".jsonl":
            rows = [json.loads(line) for line in _nonempty_lines(path)]
        elif suffix == ".json":
            data = json.loads(path.read_text())
            rows = data["cases"] if isinstance(data, dict) and "cases" in data else data
        elif suffix in (".yaml", ".yml"):
            rows = _load_yaml(path)
        else:
            raise ValueError(f"Unsupported dataset format: {suffix}")

        return cls(name=name, cases=[TestCase(**row) for row in rows])

    def save(self, path: Union[str, Path]) -> None:
        path = Path(path)
        if path.suffix.lower() == ".jsonl":
            with path.open("w") as f:
                for case in self.cases:
                    f.write(case.model_dump_json() + "\n")
        else:
            path.write_text(self.model_dump_json(indent=2))


def _nonempty_lines(path: Path) -> Iterable[str]:
    for line in path.read_text().splitlines():
        if line.strip():
            yield line


def _load_yaml(path: Path) -> list[dict]:
    try:
        import yaml
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "PyYAML is required to load YAML datasets. Install with: pip install pyyaml"
        ) from e
    data = yaml.safe_load(path.read_text())
    if isinstance(data, dict) and "cases" in data:
        return data["cases"]
    return data
