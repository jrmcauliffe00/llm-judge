"""The process the judge looks for.

Two files, one shape:

- Project process: ``data/rubric.json`` — shared, OK to commit.
- Use-case rules: ``data/use-cases/<id>.json`` — unique to you, gitignored.

Both use the same readable ``criteria`` list. The CLI writes these files so you
never invent a format. See ``data/use-case.example.json``.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional, Union

from pydantic import BaseModel, Field

from .judges.base import Criterion

DEFAULT_PROJECT_PATH = Path("data/rubric.json")
USE_CASES_DIRNAME = "use-cases"


def use_cases_dir(project_path: Union[str, Path]) -> Path:
    """``data/use-cases/`` next to ``data/rubric.json``."""
    return Path(project_path).parent / USE_CASES_DIRNAME


def case_filename(case_id: str) -> str:
    safe = re.sub(r"[^a-zA-Z0-9._-]+", "-", case_id).strip("-._")
    return f"{safe or 'case'}.json"


class UseCaseRubric(BaseModel):
    """One use-case file. Same ``criteria`` shape as the project rubric."""

    id: str
    criteria: list[Criterion] = Field(default_factory=list)

    def add(self, criterion: Criterion) -> str:
        for i, existing in enumerate(self.criteria):
            if existing.name == criterion.name:
                self.criteria[i] = criterion
                return "updated"
        self.criteria.append(criterion)
        return "added"

    def remove(self, name: str) -> bool:
        kept = [c for c in self.criteria if c.name != name]
        if len(kept) == len(self.criteria):
            return False
        self.criteria = kept
        return True

    def get(self, name: str) -> Optional[Criterion]:
        for c in self.criteria:
            if c.name == name:
                return c
        return None

    @classmethod
    def load(cls, path: Union[str, Path]) -> "UseCaseRubric":
        path = Path(path)
        data = _read_payload(path)
        if "id" not in data:
            data["id"] = path.stem
        if isinstance(data.get("criteria"), dict):
            raise ValueError(f"{path} criteria must be a list")
        return cls.model_validate(data)

    def save(self, path: Union[str, Path]) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        _write_payload(path, self.model_dump(mode="json"))


class Rubric(BaseModel):
    """Project process, plus in-memory use-case overlays (loaded from disk).

    ``cases`` is never written back to the project file. Unique use-case rules
    live as their own files under ``use-cases/``.
    """

    criteria: list[Criterion] = Field(default_factory=list)
    cases: dict[str, list[Criterion]] = Field(default_factory=dict)

    def for_case(self, case_id: Optional[str] = None) -> list[Criterion]:
        """Project process, with this use case's rules applied on top."""
        merged = {c.name: c for c in self.criteria}
        if case_id:
            for c in self.cases.get(case_id) or []:
                merged[c.name] = c
        return list(merged.values())

    def add(self, criterion: Criterion, case_id: Optional[str] = None) -> str:
        """Insert or replace a project criterion. Use-case adds go through
        :class:`UseCaseRubric` / :func:`save_use_case`."""
        if case_id is not None:
            raise ValueError("Use-case rules are files. Call save_use_case().")
        for i, existing in enumerate(self.criteria):
            if existing.name == criterion.name:
                self.criteria[i] = criterion
                return "updated"
        self.criteria.append(criterion)
        return "added"

    def remove(self, name: str) -> bool:
        kept = [c for c in self.criteria if c.name != name]
        if len(kept) == len(self.criteria):
            return False
        self.criteria = kept
        return True

    def get(self, name: str) -> Optional[Criterion]:
        for c in self.criteria:
            if c.name == name:
                return c
        return None

    @classmethod
    def load(cls, path: Union[str, Path]) -> "Rubric":
        path = Path(path)
        data = _read_payload(path)
        if isinstance(data, list):
            data = {"criteria": data}
        # Ignore any legacy in-file ``cases`` — those belong on disk now.
        if isinstance(data, dict):
            data = {k: v for k, v in data.items() if k != "cases"}
        return cls.model_validate(data)

    @classmethod
    def load_project(
        cls,
        path: Union[str, Path],
        cases_dir: Optional[Union[str, Path]] = None,
    ) -> "Rubric":
        path = Path(path)
        rubric = cls.load(path) if path.exists() else cls()
        directory = Path(cases_dir) if cases_dir else use_cases_dir(path)
        rubric.cases = _load_use_case_dir(directory)
        return rubric

    def save(self, path: Union[str, Path]) -> None:
        """Write only the project process. Use-case files are saved separately."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        _write_payload(
            path,
            {"criteria": [c.model_dump(mode="json") for c in self.criteria]},
        )


def load_use_case(
    project_path: Union[str, Path],
    case_id: str,
    cases_dir: Optional[Union[str, Path]] = None,
) -> Optional[UseCaseRubric]:
    path = _use_case_path(project_path, case_id, cases_dir)
    if not path.exists():
        return None
    return UseCaseRubric.load(path)


def save_use_case(
    project_path: Union[str, Path],
    use_case: UseCaseRubric,
    cases_dir: Optional[Union[str, Path]] = None,
) -> Path:
    path = _use_case_path(project_path, use_case.id, cases_dir)
    use_case.save(path)
    return path


def delete_use_case_file(
    project_path: Union[str, Path],
    case_id: str,
    cases_dir: Optional[Union[str, Path]] = None,
) -> None:
    path = _use_case_path(project_path, case_id, cases_dir)
    if path.exists():
        path.unlink()


def _use_case_path(
    project_path: Union[str, Path],
    case_id: str,
    cases_dir: Optional[Union[str, Path]] = None,
) -> Path:
    directory = Path(cases_dir) if cases_dir else use_cases_dir(project_path)
    return directory / case_filename(case_id)


def _load_use_case_dir(directory: Path) -> dict[str, list[Criterion]]:
    if not directory.is_dir():
        return {}
    out: dict[str, list[Criterion]] = {}
    for path in sorted(directory.iterdir()):
        if path.suffix.lower() not in {".json", ".yaml", ".yml"}:
            continue
        if path.name.startswith("."):
            continue
        loaded = UseCaseRubric.load(path)
        out[loaded.id] = list(loaded.criteria)
    return out


def _read_payload(path: Path) -> dict:
    suffix = path.suffix.lower()
    text = path.read_text()
    if suffix in {".yaml", ".yml"}:
        data = _load_yaml(path, text)
    elif suffix == ".json":
        data = json.loads(text) if text.strip() else {}
    else:
        raise ValueError(f"Unsupported rubric format: {suffix} (use .json or .yaml)")
    if data is None:
        return {}
    if not isinstance(data, (dict, list)):
        raise ValueError(f"Rubric {path} must be a mapping or a list of criteria.")
    return data


def _write_payload(path: Path, payload: dict) -> None:
    suffix = path.suffix.lower()
    if suffix in {".yaml", ".yml"}:
        _save_yaml(path, payload)
        return
    if suffix == ".json":
        path.write_text(json.dumps(payload, indent=2) + "\n")
        return
    raise ValueError(f"Unsupported rubric format: {suffix} (use .json or .yaml)")


def _load_yaml(path: Path, text: str) -> dict:
    try:
        import yaml
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "PyYAML is required to load YAML rubrics. Install with: pip install pyyaml"
        ) from e
    data = yaml.safe_load(text)
    if data is None:
        return {}
    if not isinstance(data, (dict, list)):
        raise ValueError(f"Rubric {path} must be a mapping or a list of criteria.")
    return data


def _save_yaml(path: Path, payload: dict) -> None:
    try:
        import yaml
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "PyYAML is required to save YAML rubrics. Install with: pip install pyyaml"
        ) from e
    path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True))
