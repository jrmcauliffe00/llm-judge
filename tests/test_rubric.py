"""Project process + gitignored use-case rubric files."""

from __future__ import annotations

import json
from pathlib import Path

from llm_judge import Criterion, Rubric, RubricJudge, UseCaseRubric
from llm_judge.cli import main
from llm_judge.judges.rubric import DEFAULT_CRITERIA
from llm_judge.models.mock import MockClient
from llm_judge.rubric import case_filename, use_cases_dir
from llm_judge.schema import ModelSpec


def test_example_files_match_the_common_shape():
    project = Rubric.load("data/rubric.example.json")
    assert [c.name for c in project.criteria] == ["task", "grounding"]
    assert "cases" not in json.loads(Path("data/rubric.example.json").read_text())

    use_case = UseCaseRubric.load("data/use-case.example.json")
    assert use_case.id == "hamlet-author"
    assert use_case.criteria[0].name == "author"


def test_project_save_never_writes_use_cases(tmp_path):
    path = tmp_path / "rubric.json"
    rubric = Rubric(criteria=[Criterion(name="task", description="do the job")])
    rubric.cases = {"hamlet": [Criterion(name="author", description="Shakespeare")]}
    rubric.save(path)
    raw = json.loads(path.read_text())
    assert list(raw) == ["criteria"]
    assert "cases" not in raw


def test_use_case_file_roundtrip_and_merge(tmp_path):
    project = tmp_path / "rubric.json"
    Rubric(criteria=[Criterion(name="task", description="do the job")]).save(project)
    assert main([
        "rubric", "add", str(project),
        "--case", "hamlet-author",
        "--name", "author",
        "--description", "Must name Shakespeare",
    ]) == 0

    case_path = use_cases_dir(project) / case_filename("hamlet-author")
    assert case_path.exists()
    payload = json.loads(case_path.read_text())
    assert payload["id"] == "hamlet-author"
    assert payload["criteria"][0]["name"] == "author"

    loaded = Rubric.load_project(project)
    assert [c.name for c in loaded.for_case("other")] == ["task"]
    names = [c.name for c in loaded.for_case("hamlet-author")]
    assert names == ["task", "author"]


def test_cli_update_and_remove_use_case(tmp_path):
    project = tmp_path / "rubric.json"
    assert main(["rubric", "init", str(project)]) == 0
    assert main([
        "rubric", "add", str(project),
        "--name", "task",
        "--description", "Did it do the job?",
    ]) == 0
    assert main([
        "rubric", "add", str(project),
        "--name", "task",
        "--description", "Did it finish the booking?",
    ]) == 0
    assert Rubric.load(project).criteria[0].description == "Did it finish the booking?"

    assert main([
        "rubric", "add", str(project),
        "--case", "booking-1",
        "--name", "when",
        "--description", "Must include a time",
    ]) == 0
    assert main([
        "rubric", "remove", str(project),
        "--case", "booking-1",
        "--name", "when",
    ]) == 0
    assert list(use_cases_dir(project).glob("*.json")) == []


def test_cli_list_and_run_with_rubric(tmp_path, capsys):
    project = tmp_path / "rubric.json"
    dataset = tmp_path / "ds.jsonl"
    dataset.write_text(
        json.dumps({"id": "hamlet-author", "input": "Who wrote Hamlet?", "reference": "Shakespeare"})
        + "\n"
    )
    assert main(["rubric", "init", str(project)]) == 0
    assert main([
        "rubric", "add", str(project),
        "--name", "task",
        "--description", "Name the author",
    ]) == 0
    assert main([
        "rubric", "add", str(project),
        "--case", "hamlet-author",
        "--name", "author",
        "--description", "Must say Shakespeare",
    ]) == 0
    assert main(["rubric", "list", str(project)]) == 0
    listed = capsys.readouterr().out
    assert "task" in listed
    assert "hamlet-author" in listed
    assert "author" in listed

    assert main([
        "run",
        "--dataset", str(dataset),
        "--rubric", str(project),
        "--format", "json",
    ]) == 0


def test_rubric_judge_uses_use_case_overlay():
    rubric = Rubric(
        criteria=[Criterion(name="task", description="do the job")],
        cases={"c1": [Criterion(name="author", description="Shakespeare")]},
    )
    judge = RubricJudge(
        rubric=rubric,
        client=MockClient(ModelSpec(name="mock-judge", provider="mock")),
    )
    from llm_judge import TestCase

    assert [c.name for c in judge.criteria_for(TestCase(id="c1", input="q"))] == [
        "task",
        "author",
    ]
    assert [c.name for c in judge.criteria_for(TestCase(id="other", input="q"))] == ["task"]


def test_empty_rubric_falls_back_to_defaults():
    judge = RubricJudge(
        rubric=Rubric(),
        client=MockClient(ModelSpec(name="mock-judge", provider="mock")),
    )
    assert [c.name for c in judge.criteria_for()] == [c.name for c in DEFAULT_CRITERIA]


def test_heuristic_plus_rubric_is_rejected(tmp_path):
    project = tmp_path / "rubric.json"
    Rubric().save(project)
    ds = tmp_path / "ds.jsonl"
    ds.write_text(json.dumps({"input": "q", "reference": "a"}) + "\n")
    assert main([
        "run",
        "--dataset", str(ds),
        "--rubric", str(project),
        "--judge", "heuristic",
    ]) == 2
