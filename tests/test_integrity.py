"""P2-T5: corrupted-trace fixtures are rejected, with clear errors; sound traces pass."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest
from pydantic import ValidationError
from sqlalchemy import text
from test_export import failed_run, foreign_doc
from test_store import traced_run

from agenttrace import cli
from agenttrace.schemas.run import Run
from agenttrace.store import TraceStore
from agenttrace.tracing.export import from_otel, import_runs
from agenttrace.tracing.integrity import (
    TraceIntegrityError,
    check_integrity,
    validate_run,
)

FIXTURES = Path(__file__).parent / "fixtures" / "traces"


def _load_generator() -> ModuleType:
    spec = importlib.util.spec_from_file_location("make_traces", FIXTURES / "make_traces.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


GEN = _load_generator()
INTEGRITY_CASES = sorted(n for n in GEN.CORRUPTIONS if not n.startswith("schema_"))
SCHEMA_CASES = sorted(n for n in GEN.CORRUPTIONS if n.startswith("schema_"))


def fixture_run(name: str) -> Run:
    doc = json.loads((FIXTURES / f"{name}.json").read_text(encoding="utf-8"))
    return Run.model_validate(doc["run"])


# ---------------------------------------------------------------- the fixtures themselves


def test_fixture_files_match_the_generator() -> None:
    """Committed fixtures are exactly what make_traces.py produces (no silent drift)."""
    docs = GEN.documents()
    on_disk = {p.name for p in FIXTURES.glob("*.json")}
    assert on_disk == set(docs)
    for name, doc in docs.items():
        assert (FIXTURES / name).read_text(encoding="utf-8") == GEN.render(doc), name


def test_valid_fixture_passes() -> None:
    (run,) = import_runs(FIXTURES / "valid.json")
    assert check_integrity(run) == []
    assert validate_run(run) is run


# ---------------------------------------------------------------- the done-when


@pytest.mark.parametrize("name", INTEGRITY_CASES)
def test_corrupted_fixture_is_rejected_with_its_code(name: str) -> None:
    with pytest.raises(TraceIntegrityError) as caught:
        import_runs(FIXTURES / f"{name}.json")
    codes = {i.code for i in caught.value.issues}
    assert name in codes, caught.value
    assert f"[{name}]" in str(caught.value)  # the message names the problem


@pytest.mark.parametrize("name", SCHEMA_CASES)
def test_schema_level_corruption_is_rejected(name: str) -> None:
    with pytest.raises(ValidationError):
        import_runs(FIXTURES / f"{name}.json")


@pytest.mark.parametrize("name", INTEGRITY_CASES)
def test_store_refuses_corrupted_runs(name: str) -> None:
    store = TraceStore.in_memory()
    with pytest.raises(TraceIntegrityError):
        store.save_run(fixture_run(name))
    assert store.run_ids() == []  # nothing was written


# ---------------------------------------------------------------- what each check reports


def test_issue_details_point_at_the_broken_span() -> None:
    (issue,) = check_integrity(fixture_run("orphan_ref"))
    assert issue.code == "orphan_ref" and issue.span_id == "fx/b"
    assert "'fx/ghost'" in issue.message

    (issue,) = check_integrity(fixture_run("ref_from_future"))
    assert issue.span_id == "fx/b" and "finished at 2026-11-03T10:00:00.040000" in issue.message

    (issue,) = check_integrity(fixture_run("child_outside_parent"))
    assert issue.span_id == "fx/b.tool" and "outside parent 'fx/b'" in issue.message


def test_cycles_are_reported_as_paths() -> None:
    issues = {i.code: i for i in check_integrity(fixture_run("parent_cycle"))}
    assert issues["parent_cycle"].message.endswith("fx/a -> fx/a.llm -> fx/a")
    issues = {i.code: i for i in check_integrity(fixture_run("ref_cycle"))}
    assert "fx/a -> fx/b -> fx/a" in issues["ref_cycle"].message
    assert "ref_from_future" in issues  # a loop always also breaks time order


def test_every_issue_is_listed_at_once() -> None:
    run = fixture_run("orphan_ref")
    data = run.model_dump(mode="json")
    data["spans"][2]["parent_span_id"] = "fx/ghost"  # also orphan a.llm
    with pytest.raises(TraceIntegrityError) as caught:
        validate_run(Run.model_validate(data))
    assert [i.code for i in caught.value.issues] == ["orphan_parent", "orphan_ref"]
    assert "failed 2 integrity check(s)" in str(caught.value)


# ---------------------------------------------------------------- sound traces pass


def test_real_workflow_runs_pass(tmp_path: Path) -> None:
    assert check_integrity(traced_run(tmp_path, "ok")) == []
    assert check_integrity(failed_run(tmp_path)) == []  # failure is not corruption


def test_foreign_otel_trace_passes() -> None:
    (run,) = from_otel(foreign_doc())
    assert check_integrity(run) == []


# ---------------------------------------------------------------- store and CLI


def test_corrupted_store_row_fails_on_load(tmp_path: Path) -> None:
    store = TraceStore.open(tmp_path / "s.db")
    store.save_run(traced_run(tmp_path, "r1"))
    with store.engine.begin() as conn:  # delete a parent span behind the store's back
        conn.execute(text("DELETE FROM spans WHERE span_id = 'r1/policy'"))
    with pytest.raises(TraceIntegrityError, match="orphan_parent"):
        store.get_run("r1")


def test_cli_import_reports_integrity_errors(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    db = str(tmp_path / "c.db")
    assert cli.main(["import", str(FIXTURES / "ref_hash_mismatch.json"), "--db", db]) == 2
    out = capsys.readouterr().out
    assert "cannot import" in out and "[ref_hash_mismatch] fx/b:" in out
    assert cli.main(["import", str(FIXTURES / "valid.json"), "--db", db]) == 0
