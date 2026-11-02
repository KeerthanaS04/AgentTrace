"""P2-T3: trace store. Runs persist and reload byte-identically; queries work."""

from __future__ import annotations

import itertools
from pathlib import Path

import pytest
from fake_llm import TICKET, make_gateway
from pydantic import ValidationError
from sqlalchemy import text
from test_schemas import make_diagnosis

from agenttrace import cli
from agenttrace.gateway.gateway import Gateway
from agenttrace.schemas import ExplainabilityScore, OverrideInputsPatch, ReplayResult, Run
from agenttrace.store import RunExistsError, StoreError, TraceStore
from agenttrace.store.models import SCHEMA_VERSION
from agenttrace.workflow.support import build_support_workflow, make_resources

# Characters that trip naive storage: non-breaking hyphen, curly quote, emoji.
UNICODE_TICKET = {**TICKET, "body": TICKET["body"] + " It’s a 5‑7 day wait \U0001f600"}


def traced_run(tmp_path: Path, run_id: str, scenario_id: str | None = "clean-01") -> Run:
    gateway = make_gateway(tmp_path / run_id, usage={"prompt_tokens": 11, "completion_tokens": 7})
    result = build_support_workflow().run(
        {"ticket": UNICODE_TICKET}, run_id=run_id, resources=make_resources(gateway)
    )
    return result.to_run(scenario_id=scenario_id, final_correct=True)


@pytest.fixture
def store(tmp_path: Path) -> TraceStore:
    return TraceStore.open(tmp_path / "t.db")


# ---------------------------------------------------------------- the done-when: byte-identical


def test_run_reloads_byte_identically_from_a_reopened_file(tmp_path: Path) -> None:
    run = traced_run(tmp_path, "r1")
    db = tmp_path / "store.db"
    with TraceStore.open(db) as store:
        store.save_run(run)
    with TraceStore.open(db) as reopened:  # a fresh connection: nothing cached in memory
        loaded = reopened.get_run("r1")
    assert loaded.model_dump_json() == run.model_dump_json()
    assert loaded == run
    assert [s.span_id for s in loaded.spans] == [s.span_id for s in run.spans]
    assert "‑" in loaded.spans[0].output["ticket"]["body"]


def test_every_span_reloads_byte_identically(tmp_path: Path, store: TraceStore) -> None:
    run = traced_run(tmp_path, "r1")
    store.save_run(run)
    for span in run.spans:
        assert store.get_span("r1", span.span_id).model_dump_json() == span.model_dump_json()


def test_save_load_save_is_stable(tmp_path: Path) -> None:
    run = traced_run(tmp_path, "r1")
    a, b = TraceStore.in_memory(), TraceStore.in_memory()
    a.save_run(run)
    b.save_run(a.get_run("r1"))
    assert b.get_run("r1").model_dump_json() == run.model_dump_json()


# ---------------------------------------------------------------- queries


@pytest.fixture
def three_runs(tmp_path: Path, store: TraceStore) -> TraceStore:
    store.save_run(traced_run(tmp_path, "a1", "clean-01"))
    store.save_run(traced_run(tmp_path, "b1", "clean-02"))
    store.save_run(traced_run(tmp_path, "a2", "clean-01"))
    return store


def test_query_by_run(three_runs: TraceStore) -> None:
    spans = three_runs.spans(run_id="b1")
    assert spans and {s.run_id for s in spans} == {"b1"}
    assert spans == three_runs.get_run("b1").spans


def test_query_by_scenario(three_runs: TraceStore) -> None:
    assert three_runs.run_ids(scenario_id="clean-01") == ["a1", "a2"]  # oldest first
    assert [r.run_id for r in three_runs.runs(scenario_id="clean-02")] == ["b1"]
    assert three_runs.run_ids(scenario_id="nope") == []
    assert three_runs.run_ids() == ["a1", "b1", "a2"]


def test_query_by_node_across_runs(three_runs: TraceStore) -> None:
    policy = three_runs.spans(node_id="policy")
    assert [s.run_id for s in policy] == ["a1", "b1", "a2"]
    assert all(s.node_id == "policy" and s.kind == "agent" for s in policy)
    # Node + scenario: the Policy Agent in every run of one scenario.
    assert [s.span_id for s in three_runs.spans(node_id="policy", scenario_id="clean-01")] == [
        "a1/policy",
        "a2/policy",
    ]
    # Repeated nodes come back in recorded order.
    assert [s.span_id for s in three_runs.spans(run_id="a1", node_id="tool_agent.llm")] == [
        "a1/tool_agent.llm",
        "a1/tool_agent.llm#2",
    ]


def test_query_by_kind(three_runs: TraceStore) -> None:
    llm = three_runs.spans(run_id="a1", kind="llm")
    assert llm and all(s.kind == "llm" for s in llm)
    assert sum(s.tokens_in for s in llm) == 11 * len(llm)


def test_summaries(three_runs: TraceStore) -> None:
    rows = three_runs.summaries(scenario_id="clean-01")
    n = len(three_runs.get_run("a1").spans)
    assert [(r["run_id"], r["spans"], r["final_correct"]) for r in rows] == [
        ("a1", n, True),
        ("a2", n, True),
    ]


# ---------------------------------------------------------------- writes


def test_duplicate_run_is_refused_unless_replaced(tmp_path: Path, store: TraceStore) -> None:
    run = traced_run(tmp_path, "r1")
    store.save_run(run)
    with pytest.raises(RunExistsError):
        store.save_run(run)
    shorter = run.model_copy(update={"spans": run.spans[:3], "final_correct": False})
    store.save_run(shorter, replace=True)
    assert store.get_run("r1") == shorter
    assert len(store.spans(run_id="r1")) == 3  # old spans are gone


def test_delete_cascades_to_spans(tmp_path: Path, store: TraceStore) -> None:
    store.save_run(traced_run(tmp_path, "r1"))
    store.delete_run("r1")
    assert not store.has_run("r1") and store.spans(run_id="r1") == []
    with pytest.raises(KeyError):
        store.get_run("r1")


def test_next_run_id(tmp_path: Path, store: TraceStore) -> None:
    assert store.next_run_id("clean-07-default") == "clean-07-default-r1"
    store.save_run(traced_run(tmp_path, "clean-07-default-r1"))
    assert store.next_run_id("clean-07-default") == "clean-07-default-r2"


def test_replay_runs_are_queryable_by_origin(tmp_path: Path, store: TraceStore) -> None:
    store.save_run(traced_run(tmp_path, "r1"))
    replay = traced_run(tmp_path, "r1-replay").model_copy(update={"replay_of": "r1"})
    store.save_run(replay)
    assert store.run_ids(replay_of="r1") == ["r1-replay"]


# ---------------------------------------------------------------- scores, diagnoses, replays


def test_scores_round_trip_and_filter(store: TraceStore) -> None:
    span_score = ExplainabilityScore(
        scope="span", run_id="r1", span_id="r1/policy", metric="G", value=0.8, judge="judge_fast"
    )
    run_score = ExplainabilityScore(scope="run", run_id="r1", metric="XI", value=0.7)
    agent_score = ExplainabilityScore(scope="agent", agent="policy", metric="G", value=0.6)
    for score in (span_score, run_score, agent_score):
        store.add_score(score)
    assert store.scores(span_id="r1/policy") == [span_score]
    assert store.scores(run_id="r1") == [span_score, run_score]
    assert store.scores(metric="G") == [span_score, agent_score]
    assert store.scores(agent="policy") == [agent_score]
    assert store.scores()[0].model_dump_json() == span_score.model_dump_json()


def test_diagnoses_round_trip(store: TraceStore) -> None:
    diagnosis = make_diagnosis()
    store.save_diagnosis(diagnosis)
    assert store.get_diagnosis("d1").model_dump_json() == diagnosis.model_dump_json()
    assert store.diagnoses(run_id="r1") == [diagnosis] and store.diagnoses(run_id="x") == []
    with pytest.raises(StoreError):
        store.save_diagnosis(diagnosis)
    store.save_diagnosis(diagnosis.model_copy(update={"narrative": "v2"}), replace=True)
    assert store.get_diagnosis("d1").narrative == "v2"


def test_replays_round_trip(store: TraceStore) -> None:
    replay = ReplayResult(
        original_run_id="r1",
        replay_run_id="r1-replay",
        diagnosis_id="d1",
        patch=OverrideInputsPatch(target_span_id="s2", source="model", input_overrides={"q": "x"}),
        patched_span_id="s2",
        patch_source="model",
        recovered=True,
        recovery_score=1.0,
    )
    store.add_replay(replay)
    assert store.replays(original_run_id="r1") == [replay]
    assert store.replays(diagnosis_id="d1")[0].model_dump_json() == replay.model_dump_json()
    assert store.replays(original_run_id="other") == []


# ---------------------------------------------------------------- failing loudly


def test_corrupted_row_fails_loudly(tmp_path: Path, store: TraceStore) -> None:
    store.save_run(traced_run(tmp_path, "r1"))
    with store.engine.begin() as conn:
        conn.execute(text('UPDATE spans SET doc = \'{"span_id": "x"}\' WHERE seq = 1'))
    with pytest.raises(ValidationError):
        store.get_run("r1")


def test_schema_version_mismatch_is_refused(tmp_path: Path) -> None:
    db = tmp_path / "old.db"
    with TraceStore.open(db) as store, store.engine.begin() as conn:
        conn.execute(text("UPDATE meta SET value = '0' WHERE key = 'schema_version'"))
    with pytest.raises(StoreError, match=f"expects v{SCHEMA_VERSION}"):
        TraceStore.open(db)


def test_open_uses_env_var(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AGENTTRACE_DB", str(tmp_path / "env.db"))
    with TraceStore.open() as store:
        assert store.url.endswith("env.db")


# ---------------------------------------------------------------- CLI


def use_fake_gateway(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Each `Gateway.from_env()` gets a fresh fake (the CLI closes its gateway)."""
    count = itertools.count()
    monkeypatch.setattr(
        Gateway,
        "from_env",
        classmethod(lambda cls, *a, **k: make_gateway(tmp_path / f"gw{next(count)}")),
    )


def test_cli_run_runs_and_show(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    use_fake_gateway(tmp_path, monkeypatch)
    db = str(tmp_path / "cli.db")

    assert cli.main(["run", "--scenario", "clean-01", "--db", db]) == 0
    out = capsys.readouterr().out
    assert "run       : clean-01-default-r1" in out and "provenance: exact" in out
    assert cli.main(["run", "--scenario", "clean-01", "--db", db]) == 0  # second run, new id
    assert "clean-01-default-r2" in capsys.readouterr().out

    assert cli.main(["runs", "--scenario", "clean-01", "--db", db]) == 0
    listing = capsys.readouterr().out
    assert "clean-01-default-r1" in listing and "clean-01-default-r2" in listing

    assert cli.main(["show", "clean-01-default-r1", "--db", db]) == 0
    tree = capsys.readouterr().out
    assert "scenario=clean-01" in tree and "tool_agent.order_lookup" in tree

    assert cli.main(["show", "clean-01-default-r1", "--node", "policy", "--db", db]) == 0
    assert '"node_id": "policy"' in capsys.readouterr().out

    assert cli.main(["show", "missing", "--db", db]) == 2
    assert cli.main(["run", "--scenario", "nope", "--db", db]) == 2


def test_cli_run_stores_exactly_what_ran(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    use_fake_gateway(tmp_path, monkeypatch)
    db = tmp_path / "cli.db"
    assert cli.main(["run", "--scenario", "clean-05", "--db", str(db)]) == 0
    capsys.readouterr()
    with TraceStore.open(db) as store:
        run = store.get_run("clean-05-default-r1")
    names = {s.node_id for s in run.spans}
    assert {"router", "retriever", "tool_agent", "policy", "writer"} <= names
    assert run.scenario_id == "clean-05" and run.final_correct is not None
