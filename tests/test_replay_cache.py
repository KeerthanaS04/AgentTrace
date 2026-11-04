"""P5-T1: the record/replay cache reproduces an unmodified run exactly, with no model call."""

from __future__ import annotations

import itertools
from pathlib import Path
from typing import Any

import pytest
from fake_llm import default_script, make_gateway

from agenttrace.gateway.gateway import Gateway
from agenttrace.graph import align_runs
from agenttrace.replay import ReplayCache, exact_diffs, node_key, replay_run, run_inputs
from agenttrace.replay import check as replay_check
from agenttrace.scenarios.ground_truth import Scenario, load_dataset
from agenttrace.scenarios.verify import verify_scenario
from agenttrace.schemas.run import Run
from agenttrace.tracing.context import ReplayMissError
from agenttrace.workflow.agent_config import load_agents_config
from agenttrace.workflow.support import build_support_workflow, make_resources

DATASET = load_dataset()
ROLES = load_agents_config().profile("default").roles()
WORKFLOW = build_support_workflow(ROLES)


def find(code: str, injector: str | None = None, ticket: str = "T-0101", **params: Any) -> Scenario:
    """A scenario on `ticket` (default T-0101, the ticket the scripted model is written for)."""
    for s in DATASET:
        if s.ground_truth.fault_code != code or s.ticket.ticket_id != ticket:
            continue
        if injector is None or any(
            f.injector == injector and all(f.params.get(k) == v for k, v in params.items())
            for f in s.faults
        ):
            return s
    raise LookupError((code, injector, ticket, params))


CONTROL = find("F0")


def recorded(tmp_path: Path, scenario: Scenario, model_params: Any = None, **gw: Any) -> Run:
    """Record a scenario's run (the faulty one, or the control itself)."""
    _, run = verify_scenario(
        make_gateway(tmp_path / "record", **gw), scenario, ROLES, "default", {}, model_params
    )
    assert run is not None
    return run


def offline(tmp_path: Path, bodies: list[dict[str, Any]]) -> dict[str, Any]:
    """Resources whose gateway has an empty cache and logs any request it would send."""
    return make_resources(make_gateway(tmp_path / "replay", bodies=bodies))


def assert_identical(original: Run, replay: Run) -> None:
    assert exact_diffs(original, replay) == []
    assert align_runs(original, replay).identical


# ---------------------------------------------------------------- the done-when


def test_unmodified_run_replays_exactly_without_any_model_call(tmp_path: Path) -> None:
    run = recorded(tmp_path, CONTROL)
    bodies: list[dict[str, Any]] = []
    result, cache = replay_run(run, WORKFLOW, offline(tmp_path, bodies))
    assert_identical(run, result.to_run())
    assert bodies == []  # served from the record, never sent to a provider
    stats = cache.stats()
    assert stats["llm"]["hits"] == sum(s.kind == "llm" for s in run.spans)
    assert stats["agent"]["hits"] == 5 and stats["llm"]["misses"] == 0
    # Replayed calls spent nothing.
    assert all(s.cost_usd == 0 for s in result.spans)


@pytest.mark.parametrize(
    "scenario",
    [
        find("F1", ticket="T-0102"),  # no F1 on T-0101 survived P4-T4
        find("F4", ticket="T-0107"),
        find("F2", "retrieval_empty"),
        find("F3"),
        find("F5", "set_field", field="decision"),
        find("F6"),
        find("F7"),
        find("F8"),
    ],
    ids=lambda s: f"{s.ground_truth.fault_code}-{s.scenario_id}",
)
def test_faulty_runs_replay_exactly_without_their_fault_plan(
    tmp_path: Path, scenario: Scenario
) -> None:
    # The fault is only in the record: tool, search, and agent outputs it changed come
    # back from the cache (an agent's from its nested calls' fingerprints).
    run = recorded(tmp_path, scenario)
    result, _ = replay_run(run, WORKFLOW, offline(tmp_path, []))
    assert_identical(run, result.to_run())


def test_agent_level_fault_is_reproduced_not_recomputed(tmp_path: Path) -> None:
    run = recorded(tmp_path, find("F5", "set_field", field="decision"))
    policy = next(s for s in run.spans if s.node_id == "policy")
    assert policy.output["decision"] == "deny"  # the scripted model says approve
    result, cache = replay_run(run, WORKFLOW, offline(tmp_path, []))
    assert result.outputs["decision"] == "deny"
    assert cache.stats()["agent"]["misses"] == 0


# ---------------------------------------------------------------- seeds


def deny_on_seed_2(prompt: str, seed: Any) -> dict[str, Any]:
    out = default_script(prompt)
    if seed == 2 and prompt.startswith("You are the Policy Agent"):
        out = {**out, "decision": "deny", "refund_amount": None}
    return out


def test_a_seeded_run_replays_with_its_own_seed(tmp_path: Path) -> None:
    run = recorded(tmp_path, CONTROL, {"seed": 2}, seeded=deny_on_seed_2)
    assert run.model_params == {"seed": 2} and run.final_output["decision"] == "deny"
    result, _ = replay_run(run, WORKFLOW, offline(tmp_path, []))
    assert_identical(run, result.to_run())
    # Without its seed no recorded model call matches.
    unseeded = run.model_copy(update={"model_params": {}, "seed": None})
    with pytest.raises(ReplayMissError, match="router.llm"):
        replay_run(unseeded, WORKFLOW, offline(tmp_path, []))


def test_runs_of_different_seeds_share_a_cache_safely(tmp_path: Path) -> None:
    # Same node, inputs and step params; only the nested calls differ. The agent key's
    # fingerprints keep each seed's Policy output apart.
    one = recorded(tmp_path / "1", CONTROL, {"seed": 1}, seeded=deny_on_seed_2)
    two = recorded(tmp_path / "2", CONTROL, {"seed": 2}, seeded=deny_on_seed_2)
    for first, then in ((one, two), (two, one)):
        cache = ReplayCache.from_runs([then, first], strict=True)
        result, _ = replay_run(first, WORKFLOW, offline(tmp_path, []), cache)
        assert_identical(first, result.to_run())


# ---------------------------------------------------------------- misses


def edited(run: Run, body: str) -> Run:
    """The run with a different ticket body in its recorded input."""
    spans = []
    for span in run.spans:
        if span.node_id == "__input__":
            ticket = {**span.output["ticket"], "body": body}
            span = span.model_copy(update={"output": {"ticket": ticket}})
        spans.append(span)
    return run.model_copy(update={"spans": spans})


def test_strict_replay_of_a_changed_input_raises_instead_of_calling(tmp_path: Path) -> None:
    run = recorded(tmp_path, CONTROL)
    changed = edited(run, "Return the wool sweater from order O1001, please.")
    bodies: list[dict[str, Any]] = []
    cache = ReplayCache.from_runs([run], strict=True)
    with pytest.raises(ReplayMissError, match="router.llm"):  # not an agent failure
        replay_run(changed, WORKFLOW, offline(tmp_path, bodies), cache)
    assert bodies == []


def test_record_mode_runs_misses_live_and_files_them(tmp_path: Path) -> None:
    run = recorded(tmp_path, CONTROL)
    changed = edited(run, "Return the wool sweater from order O1001, please.")
    bodies: list[dict[str, Any]] = []
    cache = ReplayCache.from_runs([run])  # record mode
    result, _ = replay_run(changed, WORKFLOW, offline(tmp_path, bodies), cache, strict=False)
    assert bodies  # the changed prompts went to the model
    assert cache.stats()["llm"]["misses"] > 0
    assert cache.stats()["retrieval"]["hits"] > 0  # the standing query did not change
    # Everything that ran is now recorded: a strict replay of the new run is exact.
    new = result.to_run()
    cache.strict = True
    again, _ = replay_run(new, WORKFLOW, offline(tmp_path, []), cache)
    assert_identical(new, again.to_run())


# ---------------------------------------------------------------- keys and storage


def test_keys_depend_on_node_model_params_and_input_only() -> None:
    base = node_key("router.llm", "llm", "m", {"temperature": 0, "seed": 1}, {"a": 1, "b": 2})
    assert base == node_key(
        "router.llm", "llm", "m", {"seed": 1, "temperature": 0}, {"b": 2, "a": 1}
    )
    assert base != node_key(
        "writer.llm", "llm", "m", {"temperature": 0, "seed": 1}, {"a": 1, "b": 2}
    )
    assert base != node_key(
        "router.llm", "llm", "m2", {"temperature": 0, "seed": 1}, {"a": 1, "b": 2}
    )
    assert base != node_key(
        "router.llm", "llm", "m", {"temperature": 0, "seed": 2}, {"a": 1, "b": 2}
    )
    assert base != node_key(
        "router.llm", "llm", "m", {"temperature": 0, "seed": 1}, {"a": 1, "b": 3}
    )


def test_the_cache_persists_on_disk(tmp_path: Path) -> None:
    run = recorded(tmp_path, CONTROL)
    path = tmp_path / "replay.sqlite"
    first = ReplayCache.from_runs([run], path=path)
    size = len(first)
    first.close()
    reopened = ReplayCache(strict=True, path=path)
    assert len(reopened) == size > 0
    result, _ = replay_run(run, WORKFLOW, offline(tmp_path, []), reopened)
    assert_identical(run, result.to_run())


def test_runs_record_their_model_params_and_old_runs_still_load(tmp_path: Path) -> None:
    run = recorded(tmp_path, CONTROL, {"seed": 3})
    assert run.model_params == {"seed": 3} and run.seed == 3
    doc = run.model_dump(mode="json")
    del doc["model_params"]  # a run stored before P5-T1
    old = Run.model_validate(doc)
    assert old.model_params == {} and old.seed == 3
    assert run_inputs(old)["ticket"]["ticket_id"] == "T-0101"


# ---------------------------------------------------------------- the command


def test_replay_check_command(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from agenttrace.store import TraceStore

    db = tmp_path / "store.db"
    with TraceStore.open(db) as store:
        for scenario in (CONTROL, find("F5", "set_field", field="decision"), find("F3")):
            store.save_run(recorded(tmp_path / scenario.scenario_id, scenario))
    count = itertools.count()
    bodies: list[dict[str, Any]] = []
    monkeypatch.setattr(
        Gateway,
        "from_env",
        classmethod(lambda cls, *a, **k: make_gateway(tmp_path / f"g{next(count)}", bodies=bodies)),
    )
    assert replay_check.main(["--db", str(db)]) == 0
    out = capsys.readouterr().out
    assert "identical (zero diffs): 3/3" in out and "PASS" in out
    assert bodies == []


def test_profile_is_inferred_from_the_models_used(tmp_path: Path) -> None:
    run = recorded(tmp_path, CONTROL)
    gateway = make_gateway(tmp_path / "g")
    agents = load_agents_config()
    assert replay_check.infer_profile(run, agents, gateway) == "default"
    other = run.model_copy(update={"agent_models": {"router": "no/such-model"}})
    assert replay_check.infer_profile(other, agents, gateway) is None
