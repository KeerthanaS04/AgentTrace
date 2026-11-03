"""P3-T3: aligning an original and a replay run yields the correct changed/unchanged set."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from fake_llm import TICKET, default_script, make_gateway

from agenttrace.graph import AlignmentError, align_runs
from agenttrace.schemas.run import Run
from agenttrace.workflow.support import build_support_workflow, make_resources

UPSTREAM = {
    "__input__",
    "router",
    "router.llm",
    "retriever",
    "retriever.llm",
    "retriever.search",
    "retriever.search#2",
    "retriever.search#3",
    "tool_agent",
    "tool_agent.llm",
    "tool_agent.order_lookup",
    "tool_agent.account_status",
    "tool_agent.llm#2",
    "tool_agent.refund_api",
}
DOWNSTREAM = {"policy", "policy.llm", "writer", "writer.llm"}


def deny_policy(prompt: str) -> dict[str, Any]:
    out = default_script(prompt)
    if prompt.startswith("You are the Policy Agent"):
        out = {**out, "decision": "deny", "refund_amount": None}
    return out


def run(
    tmp_path: Path,
    run_id: str,
    script: Any = default_script,
    roles: dict[str, str] | None = None,
) -> Run:
    gateway = make_gateway(tmp_path, script=script)
    result = build_support_workflow(roles).run(
        {"ticket": TICKET}, run_id=run_id, resources=make_resources(gateway)
    )
    return result.to_run()


def as_replay(original: Run, replay: Run, patched_node: str) -> Run:
    """Mark `replay` as a replay of `original` with `patched_node`'s output replaced."""
    spans = [s.model_copy(update={"patched": s.node_id == patched_node}) for s in replay.spans]
    return replay.model_copy(update={"replay_of": original.run_id, "spans": spans})


# ---------------------------------------------------------------- the done-when


def test_original_vs_replay_changed_and_unchanged_sets(tmp_path: Path) -> None:
    original = run(tmp_path / "o", "orig")
    replay = as_replay(original, run(tmp_path / "r", "orig-replay", deny_policy), "policy")
    al = align_runs(original, replay)

    # Everything upstream of the patched Policy Agent is untouched; it and everything
    # downstream of it changed.
    assert al.unchanged == UPSTREAM
    assert al.changed == DOWNSTREAM
    assert al.only_a == set() and al.only_b == set()
    assert al.unchanged | al.changed == {s.span_id.removeprefix("orig/") for s in original.spans}
    assert not al.identical


def test_changes_say_what_differed(tmp_path: Path) -> None:
    original = run(tmp_path / "o", "orig")
    replay = as_replay(original, run(tmp_path / "r", "orig-replay", deny_policy), "policy")
    al = align_runs(original, replay)
    # The model's reply changed (same prompt); the agent's output changed and it is the
    # patched span; the writer got a different decision, and (with this scripted model)
    # wrote the same reply, so only its input changed.
    assert al.pair("policy.llm").changes == ("output",)
    assert al.pair("policy").changes == ("output", "patched")
    assert al.pair("writer").changes == ("input",)
    assert al.pair("writer.llm").changes == ("input",)
    assert al.pair("router").changes == () and al.pair("router").status == "unchanged"


def test_pairs_follow_causal_order_and_name_both_spans(tmp_path: Path) -> None:
    original = run(tmp_path / "o", "orig")
    replay = run(tmp_path / "r", "orig-replay", deny_policy)
    al = align_runs(original, replay)
    locals_ = [p.local_id for p in al.pairs]
    assert locals_[0] == "__input__" and locals_.index("policy") < locals_.index("writer")
    first = al.pair("policy")
    assert (first.span_a, first.span_b, first.node_id) == (
        "orig/policy",
        "orig-replay/policy",
        "policy",
    )


# ---------------------------------------------------------------- other comparisons


def test_identical_runs_align_as_unchanged(tmp_path: Path) -> None:
    # Same cache dir: the second run is served from cache, which must not count.
    a = run(tmp_path, "a")
    b = run(tmp_path, "b")
    al = align_runs(a, b)
    assert al.identical and al.changed == set() and len(al.unchanged) == len(a.spans)


def test_extra_call_in_b_is_only_b_and_marks_its_node_changed(tmp_path: Path) -> None:
    def three_rounds(prompt: str) -> dict[str, Any]:
        out = default_script(prompt)
        calls = out.get("calls") if prompt.startswith("You are the Tool Agent") else None
        if calls and calls[0]["tool"] == "refund_api":
            out = {**out, "done": False}  # ask for one more round
        return out

    a = run(tmp_path / "a", "a")
    b = run(tmp_path / "b", "b", three_rounds)
    al = align_runs(a, b)
    assert al.only_b == {"tool_agent.llm#3"}
    nodes = al.node_status()
    assert nodes["tool_agent.llm"] == "changed"  # one more occurrence than in A
    assert nodes["router"] == "unchanged"
    assert al.pairs[-1].local_id == "tool_agent.llm#3"  # B-only spans come last
    # Reversed, the extra call is only in A.
    assert align_runs(b, a).only_a == {"tool_agent.llm#3"}


def test_model_swap_shows_as_model_and_params_changes(tmp_path: Path) -> None:
    """Regression comparison across models: same answers, different model behind policy."""
    a = run(tmp_path / "a", "a")
    b = run(tmp_path / "b", "b", roles={"policy": "agent_alt_2"})
    al = align_runs(a, b)
    assert al.pair("policy.llm").changes == ("input", "model", "params")
    # The agent span reports its LLM calls' model (P2-T1), and its role param differs.
    assert al.pair("policy").changes == ("model", "params")
    assert al.changed == {"policy", "policy.llm"}


def test_different_workflows_are_refused(tmp_path: Path) -> None:
    a = run(tmp_path / "a", "a")
    other = a.model_copy(update={"run_id": "x", "workflow_version": "support@2"})
    with pytest.raises(AlignmentError, match="different workflows"):
        align_runs(a, other)


# ---------------------------------------------------------------- CLI


def test_cli_diff(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    from agenttrace import cli
    from agenttrace.store import TraceStore

    original = run(tmp_path / "o", "orig")
    replay = as_replay(original, run(tmp_path / "r", "orig-replay", deny_policy), "policy")
    db = str(tmp_path / "d.db")
    with TraceStore.open(db) as store:
        store.save_run(original)
        store.save_run(replay)
    assert cli.main(["diff", "orig", "orig-replay", "--db", db]) == 0
    out = capsys.readouterr().out
    assert "unchanged 14, changed 4, only in A 0, only in B 0" in out
    assert "CHANGED  policy  (output, patched)" in out
    assert "router" not in out.split("\n\n")[1]  # unchanged spans hidden by default
    assert "first divergence: policy.llm (output_changed)" in out
    assert cli.main(["diff", "orig", "orig", "--all", "--db", db]) == 0
    assert "every span matches" in capsys.readouterr().out
    assert cli.main(["diff", "orig", "missing", "--db", db]) == 2
