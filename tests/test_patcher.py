"""P5-T2: a patch's replay plan freezes, patches, and re-runs exactly the right spans."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import pytest
from fake_llm import make_gateway

from agenttrace.graph import build_graph, downstream
from agenttrace.replay.patcher import (
    Patch,
    PatchError,
    ReplayPlan,
    downstream_steps,
    patch_fields,
    plan_patch,
)
from agenttrace.schemas.run import Run
from agenttrace.schemas.span import Explanation
from agenttrace.tracing import traced
from agenttrace.workflow.runtime import (
    INPUT_STEP,
    AgentOutput,
    FromInput,
    FromStep,
    Step,
    StepContext,
    Workflow,
)

EXPL = Explanation(rationale="r", confidence=1.0)


class Lookup:
    """A traced tool: each call is a span nested in the calling step (`<step>.lookup`)."""

    @traced("tool", "lookup")
    def call(self, key: str) -> dict[str, Any]:
        return {"value": key.upper()}


class Fn:
    """A deterministic agent: `fn(inputs, lookup)` -> output dict."""

    def __init__(
        self,
        name: str,
        inputs: tuple[str, ...],
        outputs: tuple[str, ...],
        fn: Callable[[Mapping[str, Any], Lookup], dict[str, Any]],
    ) -> None:
        self.name, self.input_keys, self.output_keys, self.fn = name, inputs, outputs, fn

    def run(self, inputs: Mapping[str, Any], ctx: StepContext) -> AgentOutput:
        return AgentOutput(output=self.fn(inputs, ctx.resource("lookup", Lookup)), explanation=EXPL)


def concat(*keys: str) -> Callable[[Mapping[str, Any], Lookup], dict[str, Any]]:
    """An agent that joins its inputs (and names its single output after its step)."""

    def fn(inputs: Mapping[str, Any], _: Lookup) -> dict[str, Any]:
        return {k: "+".join(str(inputs[i]) for i in sorted(inputs)) for k in keys}

    return fn


def step(name: str, sources: dict[str, Any], fn: Any = None) -> Step:
    out = name.lower()
    agent = Fn(name, tuple(sources), (out,), fn or concat(out))
    return Step(name, agent, sources)


def build(name: str, steps: list[Step], final: str) -> Workflow:
    return Workflow(name, ["x"], steps, {"out": FromStep(final, final.lower())})


def record(workflow: Workflow, x: str = "hi") -> Run:
    result = workflow.run({"x": x}, run_id="r", resources={"lookup": Lookup()})
    return result.to_run()


X = FromInput("x")
CHAIN = build(
    "chain",
    [
        step("A", {"x": X}),
        step("B", {"a": FromStep("A", "a")}),
        step("C", {"b": FromStep("B", "b")}),
    ],
    "C",
)
DIAMOND = build(
    "diamond",
    [
        step("A", {"x": X}),
        step("B", {"a": FromStep("A", "a")}),
        step("C", {"a": FromStep("A", "a")}),
        step("D", {"b": FromStep("B", "b"), "c": FromStep("C", "c")}),
    ],
    "D",
)
BRANCH = build(  # B never reads A
    "branch",
    [step("A", {"x": X}), step("B", {"x": X}), step("C", {"a": FromStep("A", "a")})],
    "C",
)


def two_lookups(inputs: Mapping[str, Any], lookup: Lookup) -> dict[str, Any]:
    first = lookup.call(str(inputs["x"]))
    second = lookup.call(first["value"] + "!")
    return {"a": second["value"]}


NESTED = build(
    "nested", [step("A", {"x": X}, two_lookups), step("B", {"a": FromStep("A", "a")})], "B"
)


def fails_on_boom(inputs: Mapping[str, Any], _: Lookup) -> dict[str, Any]:
    if inputs["x"] == "boom":
        raise RuntimeError("A failed")
    return {"a": inputs["x"]}


FAILING = build(
    "failing",
    [
        step("A", {"x": X}, fails_on_boom),
        step("B", {"a": FromStep("A", "a")}),
        step("C", {"b": FromStep("B", "b")}),
    ],
    "C",
)


def sets(plan: ReplayPlan) -> tuple[set[str], str, set[str]]:
    """(frozen, patched, re-run) as local ids."""
    return (
        {plan.local(s) for s in plan.frozen},
        plan.target_local,
        {plan.local(s) for s in plan.rerun},
    )


def plan_keys(workflow: Workflow, target: str, run: Run | None = None) -> ReplayPlan:
    """Patch a step with a changed value under its own output key."""
    run = run or record(workflow)
    key = target.lower()
    return plan_patch(run, workflow, patch_fields(run, target, {key: "patched"}))


# ---------------------------------------------------------------- hand-built graphs


def test_chain() -> None:
    assert sets(plan_keys(CHAIN, "B")) == ({INPUT_STEP, "A"}, "B", {"C"})
    assert sets(plan_keys(CHAIN, "A")) == ({INPUT_STEP}, "A", {"B", "C"})
    assert sets(plan_keys(CHAIN, "C")) == ({INPUT_STEP, "A", "B"}, "C", set())
    plan = plan_keys(CHAIN, "B")
    assert (plan.frozen_steps, plan.rerun_steps, plan.target_step) == (["A"], ["C"], "B")


def test_diamond() -> None:
    assert sets(plan_keys(DIAMOND, "B")) == ({INPUT_STEP, "A", "C"}, "B", {"D"})
    assert sets(plan_keys(DIAMOND, "C")) == ({INPUT_STEP, "A", "B"}, "C", {"D"})
    assert sets(plan_keys(DIAMOND, "A")) == ({INPUT_STEP}, "A", {"B", "C", "D"})


def test_independent_branch_stays_frozen() -> None:
    assert sets(plan_keys(BRANCH, "A")) == ({INPUT_STEP, "B"}, "A", {"C"})
    assert sets(plan_keys(BRANCH, "B")) == ({INPUT_STEP, "A", "C"}, "B", set())


def test_patching_the_workflow_input_reruns_its_consumers() -> None:
    run = record(BRANCH)
    plan = plan_patch(run, BRANCH, Patch(target=INPUT_STEP, output={"x": "other"}))
    assert sets(plan) == (set(), INPUT_STEP, {"A", "B", "C"})
    assert plan.target_kind == "message" and plan.changed_keys == ["x"]


def test_a_nested_call_reruns_its_step_but_not_earlier_calls() -> None:
    run = record(NESTED)
    first = plan_patch(run, NESTED, Patch(target="A.lookup", output={"value": "X"}))
    assert sets(first) == ({INPUT_STEP}, "A.lookup", {"A", "A.lookup#2", "B"})
    second = plan_patch(run, NESTED, Patch(target="A.lookup#2", output={"value": "X"}))
    # The first lookup ran before the second's output existed.
    assert sets(second) == ({INPUT_STEP, "A.lookup"}, "A.lookup#2", {"A", "B"})
    assert second.rerun_steps == ["A", "B"] and second.target_step == "A"


def test_a_patched_step_freezes_its_own_calls() -> None:
    plan = plan_keys(NESTED, "A")
    assert sets(plan) == ({INPUT_STEP, "A.lookup", "A.lookup#2"}, "A", {"B"})


def test_steps_skipped_in_the_original_rerun_when_their_cause_is_patched() -> None:
    run = record(FAILING, "boom")
    status = {s.node_id: (s.error or "")[:7] for s in run.spans if s.parent_span_id is None}
    assert status == {INPUT_STEP: "", "A": "Runtime", "B": "skipped", "C": "skipped"}
    plan = plan_patch(run, FAILING, Patch(target="A", output={"a": "fixed"}))
    assert sets(plan) == ({INPUT_STEP}, "A", {"B", "C"})
    # The recorded refs alone would miss them: skipped steps consumed nothing.
    assert downstream(build_graph(run), "r/A") == set()


def test_declared_downstream_steps() -> None:
    assert downstream_steps(DIAMOND, "A") == ["B", "C", "D"]
    assert downstream_steps(DIAMOND, "C") == ["D"]
    assert downstream_steps(BRANCH, INPUT_STEP) == ["A", "B", "C"]
    assert downstream_steps(CHAIN, "C") == []


def test_every_span_gets_exactly_one_action() -> None:
    for workflow, target in ((DIAMOND, "B"), (NESTED, "A.lookup"), (NESTED, "A")):
        run = record(workflow)
        plan = plan_patch(
            run,
            workflow,
            patch_fields(run, target, {"z": 1})
            if "." in target
            else patch_fields(run, target, {target.lower(): 1}),
        )
        actions = [plan.action(s.span_id) for s in run.spans]
        assert actions.count("patched") == 1
        assert len(plan.frozen) + len(plan.rerun) + 1 == len(run.spans)
        assert not set(plan.frozen) & set(plan.rerun)
    with pytest.raises(KeyError):
        plan.action("r/nope")


def test_changed_keys() -> None:
    run = record(CHAIN)
    assert plan_patch(run, CHAIN, patch_fields(run, "B", {"b": "new"})).changed_keys == ["b"]
    noop = patch_fields(run, "B", {})
    assert plan_patch(run, CHAIN, noop).changed_keys == []


# ---------------------------------------------------------------- refused patches


def test_bad_patches_are_refused() -> None:
    run = record(CHAIN)
    with pytest.raises(PatchError, match="no span 'Z'"):
        plan_patch(run, CHAIN, Patch(target="Z", output={"z": 1}))
    with pytest.raises(PatchError, match=r"exactly the keys \['b'\]"):
        plan_patch(run, CHAIN, Patch(target="B", output={"wrong": 1}))
    with pytest.raises(PatchError, match="is chain@1, the workflow is diamond@1"):
        plan_patch(run, DIAMOND, Patch(target="B", output={"b": 1}))
    nested = record(NESTED)
    with pytest.raises(PatchError, match="empty"):
        plan_patch(nested, NESTED, Patch(target="A.lookup", output={}))
    # A span id works as well as a local id.
    assert plan_patch(run, CHAIN, Patch(target="r/B", output={"b": 1})).target_local == "B"


# ---------------------------------------------------------------- the support workflow


def test_support_workflow_tool_patch(tmp_path: Path) -> None:
    from agenttrace.workflow.support import build_support_workflow, make_resources

    workflow = build_support_workflow()
    ticket = {
        "ticket_id": "T-1",
        "customer_id": "C001",
        "subject": "Return a sweater",
        "body": "I want to return the wool sweater from order O1001 for a refund.",
    }
    result = workflow.run(
        {"ticket": ticket}, run_id="s", resources=make_resources(make_gateway(tmp_path))
    )
    run = result.to_run()
    plan = plan_patch(
        run, workflow, Patch(target="tool_agent.order_lookup", output={"found": False})
    )
    frozen, patched, rerun = sets(plan)
    assert {INPUT_STEP, "router", "router.llm", "retriever", "retriever.llm"} <= frozen
    assert "tool_agent.llm" in frozen  # the round that asked for the lookup came first
    assert {"tool_agent", "tool_agent.account_status", "tool_agent.llm#2"} <= rerun
    assert {"policy", "policy.llm", "writer", "writer.llm"} <= rerun
    assert plan.frozen_steps == ["router", "retriever"]
    assert plan.rerun_steps == ["tool_agent", "policy", "writer"]
    # The plan is P3's blast radius minus the call that preceded the target.
    radius = {plan.local(s) for s in downstream(build_graph(run), plan.target)}
    assert radius - rerun == {"tool_agent.llm"} and rerun <= radius


def test_cli_plan(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    from agenttrace.cli import main
    from agenttrace.store import TraceStore
    from agenttrace.workflow.support import build_support_workflow, make_resources

    ticket = {
        "ticket_id": "T-1",
        "customer_id": "C001",
        "subject": "s",
        "body": "Return the wool sweater from order O1001.",
    }
    result = build_support_workflow().run(
        {"ticket": ticket}, run_id="cli-1", resources=make_resources(make_gateway(tmp_path))
    )
    db = tmp_path / "s.db"
    with TraceStore.open(db) as store:
        store.save_run(result.to_run())
    assert main(["plan", "cli-1", "policy", "--set", 'decision="deny"', "--db", str(db)]) == 0
    out = capsys.readouterr().out
    assert "changed keys : decision" in out and "PATCHED  policy" in out
    assert "re-run   writer.llm" in out and "frozen   policy.llm" in out
    assert main(["plan", "cli-1", "policy", "--set", "nope=1", "--db", str(db)]) == 2
    assert "cannot plan" in capsys.readouterr().out
