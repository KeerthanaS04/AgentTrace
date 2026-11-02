"""P2-T2: span input_refs exactly match the declared data flow; tampering is caught."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import pytest
from fake_llm import EXPL, TICKET, default_script, make_gateway

from agenttrace.gateway.cache import content_hash
from agenttrace.schemas.run import Run
from agenttrace.schemas.span import Explanation, Ref, Span
from agenttrace.workflow import toy
from agenttrace.workflow.provenance import check_provenance, declared_inputs
from agenttrace.workflow.runtime import (
    INPUT_STEP,
    AgentOutput,
    FromInput,
    FromStep,
    Step,
    StepContext,
    Workflow,
)
from agenttrace.workflow.support import build_support_workflow, make_resources


def support_run(tmp_path: Path, script: Any = default_script) -> Run:
    gateway = make_gateway(tmp_path, script=script)
    result = build_support_workflow().run(
        {"ticket": TICKET}, run_id="r1", resources=make_resources(gateway)
    )
    return result.to_run()


def flow_of(run: Run) -> set[tuple[str, str, str, str]]:
    """(consumer, input_key, producer, key) for every ref in the run."""
    nodes = {s.span_id: s.node_id for s in run.spans}
    return {
        (s.node_id, r.input_key or "?", nodes[r.span_id], r.key)
        for s in run.spans
        for r in s.input_refs
    }


def declared_flow(wf: Workflow) -> set[tuple[str, str, str, str]]:
    return {(d.consumer, d.input_key, d.producer, d.key) for d in declared_inputs(wf)}


# ---------------------------------------------------------------- the done-when


def test_clean_run_refs_exactly_match_declared_data_flow(tmp_path: Path) -> None:
    run = support_run(tmp_path)
    wf = build_support_workflow()
    assert flow_of(run) == declared_flow(wf)  # exact, input by input
    assert check_provenance(run, wf) == []
    # Spot checks against the wiring in support.py.
    flow = flow_of(run)
    assert ("writer", "decision", "policy", "decision") in flow
    assert ("policy", "tool_calls", "tool_agent", "tool_calls") in flow
    assert not any(c == "writer" and p == "tool_agent" for c, _, p, _ in flow)


def test_ref_hashes_chain_producer_output_to_consumer_input(tmp_path: Path) -> None:
    run = support_run(tmp_path)
    spans = {s.span_id: s for s in run.spans}
    n = 0
    for s in run.spans:
        for r in s.input_refs:
            assert r.input_key is not None
            assert content_hash(spans[r.span_id].output[r.key]) == r.content_hash
            assert content_hash(s.input[r.input_key]) == r.content_hash
            n += 1
    assert n == len(declared_inputs(build_support_workflow())) == 16  # 1 + 2 + 3 + 4 + 6 inputs


def test_nested_spans_carry_no_refs(tmp_path: Path) -> None:
    """Data flow is between steps; llm/tool/search spans relate to their step by nesting."""
    run = support_run(tmp_path)
    assert all(not s.input_refs for s in run.spans if s.kind != "agent")


def test_failed_and_skipped_steps_still_have_exact_provenance(tmp_path: Path) -> None:
    def broken_policy(prompt: str) -> dict[str, Any]:
        if prompt.startswith("You are the Policy Agent"):
            return {**EXPL}
        return default_script(prompt)

    run = support_run(tmp_path, broken_policy)
    nodes = {s.node_id: s for s in run.spans}
    assert nodes["policy"].error and nodes["policy"].input_refs  # it ran and failed
    assert nodes["writer"].error and not nodes["writer"].input_refs  # skipped
    assert check_provenance(run, build_support_workflow()) == []


# ---------------------------------------------------------------- not execution order


class Emit:
    """Agent that returns fixed outputs computed from its inputs."""

    def __init__(
        self,
        name: str,
        inputs: tuple[str, ...],
        outputs: tuple[str, ...],
        fn: Callable[[Mapping[str, Any]], dict[str, Any]],
    ) -> None:
        self.name, self.input_keys, self.output_keys, self.fn = name, inputs, outputs, fn

    def run(self, inputs: Mapping[str, Any], ctx: StepContext) -> AgentOutput:
        return AgentOutput(
            output=self.fn(inputs), explanation=Explanation(rationale="", confidence=1.0)
        )


def test_refs_follow_the_declaration_not_the_most_recent_producer() -> None:
    """Both `a` and `b` emit `y`; `c` is wired to `a`. Execution order (a, b, c) would
    suggest `b` (the latest `y`); the ref must still name `a`."""
    wf = Workflow(
        "latest",
        inputs=["x"],
        steps=[
            Step(
                "a", Emit("a", ("x",), ("y",), lambda i: {"y": i["x"] + 1}), {"x": FromInput("x")}
            ),
            Step(
                "b", Emit("b", ("x",), ("y",), lambda i: {"y": i["x"] + 2}), {"x": FromInput("x")}
            ),
            Step(
                "c", Emit("c", ("v",), ("z",), lambda i: {"z": i["v"]}), {"v": FromStep("a", "y")}
            ),
        ],
        outputs={"z": FromStep("c", "z")},
    )
    run = wf.run({"x": 1}, run_id="o").to_run()
    (ref,) = next(s for s in run.spans if s.node_id == "c").input_refs
    assert (ref.span_id, ref.key, ref.input_key) == ("o/a", "y", "v")
    assert ref.content_hash == content_hash(2) and run.final_output == {"z": 2}
    assert check_provenance(run, wf) == []


def test_renamed_inputs_and_diamond_shapes() -> None:
    """Consumer input names differ from producer keys; declared order != run order."""
    wf = Workflow(
        "diamond",
        inputs=["x"],
        steps=[
            Step(
                "join",
                Emit("join", ("l", "r"), ("sum",), lambda i: {"sum": i["l"] + i["r"]}),
                {"l": FromStep("left", "y"), "r": FromStep("right", "y")},
            ),
            Step(
                "left",
                Emit("left", ("x",), ("y",), lambda i: {"y": i["x"] * 2}),
                {"x": FromInput("x")},
            ),
            Step(
                "right",
                Emit("right", ("x",), ("y",), lambda i: {"y": i["x"] * 3}),
                {"x": FromInput("x")},
            ),
        ],
        outputs={"total": FromStep("join", "sum")},
    )
    run = wf.run({"x": 1}, run_id="d").to_run()
    assert (
        flow_of(run)
        == declared_flow(wf)
        == {
            ("left", "x", INPUT_STEP, "x"),
            ("right", "x", INPUT_STEP, "x"),
            ("join", "l", "left", "y"),
            ("join", "r", "right", "y"),
        }
    )
    assert check_provenance(run, wf) == []


def test_toy_workflow_provenance_is_exact() -> None:
    run = toy.run_toy("I want to return my headphones for a refund", run_id="t").to_run()
    assert check_provenance(run, toy.build_toy_workflow()) == []


# ---------------------------------------------------------------- tampered traces


def tamper(run: Run, node: str, **update: Any) -> Run:
    spans = [s.model_copy(update=update) if s.node_id == node else s for s in run.spans]
    return run.model_copy(update={"spans": spans})


def writer_refs(run: Run) -> list[Ref]:
    return list(next(s for s in run.spans if s.node_id == "writer").input_refs)


def edit_ref(run: Run, which: str, **update: Any) -> Run:
    refs = [r.model_copy(update=update) if r.input_key == which else r for r in writer_refs(run)]
    return tamper(run, "writer", input_refs=refs)


@pytest.fixture
def clean(tmp_path: Path) -> Run:
    return support_run(tmp_path)


WF = build_support_workflow()


def test_wrong_producer_is_caught(clean: Run) -> None:
    bad = edit_ref(clean, "decision", span_id="r1/router")
    assert check_provenance(bad, WF) == [
        "writer.decision: came from router.decision, declared policy.decision"
    ]


def test_wrong_producer_key_is_caught(clean: Run) -> None:
    bad = edit_ref(clean, "decision", key="justification")
    assert check_provenance(bad, WF)[0].startswith(
        "writer.decision: came from policy.justification"
    )


def test_missing_and_extra_refs_are_caught(clean: Run) -> None:
    refs = [r for r in writer_refs(clean) if r.input_key != "docs"]
    assert check_provenance(tamper(clean, "writer", input_refs=refs), WF) == [
        "writer.docs: declared input has no ref"
    ]
    extra = Ref(span_id="r1/tool_agent", key="tool_calls", content_hash="h", input_key="tool_calls")
    refs = [*writer_refs(clean), extra]
    assert check_provenance(tamper(clean, "writer", input_refs=refs), WF) == [
        "writer.tool_calls: not a declared input"
    ]


def test_duplicate_and_unbound_refs_are_caught(clean: Run) -> None:
    first = writer_refs(clean)[0]
    dup = tamper(clean, "writer", input_refs=[*writer_refs(clean), first])
    assert check_provenance(dup, WF) == [f"writer.{first.input_key}: more than one ref"]
    unbound = edit_ref(clean, "decision", input_key=None)
    assert "writer: ref to policy.decision names no input" in check_provenance(unbound, WF)


def test_hash_mismatches_are_caught(clean: Run) -> None:
    bad_hash = edit_ref(clean, "decision", content_hash="sha256:forged")
    assert check_provenance(bad_hash, WF) == [
        "writer.decision: ref hash differs from what policy produced",
        "writer.decision: recorded input differs from the value referenced",
    ]
    # The value the writer recorded is not the one policy produced (silent swap).
    writer = next(s for s in clean.spans if s.node_id == "writer")
    swapped = tamper(clean, "writer", input={**writer.input, "decision": "deny"})
    assert check_provenance(swapped, WF) == [
        "writer.decision: recorded input differs from the value referenced"
    ]
    # The producer's recorded output was edited after the fact.
    policy = next(s for s in clean.spans if s.node_id == "policy")
    edited = tamper(clean, "policy", output={**policy.output, "decision": "deny"})
    assert "writer.decision: ref hash differs from what policy produced" in check_provenance(
        edited, WF
    )


def test_missing_spans_are_caught(clean: Run) -> None:
    no_policy = clean.model_copy(
        update={"spans": [s for s in clean.spans if s.node_id not in ("policy", "policy.llm")]}
    )
    problems = check_provenance(no_policy, WF)
    assert "policy: no agent span" in problems
    assert "writer.decision: producer span policy is missing" in problems
    no_input = clean.model_copy(
        update={"spans": [s for s in clean.spans if s.node_id != INPUT_STEP]}
    )
    assert "no span for the workflow input" in check_provenance(no_input, WF)


def test_skipped_step_with_refs_and_version_mismatch_are_caught(clean: Run) -> None:
    writer = next(s for s in clean.spans if s.node_id == "writer")
    fake_skip: Span = writer.model_copy(update={"error": "skipped: upstream not ok"})
    run = clean.model_copy(update={"spans": [fake_skip if s is writer else s for s in clean.spans]})
    assert check_provenance(run, WF) == ["writer: skipped, but has input refs"]
    other = clean.model_copy(update={"workflow_version": "support@2"})
    assert check_provenance(other, WF) == ["run is 'support@2', workflow is 'support@1'"]
