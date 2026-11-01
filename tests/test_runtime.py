from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pytest

from agenttrace.gateway.cache import content_hash
from agenttrace.schemas.span import Explanation
from agenttrace.workflow import toy
from agenttrace.workflow.runtime import (
    INPUT_STEP,
    AgentOutput,
    FromInput,
    FromStep,
    Step,
    StepContext,
    Workflow,
    step_id,
)

EXPL = Explanation(rationale="test", confidence=1.0)


class Fn:
    """Agent built from a plain function, for wiring tests."""

    def __init__(
        self, name: str, inputs: tuple[str, ...], outputs: tuple[str, ...], fn: Any
    ) -> None:
        self.name, self.input_keys, self.output_keys, self.fn = name, inputs, outputs, fn

    def run(self, inputs: Mapping[str, Any], ctx: StepContext) -> AgentOutput:
        return AgentOutput(output=self.fn(inputs, ctx), explanation=EXPL)


def double() -> Fn:
    return Fn("double", ("x",), ("y",), lambda i, c: {"y": i["x"] * 2})


def add() -> Fn:
    return Fn("add", ("a", "b"), ("sum",), lambda i, c: {"sum": i["a"] + i["b"]})


def diamond() -> Workflow:
    # x -> left, x -> right, (left, right) -> join
    return Workflow(
        "diamond",
        inputs=["x"],
        steps=[
            Step("join", add(), {"a": FromStep("left", "y"), "b": FromStep("right", "y")}),
            Step("left", double(), {"x": FromInput("x")}),
            Step("right", double(), {"x": FromInput("x")}),
        ],
        outputs={"total": FromStep("join", "sum")},
    )


# ---------------------------------------------------------------- toy workflow (done-when)


def test_toy_workflow_runs_end_to_end() -> None:
    result = toy.run_toy("I want to return my headphones for a refund", run_id="r1")
    assert result.status == "ok"
    assert [s.step for s in result.steps] == ["triage", "answer"]
    assert result.outputs["category"] == "refund"
    answer = result.step("answer")
    assert answer.output["doc_id"].startswith("KB-")
    assert answer.explanation is not None
    assert answer.explanation.cited_evidence == [answer.output["doc_id"]]
    assert answer.params == {"k": 3}


def test_toy_refs_follow_declared_data_flow() -> None:
    result = toy.run_toy("where is my delivery tracking", run_id="r2")
    answer = result.step("answer")
    by_key = {r.key: r for r in answer.input_refs}
    assert by_key["ticket_text"].span_id == step_id("r2", INPUT_STEP)
    assert by_key["category"].span_id == step_id("r2", "triage")
    assert by_key["category"].content_hash == content_hash("shipping")
    # Every ref points at a span that exists (or the input), matching the declared edges.
    edges = {(r.span_id.split("/")[1], rec.step) for rec in result.steps for r in rec.input_refs}
    assert edges == set(toy.build_toy_workflow().edges())


def test_toy_is_deterministic() -> None:
    a = toy.run_toy("my account is locked", run_id="same")
    b = toy.run_toy("my account is locked", run_id="same")
    strip = {"started_at", "ended_at"}
    assert [s.model_dump(exclude=strip) for s in a.steps] == [
        s.model_dump(exclude=strip) for s in b.steps
    ]


def test_toy_main_prints_report(capsys: pytest.CaptureFixture[str]) -> None:
    assert toy.main() == 0
    out = capsys.readouterr().out
    assert "[triage]" in out and "[answer]" in out and "final" in out


# ---------------------------------------------------------------- execution


def test_runs_in_dependency_order_not_declaration_order() -> None:
    wf = diamond()
    assert wf.order == ["left", "right", "join"]
    result = wf.run({"x": 3}, run_id="d")
    assert result.status == "ok" and result.outputs == {"total": 12}


def test_error_skips_only_downstream_steps() -> None:
    def boom(i: Mapping[str, Any], c: StepContext) -> dict[str, Any]:
        raise RuntimeError("tool timed out")

    wf = Workflow(
        "partial",
        inputs=["x"],
        steps=[
            Step("bad", Fn("bad", ("x",), ("y",), boom), {"x": FromInput("x")}),
            Step("after_bad", double(), {"x": FromStep("bad", "y")}),
            Step("independent", double(), {"x": FromInput("x")}),
        ],
        outputs={"y": FromStep("after_bad", "y"), "z": FromStep("independent", "y")},
    )
    result = wf.run({"x": 1}, run_id="p")
    assert result.status == "failed"
    assert result.step("bad").status == "error"
    assert result.step("bad").error == "RuntimeError: tool timed out"
    assert result.step("bad").input_refs  # inputs are still recorded for diagnosis
    assert result.step("after_bad").status == "skipped"
    assert result.step("independent").status == "ok"
    assert result.outputs == {"z": 2}


@pytest.mark.parametrize(
    "fn, match",
    [
        (lambda i, c: {"wrong": 1}, "output keys"),
        (lambda i, c: {"y": object()}, "not JSON-serializable"),
    ],
)
def test_bad_agent_output_is_a_step_error(fn: Any, match: str) -> None:
    wf = Workflow("w", ["x"], [Step("s", Fn("f", ("x",), ("y",), fn), {"x": FromInput("x")})], {})
    rec = wf.run({"x": 1}, run_id="b").step("s")
    assert rec.status == "error" and rec.error is not None and match in rec.error


def test_agent_cannot_mutate_upstream_output() -> None:
    def mutate(i: Mapping[str, Any], c: StepContext) -> dict[str, Any]:
        i["items"].append("hacked")
        return {"n": len(i["items"])}

    wf = Workflow(
        "m",
        ["items"],
        [
            Step("src", Fn("src", ("items",), ("items",), lambda i, c: {"items": i["items"]}),
                 {"items": FromInput("items")}),
            Step("mut", Fn("mut", ("items",), ("n",), mutate), {"items": FromStep("src", "items")}),
        ],
        {},
    )  # fmt: skip
    result = wf.run({"items": ["a"]}, run_id="m")
    assert result.step("src").output == {"items": ["a"]}
    assert result.step("mut").inputs == {"items": ["a"]}


def test_context_exposes_params_and_typed_resources() -> None:
    seen: dict[str, Any] = {}

    def use(i: Mapping[str, Any], c: StepContext) -> dict[str, Any]:
        seen.update(run_id=c.run_id, step=c.step, k=c.params["k"], db=c.resource("db", dict))
        return {"y": 0}

    wf = Workflow(
        "c", ["x"], [Step("s", Fn("f", ("x",), ("y",), use), {"x": FromInput("x")}, {"k": 5})], {}
    )
    assert wf.run({"x": 1}, run_id="c", resources={"db": {"ok": True}}).status == "ok"
    assert seen == {"run_id": "c", "step": "s", "k": 5, "db": {"ok": True}}

    wf_missing = Workflow(
        "c2", ["x"], [Step("s", Fn("f", ("x",), ("y",), use), {"x": FromInput("x")}, {"k": 5})], {}
    )
    rec = wf_missing.run({"x": 1}, run_id="c2").step("s")
    assert rec.status == "error" and "needs resource 'db'" in (rec.error or "")


def test_run_rejects_wrong_workflow_inputs() -> None:
    with pytest.raises(ValueError, match="missing"):
        diamond().run({}, run_id="x")
    with pytest.raises(ValueError, match="unexpected"):
        diamond().run({"x": 1, "y": 2}, run_id="x")


# ---------------------------------------------------------------- static validation


A_FROM_INPUT = Step("a", double(), {"x": FromInput("x")})
A_FROM_B = Step("a", double(), {"x": FromStep("b", "y")})
B_FROM_A = Step("b", double(), {"x": FromStep("a", "y")})
B_FROM_A_BAD_KEY = Step("b", double(), {"x": FromStep("a", "zz")})


@pytest.mark.parametrize(
    "steps, outputs, match",
    [
        ([Step("s", double(), {})], {}, "do not match agent"),
        ([Step("s", double(), {"x": FromInput("nope")})], {}, "unknown workflow input"),
        ([Step("s", double(), {"x": FromStep("ghost", "y")})], {}, "unknown step"),
        ([A_FROM_INPUT, B_FROM_A_BAD_KEY], {}, "has no output"),
        ([A_FROM_B, B_FROM_A], {}, "cycle"),
        ([A_FROM_INPUT, A_FROM_INPUT], {}, "duplicate"),
        ([Step(INPUT_STEP, double(), {"x": FromInput("x")})], {}, "reserved"),
        ([A_FROM_INPUT], {"o": FromStep("a", "nope")}, "has no output"),
    ],
)
def test_invalid_workflows_rejected(steps: list[Step], outputs: dict[str, Any], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        Workflow("bad", ["x"], steps, outputs)


def test_edges_include_input_and_dedupe() -> None:
    assert diamond().edges() == [
        (INPUT_STEP, "left"),
        (INPUT_STEP, "right"),
        ("left", "join"),
        ("right", "join"),
    ]
