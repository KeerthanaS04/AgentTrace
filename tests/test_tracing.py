"""P2-T1: @traced + TraceContext. A workflow run yields a complete, nested `Run`."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from fake_llm import EXPL, TICKET, default_script, make_gateway

from agenttrace.schemas.run import Run
from agenttrace.schemas.span import Span
from agenttrace.tracing import TraceContext, current_span, current_trace, traced
from agenttrace.tracing.tree import format_tree
from agenttrace.workflow.kb.index import BM25Index
from agenttrace.workflow.kb.loader import load_kb
from agenttrace.workflow.runtime import INPUT_STEP, WorkflowResult, step_id
from agenttrace.workflow.support import STEPS, build_support_workflow, make_resources
from agenttrace.workflow.tools import ToolBox, ToolInputError, load_db

USAGE = {"prompt_tokens": 120, "completion_tokens": 30}


def run_support(tmp_path: Path, script: Any = default_script, run_id: str = "r1") -> WorkflowResult:
    gateway = make_gateway(tmp_path, script=script, usage=USAGE)
    return build_support_workflow().run(
        {"ticket": TICKET}, run_id=run_id, resources=make_resources(gateway)
    )


def by_node(run: Run) -> dict[str, list[Span]]:
    out: dict[str, list[Span]] = {}
    for s in run.spans:
        out.setdefault(s.node_id, []).append(s)
    return out


# ---------------------------------------------------------------- the done-when


def test_run_has_all_five_agents_and_nested_llm_tool_retrieval_spans(tmp_path: Path) -> None:
    result = run_support(tmp_path)
    run = Run.model_validate(result.to_run(scenario_id="s1").model_dump())  # re-validates
    nodes = by_node(run)
    spans = {s.span_id: s for s in run.spans}

    # One agent span per step, with the step's deterministic id, at the top level.
    agents = [s for s in run.spans if s.kind == "agent"]
    assert [s.node_id for s in agents] == list(STEPS)
    for s in agents:
        assert s.span_id == step_id("r1", s.node_id) and s.parent_span_id is None
        assert s.error is None and s.explanation is not None and s.output

    # Every agent made at least one model call, nested under it.
    for step in STEPS:
        llm = nodes[f"{step}.llm"]
        assert all(s.kind == "llm" and s.parent_span_id == step_id("r1", step) for s in llm)
    # Tool and search calls nest under the agents that made them.
    tools = [s for s in run.spans if s.kind == "tool"]
    assert {s.node_id for s in tools} == {
        "tool_agent.order_lookup",
        "tool_agent.account_status",
        "tool_agent.refund_api",
    }
    assert all(s.parent_span_id == step_id("r1", "tool_agent") for s in tools)
    searches = nodes["retriever.search"]
    assert len(searches) == 3  # 1 model query + the ticket text + the standing query
    assert all(s.kind == "retrieval" and s.parent_span_id == "r1/retriever" for s in searches)

    # The workflow input is a span too, so every input_ref points at a real span.
    assert nodes[INPUT_STEP][0].output == {"ticket": TICKET}
    for s in run.spans:
        for ref in s.input_refs:
            assert ref.span_id in spans and ref.key in spans[ref.span_id].output

    assert run.scenario_id == "s1" and run.workflow_version == "support@1"
    assert run.final_output == result.outputs
    assert set(run.agent_models) == set(STEPS)


def test_llm_span_records_model_params_tokens_cost_and_io(tmp_path: Path) -> None:
    run = run_support(tmp_path).to_run()
    llm = by_node(run)["policy.llm"][0]
    assert llm.model == "nvidia/nemotron-3.5-lightning-30b-a3b"
    assert llm.params["temperature"] == 0 and "max_tokens" in llm.params
    assert (llm.tokens_in, llm.tokens_out) == (120, 30)
    assert llm.input["role"] == "agent_default" and llm.input["json_mode"] is True
    assert llm.input["messages"][-1]["content"].startswith("You are the Policy Agent")
    assert '"decision": "approve"' in llm.output["text"]
    assert llm.output["cache_hit"] is False


def test_priced_call_records_cost(tmp_path: Path) -> None:
    gateway = make_gateway(tmp_path, usage=USAGE)
    with TraceContext("c") as trace:
        result = gateway.complete("judge_fast", [{"role": "user", "content": "You are the Router"}])
    (span,) = trace.spans
    assert span.cost_usd == result.cost_usd > 0
    assert span.node_id == "llm" and span.model == result.model


def test_agent_span_has_explanation_prompt_version_params_and_model(tmp_path: Path) -> None:
    run = run_support(tmp_path).to_run()
    policy = by_node(run)["policy"][0]
    assert policy.prompt_version == "policy-v1"
    assert policy.params == {"role": "agent_default"}
    assert policy.model == "nvidia/nemotron-3.5-lightning-30b-a3b"
    assert policy.explanation is not None and policy.explanation.cited_evidence == ["KB-REF-001"]
    assert policy.output["decision"] == "approve"
    assert set(policy.input) == {"ticket", "category", "docs", "tool_calls"}
    # Tokens and cost live on the llm spans only, so summing all spans never double counts.
    assert policy.tokens_in == 0 and policy.cost_usd == 0


def test_tool_and_retrieval_spans_record_io(tmp_path: Path) -> None:
    nodes = by_node(run_support(tmp_path).to_run())
    lookup = nodes["tool_agent.order_lookup"][0]
    assert lookup.input == {"tool": "order_lookup", "args": {"order_id": "O1001"}}
    assert lookup.output["found"] is True
    search = nodes["retriever.search"][0]
    assert search.input == {"query": "return window refund apparel", "k": 3}
    assert search.output["hits"] and {"doc_id", "score", "title"} <= set(search.output["hits"][0])


def test_children_lie_within_their_parent_in_time(tmp_path: Path) -> None:
    run = run_support(tmp_path).to_run()
    spans = {s.span_id: s for s in run.spans}
    for s in run.spans:
        if s.parent_span_id:
            parent = spans[s.parent_span_id]
            assert parent.started_at <= s.started_at <= s.ended_at <= parent.ended_at
    # Parents are listed before their children.
    seen: set[str] = set()
    for s in run.spans:
        assert s.parent_span_id is None or s.parent_span_id in seen
        seen.add(s.span_id)


# ---------------------------------------------------------------- stable ids


def test_node_and_span_ids_are_stable_across_runs(tmp_path: Path) -> None:
    a = run_support(tmp_path / "a", run_id="same").to_run()
    b = run_support(tmp_path / "b", run_id="same").to_run()
    assert [s.span_id for s in a.spans] == [s.span_id for s in b.spans]
    assert [s.node_id for s in a.spans] == [s.node_id for s in b.spans]
    other = run_support(tmp_path / "c", run_id="other").to_run()
    assert [s.node_id for s in other.spans] == [s.node_id for s in a.spans]


def test_repeated_calls_share_a_node_id_with_numbered_span_ids(tmp_path: Path) -> None:
    nodes = by_node(run_support(tmp_path).to_run())
    assert [s.span_id for s in nodes["tool_agent.llm"]] == [
        "r1/tool_agent.llm",
        "r1/tool_agent.llm#2",
    ]


def test_json_repair_attempts_are_separate_llm_spans(tmp_path: Path) -> None:
    attempts = {"n": 0}

    def flaky_router(prompt: str) -> dict[str, Any]:
        if prompt.startswith("You are the Router"):
            attempts["n"] += 1
            if attempts["n"] == 1:
                return {"oops": True}  # invalid: triggers one repair
        return default_script(prompt)

    nodes = by_node(run_support(tmp_path, script=flaky_router).to_run())
    assert [s.span_id for s in nodes["router.llm"]] == ["r1/router.llm", "r1/router.llm#2"]
    assert nodes["router"][0].error is None


# ---------------------------------------------------------------- failures


def test_failed_agent_and_skipped_steps_are_recorded(tmp_path: Path) -> None:
    def broken_policy(prompt: str) -> dict[str, Any]:
        if prompt.startswith("You are the Policy Agent"):
            return {**EXPL}  # never valid: missing the decision fields
        return default_script(prompt)

    result = run_support(tmp_path, script=broken_policy)
    nodes = by_node(result.to_run())
    assert result.status == "failed"
    assert "AgentFailure" in (nodes["policy"][0].error or "")
    assert len(nodes["policy.llm"]) == 3  # first try + 2 repairs, all traced
    writer = nodes["writer"][0]
    assert writer.error == "skipped: upstream not ok: ['policy']"
    assert writer.span_id == "r1/writer" and not writer.output


def test_tool_input_error_is_recorded_on_the_tool_span(tmp_path: Path) -> None:
    box = ToolBox(load_db())
    with (
        TraceContext("t") as trace,
        trace.span("agent", "tool_agent"),
        pytest.raises(ToolInputError),
    ):
        box.call("order_lookup", {"order": "O1001"})
    tool = trace.spans[1]
    assert tool.node_id == "tool_agent.order_lookup" and tool.parent_span_id == "t/tool_agent"
    assert tool.error is not None and tool.error.startswith("ToolInputError")


# ---------------------------------------------------------------- the decorator itself


def test_untraced_calls_are_unchanged() -> None:
    index = BM25Index.from_kb(load_kb())
    assert current_trace() is None and current_span() is None
    assert index.search("gift card refund", 2)[0].doc_id == "KB-REF-003"


def test_traced_decorator_defaults_and_context_restored() -> None:
    @traced("decision", name="pick")
    def pick(options: list[str], first: bool = True) -> str:
        return options[0] if first else options[-1]

    with TraceContext("d") as trace:
        assert pick(["a", "b"]) == "a"
        with trace.span("agent", "outer", agent="outer_agent"):
            pick(["a", "b"], first=False)
    assert current_trace() is None
    top, outer, nested = trace.spans
    assert (top.node_id, top.input, top.output) == (
        "pick",
        {"options": ["a", "b"], "first": True},
        {"result": "a"},
    )
    assert nested.node_id == "outer.pick" and nested.agent == "outer_agent"
    assert nested.output == {"result": "b"} and nested.parent_span_id == outer.span_id


def test_format_tree_shows_nesting(tmp_path: Path) -> None:
    text = format_tree(run_support(tmp_path).to_run())
    lines = text.splitlines()
    assert lines[0].startswith("message   __input__")
    assert any(line.startswith("agent     tool_agent") for line in lines)
    assert any(line.startswith("  tool      tool_agent.order_lookup") for line in lines)
    assert "[tool_agent.llm#2]" in text
