"""Offline tests for the five support agents, the workflow wiring, and the contracts.

A scripted fake model server answers each agent (picked by the prompt's first line),
so these run without API keys. Live accuracy is measured by `python tasks.py clean`.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx
import pytest

from agenttrace.gateway import CompletionCache, Gateway, load_config
from agenttrace.schemas.ground_truth import Scenario
from agenttrace.workflow import clean_eval
from agenttrace.workflow.agents.base import ExplainedReply
from agenttrace.workflow.agents.retriever import STANDING_QUERY
from agenttrace.workflow.contracts import CHECKS, CONTRACTS, run_checks
from agenttrace.workflow.kb.index import BM25Index
from agenttrace.workflow.kb.loader import load_kb
from agenttrace.workflow.runtime import INPUT_STEP, WorkflowResult
from agenttrace.workflow.support import STEPS, build_support_workflow, make_resources
from agenttrace.workflow.tools import load_db

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "models.yaml"
EXPL = {"rationale": "r", "cited_evidence": [], "confidence": 0.9}

TICKET = {
    "ticket_id": "T-1",
    "customer_id": "C001",
    "subject": "Return a sweater",
    "body": "I want to return the wool sweater from order O1001 for a refund.",
}

Script = Callable[[str], dict[str, Any]]


def default_script(prompt: str) -> dict[str, Any]:
    if prompt.startswith("You are the Router"):
        return {**EXPL, "category": "refund", "order_id": "O1001"}
    if prompt.startswith("You are the Retriever"):
        return {**EXPL, "queries": ["return window refund apparel"]}
    if prompt.startswith("You are the Tool Agent"):
        if "(none yet)" in prompt:
            return {
                **EXPL,
                "calls": [
                    {"tool": "order_lookup", "args": {"order_id": "O1001"}},
                    {"tool": "account_status", "args": {"customer_id": "C001"}},
                ],
                "done": False,
            }
        if '"refund_api"' not in prompt.split("Tool results so far:")[1]:
            quote = {"order_id": "O1001", "amount": 48, "mode": "quote", "reason": "return"}
            return {**EXPL, "calls": [{"tool": "refund_api", "args": quote}], "done": True}
        return {**EXPL, "calls": [], "done": True}
    if prompt.startswith("You are the Policy Agent"):
        return {
            **EXPL,
            "decision": "approve",
            "refund_amount": 41.05,
            "escalate_to": None,
            "cited_doc_ids": ["KB-REF-001"],
            "justification": "Within 30 days; $6.95 return fee deducted.",
        }
    if prompt.startswith("You are the Response Writer"):
        return {**EXPL, "reply": "Approved: $41.05 back in 5-7 business days. Brightcart Support"}
    raise AssertionError(f"unexpected prompt: {prompt[:60]}")


def make_gateway(tmp_path: Path, script: Script, prompts: list[str] | None = None) -> Gateway:
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        # The agent's own prompt is the first user message; later ones are repair requests.
        prompt = next(m["content"] for m in body["messages"] if m["role"] == "user")
        if prompts is not None:
            prompts.append(prompt)
        content = json.dumps(script(prompt))
        return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})

    config = load_config(CONFIG_PATH)
    return Gateway(
        config,
        cache=CompletionCache(tmp_path / "c.sqlite"),
        env={"NVIDIA_API_KEY": "k", "MODEL_API_KEY": "k", "OPENROUTER_API_KEY": "k"},
        transports={p: httpx.MockTransport(handler) for p in config.providers},
    )


def run(tmp_path: Path, script: Script = default_script, ticket: dict[str, Any] = TICKET,
        prompts: list[str] | None = None) -> WorkflowResult:  # fmt: skip
    gateway = make_gateway(tmp_path, script, prompts)
    return build_support_workflow().run(
        {"ticket": ticket}, run_id="t", resources=make_resources(gateway)
    )


# ---------------------------------------------------------------- end to end


def test_workflow_runs_all_five_agents(tmp_path: Path) -> None:
    result = run(tmp_path)
    assert result.status == "ok"
    assert [s.step for s in result.steps] == list(STEPS)
    assert result.outputs["decision"] == "approve"
    assert result.outputs["reply"].startswith("Approved")
    for rec in result.steps:
        assert rec.explanation is not None, rec.step
        assert run_checks(rec.step, rec.inputs, rec.output) == [], rec.step


def test_data_flow_matches_declared_graph(tmp_path: Path) -> None:
    result = run(tmp_path)
    wf = build_support_workflow()
    got = {(r.span_id.split("/")[1], rec.step) for rec in result.steps for r in rec.input_refs}
    assert got == set(wf.edges())
    assert ("tool_agent", "writer") not in got  # the writer never sees raw tool data
    assert (INPUT_STEP, "router") in got


def test_tool_agent_runs_real_tools(tmp_path: Path) -> None:
    calls = run(tmp_path).step("tool_agent").output["tool_calls"]
    assert [c["tool"] for c in calls] == ["order_lookup", "account_status", "refund_api"]
    assert calls[0]["result"]["order"]["days_since_delivery"] == 18
    assert calls[2]["result"]["status"] == "quoted"
    assert all(c["error"] is None for c in calls)


def test_workflow_input_is_only_the_ticket(tmp_path: Path) -> None:
    # Nothing scenario-level (category label, ground truth) reaches any prompt.
    prompts: list[str] = []
    run(tmp_path, prompts=prompts)
    assert prompts and not any("ground_truth" in p or "expected" in p for p in prompts)


# ---------------------------------------------------------------- individual agents


def test_router_drops_malformed_order_id(tmp_path: Path) -> None:
    def script(p: str) -> dict[str, Any]:
        if p.startswith("You are the Router"):
            return {**EXPL, "category": "account", "order_id": "order #12"}
        return default_script(p)

    rec = run(tmp_path, script).step("router")
    assert rec.output == {"category": "account", "order_id": None}


def test_retriever_always_includes_standing_escalation_docs(tmp_path: Path) -> None:
    docs = [d["doc_id"] for d in run(tmp_path).step("retriever").output["docs"]]
    index = BM25Index.from_kb(load_kb())
    standing = [h.doc_id for h in index.search(STANDING_QUERY, 3)]
    assert set(standing) <= set(docs)
    assert len(docs) == len(set(docs))
    assert "KB-REF-010" not in docs  # superseded docs are never served


def test_tool_agent_blocks_issue_mode_and_records_bad_args(tmp_path: Path) -> None:
    def script(p: str) -> dict[str, Any]:
        if p.startswith("You are the Tool Agent"):
            issue = {"order_id": "O1001", "amount": 48, "mode": "issue", "reason": "x"}
            return {
                **EXPL,
                "calls": [
                    {"tool": "refund_api", "args": issue},
                    {"tool": "order_lookup", "args": {"orderid": "O1001"}},
                    {"tool": "order_lookup", "args": {"orderid": "O1001"}},  # duplicate
                ],
                "done": True,
            }
        return default_script(p)

    result = run(tmp_path, script)
    calls = result.step("tool_agent").output["tool_calls"]
    assert [c["error"] is not None for c in calls] == [True, True]
    assert calls[0]["error"] == "issue mode not allowed"
    violations = run_checks("tool_agent", result.step("tool_agent").inputs, {"tool_calls": calls})
    assert any(v.startswith("tool_calls_succeeded") for v in violations)
    assert any(v.startswith("order_looked_up") for v in violations)
    # A refund was never issued: the run's DB copy is untouched.
    assert result.step("policy").status == "ok"


def test_extra_keys_in_model_reply_are_ignored() -> None:
    reply = ExplainedReply.model_validate({**EXPL, "reply_length_check": 84})
    assert reply.confidence == 0.9


def test_unparseable_reply_fails_step_and_skips_downstream(tmp_path: Path) -> None:
    def script(p: str) -> dict[str, Any]:
        if p.startswith("You are the Policy Agent"):
            return {"decision": "maybe"}
        return default_script(p)

    result = run(tmp_path, script)
    assert result.status == "failed"
    assert result.step("policy").status == "error"
    assert "no valid PolicyReply after 3 tries" in (result.step("policy").error or "")
    assert result.step("writer").status == "skipped"
    assert "decision" not in result.outputs


def test_escalate_to_is_cleared_unless_escalating(tmp_path: Path) -> None:
    def script(p: str) -> dict[str, Any]:
        out = default_script(p)
        if p.startswith("You are the Policy Agent"):
            out["escalate_to"] = "Finance Review"
        return out

    assert run(tmp_path, script).outputs["escalate_to"] is None


def test_roles_are_per_step_and_validated() -> None:
    wf = build_support_workflow({"policy": "agent_alt_1"})
    assert wf.steps["policy"].params["role"] == "agent_alt_1"
    assert wf.steps["router"].params["role"] == "agent_default"
    with pytest.raises(ValueError, match="unknown steps"):
        build_support_workflow({"plannr": "agent_default"})


# ---------------------------------------------------------------- contracts


def test_every_step_has_a_contract_with_known_checks() -> None:
    assert set(CONTRACTS) == set(STEPS)
    for contract in CONTRACTS.values():
        assert contract.programmatic_checks
        assert set(contract.programmatic_checks) <= set(CHECKS)


@pytest.mark.parametrize(
    "node, inputs, output, check",
    [
        ("router", {}, {"category": "billing", "order_id": None}, "category_in_taxonomy"),
        ("router", {}, {"category": "refund", "order_id": "1001"}, "order_id_format"),
        ("retriever", {}, {"docs": []}, "docs_nonempty"),
        ("retriever", {}, {"docs": [{"doc_id": "x"}]}, "doc_ids_well_formed"),
        ("retriever", {}, {"docs": [{"doc_id": "KB-REF-001"}] * 2}, "no_duplicate_docs"),
        ("tool_agent", {"order_id": None}, {"tool_calls": [{"tool": "rm"}]}, "tools_known"),
        ("policy", {"docs": [{"doc_id": "KB-REF-001"}]},
         {"decision": "escalate", "escalate_to": None, "cited_doc_ids": ["KB-REF-001"]},
         "escalation_names_team"),
        ("policy", {"docs": [{"doc_id": "KB-REF-001"}]},
         {"decision": "deny", "cited_doc_ids": ["KB-REF-003"]}, "cites_retrieved_docs"),
        ("policy",
         {"docs": [{"doc_id": "KB-REF-001"}],
          "tool_calls": [{"tool": "order_lookup",
                          "result": {"order": {"total": 48.0, "already_refunded": 40.0}}}]},
         {"decision": "approve", "refund_amount": 20.0, "cited_doc_ids": ["KB-REF-001"]},
         "refund_within_refundable"),
        ("writer", {"decision": "approve", "refund_amount": 41.05}, {"reply": "Approved!"},
         "reply_states_amount"),
        ("writer", {"decision": "escalate", "escalate_to": "Payments team"},
         {"reply": "Someone will help."}, "reply_names_team"),
        ("writer", {}, {"reply": "  "}, "reply_nonempty"),
    ],
)  # fmt: skip
def test_checks_flag_violations(
    node: str, inputs: dict[str, Any], output: dict[str, Any], check: str
) -> None:
    assert any(v.startswith(check) for v in run_checks(node, inputs, output))


def test_reply_amount_check_accepts_common_formats() -> None:
    inputs = {"decision": "approve", "refund_amount": 1116.1}
    assert run_checks("writer", inputs, {"reply": "We'll refund $1,116.10."}) == []
    assert run_checks("writer", {**inputs, "refund_amount": 48.0}, {"reply": "$48 back"}) == []


# ---------------------------------------------------------------- clean ticket set


def test_clean_tickets_are_valid_and_balanced() -> None:
    scenarios = clean_eval.load_scenarios()
    assert len(scenarios) == 20
    assert len({s.scenario_id for s in scenarios}) == 20
    assert all(isinstance(s, Scenario) and s.ground_truth.fault_code == "F0" for s in scenarios)
    decisions = [s.ground_truth.expected_final.decision for s in scenarios]
    assert min(decisions.count(d) for d in ("approve", "deny", "escalate")) >= 5


def test_clean_tickets_reference_real_fixtures() -> None:
    db = load_db()
    for sc in clean_eval.load_scenarios():
        assert db.customer(sc.ticket.customer_id), sc.scenario_id
        for word in sc.ticket.body.replace(".", " ").replace(")", " ").split():
            if word.startswith("O1") and len(word) == 5:
                order = db.order(word)
                assert order is not None, (sc.scenario_id, word)
                assert order.customer_id == sc.ticket.customer_id, sc.scenario_id
