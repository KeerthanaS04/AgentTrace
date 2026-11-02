"""Node contracts for the support workflow: what each step should do, plus cheap checks.

The prose fields are given to judges (Phase 6). `programmatic_checks` name the
deterministic validators below; `run_checks` runs them on one step's inputs and
output and returns violation messages (an empty list means the step passed).
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from typing import Any

from agenttrace.schemas.span import NodeContract
from agenttrace.workflow.tools import TOOLS

Check = Callable[[Mapping[str, Any], Mapping[str, Any]], str | None]

CATEGORIES = {"refund", "shipping", "account", "other"}
DECISIONS = {"approve", "deny", "escalate"}
DOC_ID_RE = re.compile(r"^KB-[A-Z]{3}-\d{3}$")
ORDER_ID_RE = re.compile(r"^O\d{4}$")


# ---------------------------------------------------------------- checks


def category_in_taxonomy(inputs: Mapping[str, Any], out: Mapping[str, Any]) -> str | None:
    return None if out.get("category") in CATEGORIES else f"category {out.get('category')!r}"


def order_id_format(inputs: Mapping[str, Any], out: Mapping[str, Any]) -> str | None:
    oid = out.get("order_id")
    return None if oid is None or ORDER_ID_RE.match(str(oid)) else f"order_id {oid!r}"


def docs_nonempty(inputs: Mapping[str, Any], out: Mapping[str, Any]) -> str | None:
    return None if out.get("docs") else "no docs retrieved"


def doc_ids_well_formed(inputs: Mapping[str, Any], out: Mapping[str, Any]) -> str | None:
    bad = [
        d.get("doc_id") for d in out.get("docs", []) if not DOC_ID_RE.match(str(d.get("doc_id")))
    ]
    return f"malformed doc ids {bad}" if bad else None


def no_duplicate_docs(inputs: Mapping[str, Any], out: Mapping[str, Any]) -> str | None:
    ids = [d.get("doc_id") for d in out.get("docs", [])]
    return "duplicate docs" if len(ids) != len(set(ids)) else None


def tools_known(inputs: Mapping[str, Any], out: Mapping[str, Any]) -> str | None:
    bad = [c.get("tool") for c in out.get("tool_calls", []) if c.get("tool") not in TOOLS]
    return f"unknown tools {bad}" if bad else None


def tool_calls_succeeded(inputs: Mapping[str, Any], out: Mapping[str, Any]) -> str | None:
    errs = [
        f"{c.get('tool')}: {c.get('error')}" for c in out.get("tool_calls", []) if c.get("error")
    ]
    return f"tool errors {errs}" if errs else None


def order_looked_up(inputs: Mapping[str, Any], out: Mapping[str, Any]) -> str | None:
    if inputs.get("order_id") is None:
        return None
    looked_up = any(
        c.get("tool") == "order_lookup"
        and not c.get("error")
        and (c.get("args") or {}).get("order_id") == inputs["order_id"]
        for c in out.get("tool_calls", [])
    )
    return None if looked_up else f"order {inputs['order_id']} was never looked up successfully"


def decision_valid(inputs: Mapping[str, Any], out: Mapping[str, Any]) -> str | None:
    return None if out.get("decision") in DECISIONS else f"decision {out.get('decision')!r}"


def escalation_names_team(inputs: Mapping[str, Any], out: Mapping[str, Any]) -> str | None:
    if out.get("decision") == "escalate" and not out.get("escalate_to"):
        return "escalated without naming a team"
    return None


def cites_retrieved_docs(inputs: Mapping[str, Any], out: Mapping[str, Any]) -> str | None:
    retrieved = {d.get("doc_id") for d in inputs.get("docs", [])}
    cited = list(out.get("cited_doc_ids", []))
    if not cited:
        return "no policy cited"
    outside = [c for c in cited if c not in retrieved]
    return f"cites docs that were not retrieved: {outside}" if outside else None


def refund_within_refundable(inputs: Mapping[str, Any], out: Mapping[str, Any]) -> str | None:
    amount = out.get("refund_amount")
    if out.get("decision") != "approve" or not amount:
        return None
    for call in inputs.get("tool_calls", []):
        order = (
            ((call.get("result") or {}).get("order"))
            if call.get("tool") == "order_lookup"
            else None
        )
        if order:
            refundable = round(order["total"] - order["already_refunded"], 2)
            if amount > refundable + 1e-9:
                return f"refund ${amount} exceeds refundable ${refundable}"
    return None


def reply_nonempty(inputs: Mapping[str, Any], out: Mapping[str, Any]) -> str | None:
    return None if str(out.get("reply", "")).strip() else "empty reply"


def reply_states_amount(inputs: Mapping[str, Any], out: Mapping[str, Any]) -> str | None:
    amount = inputs.get("refund_amount")
    if inputs.get("decision") != "approve" or not amount:
        return None
    reply = str(out.get("reply", "")).replace(",", "")
    shown = {f"{amount:.2f}", f"{amount:g}"}
    return None if any(s in reply for s in shown) else f"reply does not state ${amount:.2f}"


def reply_names_team(inputs: Mapping[str, Any], out: Mapping[str, Any]) -> str | None:
    team = inputs.get("escalate_to")
    if inputs.get("decision") != "escalate" or not team:
        return None
    return (
        None if str(team).lower() in str(out.get("reply", "")).lower() else f"reply omits {team!r}"
    )


CHECKS: dict[str, Check] = {
    fn.__name__: fn
    for fn in (
        category_in_taxonomy,
        order_id_format,
        docs_nonempty,
        doc_ids_well_formed,
        no_duplicate_docs,
        tools_known,
        tool_calls_succeeded,
        order_looked_up,
        decision_valid,
        escalation_names_team,
        cites_retrieved_docs,
        refund_within_refundable,
        reply_nonempty,
        reply_states_amount,
        reply_names_team,
    )
}


# ---------------------------------------------------------------- contracts


CONTRACTS: dict[str, NodeContract] = {
    c.node_id: c
    for c in (
        NodeContract(
            node_id="router",
            description="Classify the support ticket and extract the order ID it refers to.",
            input_expectations="A customer ticket with subject and body.",
            output_expectations=(
                "category is the one of refund/shipping/account/other that matches the main "
                "request; order_id is the order the customer names, or null if none."
            ),
            programmatic_checks=["category_in_taxonomy", "order_id_format"],
        ),
        NodeContract(
            node_id="retriever",
            description="Write search queries and select the policy docs the ticket needs.",
            input_expectations="The ticket and its category.",
            output_expectations=(
                "docs include every current (non-superseded) policy that governs the outcome, "
                "including rules that could force an escalation; no duplicates."
            ),
            programmatic_checks=["docs_nonempty", "doc_ids_well_formed", "no_duplicate_docs"],
        ),
        NodeContract(
            node_id="tool_agent",
            description="Call the order, account, and refund tools to gather facts for the ticket.",
            input_expectations="The ticket, its category, and the order ID from the router.",
            output_expectations=(
                "tool_calls look up the named order and the customer's account; refund "
                "requests include a refund_api quote. Results are passed through unchanged."
            ),
            programmatic_checks=["tools_known", "tool_calls_succeeded", "order_looked_up"],
        ),
        NodeContract(
            node_id="policy",
            description="Decide approve, deny, or escalate from retrieved policy and tool data.",
            input_expectations="Current policy docs and correct tool results for this customer.",
            output_expectations=(
                "The decision follows the cited policies applied to the tool facts; escalation "
                "rules take priority; refund_amount applies any required deductions; every "
                "cited doc was retrieved."
            ),
            programmatic_checks=[
                "decision_valid",
                "escalation_names_team",
                "cites_retrieved_docs",
                "refund_within_refundable",
            ],
        ),
        NodeContract(
            node_id="writer",
            description="Write the customer reply that communicates the policy decision.",
            input_expectations="The policy decision with amount, team, and justification.",
            output_expectations=(
                "The reply states the same decision, the exact refund amount and timing if "
                "approved, the team if escalated, and the reason if denied, with no claims "
                "beyond the decision and policies."
            ),
            programmatic_checks=["reply_nonempty", "reply_states_amount", "reply_names_team"],
        ),
    )
}


def run_checks(node_id: str, inputs: Mapping[str, Any], output: Mapping[str, Any]) -> list[str]:
    """Violations for one step (empty list = all checks passed)."""
    contract = CONTRACTS[node_id]
    violations: list[str] = []
    for name in contract.programmatic_checks:
        message = CHECKS[name](inputs, output)
        if message:
            violations.append(f"{name}: {message}")
    return violations
