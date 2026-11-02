"""The customer-support workflow: Router -> Retriever -> Tool Agent -> Policy Agent -> Writer.

Data flow (every edge is declared below, so the trace's data flow is exactly this):
    router      <- ticket
    retriever   <- ticket, router.category
    tool_agent  <- ticket, router.category, router.order_id
    policy      <- ticket, router.category, retriever.docs, tool_agent.tool_calls
    writer      <- ticket, policy.{decision, refund_amount, escalate_to, justification},
                   retriever.docs
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from agenttrace.gateway.gateway import Gateway
from agenttrace.workflow.agents.base import DEFAULT_ROLE
from agenttrace.workflow.agents.policy import PolicyAgent
from agenttrace.workflow.agents.retriever import RetrieverAgent
from agenttrace.workflow.agents.router import RouterAgent
from agenttrace.workflow.agents.tool_agent import ToolAgent
from agenttrace.workflow.agents.writer import WriterAgent
from agenttrace.workflow.kb.index import BM25Index
from agenttrace.workflow.kb.loader import KnowledgeBase, load_kb
from agenttrace.workflow.runtime import FromInput, FromStep, Step, Workflow
from agenttrace.workflow.tools import FixtureDB, ToolBox, load_db

STEPS = ("router", "retriever", "tool_agent", "policy", "writer")


def build_support_workflow(roles: Mapping[str, str] | None = None) -> Workflow:
    """Build the workflow. `roles` maps step name -> model role (default `agent_default`)."""
    roles = dict(roles or {})
    unknown = set(roles) - set(STEPS)
    if unknown:
        raise ValueError(f"unknown steps in roles: {sorted(unknown)}")

    def params(step: str) -> dict[str, Any]:
        return {"role": roles.get(step, DEFAULT_ROLE)}

    ticket = FromInput("ticket")
    return Workflow(
        name="support",
        inputs=["ticket"],
        steps=[
            Step("router", RouterAgent(), {"ticket": ticket}, params("router")),
            Step(
                "retriever",
                RetrieverAgent(),
                {"ticket": ticket, "category": FromStep("router", "category")},
                params("retriever"),
            ),
            Step(
                "tool_agent",
                ToolAgent(),
                {
                    "ticket": ticket,
                    "category": FromStep("router", "category"),
                    "order_id": FromStep("router", "order_id"),
                },
                params("tool_agent"),
            ),
            Step(
                "policy",
                PolicyAgent(),
                {
                    "ticket": ticket,
                    "category": FromStep("router", "category"),
                    "docs": FromStep("retriever", "docs"),
                    "tool_calls": FromStep("tool_agent", "tool_calls"),
                },
                params("policy"),
            ),
            Step(
                "writer",
                WriterAgent(),
                {
                    "ticket": ticket,
                    "decision": FromStep("policy", "decision"),
                    "refund_amount": FromStep("policy", "refund_amount"),
                    "escalate_to": FromStep("policy", "escalate_to"),
                    "justification": FromStep("policy", "justification"),
                    "docs": FromStep("retriever", "docs"),
                },
                params("writer"),
            ),
        ],
        outputs={
            "category": FromStep("router", "category"),
            "decision": FromStep("policy", "decision"),
            "refund_amount": FromStep("policy", "refund_amount"),
            "escalate_to": FromStep("policy", "escalate_to"),
            "reply": FromStep("writer", "reply"),
        },
    )


def make_resources(
    gateway: Gateway,
    *,
    db: FixtureDB | None = None,
    kb: KnowledgeBase | None = None,
    index: BM25Index | None = None,
) -> dict[str, Any]:
    """Resources for one run. The ToolBox gets a fresh DB copy, so runs never share state."""
    db = db or load_db()
    kb = kb or load_kb()
    return {
        "gateway": gateway,
        "kb": kb,
        "kb_index": index or BM25Index.from_kb(kb),
        "tools": ToolBox(db),
        "today": db.as_of.isoformat(),
    }
