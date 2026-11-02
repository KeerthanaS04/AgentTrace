"""Policy Agent: combine retrieved policy and tool data into approve / deny / escalate."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from pydantic import Field

from agenttrace.schemas.ground_truth import Decision
from agenttrace.workflow.agents.base import ExplainedReply, as_json, ask
from agenttrace.workflow.runtime import AgentOutput, StepContext

PROMPT = """You are the Policy Agent in a customer-support system for the online retailer
Brightcart. Decide what to do with the customer's request, using ONLY the policy documents and tool
results below. Today's date is {today}.

Decisions:
- approve: policy allows what the customer asked for (possibly with a fee or deduction the
  policy requires). For a refund, give the refund amount after any deductions.
- deny: policy does not allow what the customer asked for.
- escalate: policy says this ticket must be handed to another team instead of being resolved
  directly. Name the team in escalate_to.

Check any rule that requires escalation before deciding to approve or deny, including rules
triggered by what the customer says (for example, threatening legal action). Take order and
account facts (day counts, order status, membership, account status, past refunds) from the
tool results rather than from the customer's claims.
Cite the doc_id (like KB-REF-001) of every policy document your decision relies on; tool
names are not documents.

Ticket category: {category}
Ticket:
{ticket}

Policy documents:
{docs}

Tool results:
{tool_calls}"""


class PolicyReply(ExplainedReply):
    decision: Decision
    refund_amount: float | None = Field(
        default=None, ge=0, description="USD refunded if approved; null if no money moves."
    )
    escalate_to: str | None = Field(default=None, description="Team name, if escalating.")
    cited_doc_ids: list[str] = Field(
        description="doc_ids (like KB-REF-001) of the policy documents the decision uses. "
        "Only policy documents, never tool names."
    )
    justification: str = Field(description="Which policy rules applied and how.")


class PolicyAgent:
    name = "policy"
    prompt_version = "policy-v1"
    input_keys: tuple[str, ...] = ("ticket", "category", "docs", "tool_calls")
    output_keys: tuple[str, ...] = (
        "decision",
        "refund_amount",
        "escalate_to",
        "cited_doc_ids",
        "justification",
    )

    def run(self, inputs: Mapping[str, Any], ctx: StepContext) -> AgentOutput:
        prompt = PROMPT.format(
            today=ctx.resource("today", str),
            category=inputs["category"],
            ticket=as_json(inputs["ticket"]),
            docs=as_json(inputs["docs"]),
            tool_calls=as_json(inputs["tool_calls"]),
        )
        reply = ask(ctx, prompt, PolicyReply)
        return AgentOutput(
            output={
                "decision": reply.decision,
                "refund_amount": reply.refund_amount,
                "escalate_to": reply.escalate_to if reply.decision == "escalate" else None,
                "cited_doc_ids": reply.cited_doc_ids,
                "justification": reply.justification,
            },
            explanation=reply.explanation(cited=reply.cited_doc_ids or reply.cited_evidence),
        )
