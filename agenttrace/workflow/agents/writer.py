"""Response Writer: draft the customer reply from the Policy Agent's decision."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from pydantic import Field

from agenttrace.workflow.agents.base import ExplainedReply, as_json, ask
from agenttrace.workflow.runtime import AgentOutput, StepContext

PROMPT = """You are the Response Writer in a customer-support system for the online
retailer Brightcart. Write the reply to the customer. It must:
- state the decision plainly (approved, declined, or passed to another team);
- give the exact refund amount in dollars if one was approved, and how long it takes to arrive;
- if escalated, name the team that will follow up and when;
- if declined, give the policy reason;
- use only facts from the decision and policy documents below. Do not promise anything else.
Keep it under 150 words, friendly and direct. Sign off as "Brightcart Support".

Ticket:
{ticket}

Decision: {decision}
Refund amount: {refund_amount}
Escalate to: {escalate_to}
Justification: {justification}

Policy documents:
{docs}"""


class WriterReply(ExplainedReply):
    reply: str = Field(min_length=1, description="The full message to the customer.")


class WriterAgent:
    name = "writer"
    prompt_version = "writer-v1"
    input_keys: tuple[str, ...] = (
        "ticket",
        "decision",
        "refund_amount",
        "escalate_to",
        "justification",
        "docs",
    )
    output_keys: tuple[str, ...] = ("reply",)

    def run(self, inputs: Mapping[str, Any], ctx: StepContext) -> AgentOutput:
        prompt = PROMPT.format(
            ticket=as_json(inputs["ticket"]),
            decision=inputs["decision"],
            refund_amount=inputs["refund_amount"],
            escalate_to=inputs["escalate_to"],
            justification=inputs["justification"],
            docs=as_json(inputs["docs"]),
        )
        reply = ask(ctx, prompt, WriterReply)
        return AgentOutput(output={"reply": reply.reply}, explanation=reply.explanation())
