"""Router: classify the ticket and pull out the order ID it refers to."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

from pydantic import Field

from agenttrace.schemas.scenario import TicketCategory
from agenttrace.workflow.agents.base import ExplainedReply, as_json, ask
from agenttrace.workflow.runtime import AgentOutput, StepContext

ORDER_ID_RE = re.compile(r"^O\d{4}$")

PROMPT = """You are the Router in a customer-support system for the online retailer
Brightcart. Classify the ticket into exactly one category:
- refund: the customer wants money back, a return, a refund, or a price difference
- shipping: delivery status, late or missing packages, shipping fees, cancelling or
  changing an order before it arrives
- account: sign-in, passwords, account email, Brightcart Plus membership, store credit,
  payment methods
- other: anything else
If the customer wants money back because of a late delivery or shipping fee, use shipping.
Also extract the order ID the ticket refers to (format O followed by 4 digits), or null if none.

Ticket:
{ticket}"""


class RouterReply(ExplainedReply):
    category: TicketCategory
    order_id: str | None = Field(description="Order ID like O1001, or null.")


class RouterAgent:
    name = "router"
    prompt_version = "router-v1"
    input_keys: tuple[str, ...] = ("ticket",)
    output_keys: tuple[str, ...] = ("category", "order_id")

    def run(self, inputs: Mapping[str, Any], ctx: StepContext) -> AgentOutput:
        reply = ask(ctx, PROMPT.format(ticket=as_json(inputs["ticket"])), RouterReply)
        order_id = (reply.order_id or "").strip().upper() or None
        if order_id is not None and not ORDER_ID_RE.match(order_id):
            order_id = None
        return AgentOutput(
            output={"category": reply.category, "order_id": order_id},
            explanation=reply.explanation(),
        )
