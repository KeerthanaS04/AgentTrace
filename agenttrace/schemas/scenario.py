"""The part of a scenario the workflow is allowed to see.

Injected faults and ground truth live in `schemas.ground_truth` and must never
reach the diagnosis or explainability code.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field

from agenttrace.schemas.base import NonEmptyStr, StrictModel

TicketCategory = Literal["refund", "shipping", "account", "other"]


class Ticket(StrictModel):
    ticket_id: NonEmptyStr
    customer_id: NonEmptyStr
    subject: str
    body: NonEmptyStr


class ScenarioInput(StrictModel):
    scenario_id: NonEmptyStr
    category: TicketCategory
    ticket: Ticket
    # Rows for the mock tool database (orders, accounts, refunds).
    fixtures: dict[str, Any] = Field(default_factory=dict)
