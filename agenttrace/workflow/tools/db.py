"""Fixture database behind the mock tools: customers, orders, and past refunds.

Every date is relative to `as_of` (the scenario's "today"); nothing reads the real
clock, so tool outputs are identical on every run and replay.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from agenttrace.schemas.base import NonEmptyStr, NonNegativeFloat, StrictModel

DEFAULT_DB_PATH = Path(__file__).resolve().parents[3] / "data" / "fixtures" / "default_db.json"

ItemCategory = Literal["electronics", "home", "apparel", "gift_card", "digital"]
OrderStatus = Literal["processing", "shipped", "delivered", "cancelled"]
ShippingMethod = Literal["standard", "express", "international"]
PaymentMethod = Literal["card", "paypal", "store_credit"]
AccountState = Literal["active", "locked", "suspended"]


class Item(StrictModel):
    sku: NonEmptyStr
    name: NonEmptyStr
    category: ItemCategory
    price: NonNegativeFloat
    quantity: int = Field(default=1, ge=1)
    final_sale: bool = False
    opened: bool = False


class Order(StrictModel):
    order_id: NonEmptyStr
    customer_id: NonEmptyStr
    email: NonEmptyStr
    status: OrderStatus
    placed_at: date
    shipped_at: date | None = None
    delivered_at: date | None = None
    promised_delivery: date | None = None
    shipping_method: ShippingMethod = "standard"
    shipping_fee: NonNegativeFloat = 0.0
    payment_method: PaymentMethod = "card"
    card_last4: str | None = Field(default=None, pattern=r"^\d{4}$")
    chargeback_filed: bool = False
    items: list[Item] = Field(min_length=1)

    @model_validator(mode="after")
    def _check_dates(self) -> Order:
        shipped, delivered = self.shipped_at, self.delivered_at
        if shipped and shipped < self.placed_at:
            raise ValueError(f"{self.order_id}: shipped before placed")
        if delivered and (shipped is None or delivered < shipped):
            raise ValueError(f"{self.order_id}: delivered without or before shipping")
        expected = {
            "processing": (False, False),
            "shipped": (True, False),
            "delivered": (True, True),
        }.get(self.status)
        if expected and (bool(shipped), bool(delivered)) != expected:
            raise ValueError(f"{self.order_id}: dates inconsistent with status {self.status!r}")
        if self.payment_method == "card" and self.card_last4 is None:
            raise ValueError(f"{self.order_id}: card payments need card_last4")
        return self

    @property
    def subtotal(self) -> float:
        return round(sum(i.price * i.quantity for i in self.items), 2)

    @property
    def total(self) -> float:
        return round(self.subtotal + self.shipping_fee, 2)


class Customer(StrictModel):
    customer_id: NonEmptyStr
    name: NonEmptyStr
    email: NonEmptyStr
    account_status: AccountState = "active"
    # Suspended accounts are under fraud review (KB-ACC-003).
    plus_member: bool = False
    plus_since: date | None = None

    @model_validator(mode="after")
    def _check_plus(self) -> Customer:
        if self.plus_member != (self.plus_since is not None):
            raise ValueError(f"{self.customer_id}: plus_since is required iff plus_member")
        return self


class RefundRecord(StrictModel):
    refund_id: NonEmptyStr
    order_id: NonEmptyStr
    amount: float = Field(gt=0)
    issued_at: date


class FixtureDB(StrictModel):
    # The scenario's "today". All day counts are measured from here.
    as_of: date
    customers: list[Customer] = Field(default_factory=list)
    orders: list[Order] = Field(default_factory=list)
    refunds: list[RefundRecord] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check_references(self) -> FixtureDB:
        for kind, ids in (
            ("customer", [c.customer_id for c in self.customers]),
            ("order", [o.order_id for o in self.orders]),
            ("refund", [r.refund_id for r in self.refunds]),
        ):
            if len(ids) != len(set(ids)):
                raise ValueError(f"duplicate {kind} ids")
        customers = {c.customer_id: c for c in self.customers}
        orders = {o.order_id: o for o in self.orders}
        for o in self.orders:
            customer = customers.get(o.customer_id)
            if customer is None:
                raise ValueError(f"{o.order_id}: unknown customer {o.customer_id}")
            if o.email != customer.email:
                raise ValueError(f"{o.order_id}: email does not match customer {o.customer_id}")
            if o.placed_at > self.as_of:
                raise ValueError(f"{o.order_id}: placed after as_of")
        for r in self.refunds:
            if r.order_id not in orders:
                raise ValueError(f"{r.refund_id}: unknown order {r.order_id}")
            if r.issued_at > self.as_of:
                raise ValueError(f"{r.refund_id}: issued after as_of")
        for o in self.orders:
            refunded = sum(r.amount for r in self.refunds if r.order_id == o.order_id)
            if refunded > o.total + 1e-9:
                raise ValueError(f"{o.order_id}: refunds exceed order total")
        return self

    def customer(self, customer_id: str) -> Customer | None:
        return next((c for c in self.customers if c.customer_id == customer_id), None)

    def order(self, order_id: str) -> Order | None:
        return next((o for o in self.orders if o.order_id == order_id), None)

    def refunds_for_order(self, order_id: str) -> list[RefundRecord]:
        return [r for r in self.refunds if r.order_id == order_id]

    def refunds_for_customer(self, customer_id: str, within_days: int) -> list[RefundRecord]:
        order_ids = {o.order_id for o in self.orders if o.customer_id == customer_id}
        return [
            r
            for r in self.refunds
            if r.order_id in order_ids and (self.as_of - r.issued_at).days <= within_days
        ]


def load_db(path: Path | str = DEFAULT_DB_PATH) -> FixtureDB:
    return FixtureDB.model_validate(json.loads(Path(path).read_text(encoding="utf-8")))
