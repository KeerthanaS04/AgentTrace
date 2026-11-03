"""Generate the labelled scenario dataset (P4-T2).

    uv run python -m agenttrace.scenarios.generator           # write data/scenarios/
    uv run python -m agenttrace.scenarios.generator --check   # verify the committed file

A scenario = a base ticket (one of the clean tickets the default model resolves
correctly) + a fault spec + ground truth. Every fault is written out below, by hand,
aimed at a fact the base ticket's decision actually depends on. Ground truth:

    root_cause_node_ids  the fault targets (logical node ids, e.g. "tool_agent.order_lookup")
    affected_node_ids    the steps the corrupted value reaches, computed from the declared
                         wiring down to the field: a wrong `router.order_id` reaches the
                         Tool Agent, Policy and Writer, but not the Retriever
    expected_final       the *correct* outcome (the base ticket's), which a faulty run
                         should miss (F1-F7) or still reach (F8, F0)

Scenario ids are opaque (`sc-001`, ...) and assigned by a seeded shuffle: diagnosis can see
a run's scenario id, so the id must not reveal the fault.

Every spec is checked by running it (P4-T4, `python tasks.py verify`). Specs whose fault
turned out not to matter are moved to DROPPED with the measured reason.
"""

from __future__ import annotations

import argparse
import json
import random
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from agenttrace.scenarios.ground_truth import SCENARIOS_PATH
from agenttrace.schemas.ground_truth import (
    ExpectedFinal,
    FaultCode,
    FaultInjection,
    GroundTruth,
    Scenario,
)
from agenttrace.workflow.clean_eval import load_scenarios
from agenttrace.workflow.runtime import FromStep, Workflow
from agenttrace.workflow.support import build_support_workflow

ROOT = Path(__file__).resolve().parents[2]
SEED = 20261103

# Clean tickets used as bases. clean-05 and clean-18 are left out: the default model
# already gets them wrong (findings §6), so a fault there could not be told apart from
# the model's own error.
EXCLUDED_BASES = ("clean-05", "clean-18")

Spec = tuple[FaultCode, str, list[FaultInjection]]


def search(injector: str, **params: Any) -> FaultInjection:
    return FaultInjection(target_node_id="retriever.search", injector=injector, params=params)


def tool(name: str, injector: str, **params: Any) -> FaultInjection:
    # occurrence 1: the first call to the tool is the lookup the ticket is about.
    return FaultInjection(
        target_node_id=f"tool_agent.{name}", injector=injector, params={"occurrence": 1, **params}
    )


def agent(step: str, injector: str, **params: Any) -> FaultInjection:
    return FaultInjection(target_node_id=step, injector=injector, params=params)


def wrong_doc(replace: str, with_: str) -> FaultInjection:
    return search("retrieval_wrong_doc", replace=replace, **{"with": with_})


def claim(text: str) -> FaultInjection:
    return agent("policy", "hallucinated_claim", field="justification", claim=text)


def invent(text: str) -> FaultInjection:
    """A fabricated justification: the claim replaces the Policy Agent's reasoning."""
    return agent("policy", "hallucinated_claim", field="justification", claim=text, mode="replace")


IRRELEVANT = ["KB-GEN-001", "KB-SHP-006"]


def order_value(path: str, value: Any) -> FaultInjection:
    return tool("order_lookup", "tool_wrong_value", path=f"order.{path}", value=value)


def order_values(**values: Any) -> FaultInjection:
    """Several order fields changed together, so no sibling field contradicts the fault."""
    paths = {f"order.{k}": v for k, v in values.items()}
    return tool("order_lookup", "tool_wrong_value", values=paths)


# O1013 (clean-13) rewritten as delivered on its promised date: every date field agrees.
ON_TIME_O1013 = order_values(delivered_at="2026-01-16", delivered_late=False, days_since_delivery=4)


def account_value(path: str, value: Any) -> FaultInjection:
    return tool("account_status", "tool_wrong_value", path=path, value=value)


def broken(name: str, mode: str, **params: Any) -> FaultInjection:
    return tool(name, "tool_malformed", mode=mode, **params)


def set_out(step: str, field: str, value: Any) -> FaultInjection:
    return agent(step, "set_field", field=field, value=value)


# (fault code, base ticket, injections). Comments say what the fault should break.
#
# Rule learned from verification (P4-T4, findings §6): the workflow is redundant, so a
# fault only matters when it removes or corrupts the *only* source of the fact the outcome
# rests on. Second sources seen in faulty runs:
#   - the ticket text: the Policy Agent falls back on the customer's own claims ("I'm a
#     Plus member", "it arrived late") when a tool fact is missing, despite its prompt;
#   - the refund_api quote, computed from the database, not from order_lookup's output;
#   - a sibling field, e.g. the dates next to `delivered_late`;
#   - a second KB document restating the rule, or the model's own knowledge of it.
# Superseded KB documents say so ("Superseded by ..."), so the model rightly ignores them.
SPECS: list[Spec] = [
    # ---- F1 wrong / stale retrieval: the governing policy swapped for a plausible wrong one
    ("F1", "clean-10", [wrong_doc("KB-REF-001", "KB-REF-008")]),  # holiday rule, wrong dates
    ("F1", "clean-19", [wrong_doc("KB-REF-008", "KB-REF-001")]),  # no holiday extension
    ("F1", "clean-02", [wrong_doc("KB-REF-002", "KB-REF-001")]),  # loses the 10% fee
    ("F1", "clean-14", [wrong_doc("KB-REF-003", "KB-REF-001")]),  # gift cards look returnable
    ("F1", "clean-17", [wrong_doc("KB-ACC-004", "KB-REF-011")]),  # no Plus refund window
    ("F1", "clean-12", [wrong_doc("KB-ACC-001", "KB-ACC-002")]),  # locked, not password reset
    ("F1", "clean-12", [wrong_doc("KB-ACC-001", "KB-ACC-005")]),  # 10-minute email code instead
    ("F1", "clean-07", [wrong_doc("KB-ESC-003", "KB-ESC-004")]),  # no repeat-refund rule
    ("F1", "clean-20", [wrong_doc("KB-ACC-005", "KB-ACC-002")]),  # phone code, not T&S
    ("F1", "clean-02", [wrong_doc("KB-REF-002", "KB-REF-004")]),  # damaged rule: no fee
    # ---- F2 empty or irrelevant retrieval
    ("F2", "clean-01", [search("retrieval_empty")]),
    ("F2", "clean-03", [search("retrieval_empty")]),
    ("F2", "clean-07", [search("retrieval_empty")]),
    ("F2", "clean-08", [search("retrieval_empty")]),
    ("F2", "clean-20", [search("retrieval_empty")]),
    ("F2", "clean-12", [search("retrieval_empty")]),  # the 30-minute link is KB-only
    ("F2", "clean-06", [search("retrieval_empty")]),  # Trust & Safety is named only in the KB
    ("F2", "clean-02", [search("retrieval_irrelevant", docs=IRRELEVANT)]),
    ("F2", "clean-14", [search("retrieval_irrelevant", docs=IRRELEVANT)]),
    ("F2", "clean-17", [search("retrieval_irrelevant", docs=IRRELEVANT)]),
    # ---- F3 wrong value in a tool response: the fact the decision rests on
    ("F3", "clean-01", [order_value("days_since_delivery", 45)]),
    ("F3", "clean-02", [order_value("days_since_delivery", 20)]),
    ("F3", "clean-03", [account_value("plus_member", False)]),
    ("F3", "clean-04", [order_value("days_since_delivery", 12)]),
    ("F3", "clean-06", [account_value("under_fraud_review", False)]),
    ("F3", "clean-07", [account_value("refunds_last_90_days", 1)]),
    ("F3", "clean-09", [account_value("plus_member", False)]),
    ("F3", "clean-10", [order_value("days_since_delivery", 10)]),
    ("F3", "clean-11", [order_value("status", "shipped")]),
    ("F3", "clean-17", [account_value("plus_since", "2026-01-12")]),  # joined 8 days ago
    # ---- F4 malformed / error tool response, on a tool that is the fact's only source
    ("F4", "clean-06", [broken("account_status", "error")]),
    (
        "F4",
        "clean-07",
        [broken("account_status", "missing_fields", fields=["refunds_last_90_days"])],
    ),
    ("F4", "clean-11", [broken("order_lookup", "error", message="service unavailable")]),
    ("F4", "clean-17", [broken("account_status", "error")]),
    ("F4", "clean-07", [broken("account_status", "error")]),
    (
        "F4",
        "clean-07",
        [broken("account_status", "wrong_type", path="refunds_last_90_days", value="N/A")],
    ),
    # The join date is the deny's only basis, and the customer never states it.
    ("F4", "clean-17", [broken("account_status", "missing_fields", fields=["plus_since"])]),
    ("F4", "clean-17", [broken("account_status", "wrong_type", path="plus_since", value="N/A")]),
    # The delivery date is the deny's only basis, and the customer never states it.
    ("F4", "clean-10", [broken("order_lookup", "error")]),
    (
        "F4",
        "clean-10",
        [
            broken(
                "order_lookup",
                "missing_fields",
                fields=[
                    "order.placed_at",
                    "order.shipped_at",
                    "order.delivered_at",
                    "order.days_since_placed",
                    "order.business_days_since_placed",
                    "order.days_since_delivery",
                ],
            )
        ],
    ),
    (
        "F4",
        "clean-06",
        [
            broken(
                "account_status", "missing_fields", fields=["account_status", "under_fraud_review"]
            )
        ],
    ),
    # ---- F5 invalid intermediate decision
    ("F5", "clean-01", [set_out("policy", "decision", "deny")]),
    ("F5", "clean-04", [set_out("policy", "decision", "deny")]),
    ("F5", "clean-09", [set_out("policy", "decision", "deny")]),
    ("F5", "clean-10", [set_out("policy", "decision", "approve")]),
    ("F5", "clean-12", [set_out("policy", "decision", "approve")]),
    ("F5", "clean-14", [set_out("policy", "decision", "approve")]),
    ("F5", "clean-15", [set_out("policy", "decision", "approve")]),
    ("F5", "clean-16", [set_out("policy", "decision", "approve")]),
    ("F5", "clean-02", [set_out("router", "order_id", "O1999")]),
    ("F5", "clean-07", [set_out("policy", "escalate_to", "Payments team")]),
    ("F5", "clean-13", [set_out("policy", "decision", "deny")]),
    ("F5", "clean-17", [set_out("policy", "decision", "approve")]),
    # ---- F6 hallucinated intermediate claim on the reply's required fact. Appended claims
    # work only where the justification does not already state the true fact; elsewhere the
    # whole justification is fabricated (same decision, wrong fact).
    ("F6", "clean-01", [claim("Per KB-REF-007, card refunds take 10-14 business days.")]),
    (
        "F6",
        "clean-09",
        [claim("Per KB-REF-009, rugs carry a $25 return fee, so the refund is $115.")],
    ),
    (
        "F6",
        "clean-01",
        [
            invent(
                "Per KB-REF-001, the wool sweater is unused and was delivered 18 days ago, "
                "within the 30-day window, so a full refund of $48.00 is approved. Per "
                "KB-REF-007, it is issued as Brightcart store credit, available immediately."
            )
        ],
    ),
    (
        "F6",
        "clean-02",
        [
            invent(
                "Per KB-REF-002, opened electronics in working condition are refunded minus a "
                "flat $12.90 handling charge. The headphones were delivered 8 days ago, within "
                "the 15-day window, so the refund is $116.10."
            )
        ],
    ),
    (
        "F6",
        "clean-10",
        [
            invent(
                "KB-REF-001 allows returns within 60 days of delivery for most items. The desk "
                "lamp was delivered on 2025-11-01, 80 days ago, past the 60-day window, and the "
                "customer is not a Plus member, so the return is declined."
            )
        ],
    ),
    (
        "F6",
        "clean-11",
        [
            invent(
                "Per KB-SHP-007, an order cancelled before it ships is refunded minus a $5.00 "
                "cancellation fee. Order O1011 has not shipped, so it is cancelled with a "
                "refund of $49.00."
            )
        ],
    ),
    (
        "F6",
        "clean-14",
        [
            invent(
                "Per KB-REF-003, gift cards can be returned only within 7 days of purchase. "
                "This gift card was bought more than 7 days ago, so the refund is declined."
            )
        ],
    ),
    (
        "F6",
        "clean-14",
        [
            invent(
                "Per KB-REF-003, a gift card cannot be returned once its code has been emailed "
                "to the recipient. This card's code was already sent, so the refund is declined."
            )
        ],
    ),
    (
        "F6",
        "clean-01",
        [
            invent(
                "Per KB-REF-001, the wool sweater is unused and was delivered 18 days ago, "
                "within the 30-day window, so a full refund of $48.00 is approved. Per "
                "KB-REF-007, it goes back to the customer's PayPal account within 3-5 business "
                "days."
            )
        ],
    ),
    (
        "F6",
        "clean-09",
        [
            invent(
                "Per KB-REF-011, Plus members can return most items within 60 days, and the rug "
                "was delivered 56 days ago. Per KB-REF-009, a $25.00 oversize return fee applies "
                "to rugs, so the refund is $115.00."
            )
        ],
    ),
    (
        "F6",
        "clean-17",
        [
            invent(
                "Per KB-ACC-004, a Brightcart Plus membership can be cancelled for a refund only "
                "within 7 days of joining. This membership started on 2025-06-01, so cancelling "
                "stops the next renewal but no refund is given."
            )
        ],
    ),
    # ---- F7 multi-fault: two independent roots
    (
        "F7",
        "clean-01",
        [wrong_doc("KB-REF-001", "KB-REF-010"), order_value("days_since_delivery", 45)],
    ),
    (
        "F7",
        "clean-02",
        [
            order_value("days_since_delivery", 20),
            claim("Per KB-REF-002, opened electronics get a full refund with no restocking fee."),
        ],
    ),
    (
        "F7",
        "clean-03",
        [
            account_value("plus_member", False),
            set_out("policy", "escalate_to", "Finance Review team"),
        ],
    ),
    (
        "F7",
        "clean-04",
        [order_value("days_since_delivery", 12), account_value("refunds_last_90_days", 5)],
    ),
    (
        "F7",
        "clean-07",
        [account_value("refunds_last_90_days", 1), search("retrieval_irrelevant", docs=IRRELEVANT)],
    ),
    (
        "F7",
        "clean-09",
        [account_value("plus_member", False), wrong_doc("KB-REF-011", "KB-REF-010")],
    ),
    (
        "F7",
        "clean-10",
        [
            order_value("days_since_delivery", 10),
            claim("Per KB-REF-001, the standard return window is 90 days."),
        ],
    ),
    ("F7", "clean-11", [order_value("status", "shipped"), search("retrieval_empty")]),
    (
        "F7",
        "clean-14",
        [order_value("items.0.category", "apparel"), wrong_doc("KB-REF-003", "KB-REF-001")],
    ),
    (
        "F7",
        "clean-17",
        [account_value("plus_since", "2026-01-12"), search("retrieval_empty")],
    ),
    # ---- F8 silent: a real fault in something nothing downstream uses
    ("F8", "clean-01", [order_value("email", "x@example.com")]),
    ("F8", "clean-04", [order_value("email", "x@example.com")]),
    ("F8", "clean-10", [order_value("email", "x@example.com")]),
    ("F8", "clean-19", [order_value("email", "x@example.com")]),
    ("F8", "clean-02", [order_value("card_last4", "0000")]),
    ("F8", "clean-09", [order_value("card_last4", "0000")]),
    ("F8", "clean-07", [account_value("name", "Alex Doe")]),
    ("F8", "clean-17", [account_value("name", "Alex Doe")]),
    ("F8", "clean-03", [set_out("retriever", "queries", ["refund"])]),
    ("F8", "clean-14", [set_out("retriever", "queries", ["refund"])]),
    (
        "F8",
        "clean-11",
        [agent("policy", "hallucinated_claim", field="cited_doc_ids", claim="KB-GEN-001")],
    ),
    (
        "F8",
        "clean-13",
        [agent("policy", "hallucinated_claim", field="cited_doc_ids", claim="KB-GEN-001")],
    ),
]

# Specs removed after verification (P4-T4: `python tasks.py verify`, default profile,
# 2026-11-03), with what the faulty run showed. Every one landed and changed its root's
# output, but the final outcome (decision, required facts) stayed the same, because the
# model had a second source for the fact. Kept here so the removal is documented and
# a removed spec is never re-added by accident.
_SUPERSEDED = "the swapped-in doc says it is superseded, so the model ignored it"
_CUSTOMER = (
    "with the tool fact gone or changed, the Policy Agent took the customer's own claim "
    "(Plus member / arrived late) despite its prompt"
)
_APPENDED = "the claim was appended after the agent's correct reasoning; the writer followed that"
DROPPED: list[tuple[Spec, str]] = [
    (("F1", "clean-01", [wrong_doc("KB-REF-001", "KB-REF-010")]), _SUPERSEDED),
    (("F1", "clean-09", [wrong_doc("KB-REF-011", "KB-REF-010")]), _SUPERSEDED),
    (
        ("F1", "clean-03", [wrong_doc("KB-REF-011", "KB-REF-002")]),
        "the $200 approval limit alone forces the escalation",
    ),
    (
        ("F1", "clean-16", [wrong_doc("KB-REF-013", "KB-REF-012")]),
        "the clean reply already lacks the fact; the deny did not change",
    ),
    (
        ("F1", "clean-13", [wrong_doc("KB-SHP-005", "KB-SHP-001")]),
        "KB-SHP-002 and the refund quote still supported refunding the fee",
    ),
    (
        ("F2", "clean-11", [search("retrieval_empty")]),
        "an unshipped order was cancelled without any policy document",
    ),
    (
        ("F2", "clean-13", [search("retrieval_empty")]),
        "the order dates and the refund quote were enough to refund the fee",
    ),
    (
        ("F2", "clean-09", [search("retrieval_irrelevant", docs=IRRELEVANT)]),
        "the refund quote's eligibility was taken as the policy",
    ),
    (
        ("F3", "clean-08", [order_value("chargeback_filed", False)]),
        "the ticket and refund_api (chargeback_open) still showed the chargeback",
    ),
    (
        ("F3", "clean-13", [order_value("delivered_late", False)]),
        "the delivery dates still showed it was late",
    ),
    (
        ("F3", "clean-14", [order_value("items.0.category", "apparel")]),
        "the ticket and item name still say gift card",
    ),
    (
        ("F4", "clean-01", [broken("order_lookup", "error")]),
        "the refund quote still gave the amount; the approve stood",
    ),
    (
        (
            "F4",
            "clean-03",
            [broken("order_lookup", "missing_fields", fields=["order.items.0.price"])],
        ),
        "the order total and the ticket still give $899",
    ),
    (
        ("F4", "clean-04", [broken("refund_api", "error")]),
        "the order data alone supported the full refund",
    ),
    (
        (
            "F4",
            "clean-08",
            [broken("order_lookup", "missing_fields", fields=["order.chargeback_filed"])],
        ),
        "the ticket and refund_api still showed the chargeback",
    ),
    (
        (
            "F4",
            "clean-09",
            [broken("account_status", "wrong_type", path="plus_member", value="N/A")],
        ),
        "plus_since was still there",
    ),
    (
        (
            "F4",
            "clean-10",
            [
                broken(
                    "order_lookup", "wrong_type", path="order.days_since_delivery", value="unknown"
                )
            ],
        ),
        "delivered_at was still there; the model counted the days itself",
    ),
    (
        (
            "F4",
            "clean-13",
            [broken("order_lookup", "missing_fields", fields=["order.delivered_late"])],
        ),
        "the delivery dates still showed it was late",
    ),
    (
        (
            "F4",
            "clean-19",
            [broken("order_lookup", "wrong_type", path="order.delivered_at", value="N/A")],
        ),
        "days_since_delivery was still there; the clean reply already lacks the fact",
    ),
    (
        ("F5", "clean-11", [set_out("router", "order_id", "O1999")]),
        "the Tool Agent took the order id from the ticket text",
    ),
    (
        ("F5", "clean-20", [set_out("policy", "escalate_to", "support supervisor")]),
        "the writer named the team from KB-ACC-005 instead",
    ),
    (
        (
            "F6",
            "clean-02",
            [claim("Per KB-REF-002, opened electronics get a full refund with no restocking fee.")],
        ),
        _APPENDED,
    ),
    (
        (
            "F6",
            "clean-03",
            [claim("Per KB-REF-005, refunds over $200 go to the Finance Review team.")],
        ),
        _APPENDED,
    ),
    (
        ("F6", "clean-04", [claim("Per KB-REF-004, damaged items get a partial refund of $59.")]),
        _APPENDED,
    ),
    (
        (
            "F6",
            "clean-06",
            [claim("Per KB-ACC-003, fraud reviews are handled by the Payments team.")],
        ),
        _APPENDED,
    ),
    (
        (
            "F6",
            "clean-07",
            [claim("Per KB-ESC-003, repeat-refund reviews go to a support supervisor.")],
        ),
        _APPENDED,
    ),
    (
        ("F6", "clean-08", [claim("Per KB-ESC-002, chargebacks are handled by Trust & Safety.")]),
        _APPENDED,
    ),
    (
        ("F6", "clean-10", [claim("Per KB-REF-001, the standard return window is 90 days.")]),
        _APPENDED,
    ),
    (("F6", "clean-13", [claim("Per KB-SHP-005, the late Express refund is $9.99.")]), _APPENDED),
    (
        (
            "F6",
            "clean-17",
            [claim("Per KB-ACC-004, Plus can be refunded within 30 days of joining.")],
        ),
        _APPENDED,
    ),
    (
        (
            "F6",
            "clean-19",
            [claim("Per KB-REF-008, holiday returns are accepted until February 15.")],
        ),
        _APPENDED,
    ),
    (
        ("F7", "clean-04", [search("retrieval_empty"), order_value("days_since_delivery", 12)]),
        "the faults cancel: with no documents the 7-day damage rule is gone",
    ),
    (
        (
            "F7",
            "clean-08",
            [
                order_value("chargeback_filed", False),
                claim("Per KB-ESC-002, chargebacks are handled by Trust & Safety."),
            ],
        ),
        "the chargeback is still in the ticket and refund_api; the claim was appended",
    ),
    (
        (
            "F7",
            "clean-13",
            [
                order_value("delivered_late", False),
                claim("Per KB-SHP-005, the late Express refund is $9.99."),
            ],
        ),
        "the delivery dates still showed it was late; the claim was appended",
    ),
    # Second pass (same command and profile, 2026-11-03), replacements for the above.
    (
        ("F1", "clean-11", [wrong_doc("KB-SHP-007", "KB-SHP-008")]),
        "the model read KB-SHP-008 as allowing the cancellation",
    ),
    (
        ("F2", "clean-10", [search("retrieval_empty")]),
        "the model knew the 30-day window without any document",
    ),
    (("F3", "clean-13", [ON_TIME_O1013]), _CUSTOMER),
    (("F4", "clean-09", [broken("account_status", "error")]), _CUSTOMER),
    (("F4", "clean-03", [broken("account_status", "error")]), _CUSTOMER),
    (
        (
            "F4",
            "clean-03",
            [broken("account_status", "missing_fields", fields=["plus_member", "plus_since"])],
        ),
        _CUSTOMER,
    ),
    (
        (
            "F4",
            "clean-13",
            [
                broken(
                    "order_lookup",
                    "missing_fields",
                    fields=[
                        "order.delivered_at",
                        "order.promised_delivery",
                        "order.delivered_late",
                    ],
                )
            ],
        ),
        _CUSTOMER,
    ),
    (
        (
            "F4",
            "clean-11",
            [broken("order_lookup", "missing_fields", fields=["order.status", "order.shipped_at"])],
        ),
        "the model read the empty delivered_at as not shipped",
    ),
    (
        (
            "F6",
            "clean-04",
            [
                invent(
                    "Per KB-REF-004, an item damaged on arrival and reported within 7 days with a "
                    "photo is refunded at 75% of its price. The blender cost $79.00, so the "
                    "refund is $59.25."
                )
            ],
        ),
        "the claim reached the reply ($59.25), but the reply also quotes the $79.00 price, "
        "so the required fact '79' still matches (an outcome-checker limit, findings §6)",
    ),
    (
        (
            "F6",
            "clean-12",
            [
                invent(
                    "Per KB-ACC-001, support agents cannot see or set passwords. The customer can "
                    "reset the password from the sign-in page; the reset link is sent to the "
                    "account email and stays valid for 24 hours."
                )
            ],
        ),
        "the writer took the 30-minute expiry from KB-ACC-001 in its own documents",
    ),
    (
        (
            "F6",
            "clean-13",
            [
                invent(
                    "Per KB-SHP-005, when an Express order arrives late the customer receives a "
                    "$5.00 store credit for the delay. Order O1013 arrived 3 days late, so a "
                    "$5.00 credit is approved."
                )
            ],
        ),
        "the writer followed refund_amount ($14.99) and KB-SHP-005",
    ),
    (
        ("F7", "clean-13", [ON_TIME_O1013, wrong_doc("KB-SHP-005", "KB-SHP-001")]),
        "the model approved the fee refund with no late delivery and no policy for it",
    ),
]


# ---------------------------------------------------------------- ground truth


def _step_of(node_id: str) -> str:
    return node_id.split(".", 1)[0]


def affected_steps(workflow: Workflow, faults: Iterable[FaultInjection]) -> list[str]:
    """Steps a fault's corrupted value reaches, following the declared wiring per field.

    A fault inside a step (a tool or search call) corrupts that step's outputs; an agent
    fault corrupts only the field it sets. From there, every step that consumes a
    corrupted (step, key) is affected, and all of its outputs are. Root nodes are not
    listed as affected.
    """
    corrupted: set[tuple[str, str]] = set()
    affected: set[str] = set()
    roots: set[str] = set()
    for f in faults:
        roots.add(f.target_node_id)
        step = _step_of(f.target_node_id)
        keys = set(workflow.steps[step].agent.output_keys)
        if "." in f.target_node_id:
            affected.add(step)  # the step containing the broken call
        elif "field" in f.params:
            keys = {f.params["field"]}
        corrupted |= {(step, k) for k in keys}
    changed = True
    while changed:
        changed = False
        for name in workflow.order:
            consumer = workflow.steps[name]
            hit = any(
                isinstance(src, FromStep) and (src.step, src.key) in corrupted
                for src in consumer.inputs.values()
            )
            if hit and name not in affected:
                affected.add(name)
                corrupted |= {(name, k) for k in consumer.agent.output_keys}
                changed = True
    return [s for s in workflow.order if s in affected and s not in roots]


def build_scenarios() -> list[Scenario]:
    workflow = build_support_workflow()
    bases = {s.scenario_id: s for s in load_scenarios() if s.scenario_id not in EXCLUDED_BASES}
    drafts: list[dict[str, Any]] = []
    specs: list[Spec] = [("F0", base, []) for base in bases] + SPECS
    for code, base_id, faults in specs:
        base = bases[base_id]
        truth = GroundTruth(
            fault_code=code,
            root_cause_node_ids=list(dict.fromkeys(f.target_node_id for f in faults)),
            affected_node_ids=affected_steps(workflow, faults) if faults else [],
            expected_final=ExpectedFinal.model_validate(
                base.ground_truth.expected_final.model_dump()
            ),
        )
        drafts.append(
            {
                **base.model_dump(mode="json"),
                "faults": [f.model_dump(mode="json") for f in faults],
                "ground_truth": truth.model_dump(mode="json"),
            }
        )
    random.Random(SEED).shuffle(drafts)
    return [
        Scenario.model_validate({**d, "scenario_id": f"sc-{n:03d}"})
        for n, d in enumerate(drafts, start=1)
    ]


def render(scenarios: list[Scenario]) -> str:
    doc = {
        "about": (
            "AgentTrace scenario dataset (P4-T2). Generated by "
            "`uv run python -m agenttrace.scenarios.generator`; do not edit by hand. "
            "Ground truth here must never reach diagnosis or explainability code."
        ),
        "seed": SEED,
        "scenarios": [s.model_dump(mode="json") for s in scenarios],
    }
    return json.dumps(doc, indent=1, ensure_ascii=False) + "\n"


def coverage_table(scenarios: list[Scenario]) -> str:
    """Scenario counts per fault code (rows) and ticket category (columns)."""
    categories = sorted({s.category for s in scenarios})
    lines = [f"{'code':<6}" + "".join(f"{c:>10}" for c in categories) + f"{'total':>8}"]
    codes = sorted({s.ground_truth.fault_code for s in scenarios})
    for code in codes:
        row = [s for s in scenarios if s.ground_truth.fault_code == code]
        cells = "".join(f"{sum(s.category == c for s in row):>10}" for c in categories)
        lines.append(f"{code:<6}{cells}{len(row):>8}")
    totals = "".join(f"{sum(s.category == c for s in scenarios):>10}" for c in categories)
    lines.append(f"{'total':<6}{totals}{len(scenarios):>8}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="fail if the committed file differs")
    args = parser.parse_args(argv)
    scenarios = build_scenarios()
    text = render(scenarios)
    print(coverage_table(scenarios) + "\n")
    if args.check:
        same = SCENARIOS_PATH.exists() and SCENARIOS_PATH.read_text(encoding="utf-8") == text
        print("up to date" if same else f"{SCENARIOS_PATH} is out of date: regenerate it")
        return 0 if same else 1
    SCENARIOS_PATH.parent.mkdir(parents=True, exist_ok=True)
    SCENARIOS_PATH.write_text(text, encoding="utf-8")
    print(f"wrote {len(scenarios)} scenarios to {SCENARIOS_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
