"""Retriever: write search queries for the ticket and select policy docs from the KB."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from pydantic import Field

from agenttrace.workflow.agents.base import ExplainedReply, as_json, ask
from agenttrace.workflow.kb.index import BM25Index
from agenttrace.workflow.kb.loader import KnowledgeBase
from agenttrace.workflow.runtime import AgentOutput, StepContext

HITS_PER_QUERY = 3
MAX_DOCS = 6
# Searched on every ticket, like a support playbook's "always check escalation rules"
# step. (Measured: without it, 2 of 20 clean tickets missed the escalation policy.)
STANDING_QUERY = "when to escalate refund approval limit supervisor legal action fraud chargeback"

PROMPT = """You are the Retriever in a customer-support system for the online retailer Brightcart.
Write 1 to 3 short keyword search queries that will find the company policies needed to resolve
this ticket. Cover the main request and any rule that could change the outcome, such as return
windows for the item type, damaged items, membership benefits, refund approval limits, fraud
review, chargebacks, repeated refunds, or escalation.

Ticket category: {category}
Ticket:
{ticket}"""


class RetrieverReply(ExplainedReply):
    queries: list[str] = Field(min_length=1, max_length=3)


class RetrieverAgent:
    """The model writes the queries; BM25 does the search, so results are reproducible.

    The ticket's own text and a standing escalation query are searched too, which keeps
    recall up when the model's queries miss.
    """

    name = "retriever"
    prompt_version = "retriever-v1"
    input_keys: tuple[str, ...] = ("ticket", "category")
    output_keys: tuple[str, ...] = ("queries", "docs")

    def run(self, inputs: Mapping[str, Any], ctx: StepContext) -> AgentOutput:
        ticket = inputs["ticket"]
        reply = ask(
            ctx,
            PROMPT.format(category=inputs["category"], ticket=as_json(ticket)),
            RetrieverReply,
        )
        doc_ids, docs = select_docs(
            ctx.resource("kb_index", BM25Index),
            ctx.resource("kb", KnowledgeBase),
            reply.queries,
            ticket,
            ctx.params,
        )
        return AgentOutput(
            output={"queries": reply.queries, "docs": docs},
            explanation=reply.explanation(cited=doc_ids),
        )


def select_docs(
    index: BM25Index,
    kb: KnowledgeBase,
    queries: list[str],
    ticket: Mapping[str, Any],
    params: Mapping[str, Any],
) -> tuple[list[str], list[dict[str, Any]]]:
    """Search the model's queries and the ticket text, cap, then add the standing
    escalation docs. Deterministic given the queries (used again by replay rules, P5-T4)."""
    k = int(params.get("hits_per_query", HITS_PER_QUERY))
    doc_ids: list[str] = []
    for query in [*queries, f"{ticket['subject']} {ticket['body']}"]:
        for hit in index.search(query, k):
            if hit.doc_id not in doc_ids:
                doc_ids.append(hit.doc_id)
    doc_ids = doc_ids[: int(params.get("max_docs", MAX_DOCS))]
    # Standing escalation docs are added after the cap, so they are never crowded out.
    doc_ids += [h.doc_id for h in index.search(STANDING_QUERY, k) if h.doc_id not in doc_ids]
    docs = [
        {
            "doc_id": d.doc_id,
            "title": d.title,
            "effective_date": d.effective_date.isoformat(),
            "body": d.body.strip(),
        }
        for d in (kb.get(i) for i in doc_ids)
    ]
    return doc_ids, docs
