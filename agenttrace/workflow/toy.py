"""Toy 2-agent workflow that exercises the runtime without any model calls.

    triage  : ticket text -> category (keyword rules)
    answer  : ticket text + category -> reply citing the top knowledge-base doc

Run `python tasks.py toy` to print each step's inputs, refs, output, and explanation.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from agenttrace.schemas.span import Explanation
from agenttrace.workflow.kb.index import BM25Index
from agenttrace.workflow.kb.loader import load_kb
from agenttrace.workflow.runtime import (
    AgentOutput,
    FromInput,
    FromStep,
    Step,
    StepContext,
    Workflow,
    WorkflowResult,
)

KEYWORDS = {
    "refund": ("refund", "return", "money back"),
    "shipping": ("shipping", "delivery", "arrive", "tracking"),
    "account": ("password", "login", "account", "locked"),
}


class TriageAgent:
    name = "toy_triage"
    input_keys: tuple[str, ...] = ("ticket_text",)
    output_keys: tuple[str, ...] = ("category",)

    def run(self, inputs: Mapping[str, Any], ctx: StepContext) -> AgentOutput:
        text = inputs["ticket_text"].lower()
        for category, words in KEYWORDS.items():
            matched = [w for w in words if w in text]
            if matched:
                return AgentOutput(
                    output={"category": category},
                    explanation=Explanation(
                        rationale=f"matched keywords {matched}",
                        cited_evidence=matched,
                        confidence=0.9,
                    ),
                )
        return AgentOutput(
            output={"category": "other"},
            explanation=Explanation(rationale="no keyword matched", confidence=0.5),
        )


class AnswerAgent:
    name = "toy_answer"
    input_keys: tuple[str, ...] = ("ticket_text", "category")
    output_keys: tuple[str, ...] = ("reply", "doc_id")

    def run(self, inputs: Mapping[str, Any], ctx: StepContext) -> AgentOutput:
        index = ctx.resource("kb_index", BM25Index)
        hits = index.search(f"{inputs['category']} {inputs['ticket_text']}", k=int(ctx.params["k"]))
        if not hits:
            return AgentOutput(
                output={"reply": "We'll look into this and get back to you.", "doc_id": None},
                explanation=Explanation(rationale="no relevant policy found", confidence=0.2),
            )
        top = hits[0]
        return AgentOutput(
            output={
                "reply": f"Per our policy '{top.title}', here is how we can help.",
                "doc_id": top.doc_id,
            },
            explanation=Explanation(
                rationale=f"top BM25 hit for a {inputs['category']} ticket",
                cited_evidence=[top.doc_id],
                confidence=0.7,
            ),
        )


def build_toy_workflow() -> Workflow:
    return Workflow(
        name="toy",
        inputs=["ticket_text"],
        steps=[
            Step("triage", TriageAgent(), {"ticket_text": FromInput("ticket_text")}),
            Step(
                "answer",
                AnswerAgent(),
                {
                    "ticket_text": FromInput("ticket_text"),
                    "category": FromStep("triage", "category"),
                },
                params={"k": 3},
            ),
        ],
        outputs={"reply": FromStep("answer", "reply"), "category": FromStep("triage", "category")},
    )


def run_toy(ticket_text: str, run_id: str = "toy-run") -> WorkflowResult:
    index = BM25Index.from_kb(load_kb())
    return build_toy_workflow().run(
        {"ticket_text": ticket_text}, run_id=run_id, resources={"kb_index": index}
    )


def main() -> int:
    ticket = "My headphones arrived yesterday but I want to return them for a refund."
    result = run_toy(ticket)
    print(f"ticket : {ticket}")
    print(f"status : {result.status}\n")
    for rec in result.steps:
        print(f"[{rec.step}] agent={rec.agent} status={rec.status}")
        for ref in rec.input_refs:
            print(f"    input  {ref.key:<12} <- {ref.span_id}  ({ref.content_hash[:15]}...)")
        print(f"    output {json.dumps(rec.output)}")
        if rec.explanation:
            ex = rec.explanation
            print(
                f"    why    {ex.rationale} (confidence {ex.confidence}, cites {ex.cited_evidence})"
            )
        print()
    print(f"final  : {json.dumps(result.outputs)}")
    return 0 if result.status == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
