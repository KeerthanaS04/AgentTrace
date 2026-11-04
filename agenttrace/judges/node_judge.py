"""Node judge (P6-T1): one model's two-question verdict on one span, as an AnomalyScore.

    result = judge_span(gateway, run, "policy", role="judge_fast")
    result.score        # schemas.AnomalyScore, or None if the judge never gave a valid reply
    result.attempts     # 1 + repairs; result.errors holds each rejected reply's problem

The reply is validated twice: against `JudgeReply` (types, probabilities in [0, 1],
1-5 evidence items), and its citations against the context. Every evidence `ref` must be
a span id the context lists or a policy doc id that appears in it, so a judge cannot cite
what it was not shown. A rejected reply goes back to the judge with the reason (the
gateway's repair loop, up to 2 repairs); if it still fails, the score is None and the
failure is recorded, never guessed.
"""

from __future__ import annotations

import json
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any

from pydantic import Field, model_validator

from agenttrace.gateway.gateway import Gateway
from agenttrace.judges.context import JudgeContext, judge_context
from agenttrace.judges.registry import load_prompt
from agenttrace.schemas.base import NonEmptyStr, Probability
from agenttrace.schemas.diagnosis import AnomalyScore, Verdict
from agenttrace.schemas.run import Run
from agenttrace.workflow.agents.base import LLMReply
from agenttrace.workflow.tools import load_db

# The refs the reply being validated may cite (set per call; read by the validator).
_ALLOWED: ContextVar[frozenset[str] | None] = ContextVar("judge_allowed_refs", default=None)


class Citation(LLMReply):
    ref: NonEmptyStr = Field(description="A span id or policy doc id from the context.")
    claim: NonEmptyStr = Field(description="What that span or document shows.")


class JudgeReply(LLMReply):
    inputs_ok: Probability = Field(description="How sure the inputs are trustworthy (0-1).")
    inputs_reasoning: str = Field(description="(a) Why the inputs are or are not trustworthy.")
    output_ok_given_inputs: Probability = Field(
        description="How sure the output is correct given these inputs (0-1)."
    )
    output_reasoning: str = Field(description="(b) Why the output is or is not correct.")
    verdict: Verdict
    evidence: list[Citation] = Field(min_length=1, max_length=5)

    @model_validator(mode="after")
    def _cites_only_the_context(self) -> JudgeReply:
        allowed = _ALLOWED.get()
        if allowed is not None:
            unknown = [c.ref for c in self.evidence if c.ref not in allowed]
            if unknown:
                raise ValueError(
                    f"evidence cites {unknown}, which are not in the context; cite only "
                    f"these ids: {sorted(allowed)}"
                )
        return self


@dataclass
class JudgeResult:
    span_id: str
    role: str
    model: str
    prompt_version: str
    score: AnomalyScore | None
    attempts: int
    errors: list[str] = field(default_factory=list)
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0
    latency_s: float = 0.0

    @property
    def valid(self) -> bool:
        return self.score is not None


def _dump(value: Any) -> str:
    return json.dumps(value, indent=1, sort_keys=True, default=str) if value else "(none)"


def render_prompt(ctx: JudgeContext, template: str, today: str) -> str:
    citable = [f"- {s}" for s in ctx.citable]
    if ctx.doc_ids:
        citable.append(f"- policy documents: {', '.join(ctx.doc_ids)}")
    return template.format(
        today=today,
        span_id=ctx.span_id,
        kind=ctx.kind,
        node_id=ctx.node_id,
        description=ctx.description,
        input_expectations=ctx.input_expectations or "(not specified)",
        output_expectations=ctx.output_expectations or "(not specified)",
        inputs=_dump(ctx.inputs),
        output=_dump(ctx.output),
        explanation=_dump(ctx.explanation),
        parents=_dump(ctx.parents),
        calls=_dump(ctx.calls),
        citable="\n".join(citable),
    )


def to_score(reply: JudgeReply, span_id: str, judge: str, prompt_version: str) -> AnomalyScore:
    return AnomalyScore(
        span_id=span_id,
        judge=judge,
        prompt_version=prompt_version,
        output_ok_given_inputs=reply.output_ok_given_inputs,
        inputs_ok=reply.inputs_ok,
        verdict=reply.verdict,
        evidence=[f"{c.ref}: {c.claim}" for c in reply.evidence],
        reasoning=f"(a) inputs: {reply.inputs_reasoning}\n(b) output: {reply.output_reasoning}",
    )


def judge_span(
    gateway: Gateway,
    run: Run,
    target: str,
    *,
    role: str = "judge_fast",
    version: str | None = None,
    today: str | None = None,
) -> JudgeResult:
    """Judge one span. The score's `span_id` is the full span id; `judge` is the role."""
    ctx = judge_context(run, target)
    template, prompt_version = load_prompt("node_judge", version)
    prompt = render_prompt(ctx, template, today or load_db().as_of.isoformat())
    token = _ALLOWED.set(frozenset(ctx.refs()))
    try:
        result = gateway.complete_structured(
            role,
            [{"role": "user", "content": prompt}],
            JudgeReply,
            tag=f"judge/{role}/{ctx.span_id}",
        )
    finally:
        _ALLOWED.reset(token)
    full_id = f"{run.run_id}/{ctx.span_id}"
    calls = result.completions
    score = None
    if result.value is not None:
        score = to_score(result.value, full_id, role, prompt_version)
    return JudgeResult(
        span_id=full_id,
        role=role,
        model=calls[-1].model if calls else "",
        prompt_version=prompt_version,
        score=score,
        attempts=result.attempts,
        errors=result.errors,
        tokens_in=sum(c.tokens_in for c in calls),
        tokens_out=sum(c.tokens_out for c in calls),
        cost_usd=sum(c.cost_usd for c in calls),
        latency_s=sum(c.latency_s for c in calls),
    )
