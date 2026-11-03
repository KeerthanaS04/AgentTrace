"""Shared pieces for the LLM-backed support agents.

Every agent asks its model for a JSON reply that includes the three explanation
fields below, so each step's `Explanation` comes from the model that made the
decision rather than being written after the fact.
"""

from __future__ import annotations

import json
from typing import Any, TypeVar

from pydantic import ConfigDict, Field

from agenttrace.gateway.gateway import Gateway
from agenttrace.schemas.base import Probability, StrictModel
from agenttrace.schemas.span import Explanation
from agenttrace.workflow.runtime import StepContext

DEFAULT_ROLE = "agent_default"


class AgentFailure(RuntimeError):
    """The agent could not produce a valid output (e.g. its model never returned valid JSON)."""


class LLMReply(StrictModel):
    """Base for parsing model replies: extra keys a model adds are dropped, not fatal.

    (Measured: Nemotron added an unrequested "reply_length_check" field and repeated it
    on every repair attempt at temperature 0, so strict parsing failed the step.)
    """

    model_config = ConfigDict(extra="ignore", validate_assignment=True)


class ExplainedReply(LLMReply):
    """Fields every agent's LLM reply carries, in addition to its own."""

    rationale: str = Field(description="One or two sentences on why you chose this output.")
    cited_evidence: list[str] = Field(
        default_factory=list,
        description="Exact IDs or short quotes from the input that support the output.",
    )
    confidence: Probability = Field(description="0 to 1: how sure you are the output is right.")

    def explanation(self, cited: list[str] | None = None) -> Explanation:
        return Explanation(
            rationale=self.rationale,
            cited_evidence=self.cited_evidence if cited is None else cited,
            confidence=self.confidence,
        )


ReplyT = TypeVar("ReplyT", bound=ExplainedReply)


def ask(ctx: StepContext, prompt: str, schema: type[ReplyT]) -> ReplyT:
    """One structured LLM call on the step's configured role (default `agent_default`)."""
    gateway = ctx.resource("gateway", Gateway)
    role = str(ctx.params.get("role", DEFAULT_ROLE))
    result = gateway.complete_structured(
        role,
        [{"role": "user", "content": prompt}],
        schema,
        params=dict(ctx.model_params) or None,  # run-level overrides, e.g. a seed (P4-T5)
        tag=f"{ctx.run_id}/{ctx.step}",
    )
    if result.value is None:
        last = result.errors[-1] if result.errors else "no reply"
        raise AgentFailure(
            f"{ctx.step}: no valid {schema.__name__} after {result.attempts} tries; "
            f"last error: {last[:300]}"
        )
    return result.value


def as_json(value: Any) -> str:
    """Compact, stable JSON for embedding data in prompts."""
    return json.dumps(value, indent=1, sort_keys=True, default=str)
