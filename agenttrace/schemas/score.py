"""Continuous explainability scores."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field, model_validator

from agenttrace.schemas.base import Probability, StrictModel, UtcDatetime, utc_now

# G = grounding, RAC = rationale-action consistency, Fa = faithfulness,
# C = calibration, XI = composite explainability index.
Metric = Literal["G", "RAC", "Fa", "C", "XI"]
ScoreScope = Literal["span", "agent", "run"]


class ExplainabilityScore(StrictModel):
    scope: ScoreScope
    run_id: str | None = None
    span_id: str | None = None
    agent: str | None = None
    metric: Metric
    value: Probability
    # None for metrics computed without a judge (e.g. calibration).
    judge: str | None = None
    prompt_version: str | None = None
    details: dict[str, Any] = Field(default_factory=dict)
    computed_at: UtcDatetime = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def _check_scope_ids(self) -> ExplainabilityScore:
        required = {
            "span": ("run_id", "span_id"),
            "agent": ("agent",),
            "run": ("run_id",),
        }[self.scope]
        missing = [name for name in required if not getattr(self, name)]
        if missing:
            raise ValueError(f"scope={self.scope!r} requires {', '.join(missing)}")
        return self
