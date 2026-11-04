"""A full workflow execution."""

from __future__ import annotations

from typing import Any

from pydantic import Field, model_validator

from agenttrace.schemas.base import NonEmptyStr, StrictModel, UtcDatetime, utc_now
from agenttrace.schemas.span import Span


class Run(StrictModel):
    run_id: NonEmptyStr
    scenario_id: str | None = None
    workflow_version: NonEmptyStr
    agent_models: dict[str, str] = Field(default_factory=dict)
    final_output: dict[str, Any] = Field(default_factory=dict)
    final_correct: bool | None = None
    spans: list[Span] = Field(default_factory=list)
    # Set on counterfactual replays: the run this one was replayed from.
    replay_of: str | None = None
    seed: int | None = None
    # Overrides applied to every model call of the run (e.g. {"seed": 2}); a replay must
    # use the same ones, or no recorded call matches.
    model_params: dict[str, Any] = Field(default_factory=dict)
    created_at: UtcDatetime = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def _check_spans(self) -> Run:
        ids = [s.span_id for s in self.spans]
        if len(ids) != len(set(ids)):
            raise ValueError("span_ids within a run must be unique")
        foreign = {s.run_id for s in self.spans} - {self.run_id}
        if foreign:
            raise ValueError(f"spans belong to a different run: {sorted(foreign)}")
        if self.replay_of == self.run_id:
            raise ValueError("a run cannot be a replay of itself")
        return self
