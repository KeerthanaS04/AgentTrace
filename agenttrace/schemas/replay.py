"""Counterfactual replay results."""

from __future__ import annotations

from typing import Any

from pydantic import Field, model_validator

from agenttrace.schemas.base import NonEmptyStr, Probability, StrictModel
from agenttrace.schemas.diagnosis import Patch, PatchSource


class ReplayResult(StrictModel):
    original_run_id: NonEmptyStr
    replay_run_id: NonEmptyStr
    diagnosis_id: str | None = None
    patch: Patch
    patched_span_id: NonEmptyStr
    patch_source: PatchSource
    # Local ids (e.g. "policy", "tool_agent.llm#2") of spans the replay patched, changed,
    # added, or no longer has (replay.differ).
    changed_spans: list[str] = Field(default_factory=list)
    original_final: dict[str, Any] = Field(default_factory=dict)
    replay_final: dict[str, Any] = Field(default_factory=dict)
    recovered: bool
    recovery_score: Probability
    # Scored against the clean run of the same seed (P5-T5): was recovery possible at all
    # (the clean run was correct), and did the replay reach the clean run's outcome?
    clean_correct: bool | None = None
    matches_clean: bool | None = None

    @model_validator(mode="after")
    def _check_consistency(self) -> ReplayResult:
        if self.replay_run_id == self.original_run_id:
            raise ValueError("replay_run_id must differ from original_run_id")
        if self.patch.target_span_id != self.patched_span_id:
            raise ValueError("patched_span_id must match patch.target_span_id")
        if self.patch.source != self.patch_source:
            raise ValueError("patch_source must match patch.source")
        return self
