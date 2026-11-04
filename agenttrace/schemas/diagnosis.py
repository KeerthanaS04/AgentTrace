"""Diagnosis schemas: per-node anomaly scores, root causes, propagation, fixes, patches."""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import Field, model_validator

from agenttrace.schemas.base import (
    NonEmptyStr,
    NonNegativeFloat,
    Probability,
    StrictModel,
    UtcDatetime,
    utc_now,
)

# Fault taxonomy used by the diagnosis pipeline. The injection codes (F0-F8) are
# deliberately separate and live with the ground truth.
FaultType = Literal[
    "retrieval_wrong",
    "retrieval_empty",
    "tool_wrong_value",
    "tool_malformed",
    "invalid_decision",
    "hallucination",
    "other",
]
Verdict = Literal["ok", "suspicious", "faulty"]
PatchSource = Literal["oracle", "model", "rule"]


class AnomalyScore(StrictModel):
    span_id: NonEmptyStr
    judge: NonEmptyStr
    prompt_version: NonEmptyStr
    # Local correctness: is the output right given these inputs?
    output_ok_given_inputs: Probability
    # Trust in the inputs this span received.
    inputs_ok: Probability
    verdict: Verdict
    evidence: list[str] = Field(default_factory=list)
    reasoning: str

    @model_validator(mode="after")
    def _non_ok_needs_evidence(self) -> AnomalyScore:
        if self.verdict != "ok" and not self.evidence:
            raise ValueError("a suspicious or faulty verdict must cite evidence")
        return self


class ValidatorFinding(StrictModel):
    """A deterministic check that failed on one span (P6-T2): hard evidence, no model call.

    `error`: the span's output is malformed or contradicts itself or the record (a judge
    is not needed to call it wrong). `warning`: worth a judge's look (e.g. an empty
    search, a number the inputs do not contain).
    """

    span_id: NonEmptyStr
    check: NonEmptyStr
    severity: Literal["error", "warning"]
    message: NonEmptyStr
    # Span ids / doc ids the finding rests on (like a judge's evidence refs).
    evidence: list[str] = Field(default_factory=list)


class RootCause(StrictModel):
    node_id: NonEmptyStr
    span_id: NonEmptyStr
    fault_type: FaultType
    confidence: Probability
    evidence: list[NonEmptyStr] = Field(min_length=1)
    status: Literal["candidate", "verified", "unverified"] = "candidate"
    # Set by interventional ranking: does fixing this alone recover the outcome?
    role: Literal["necessary", "contributing"] | None = None


class PropagatedError(StrictModel):
    span_id: NonEmptyStr
    node_id: NonEmptyStr
    caused_by: list[NonEmptyStr] = Field(min_length=1)
    # Span ids from a root cause to this span, inclusive at both ends.
    path: list[NonEmptyStr] = Field(min_length=2)
    verified: bool | None = None

    @model_validator(mode="after")
    def _check_path(self) -> PropagatedError:
        if self.path[-1] != self.span_id:
            raise ValueError("path must end at this span")
        if self.path[0] not in self.caused_by:
            raise ValueError("path must start at one of the caused_by root spans")
        return self


class _PatchBase(StrictModel):
    target_span_id: NonEmptyStr
    source: PatchSource


class ReplaceOutputPatch(_PatchBase):
    """Replace the target span's output outright."""

    kind: Literal["replace_output"] = "replace_output"
    output: dict[str, Any]


class OverrideInputsPatch(_PatchBase):
    """Re-run the target span with some inputs overridden (e.g. a corrected query)."""

    kind: Literal["override_inputs"] = "override_inputs"
    input_overrides: dict[str, Any] = Field(min_length=1)


class OverrideParamsPatch(_PatchBase):
    """Re-run the target span with different params (e.g. enable payload validation)."""

    kind: Literal["override_params"] = "override_params"
    param_overrides: dict[str, Any] = Field(min_length=1)


Patch = Annotated[
    ReplaceOutputPatch | OverrideInputsPatch | OverrideParamsPatch,
    Field(discriminator="kind"),
]


class Fix(StrictModel):
    fix_id: NonEmptyStr
    target_span_id: NonEmptyStr
    fault_type: FaultType
    description: str
    patch: Patch

    @model_validator(mode="after")
    def _patch_targets_same_span(self) -> Fix:
        if self.patch.target_span_id != self.target_span_id:
            raise ValueError("patch.target_span_id must match target_span_id")
        return self


class Diagnosis(StrictModel):
    diagnosis_id: NonEmptyStr
    run_id: NonEmptyStr
    prompt_version: NonEmptyStr
    root_causes: list[RootCause] = Field(default_factory=list)  # ranked, best first
    propagated: list[PropagatedError] = Field(default_factory=list)
    recommended_fixes: list[Fix] = Field(default_factory=list)
    escalated_to_deep: bool = False
    escalation_reason: str | None = None
    # Ensemble agreement (Cohen's/Fleiss' kappa), so it can be negative.
    judge_agreement: float = Field(ge=-1.0, le=1.0)
    narrative: str = ""
    cost_usd: NonNegativeFloat = 0.0
    created_at: UtcDatetime = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def _check_consistency(self) -> Diagnosis:
        if self.escalated_to_deep != bool(self.escalation_reason):
            raise ValueError("escalation_reason is required if and only if escalated_to_deep")
        roots = {rc.span_id for rc in self.root_causes}
        for err in self.propagated:
            unknown = set(err.caused_by) - roots
            if unknown:
                raise ValueError(f"propagated error cites unknown root spans: {sorted(unknown)}")
        for fix in self.recommended_fixes:
            if fix.target_span_id not in roots:
                raise ValueError(f"fix {fix.fix_id} targets a span that is not a root cause")
        return self
