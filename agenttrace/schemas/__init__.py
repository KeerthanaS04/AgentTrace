"""Pydantic data contracts shared by all components.

Ground-truth schemas (`agenttrace.schemas.ground_truth`) are deliberately not
re-exported here, so importing this package never exposes them.
"""

from agenttrace.schemas.diagnosis import (
    AnomalyScore,
    Diagnosis,
    FaultType,
    Fix,
    OverrideInputsPatch,
    OverrideParamsPatch,
    Patch,
    PatchSource,
    PropagatedError,
    ReplaceOutputPatch,
    RootCause,
    ValidatorFinding,
    Verdict,
)
from agenttrace.schemas.replay import ReplayResult
from agenttrace.schemas.run import Run
from agenttrace.schemas.scenario import ScenarioInput, Ticket, TicketCategory
from agenttrace.schemas.score import ExplainabilityScore, Metric, ScoreScope
from agenttrace.schemas.span import Explanation, NodeContract, Ref, Span, SpanKind

__all__ = [
    "AnomalyScore",
    "Diagnosis",
    "Explanation",
    "ExplainabilityScore",
    "FaultType",
    "Fix",
    "Metric",
    "NodeContract",
    "OverrideInputsPatch",
    "OverrideParamsPatch",
    "Patch",
    "PatchSource",
    "PropagatedError",
    "Ref",
    "ReplaceOutputPatch",
    "ReplayResult",
    "RootCause",
    "Run",
    "ScenarioInput",
    "ScoreScope",
    "Span",
    "SpanKind",
    "Ticket",
    "TicketCategory",
    "ValidatorFinding",
    "Verdict",
]
