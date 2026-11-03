"""The one module that hands out ground truth (P4-T3).

    from agenttrace.scenarios.ground_truth import load_dataset, truth_for
    truth = truth_for("sc-042")     # root causes, affected steps, expected final outcome

Only the evaluation harness may use this (scenario generation, fault injection, scoring:
see `ALLOWED` in `agenttrace.scenarios.isolation`). Diagnosis, explainability, and the
judges must never reach it, directly or through another module; a test enforces that,
so the debugger cannot be graded on answers it was shown.

The schema classes themselves are defined in `agenttrace.schemas.ground_truth`, which
is held to the same rule.
"""

from __future__ import annotations

import json
from functools import cache
from pathlib import Path

from agenttrace.schemas.ground_truth import (
    ExpectedFinal,
    FaultCode,
    FaultInjection,
    GroundTruth,
    Scenario,
)

SCENARIOS_PATH = Path(__file__).resolve().parents[2] / "data" / "scenarios" / "scenarios.json"

__all__ = [
    "SCENARIOS_PATH",
    "ExpectedFinal",
    "FaultCode",
    "FaultInjection",
    "GroundTruth",
    "Scenario",
    "load_dataset",
    "scenario",
    "truth_for",
]


def load_dataset(path: Path = SCENARIOS_PATH) -> list[Scenario]:
    """Every scenario, validated, in dataset order."""
    data = json.loads(path.read_text(encoding="utf-8"))
    return [Scenario.model_validate(s) for s in data["scenarios"]]


@cache
def _by_id() -> dict[str, Scenario]:
    return {s.scenario_id: s for s in load_dataset()}


def scenario(scenario_id: str) -> Scenario:
    """A scenario with its faults and ground truth. Use `.to_input()` for what the
    workflow may see."""
    try:
        return _by_id()[scenario_id]
    except KeyError:
        raise KeyError(f"no scenario {scenario_id!r} in {SCENARIOS_PATH.name}") from None


def truth_for(scenario_id: str) -> GroundTruth:
    return scenario(scenario_id).ground_truth
