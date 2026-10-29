"""Hidden scenario data: injected faults and ground-truth labels.

Only the fault injector and the evaluation harness may import this module.
It is intentionally not re-exported from `agenttrace.schemas`.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field, model_validator

from agenttrace.schemas.base import NonEmptyStr, StrictModel
from agenttrace.schemas.scenario import ScenarioInput

# F0 control, F1 wrong retrieval, F2 empty retrieval, F3 wrong tool value,
# F4 malformed tool response, F5 invalid decision, F6 hallucinated claim,
# F7 multi-fault, F8 silent fault.
FaultCode = Literal["F0", "F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8"]
Decision = Literal["approve", "deny", "escalate"]


class FaultInjection(StrictModel):
    target_node_id: NonEmptyStr
    injector: NonEmptyStr
    params: dict[str, Any] = Field(default_factory=dict)


class ExpectedFinal(StrictModel):
    decision: Decision
    # Facts the customer reply must contain to count as correct.
    required_facts: list[str] = Field(default_factory=list)


class GroundTruth(StrictModel):
    fault_code: FaultCode
    root_cause_node_ids: list[NonEmptyStr] = Field(default_factory=list)
    affected_node_ids: list[NonEmptyStr] = Field(default_factory=list)
    expected_final: ExpectedFinal

    @model_validator(mode="after")
    def _check_roots(self) -> GroundTruth:
        n_roots = len(self.root_cause_node_ids)
        if self.fault_code == "F0" and (n_roots or self.affected_node_ids):
            raise ValueError("F0 controls have no root causes or affected nodes")
        if self.fault_code == "F7" and n_roots < 2:
            raise ValueError("F7 multi-fault scenarios need at least two root causes")
        if self.fault_code not in ("F0", "F7") and n_roots != 1:
            raise ValueError(f"{self.fault_code} scenarios have exactly one root cause")
        if set(self.root_cause_node_ids) & set(self.affected_node_ids):
            raise ValueError("a node cannot be both a root cause and an affected node")
        return self


class Scenario(ScenarioInput):
    faults: list[FaultInjection] = Field(default_factory=list)
    ground_truth: GroundTruth

    @model_validator(mode="after")
    def _faults_match_roots(self) -> Scenario:
        targets = {f.target_node_id for f in self.faults}
        if targets != set(self.ground_truth.root_cause_node_ids):
            raise ValueError("fault targets must equal ground_truth.root_cause_node_ids")
        return self

    def to_input(self) -> ScenarioInput:
        """Strip faults and ground truth: the view the workflow and diagnosis get."""
        return ScenarioInput.model_validate(
            self.model_dump(include=set(ScenarioInput.model_fields))
        )
