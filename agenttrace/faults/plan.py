"""FaultPlan: apply a scenario's fault injections during one run (P4-T1).

    plan = FaultPlan(scenario.faults)
    workflow.run(inputs, run_id=..., resources=..., faults=plan)
    plan.applied        # [AppliedFault(node_id, span_id, injector)] for verification
    plan.unapplied()    # injections whose target node never ran

The plan is handed to the runtime, which consults it whenever a traced node produces
its output ("at record time"): the mutated value is recorded and passed downstream as
if the node had produced it. Nothing about the injection goes into the trace, so
diagnosis has to find the fault from the evidence. Only the harness sees `applied`.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from agenttrace.faults.injectors import INJECTORS, OCCURRENCE
from agenttrace.schemas.ground_truth import FaultInjection
from agenttrace.schemas.span import SpanKind
from agenttrace.tracing.context import InjectionError


class FaultPlanError(InjectionError, ValueError):
    pass


@dataclass(frozen=True)
class AppliedFault:
    node_id: str
    span_id: str
    injector: str
    occurrence: int


class FaultPlan:
    def __init__(self, faults: Sequence[FaultInjection]) -> None:
        for f in faults:
            if f.injector not in INJECTORS:
                raise FaultPlanError(
                    f"unknown injector {f.injector!r}; known: {', '.join(sorted(INJECTORS))}"
                )
            occurrence = f.params.get(OCCURRENCE)
            if occurrence is not None and (not isinstance(occurrence, int) or occurrence < 1):
                raise FaultPlanError(f"{f.injector}: occurrence must be an integer >= 1")
        self.faults = list(faults)
        self.applied: list[AppliedFault] = []
        self._seen: dict[str, int] = {}

    def apply(
        self, node_id: str, kind: SpanKind, span_id: str, value: dict[str, Any]
    ) -> dict[str, Any]:
        """Return `value` with every injection aimed at this node applied (in order)."""
        n = self._seen[node_id] = self._seen.get(node_id, 0) + 1
        for fault in self.faults:
            if fault.target_node_id != node_id:
                continue
            want = fault.params.get(OCCURRENCE)
            if want is not None and want != n:
                continue
            injector = INJECTORS[fault.injector]
            if kind not in injector.kinds:
                raise FaultPlanError(
                    f"{fault.injector} targets {'/'.join(injector.kinds)} spans, but "
                    f"{node_id!r} is a {kind} span"
                )
            params = {k: v for k, v in fault.params.items() if k != OCCURRENCE}
            value = injector(value, params)
            self.applied.append(AppliedFault(node_id, span_id, fault.injector, n))
        return value

    def unapplied(self) -> list[FaultInjection]:
        hit = {(a.node_id, a.injector) for a in self.applied}
        return [f for f in self.faults if (f.target_node_id, f.injector) not in hit]
