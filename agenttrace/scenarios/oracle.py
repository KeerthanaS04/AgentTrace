"""Oracle patch source (P5-T4): the value a node *should* have produced.

    patch = oracle_patch(gateway, scenario, run, "tool_agent.order_lookup", roles)

The oracle is what the node produced in the scenario's clean run: same ticket, same
models, same `model_params` (seed), no faults. That run is recomputed through the
gateway, so with a warm cache it costs nothing. It is the upper bound for the other
sources: no model or rule can be more right than the run without the fault.

Harness only: it reads the scenario (the faults are what it leaves out).
"""

from __future__ import annotations

from agenttrace.gateway.gateway import Gateway
from agenttrace.replay.patcher import Patch, resolve_target
from agenttrace.replay.replay import run_model_params
from agenttrace.scenarios.ground_truth import Scenario
from agenttrace.scenarios.verify import _run, run_tag
from agenttrace.schemas.run import Run


def clean_run(gateway: Gateway, scenario: Scenario, run: Run, roles: dict[str, str]) -> Run:
    """The scenario's ticket run without faults, with the same model params as `run`."""
    params = run_model_params(run) or None
    run_id = f"{scenario.ticket.ticket_id}-oracle-clean{run_tag(params)}"
    clean, _ = _run(gateway, scenario, roles, run_id, None, params)
    return clean


def oracle_patch(
    gateway: Gateway, scenario: Scenario, run: Run, target: str, roles: dict[str, str]
) -> Patch | None:
    """The clean run's output for `target` (same local id), or None if it has none."""
    span = resolve_target(run, target)
    local = span.span_id.removeprefix(f"{run.run_id}/")
    clean = clean_run(gateway, scenario, run, roles)
    match = next((s for s in clean.spans if s.span_id == f"{clean.run_id}/{local}"), None)
    if match is None or match.error is not None:
        return None
    return Patch(
        target=span.span_id,
        output=match.output,
        patch_source="oracle",
        note="output of the clean run (same seed)",
    )
