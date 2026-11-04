"""Replay a recorded run through its workflow (P5-T1).

    result, cache = replay_run(run, workflow, resources)   # strict: no live call allowed
    diffs = exact_diffs(run, result.to_run())               # [] for an unmodified replay

The replay gets the original run's workflow input and `model_params` (its seed), and a
ReplayCache filed from the run itself, so every call and agent is served from the record.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from agenttrace.replay.cache import ReplayCache
from agenttrace.schemas.run import Run
from agenttrace.workflow.runtime import INPUT_STEP, Workflow, WorkflowResult


def run_inputs(run: Run) -> dict[str, Any]:
    """The workflow input a run was given (recorded on its `__input__` span)."""
    for span in run.spans:
        if span.node_id == INPUT_STEP and span.parent_span_id is None:
            return dict(span.output)
    raise ValueError(f"run {run.run_id} has no {INPUT_STEP} span")


def run_model_params(run: Run) -> dict[str, Any]:
    """The model-call overrides a run used (runs stored before P5-T1 recorded only a seed)."""
    if run.model_params:
        return dict(run.model_params)
    return {} if run.seed is None else {"seed": run.seed}


def replay_run(
    run: Run,
    workflow: Workflow,
    resources: Mapping[str, Any],
    cache: ReplayCache | None = None,
    *,
    run_id: str | None = None,
    strict: bool = True,
) -> tuple[WorkflowResult, ReplayCache]:
    """Re-execute `run` with its recorded outputs. Without `cache`, one is filed from the
    run itself; `strict` makes any miss an error instead of a live call."""
    if cache is None:
        cache = ReplayCache.from_runs([run], strict=strict)
    result = workflow.run(
        run_inputs(run),
        run_id=run_id or f"{run.run_id}-replay",
        resources=resources,
        model_params=run_model_params(run) or None,
        replay=cache,
    )
    return result, cache


def _local(run: Run) -> dict[str, Any]:
    prefix = f"{run.run_id}/"
    return {s.span_id.removeprefix(prefix): s for s in run.spans}


def exact_diffs(original: Run, replay: Run) -> list[str]:
    """Spans whose output, input, model, params, status, or explanation differ *at all*
    (no field ignored), by local id; plus "final_output" if that differs. Stricter than
    `align_runs`, which ignores cache-only fields."""
    a, b = _local(original), _local(replay)
    diffs = sorted(set(a) ^ set(b))
    for local_id in sorted(set(a) & set(b)):
        x, y = a[local_id], b[local_id]
        same = (
            x.output == y.output
            and x.input == y.input
            and x.model == y.model
            and x.params == y.params
            and (x.error is None) == (y.error is None)
            and x.explanation == y.explanation
        )
        if not same:
            diffs.append(local_id)
    if original.final_output != replay.final_output:
        diffs.append("final_output")
    return diffs
