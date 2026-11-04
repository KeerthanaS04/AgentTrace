"""Patcher (P5-T2): a target span + a replacement output -> a replay plan.

    plan = plan_patch(run, workflow, Patch(target="tool_agent.order_lookup", output={...}))
    plan.patched   the target: its output is replaced (the replay marks it patched=True)
    plan.frozen    spans whose recorded output the replay reuses
    plan.rerun     spans the replay re-executes

Which steps re-run is decided by the workflow's *declared* wiring, not by the run's
recorded refs: a step that was skipped because the target's step failed has no recorded
inputs, yet must run once the patch fixes that step. Reached steps are:

    target is a step (agent)       every step downstream of it
    target is a call in a step     that step, plus every step downstream of it
    target is the workflow input   every step that consumes it, and downstream

Within a reached step every span re-runs, except in the target's own step, where calls
opened *before* the target are frozen: they ran before its output existed. Spans nested
inside the target (an agent's own model calls) are frozen too: the patch replaces what
they led to, not what they were asked. Everything else is frozen.

Step-level, not field-level: a downstream step re-runs even if it reads only a field the
patch did not change; the replay cache (P5-T1) then serves it unchanged at no cost.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal

from pydantic import Field

from agenttrace.schemas.base import NonEmptyStr, StrictModel
from agenttrace.schemas.run import Run
from agenttrace.schemas.span import Span, SpanKind
from agenttrace.workflow.runtime import INPUT_STEP, FromInput, Workflow

Action = Literal["frozen", "patched", "rerun"]


class PatchError(ValueError):
    """A patch that cannot apply to this run (unknown target, wrong output keys, ...)."""


class Patch(StrictModel):
    """Replace one span's output in a replay."""

    target: NonEmptyStr  # span id, or local id within the run (e.g. "policy", "router.llm#2")
    output: dict[str, Any]
    # Where the replacement came from: "oracle", "model", or "rule" (P5-T4).
    patch_source: str | None = None
    note: str = ""


class ReplayPlan(StrictModel):
    run_id: NonEmptyStr
    target: NonEmptyStr  # span id
    target_local: NonEmptyStr
    target_node: NonEmptyStr
    target_kind: SpanKind
    target_step: NonEmptyStr  # the step it belongs to ("__input__" for the input)
    output: dict[str, Any]
    patch_source: str | None = None
    # Output keys whose value the patch changes (empty: the patch is a no-op).
    changed_keys: list[str] = Field(default_factory=list)
    frozen: list[str] = Field(default_factory=list)  # span ids, recorded order
    rerun: list[str] = Field(default_factory=list)
    frozen_steps: list[str] = Field(default_factory=list)  # workflow order
    rerun_steps: list[str] = Field(default_factory=list)

    @property
    def patched(self) -> list[str]:
        return [self.target]

    def action(self, span_id: str) -> Action:
        if span_id == self.target:
            return "patched"
        if span_id in self._rerun:
            return "rerun"
        if span_id in self._frozen:
            return "frozen"
        raise KeyError(f"span {span_id!r} is not in run {self.run_id}")

    @property
    def _rerun(self) -> set[str]:
        return set(self.rerun)

    @property
    def _frozen(self) -> set[str]:
        return set(self.frozen)

    def local(self, span_id: str) -> str:
        return span_id.removeprefix(f"{self.run_id}/")


# ---------------------------------------------------------------- helpers


def resolve_target(run: Run, target: str) -> Span:
    """The span a patch names, by span id or local id."""
    by_id = {s.span_id: s for s in run.spans}
    span = by_id.get(target) or by_id.get(f"{run.run_id}/{target}")
    if span is None:
        known = ", ".join(s.span_id.removeprefix(f"{run.run_id}/") for s in run.spans)
        raise PatchError(f"run {run.run_id} has no span {target!r}; spans: {known}")
    return span


def step_of(run: Run, span: Span) -> Span:
    """The top-level span (step, or the input message) a span is nested in."""
    by_id = {s.span_id: s for s in run.spans}
    while span.parent_span_id is not None:
        span = by_id[span.parent_span_id]
    return span


def downstream_steps(workflow: Workflow, step: str) -> list[str]:
    """Steps that consume `step`'s outputs, directly or not, by declared wiring
    (`step` may be INPUT_STEP), in workflow order."""
    reached = {step}
    for name in workflow.order:
        for src in workflow.steps[name].inputs.values():
            producer = INPUT_STEP if isinstance(src, FromInput) else src.step
            if producer in reached:
                reached.add(name)
                break
    return [n for n in workflow.order if n in reached and n != step]


def _expected_keys(workflow: Workflow, step: Span) -> set[str] | None:
    """Output keys a replacement for this top-level span must have (None: any dict)."""
    if step.node_id == INPUT_STEP:
        return set(workflow.inputs)
    if step.node_id in workflow.steps:
        return set(workflow.steps[step.node_id].agent.output_keys)
    return None


def patch_fields(
    run: Run, target: str, fields: Mapping[str, Any], *, patch_source: str | None = None
) -> Patch:
    """A patch that keeps the target's recorded output and changes only `fields`."""
    span = resolve_target(run, target)
    return Patch(target=span.span_id, output={**span.output, **fields}, patch_source=patch_source)


# ---------------------------------------------------------------- planning


def plan_patch(run: Run, workflow: Workflow, patch: Patch) -> ReplayPlan:
    """Freeze everything the target cannot affect; re-run what it can (see module doc)."""
    expected_version = f"{workflow.name}@{workflow.version}"
    if run.workflow_version != expected_version:
        raise PatchError(
            f"run {run.run_id} is {run.workflow_version}, the workflow is {expected_version}"
        )
    target = resolve_target(run, patch.target)
    top = step_of(run, target)
    nested = target.parent_span_id is not None

    if not nested:
        want = _expected_keys(workflow, top)
        if want is not None and set(patch.output) != want:
            raise PatchError(
                f"a replacement for {top.node_id!r} needs exactly the keys {sorted(want)}, "
                f"got {sorted(patch.output)}"
            )
    elif not patch.output:
        raise PatchError(f"the replacement for {target.node_id!r} is empty")

    reached = set(downstream_steps(workflow, top.node_id))
    if nested:
        reached.add(top.node_id)  # the step around a patched call must recompute

    by_id = {s.span_id: s for s in run.spans}
    order = {s.span_id: i for i, s in enumerate(run.spans)}

    def ancestors(span: Span) -> set[str]:
        out: set[str] = set()
        while span.parent_span_id is not None:
            out.add(span.parent_span_id)
            span = by_id[span.parent_span_id]
        return out

    target_ancestors = ancestors(target)
    frozen: list[str] = []
    rerun: list[str] = []
    for span in run.spans:
        if span.span_id == target.span_id:
            continue
        above = ancestors(span)
        if target.span_id in above:
            frozen.append(span.span_id)  # inside the target: replaced, not re-asked
        elif step_of(run, span).node_id not in reached:
            frozen.append(span.span_id)
        elif (
            nested
            and step_of(run, span).span_id == top.span_id
            and span.span_id not in target_ancestors
            and order[span.span_id] < order[target.span_id]
        ):
            frozen.append(span.span_id)  # an earlier call in the target's own step
        else:
            rerun.append(span.span_id)

    steps = [n for n in workflow.order]
    changed = sorted(
        k
        for k in set(patch.output) | set(target.output)
        if patch.output.get(k, _MISSING) != target.output.get(k, _MISSING)
    )
    return ReplayPlan(
        run_id=run.run_id,
        target=target.span_id,
        target_local=target.span_id.removeprefix(f"{run.run_id}/"),
        target_node=target.node_id,
        target_kind=target.kind,
        target_step=top.node_id,
        output=dict(patch.output),
        patch_source=patch.patch_source,
        changed_keys=changed,
        frozen=frozen,
        rerun=rerun,
        frozen_steps=[n for n in steps if n not in reached and n != top.node_id],
        rerun_steps=[n for n in steps if n in reached],
    )


_MISSING = object()
