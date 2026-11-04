"""Re-executor (P5-T3): run a replay plan and store the replay linked to the original.

    plan = plan_patch(run, workflow, patch)
    out = execute_plan(run, workflow, plan, make_resources(gateway))
    out.replay          # the replayed Run: replay_of = run.run_id, target span patched=True
    out.violations()    # [] when only the target and its descendants changed

The workflow runs again in topological order with a `PatchingReplay` hook:

    frozen spans   served from the original run's record (a replay cache filed from it);
                   one with no recorded output is a planning error, raised, never re-run
    the target     its output replaced by the patch, `patched=True`, nothing executed
                   (an agent target's own calls still replay from the record)
    re-run spans   executed again: served from the record if their inputs did not change
                   (cache-aware), else run live through the gateway, whose own cache
                   still applies

`faults` (a `FaultPlan`) keeps other injected faults in force during the replay, so a
counterfactual "fix this root" does not also silently fix a second root downstream (F7).
The patched target's own fault never applies: the patch replaces its output.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from agenttrace.graph import Alignment, align_runs
from agenttrace.replay.cache import ReplayCache
from agenttrace.replay.patcher import ReplayPlan
from agenttrace.replay.replay import run_inputs, run_model_params
from agenttrace.schemas.run import Run
from agenttrace.schemas.span import Explanation, SpanKind
from agenttrace.tracing.context import FaultHook, ReplayMissError, SpanBuilder, TraceContext
from agenttrace.workflow.runtime import Workflow, WorkflowResult


def _local(span_id: str) -> str:
    return span_id.split("/", 1)[1]


class PatchingReplay:
    """The replay hook for one plan: frozen from the record, target patched, rest re-run."""

    def __init__(self, plan: ReplayPlan, cache: ReplayCache) -> None:
        self.plan = plan
        self.cache = cache
        self._frozen = {plan.local(s) for s in plan.frozen}
        self.patched_spans: list[str] = []

    def _patch(self, span: SpanBuilder) -> None:
        span.output = copy.deepcopy(self.plan.output)
        span.patched = True
        span.cost_usd = 0.0
        self.patched_spans.append(span.span_id)

    def _frozen_miss(self, span: SpanBuilder) -> None:
        raise ReplayMissError(
            f"frozen span {_local(span.span_id)} has no recorded output (the plan does not "
            f"match run {self.plan.run_id})"
        )

    def lookup_call(
        self,
        node_id: str,
        kind: SpanKind,
        model: str | None,
        params: Mapping[str, Any],
        span: SpanBuilder,
    ) -> bool:
        local = _local(span.span_id)
        if local == self.plan.target_local:
            span.model, span.params = model, dict(params)
            self._patch(span)
            return True
        if self.cache.lookup_call(node_id, kind, model, params, span):
            return True
        if local in self._frozen:
            self._frozen_miss(span)
        return False

    def record_call(
        self,
        node_id: str,
        kind: SpanKind,
        model: str | None,
        params: Mapping[str, Any],
        span: SpanBuilder,
    ) -> None:
        self.cache.record_call(node_id, kind, model, params, span)

    def resolve_agent(
        self,
        trace: TraceContext,
        span: SpanBuilder,
        output: dict[str, Any],
        explanation: Explanation | None,
    ) -> tuple[dict[str, Any], Explanation | None]:
        local = _local(span.span_id)
        if local == self.plan.target_local:
            # The agent's own explanation described the output it computed, not this one.
            self._patch(span)
            return copy.deepcopy(self.plan.output), None
        misses = self.cache.misses["agent"]
        resolved = self.cache.resolve_agent(trace, span, output, explanation)
        if local in self._frozen and self.cache.misses["agent"] > misses:
            self._frozen_miss(span)
        return resolved


@dataclass
class ExecutedReplay:
    plan: ReplayPlan
    original: Run
    replay: Run
    result: WorkflowResult
    stats: dict[str, dict[str, int]] = field(default_factory=dict)

    @property
    def alignment(self) -> Alignment:
        return align_runs(self.original, self.replay)

    @property
    def live_model_calls(self) -> int:
        """Model calls the replay could not serve from the record (sent to the gateway)."""
        return self.stats.get("llm", {}).get("misses", 0)

    def violations(self) -> list[str]:
        """Ways the replay changed something the plan froze. Empty when only the target
        and its descendants changed: frozen spans unchanged, the target patched, and any
        span new to the replay or missing from it inside a re-run step."""
        plan = self.plan
        frozen = {plan.local(s) for s in plan.frozen}
        rerun_steps = set(plan.rerun_steps)
        problems: list[str] = []
        for pair in self.alignment.pairs:
            step = pair.local_id.split(".", 1)[0].split("#", 1)[0]
            if pair.local_id == plan.target_local:
                if pair.status != "changed" or "patched" not in pair.changes:
                    problems.append(f"target {pair.local_id} is not marked patched")
            elif pair.status == "changed" and pair.local_id in frozen:
                problems.append(f"frozen {pair.local_id} changed ({', '.join(pair.changes)})")
            elif pair.status in ("only_a", "only_b") and step not in rerun_steps:
                where = "missing from" if pair.status == "only_a" else "new in"
                problems.append(f"{pair.local_id} {where} the replay, outside re-run steps")
        return problems


def execute_plan(
    run: Run,
    workflow: Workflow,
    plan: ReplayPlan,
    resources: Mapping[str, Any],
    *,
    run_id: str | None = None,
    cache: ReplayCache | None = None,
    faults: FaultHook | None = None,
) -> ExecutedReplay:
    """Replay `run` with `plan`'s patch. Returns the replayed Run (`replay_of` set)."""
    if plan.run_id != run.run_id:
        raise ValueError(f"the plan is for run {plan.run_id}, not {run.run_id}")
    cache = cache if cache is not None else ReplayCache.from_runs([run])
    hook = PatchingReplay(plan, cache)
    # The workflow input is recorded, not produced by a node: a patch to it is the input.
    is_input = plan.target_kind == "message"
    result = workflow.run(
        dict(plan.output) if is_input else run_inputs(run),
        run_id=run_id or f"{run.run_id}-patch",
        resources=resources,
        faults=faults,
        model_params=run_model_params(run) or None,
        replay=hook,
    )
    replay = result.to_run(scenario_id=run.scenario_id).model_copy(
        update={"replay_of": run.run_id, "patch_source": plan.patch_source}
    )
    if is_input:
        spans = [
            s.model_copy(update={"patched": True}) if _local(s.span_id) == plan.target_local else s
            for s in replay.spans
        ]
        replay = replay.model_copy(update={"spans": spans})
    return ExecutedReplay(
        plan=plan, original=run, replay=replay, result=result, stats=cache.stats()
    )
