"""Lightweight agent runtime: agents, a declared step graph, and its executor.

Every agent is `run(inputs, ctx) -> AgentOutput` (an output dict plus an `Explanation`).
A workflow declares, for each step, exactly where each input comes from (a workflow
input or a named output of another step). The runtime executes steps in dependency
order and records, per step, the resolved inputs together with the `Ref`s they came
from, so data flow is always declared, never inferred from execution order.

Values crossing step boundaries must be JSON-serializable; they are copied on the way
in and out, so an agent can never mutate another step's recorded output.

Every run is traced (P2-T1): the workflow input is a "message" span, each step an
"agent" span whose span_id is its step id, and the model, tool, and search calls made
inside a step are nested spans (see `agenttrace.tracing`). `WorkflowResult.to_run()`
packages them as a `Run`.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, TypeVar

from pydantic import Field

from agenttrace.gateway.cache import content_hash
from agenttrace.schemas.base import NonEmptyStr, StrictModel, UtcDatetime, utc_now
from agenttrace.schemas.run import Run
from agenttrace.schemas.span import Explanation, Ref, Span
from agenttrace.tracing import TraceContext

T = TypeVar("T")

# Pseudo-step that owns the workflow's own inputs (the ticket), so they get Refs too.
INPUT_STEP = "__input__"

StepStatus = Literal["ok", "error", "skipped"]


# ---------------------------------------------------------------- data-flow sources


@dataclass(frozen=True)
class FromInput:
    """Bind a step input to a workflow input."""

    key: str


@dataclass(frozen=True)
class FromStep:
    """Bind a step input to a named output of another step."""

    step: str
    key: str


Source = FromInput | FromStep


# ---------------------------------------------------------------- agents


class AgentOutput(StrictModel):
    output: dict[str, Any]
    explanation: Explanation


@dataclass(frozen=True)
class StepContext:
    """What an agent may use besides its inputs: run identity, params, shared resources."""

    run_id: str
    step: str
    params: Mapping[str, Any] = field(default_factory=dict)
    resources: Mapping[str, Any] = field(default_factory=dict)

    def resource(self, name: str, kind: type[T]) -> T:
        if name not in self.resources:
            raise KeyError(f"step {self.step!r} needs resource {name!r}, which was not provided")
        value = self.resources[name]
        if not isinstance(value, kind):
            raise TypeError(f"resource {name!r} is {type(value).__name__}, not {kind.__name__}")
        return value


class Agent(Protocol):
    name: str
    input_keys: tuple[str, ...]
    output_keys: tuple[str, ...]

    def run(self, inputs: Mapping[str, Any], ctx: StepContext) -> AgentOutput: ...


@dataclass(frozen=True)
class Step:
    name: str
    agent: Agent
    inputs: Mapping[str, Source]
    params: Mapping[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------- results


class StepRecord(StrictModel):
    """One executed (or skipped) step; the same data as its agent span, minus nesting."""

    step_id: NonEmptyStr
    step: NonEmptyStr
    agent: NonEmptyStr
    status: StepStatus
    inputs: dict[str, Any] = Field(default_factory=dict)
    input_refs: list[Ref] = Field(default_factory=list)
    output: dict[str, Any] = Field(default_factory=dict)
    explanation: Explanation | None = None
    params: dict[str, Any] = Field(default_factory=dict)
    started_at: UtcDatetime | None = None
    ended_at: UtcDatetime | None = None
    error: str | None = None


class WorkflowResult(StrictModel):
    run_id: NonEmptyStr
    workflow: NonEmptyStr
    workflow_version: NonEmptyStr = "1"
    status: Literal["ok", "failed"]
    outputs: dict[str, Any] = Field(default_factory=dict)
    steps: list[StepRecord] = Field(default_factory=list)
    # Every span of the run, parents before children (see module docstring).
    spans: list[Span] = Field(default_factory=list)

    def step(self, name: str) -> StepRecord:
        return next(s for s in self.steps if s.step == name)

    def to_run(self, *, scenario_id: str | None = None, final_correct: bool | None = None) -> Run:
        """The run in the trace schema: all spans plus which model each agent used."""
        step_ids = {r.step_id for r in self.steps}
        return Run(
            run_id=self.run_id,
            scenario_id=scenario_id,
            workflow_version=f"{self.workflow}@{self.workflow_version}",
            agent_models={
                s.node_id: s.model for s in self.spans if s.span_id in step_ids and s.model
            },
            final_output=self.outputs,
            final_correct=final_correct,
            spans=self.spans,
        )


def step_id(run_id: str, step: str) -> str:
    """Deterministic id for a step execution, so replays line up with originals."""
    return f"{run_id}/{step}"


def _mint(
    minted: dict[tuple[str, str], Ref], run_id: str, owner: str, values: Mapping[str, Any]
) -> None:
    """Create the Ref for each value at the moment its producer emits it."""
    for key, value in values.items():
        minted[(owner, key)] = Ref(
            span_id=step_id(run_id, owner), key=key, content_hash=content_hash(value)
        )


def _copy_json(value: Any, what: str) -> Any:
    try:
        return json.loads(json.dumps(value))
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{what} is not JSON-serializable: {exc}") from exc


# ---------------------------------------------------------------- workflow


class Workflow:
    """A declared, validated graph of steps. Validation happens at construction."""

    def __init__(
        self,
        name: str,
        inputs: Sequence[str],
        steps: Sequence[Step],
        outputs: Mapping[str, Source],
        *,
        version: str = "1",
    ) -> None:
        self.name = name
        self.version = version
        self.inputs = tuple(inputs)
        self.steps = {s.name: s for s in steps}
        self.outputs = dict(outputs)
        if len(self.steps) != len(steps):
            raise ValueError("duplicate step names")
        if INPUT_STEP in self.steps:
            raise ValueError(f"{INPUT_STEP!r} is reserved")
        for step in steps:
            expected, bound = set(step.agent.input_keys), set(step.inputs)
            if expected != bound:
                raise ValueError(
                    f"step {step.name!r}: inputs {sorted(bound)} do not match agent "
                    f"{step.agent.name!r} inputs {sorted(expected)}"
                )
            for key, src in step.inputs.items():
                self._check_source(src, f"step {step.name!r} input {key!r}")
        for key, src in self.outputs.items():
            self._check_source(src, f"workflow output {key!r}")
        self.order = self._topological_order([s.name for s in steps])

    def _check_source(self, src: Source, where: str) -> None:
        if isinstance(src, FromInput):
            if src.key not in self.inputs:
                raise ValueError(f"{where}: unknown workflow input {src.key!r}")
            return
        producer = self.steps.get(src.step)
        if producer is None:
            raise ValueError(f"{where}: unknown step {src.step!r}")
        if src.key not in producer.agent.output_keys:
            raise ValueError(f"{where}: step {src.step!r} has no output {src.key!r}")

    def dependencies(self, step: str) -> list[str]:
        """Steps whose outputs this step consumes, in declaration order of its inputs."""
        deps: list[str] = []
        for src in self.steps[step].inputs.values():
            if isinstance(src, FromStep) and src.step not in deps:
                deps.append(src.step)
        return deps

    def _topological_order(self, declared: list[str]) -> list[str]:
        # Kahn's algorithm; ties broken by declaration order, so the order is stable.
        remaining = {s: set(self.dependencies(s)) for s in declared}
        order: list[str] = []
        while remaining:
            ready = [s for s in declared if s in remaining and not remaining[s]]
            if not ready:
                raise ValueError(f"cycle among steps {sorted(remaining)}")
            nxt = ready[0]
            order.append(nxt)
            del remaining[nxt]
            for deps in remaining.values():
                deps.discard(nxt)
        return order

    def edges(self) -> list[tuple[str, str]]:
        """Declared data-flow edges (producer, consumer), including from the input."""
        out: list[tuple[str, str]] = []
        for name in self.order:
            for src in self.steps[name].inputs.values():
                edge = (INPUT_STEP if isinstance(src, FromInput) else src.step, name)
                if edge not in out:
                    out.append(edge)
        return out

    # ------------------------------------------------------------ execution

    def run(
        self,
        inputs: Mapping[str, Any],
        *,
        run_id: str,
        resources: Mapping[str, Any] | None = None,
    ) -> WorkflowResult:
        missing, extra = set(self.inputs) - set(inputs), set(inputs) - set(self.inputs)
        if missing or extra:
            raise ValueError(
                f"workflow inputs: missing {sorted(missing)}, unexpected {sorted(extra)}"
            )

        values: dict[str, dict[str, Any]] = {INPUT_STEP: _copy_json(dict(inputs), "workflow input")}
        records: dict[str, StepRecord] = {}
        # (producer, key) -> the Ref minted when the value was produced. Consumers get
        # their refs from here via their declared sources, never by recomputing them.
        minted: dict[tuple[str, str], Ref] = {}
        _mint(minted, run_id, INPUT_STEP, values[INPUT_STEP])
        with TraceContext(run_id) as trace:
            trace.instant("message", INPUT_STEP, agent=self.name, output=values[INPUT_STEP])
            for name in self.order:
                step = self.steps[name]
                records[name] = self._run_step(
                    step, run_id, values, minted, records, resources, trace
                )

        outputs: dict[str, Any] = {}
        for key, src in self.outputs.items():
            owner = INPUT_STEP if isinstance(src, FromInput) else src.step
            if owner in values:
                outputs[key] = _copy_json(values[owner][src.key], "workflow output")
        ok = all(r.status == "ok" for r in records.values()) and len(outputs) == len(self.outputs)
        return WorkflowResult(
            run_id=run_id,
            workflow=self.name,
            workflow_version=self.version,
            status="ok" if ok else "failed",
            outputs=outputs,
            steps=[records[n] for n in self.order],
            spans=trace.spans,
        )

    def _run_step(
        self,
        step: Step,
        run_id: str,
        values: dict[str, dict[str, Any]],
        minted: dict[tuple[str, str], Ref],
        records: Mapping[str, StepRecord],
        resources: Mapping[str, Any] | None,
        trace: TraceContext,
    ) -> StepRecord:
        base: dict[str, Any] = {
            "step_id": step_id(run_id, step.name),
            "step": step.name,
            "agent": step.agent.name,
            "params": dict(step.params),
        }
        failed = [d for d in self.dependencies(step.name) if records[d].status != "ok"]
        if failed:
            error = f"upstream not ok: {failed}"
            trace.instant(
                "agent",
                step.name,
                agent=step.agent.name,
                params=step.params,
                error=f"skipped: {error}",
            )
            return StepRecord(**base, status="skipped", error=error)

        resolved: dict[str, Any] = {}
        refs: list[Ref] = []
        for key, src in step.inputs.items():
            owner = INPUT_STEP if isinstance(src, FromInput) else src.step
            resolved[key] = _copy_json(values[owner][src.key], f"input {key!r}")
            refs.append(minted[(owner, src.key)].model_copy(update={"input_key": key}))

        ctx = StepContext(
            run_id=run_id, step=step.name, params=step.params, resources=resources or {}
        )
        started = utc_now()
        with trace.span(
            "agent",
            step.name,
            agent=step.agent.name,
            input=resolved,
            input_refs=refs,
            params=step.params,
            prompt_version=getattr(step.agent, "prompt_version", None),
        ) as span:
            try:
                result = step.agent.run(_copy_json(resolved, "inputs"), ctx)
                if not isinstance(result, AgentOutput):
                    raise TypeError(f"agent returned {type(result).__name__}, not AgentOutput")
                got, want = set(result.output), set(step.agent.output_keys)
                if got != want:
                    raise ValueError(f"output keys {sorted(got)} != declared {sorted(want)}")
                output = _copy_json(result.output, "output")
            except Exception as exc:
                span.error = f"{type(exc).__name__}: {exc}"
                return StepRecord(
                    **base,
                    status="error",
                    inputs=resolved,
                    input_refs=refs,
                    started_at=started,
                    ended_at=utc_now(),
                    error=span.error,
                )
            span.output = output
            span.explanation = result.explanation
        values[step.name] = output
        _mint(minted, run_id, step.name, output)
        return StepRecord(
            **base,
            status="ok",
            inputs=resolved,
            input_refs=refs,
            output=output,
            explanation=result.explanation,
            started_at=started,
            ended_at=utc_now(),
        )
