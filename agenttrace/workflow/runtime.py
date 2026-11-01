"""Lightweight agent runtime: agents, a declared step graph, and its executor.

Every agent is `run(inputs, ctx) -> AgentOutput` (an output dict plus an `Explanation`).
A workflow declares, for each step, exactly where each input comes from (a workflow
input or a named output of another step). The runtime executes steps in dependency
order and records, per step, the resolved inputs together with the `Ref`s they came
from, so data flow is always declared, never inferred from execution order.

Values crossing step boundaries must be JSON-serializable; they are copied on the way
in and out, so an agent can never mutate another step's recorded output.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, TypeVar

from pydantic import Field

from agenttrace.gateway.cache import content_hash
from agenttrace.schemas.base import NonEmptyStr, StrictModel, UtcDatetime, utc_now
from agenttrace.schemas.span import Explanation, Ref

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
    """One executed (or skipped) step. Becomes an agent span once tracing lands (P2)."""

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
    status: Literal["ok", "failed"]
    outputs: dict[str, Any] = Field(default_factory=dict)
    steps: list[StepRecord] = Field(default_factory=list)

    def step(self, name: str) -> StepRecord:
        return next(s for s in self.steps if s.step == name)


def step_id(run_id: str, step: str) -> str:
    """Deterministic id for a step execution, so replays line up with originals."""
    return f"{run_id}/{step}"


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
    ) -> None:
        self.name = name
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
        for name in self.order:
            records[name] = self._run_step(self.steps[name], run_id, values, records, resources)

        outputs: dict[str, Any] = {}
        for key, src in self.outputs.items():
            owner = INPUT_STEP if isinstance(src, FromInput) else src.step
            if owner in values:
                outputs[key] = _copy_json(values[owner][src.key], "workflow output")
        ok = all(r.status == "ok" for r in records.values()) and len(outputs) == len(self.outputs)
        return WorkflowResult(
            run_id=run_id,
            workflow=self.name,
            status="ok" if ok else "failed",
            outputs=outputs,
            steps=[records[n] for n in self.order],
        )

    def _run_step(
        self,
        step: Step,
        run_id: str,
        values: dict[str, dict[str, Any]],
        records: Mapping[str, StepRecord],
        resources: Mapping[str, Any] | None,
    ) -> StepRecord:
        base: dict[str, Any] = {
            "step_id": step_id(run_id, step.name),
            "step": step.name,
            "agent": step.agent.name,
            "params": dict(step.params),
        }
        failed = [d for d in self.dependencies(step.name) if records[d].status != "ok"]
        if failed:
            return StepRecord(**base, status="skipped", error=f"upstream not ok: {failed}")

        resolved: dict[str, Any] = {}
        refs: list[Ref] = []
        for key, src in step.inputs.items():
            owner = INPUT_STEP if isinstance(src, FromInput) else src.step
            value = _copy_json(values[owner][src.key], f"input {key!r}")
            resolved[key] = value
            refs.append(
                Ref(span_id=step_id(run_id, owner), key=src.key, content_hash=content_hash(value))
            )

        ctx = StepContext(
            run_id=run_id, step=step.name, params=step.params, resources=resources or {}
        )
        started = utc_now()
        try:
            result = step.agent.run(_copy_json(resolved, "inputs"), ctx)
            if not isinstance(result, AgentOutput):
                raise TypeError(f"agent returned {type(result).__name__}, not AgentOutput")
            got, want = set(result.output), set(step.agent.output_keys)
            if got != want:
                raise ValueError(f"output keys {sorted(got)} != declared {sorted(want)}")
            output = _copy_json(result.output, "output")
        except Exception as exc:
            return StepRecord(
                **base,
                status="error",
                inputs=resolved,
                input_refs=refs,
                started_at=started,
                ended_at=utc_now(),
                error=f"{type(exc).__name__}: {exc}",
            )
        values[step.name] = output
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
