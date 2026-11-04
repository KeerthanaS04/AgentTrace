"""Verification loop (P7-T1): prove or reject root-cause candidates by experiment.

    outcome = verify(run, diagnosis, workflow, resources, gateway)
    outcome.diagnosis          # root causes now `verified` / `unverified` (demoted)
    outcome.replays            # one ReplayResult per attempt, diagnosis_id set

For each ranked root candidate, best first, up to `max_attempts` (default 3):

    1. patch    from `patch_for(run, target)`: by default a rule if one applies (tool
                re-call, search re-run) else the model's proposal (P5-T4: rules match the
                oracle on tool/retrieval faults; only a model can patch a judgement). The
                target is the root span, or its step when the node ran several times
                (a retrieval fault hits every search: patch the retriever step).
    2. replay   the patch's plan (P5-T2/T3), descendants re-run, other spans frozen.
    3. check    `recovery(original, replay)` -> RecoveryCheck. By default, truth-free:
                the final output must have changed, and no span that produces it
                (policy, writer) may be judged *faulty* (wrong given its inputs) or
                carry a validator error. "suspicious" (unsure) passes: P7-T1 measured
                that requiring "ok" rejected 7 of 10 real recoveries. The evaluation
                harness passes a checker against the expected outcome.
    4. verdict  recovered -> the candidate is `verified` and the loop stops; otherwise it
                is `unverified`, its confidence is multiplied by `demote` (0.5), and the
                next candidate is tried. A candidate no source can patch, or whose
                patch target was already tried (two spans of one node), is unverified
                without spending an attempt.

Every attempt is a `ReplayResult` with `diagnosis_id` set (stored by the caller). Nothing
here sees ground truth; hooks supplied by the harness may.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

from agenttrace.diagnosis.stage_a import score_run
from agenttrace.diagnosis.validators import ValidationContext, validate_run
from agenttrace.gateway.gateway import Gateway
from agenttrace.replay.differ import diff_runs
from agenttrace.replay.executor import ExecutedReplay, execute_plan
from agenttrace.replay.patcher import Patch, PatchError, plan_patch
from agenttrace.replay.sources import PatchSourceError, model_patch, rule_patch
from agenttrace.schemas.base import Probability, StrictModel
from agenttrace.schemas.diagnosis import Diagnosis, RootCause
from agenttrace.schemas.replay import ReplayResult
from agenttrace.schemas.run import Run
from agenttrace.tracing.context import FaultHook
from agenttrace.workflow.runtime import Workflow

MAX_ATTEMPTS = 3
DEMOTE = 0.5
FINAL_NODES = ("policy", "writer")  # the steps whose outputs are the run's final output


class RecoveryCheck(StrictModel):
    recovered: bool
    score: Probability  # partial credit, for ReplayResult.recovery_score
    detail: str = ""
    cost_usd: float = 0.0


PatchFor = Callable[[Run, str], Patch | None]
RecoveryFn = Callable[[Run, Run], RecoveryCheck]
FaultsFor = Callable[[RootCause], FaultHook | None]


@dataclass
class Attempt:
    root: RootCause
    target: str  # local span id patched
    patch: Patch | None
    recovered: bool
    detail: str
    replay_run: Run | None = None
    result: ReplayResult | None = None


@dataclass
class VerificationOutcome:
    diagnosis: Diagnosis
    attempts: list[Attempt] = field(default_factory=list)
    cost_usd: float = 0.0

    @property
    def replays(self) -> list[ReplayResult]:
        return [a.result for a in self.attempts if a.result is not None]

    @property
    def verified(self) -> RootCause | None:
        return next((r for r in self.diagnosis.root_causes if r.status == "verified"), None)


def patch_target(run: Run, root: RootCause) -> str:
    """The local span to patch for a root: the span itself, or its step if the node ran
    more than once (patching one search would leave the others' fault in place)."""
    local = root.span_id.removeprefix(f"{run.run_id}/")
    if sum(s.node_id == root.node_id for s in run.spans) > 1 and "." in root.node_id:
        return root.node_id.split(".", 1)[0]
    return local


def default_patch_for(gateway: Gateway, resources: Mapping[str, Any]) -> PatchFor:
    """Rule first (deterministic, free), else the model's proposal."""

    def patch_for(run: Run, target: str) -> Patch | None:
        patch = rule_patch(run, target, resources)
        if patch is not None:
            return patch
        try:
            return model_patch(run, target, gateway)
        except (PatchError, PatchSourceError):
            return None

    return patch_for


def judged_recovery(
    gateway: Gateway, ctx: ValidationContext | None = None, role: str = "judge_fast"
) -> RecoveryFn:
    """Truth-free recovery: the outcome changed, and the judge and validators now pass
    the final-output spans of the replay."""
    ctx = ctx or ValidationContext.default()

    def check(original: Run, replay: Run) -> RecoveryCheck:
        if replay.final_output == original.final_output:
            return RecoveryCheck(
                recovered=False, score=0.0, detail="the final output did not change"
            )
        errors = {
            f.span_id
            for f in validate_run(replay, ctx)
            if f.severity == "error" and f.span_id.split("/", 1)[1].split(".")[0] in FINAL_NODES
        }
        stage = score_run(replay, gateway, roles=(role,), ctx=ctx)
        finals = [s for s in stage.spans if s.local_id in FINAL_NODES and role in s.scores]
        bad = [
            s.local_id for s in finals if s.scores[role].verdict == "faulty" or s.span_id in errors
        ]
        verdicts = ", ".join(f"{s.local_id}={s.scores[role].verdict}" for s in finals)
        score = (
            sum(s.scores[role].output_ok_given_inputs for s in finals) / len(finals)
            if finals
            else 0.0
        )
        detail = (
            f"the judge passes the new outcome ({verdicts})"
            if not bad
            else f"still flagged: {bad} ({verdicts})"
        )
        if not finals:
            detail = "the judge scored no final span"
        return RecoveryCheck(
            recovered=not bad and bool(finals),
            score=round(score, 4),
            detail=detail,
            cost_usd=stage.cost_usd,
        )

    return check


def verify(
    run: Run,
    diagnosis: Diagnosis,
    workflow: Workflow,
    resources: Mapping[str, Any],
    *,
    patch_for: PatchFor,
    recovery: RecoveryFn,
    faults_for: FaultsFor | None = None,
    max_attempts: int = MAX_ATTEMPTS,
    demote: float = DEMOTE,
    replay_id: Callable[[int], str] | None = None,
) -> VerificationOutcome:
    """Run the loop (see module doc). Returns the updated diagnosis and the attempts."""
    roots = [r.model_copy(deep=True) for r in diagnosis.root_causes]
    outcome = VerificationOutcome(diagnosis=diagnosis)
    used = 0
    tried: set[str] = set()
    for root in roots:
        if used >= max_attempts:
            break
        target = patch_target(run, root)
        if target in tried:
            root.status = "unverified"
            root.confidence = round(root.confidence * demote, 4)
            outcome.attempts.append(
                Attempt(root, target, None, False, f"{target} was already patched and replayed")
            )
            continue
        tried.add(target)
        patch = patch_for(run, target)
        if patch is None:
            root.status = "unverified"
            root.confidence = round(root.confidence * demote, 4)
            outcome.attempts.append(Attempt(root, target, None, False, "no patch for this node"))
            continue
        try:
            plan = plan_patch(run, workflow, patch)
        except PatchError as exc:
            root.status = "unverified"
            root.confidence = round(root.confidence * demote, 4)
            outcome.attempts.append(Attempt(root, target, patch, False, f"patch refused: {exc}"))
            continue
        used += 1
        rid = replay_id(used) if replay_id else f"{run.run_id}-verify-{used}"
        faults = faults_for(root) if faults_for else None
        executed: ExecutedReplay = execute_plan(
            run, workflow, plan, resources, run_id=rid, faults=faults
        )
        check = recovery(run, executed.replay)
        outcome.cost_usd += check.cost_usd
        diff = diff_runs(run, executed.replay)
        tagged = plan if plan.patch_source else plan.model_copy(update={"patch_source": "model"})
        schema = tagged.schema_patch()
        result = ReplayResult(
            original_run_id=run.run_id,
            replay_run_id=executed.replay.run_id,
            diagnosis_id=diagnosis.diagnosis_id,
            patch=schema,
            patched_span_id=plan.target,
            patch_source=schema.source,
            changed_spans=diff.changed_spans,
            original_final=run.final_output,
            replay_final=executed.replay.final_output,
            recovered=check.recovered,
            recovery_score=check.score,
        )
        outcome.attempts.append(
            Attempt(root, target, patch, check.recovered, check.detail, executed.replay, result)
        )
        if check.recovered:
            root.status = "verified"
            break
        root.status = "unverified"
        root.confidence = round(root.confidence * demote, 4)
    # Candidates never tried keep their status ("candidate").
    outcome.diagnosis = diagnosis.model_copy(
        update={
            "root_causes": roots,
            "cost_usd": diagnosis.cost_usd + outcome.cost_usd,
        }
    )
    return outcome
