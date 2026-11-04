"""Recovery scoring (P5-T5): did a replay fix the outcome, and by how much?

    rec = score_recovery(expected, original.final_output, replay.final_output, clean_final)
    result = replay_result(executed, diff, rec)     # the stored schemas.ReplayResult

    recovered       the replay's final output is correct (deterministic checker, P1-T7)
    recovery_score  partial credit in [0, 1]: half for the right decision, half for the
                    share of required facts the reply states (the decision alone when the
                    ticket has no required facts)
    clean_correct   whether the clean run of the same seed was correct, i.e. whether
                    recovery was possible at all (P4-T5: clean runs miss on some seeds)
    matches_clean   whether the replay reached the clean run's outcome (decision and
                    missing facts), right or wrong

Separating "the patch did not help" from "the workflow gets this ticket wrong anyway"
needs the last two (findings §6, P5-T3). Harness side: this reads the expected outcome.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from agenttrace.evaluation.outcome import OutcomeCheck, check_outcome
from agenttrace.replay.differ import ReplayDiff
from agenttrace.replay.executor import ExecutedReplay
from agenttrace.schemas.base import Probability, StrictModel
from agenttrace.schemas.ground_truth import ExpectedFinal
from agenttrace.schemas.replay import ReplayResult


def outcome_score(check: OutcomeCheck) -> float:
    """Partial credit for one final output (see module doc)."""
    facts = len(check.facts_found) + len(check.facts_missing)
    if facts == 0:
        return float(check.decision_ok)
    return 0.5 * check.decision_ok + 0.5 * len(check.facts_found) / facts


class Recovery(StrictModel):
    recovered: bool
    recovery_score: Probability
    before_correct: bool
    before_score: Probability
    decision_ok: bool
    facts_missing: list[str]
    clean_correct: bool | None = None
    matches_clean: bool | None = None

    @property
    def gain(self) -> float:
        """How much the replay moved the score (negative: the patch made it worse)."""
        return self.recovery_score - self.before_score


def _outcome(check: OutcomeCheck) -> tuple[str | None, tuple[str, ...]]:
    return check.got_decision, tuple(sorted(check.facts_missing))


def score_recovery(
    expected: ExpectedFinal,
    original_final: Mapping[str, Any],
    replay_final: Mapping[str, Any],
    clean_final: Mapping[str, Any] | None = None,
) -> Recovery:
    before = check_outcome(expected, original_final)
    after = check_outcome(expected, replay_final)
    clean = check_outcome(expected, clean_final) if clean_final is not None else None
    return Recovery(
        recovered=after.correct,
        recovery_score=outcome_score(after),
        before_correct=before.correct,
        before_score=outcome_score(before),
        decision_ok=after.decision_ok,
        facts_missing=after.facts_missing,
        clean_correct=None if clean is None else clean.correct,
        matches_clean=None if clean is None else _outcome(after) == _outcome(clean),
    )


def replay_result(
    executed: ExecutedReplay,
    diff: ReplayDiff,
    recovery: Recovery,
    *,
    diagnosis_id: str | None = None,
) -> ReplayResult:
    """The stored record of one counterfactual replay (needs a tagged patch source)."""
    patch = executed.plan.schema_patch()
    return ReplayResult(
        original_run_id=executed.original.run_id,
        replay_run_id=executed.replay.run_id,
        diagnosis_id=diagnosis_id,
        patch=patch,
        patched_span_id=patch.target_span_id,
        patch_source=patch.source,
        changed_spans=diff.changed_spans,
        original_final=executed.original.final_output,
        replay_final=executed.replay.final_output,
        recovered=recovery.recovered,
        recovery_score=recovery.recovery_score,
        clean_correct=recovery.clean_correct,
        matches_clean=recovery.matches_clean,
    )
