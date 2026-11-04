"""Differ (P5-T5): what a replay changed, span by span and in the final output.

    diff = diff_runs(original, replay)
    diff.changed_spans     local ids that are patched / changed / new / missing
    diff.final_changes     [FinalChange(key, before, after)] for the workflow output

Built on `align_runs` (spans matched by local id; cache-only fields ignored). Each span
gets one status:

    patched    the patch target (its output was replaced)
    changed    re-ran and differs (`changes` says how: input, output, model, ...)
    new        only in the replay (e.g. a tool call the re-run step made for the first time)
    missing    only in the original (a call the re-run step no longer made)
    unchanged  identical

No ground truth here: whether the change *helped* is scored separately
(`evaluation.recovery`). The same diff feeds the UI overlay (`graph.ui`) and the
evaluation harness (`ReplayResult` records).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from pydantic import Field

from agenttrace.graph import align_runs, build_graph
from agenttrace.graph.ui import (
    ReplayStatus,
    UiFinalChange,
    UiGraph,
    UiReplay,
    replay_legend,
    to_ui_graph,
    with_replay_overlay,
)
from agenttrace.schemas.base import NonEmptyStr, StrictModel
from agenttrace.schemas.run import PatchSource, Run
from agenttrace.schemas.span import SpanKind


class SpanChange(StrictModel):
    local_id: NonEmptyStr
    node_id: NonEmptyStr
    kind: SpanKind
    step: NonEmptyStr  # the top-level step the span belongs to
    status: ReplayStatus
    changes: list[str] = Field(default_factory=list)


class FinalChange(StrictModel):
    key: NonEmptyStr
    before: Any = None
    after: Any = None


class ReplayDiff(StrictModel):
    original_run_id: NonEmptyStr
    replay_run_id: NonEmptyStr
    patch_source: PatchSource | None = None
    spans: list[SpanChange]  # every span of either run, in causal order
    final_changes: list[FinalChange] = Field(default_factory=list)

    @property
    def patched(self) -> list[str]:
        return [s.local_id for s in self.spans if s.status == "patched"]

    @property
    def changed_spans(self) -> list[str]:
        return [s.local_id for s in self.spans if s.status != "unchanged"]

    @property
    def final_identical(self) -> bool:
        return not self.final_changes

    def counts(self) -> dict[str, int]:
        out = dict.fromkeys(("patched", "changed", "new", "missing", "unchanged"), 0)
        for s in self.spans:
            out[s.status] += 1
        return out

    def status(self, local_id: str) -> ReplayStatus:
        for s in self.spans:
            if s.local_id == local_id:
                return s.status
        raise KeyError(local_id)


def _step(local_id: str) -> str:
    return local_id.split(".", 1)[0].split("#", 1)[0]


def diff_runs(original: Run, replay: Run) -> ReplayDiff:
    """The replay's changes against the run it replayed."""
    alignment = align_runs(original, replay)
    by_id = {s.span_id: s for s in [*original.spans, *replay.spans]}
    status_of = {"only_a": "missing", "only_b": "new", "unchanged": "unchanged"}
    spans = []
    for pair in alignment.pairs:
        span = by_id[pair.span_b or pair.span_a or ""]
        if pair.status == "changed":
            status: ReplayStatus = "patched" if "patched" in pair.changes else "changed"
        else:
            status = status_of[pair.status]  # type: ignore[assignment]
        spans.append(
            SpanChange(
                local_id=pair.local_id,
                node_id=span.node_id,
                kind=span.kind,
                step=_step(pair.local_id),
                status=status,
                changes=list(pair.changes),
            )
        )
    before, after = original.final_output, replay.final_output
    finals = [
        FinalChange(key=k, before=before.get(k), after=after.get(k))
        for k in sorted(set(before) | set(after))
        if before.get(k) != after.get(k)
    ]
    return ReplayDiff(
        original_run_id=original.run_id,
        replay_run_id=replay.run_id,
        patch_source=replay.patch_source,
        spans=spans,
        final_changes=finals,
    )


def to_ui_replay(replay: Run, diff: ReplayDiff, grade: Mapping[str, Any] | None = None) -> UiGraph:
    """The replay's React Flow graph with the diff overlaid (the UI's replay view).

    `grade` optionally adds recovery fields (`recovered`, `recovery_score`,
    `before_score`, `clean_correct`) computed by the harness.
    """
    from agenttrace.workflow.contracts import CONTRACTS

    doc = to_ui_graph(build_graph(replay, contracts=CONTRACTS), replay)
    statuses = {s.local_id: (s.status, list(s.changes)) for s in diff.spans}
    summary = UiReplay(
        original_run_id=diff.original_run_id,
        patch_source=diff.patch_source,
        patched=diff.patched,
        missing=[s.local_id for s in diff.spans if s.status == "missing"],
        final_changes=[UiFinalChange(**c.model_dump()) for c in diff.final_changes],
        counts=diff.counts(),
        legend=replay_legend(),
        **dict(grade or {}),
    )
    return with_replay_overlay(doc, statuses, summary)
