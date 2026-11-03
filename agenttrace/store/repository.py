"""TraceStore: save and query runs, spans, scores, diagnoses, and replays (SQLite).

    with TraceStore.open() as store:          # .cache/agenttrace.db, or $AGENTTRACE_DB
        store.save_run(run)
        same = store.get_run(run.run_id)      # same.model_dump_json() == run.model_dump_json()
        store.spans(node_id="policy", scenario_id="clean-07")

Records are stored as their Pydantic JSON and re-validated on load, so what comes back
is exactly what went in, and a corrupted row fails loudly instead of loading silently.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

from sqlalchemy import Engine, create_engine, event, func, select
from sqlalchemy.orm import Session

from agenttrace.schemas.diagnosis import Diagnosis
from agenttrace.schemas.replay import ReplayResult
from agenttrace.schemas.run import Run
from agenttrace.schemas.score import ExplainabilityScore, Metric
from agenttrace.schemas.span import Span, SpanKind
from agenttrace.store.models import (
    SCHEMA_VERSION,
    Base,
    DiagnosisRow,
    MetaRow,
    ReplayRow,
    RunRow,
    ScoreRow,
    SpanRow,
)
from agenttrace.tracing.integrity import validate_run

DEFAULT_DB_PATH = Path(".cache") / "agenttrace.db"


class StoreError(RuntimeError):
    pass


class RunExistsError(StoreError):
    pass


def _ts(value: datetime) -> str:
    """Fixed-width UTC timestamp, so string order is time order."""
    return value.strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _enable_foreign_keys(engine: Engine) -> None:
    @event.listens_for(engine, "connect")
    def _on_connect(conn: Any, _record: Any) -> None:
        cursor = conn.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()


class TraceStore:
    def __init__(self, url: str) -> None:
        self.url = url
        self.engine = create_engine(url)
        _enable_foreign_keys(self.engine)
        Base.metadata.create_all(self.engine)
        self._check_schema_version()

    @classmethod
    def open(cls, path: Path | str | None = None) -> TraceStore:
        """Open (creating if needed) a SQLite store at `path`, $AGENTTRACE_DB, or the default."""
        target = Path(path or os.environ.get("AGENTTRACE_DB") or DEFAULT_DB_PATH)
        target.parent.mkdir(parents=True, exist_ok=True)
        return cls(f"sqlite:///{target.as_posix()}")

    @classmethod
    def in_memory(cls) -> TraceStore:
        return cls("sqlite://")

    def __enter__(self) -> TraceStore:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        self.engine.dispose()

    @contextmanager
    def _session(self) -> Iterator[Session]:
        with Session(self.engine) as session, session.begin():
            yield session

    def _check_schema_version(self) -> None:
        with self._session() as s:
            row = s.get(MetaRow, "schema_version")
            if row is None:
                s.add(MetaRow(key="schema_version", value=str(SCHEMA_VERSION)))
            elif row.value != str(SCHEMA_VERSION):
                raise StoreError(
                    f"{self.url}: store schema v{row.value}, this code expects v{SCHEMA_VERSION}"
                )

    # ------------------------------------------------------------ runs

    def save_run(self, run: Run, *, replace: bool = False) -> None:
        """Save a run and all its spans in one transaction.

        The trace is integrity-checked first (orphan refs, cycles, timestamps); a broken
        trace raises `TraceIntegrityError` and nothing is written.
        """
        validate_run(run)
        with self._session() as s:
            existing = s.get(RunRow, run.run_id)
            if existing is not None:
                if not replace:
                    raise RunExistsError(f"run {run.run_id!r} is already stored")
                s.delete(existing)
                s.flush()
            s.add(
                RunRow(
                    run_id=run.run_id,
                    scenario_id=run.scenario_id,
                    workflow_version=run.workflow_version,
                    final_correct=run.final_correct,
                    replay_of=run.replay_of,
                    created_at=_ts(run.created_at),
                    doc=run.model_dump_json(exclude={"spans"}),
                )
            )
            s.add_all(
                SpanRow(
                    run_id=run.run_id,
                    span_id=span.span_id,
                    seq=i,
                    node_id=span.node_id,
                    kind=span.kind,
                    agent=span.agent,
                    parent_span_id=span.parent_span_id,
                    doc=span.model_dump_json(),
                )
                for i, span in enumerate(run.spans)
            )

    def has_run(self, run_id: str) -> bool:
        with self._session() as s:
            return s.get(RunRow, run_id) is not None

    def get_run(self, run_id: str) -> Run:
        with self._session() as s:
            row = s.get(RunRow, run_id)
            if row is None:
                raise KeyError(f"no stored run {run_id!r}")
            docs = s.scalars(
                select(SpanRow.doc).where(SpanRow.run_id == run_id).order_by(SpanRow.seq)
            ).all()
            data = json.loads(row.doc)
        data["spans"] = [json.loads(d) for d in docs]
        return validate_run(Run.model_validate(data))

    def runs(self, *, scenario_id: str | None = None, replay_of: str | None = None) -> list[Run]:
        """Full runs, oldest first, optionally filtered by scenario or replay origin."""
        return [self.get_run(i) for i in self.run_ids(scenario_id=scenario_id, replay_of=replay_of)]

    def run_ids(self, *, scenario_id: str | None = None, replay_of: str | None = None) -> list[str]:
        query = select(RunRow.run_id).order_by(RunRow.created_at, RunRow.run_id)
        if scenario_id is not None:
            query = query.where(RunRow.scenario_id == scenario_id)
        if replay_of is not None:
            query = query.where(RunRow.replay_of == replay_of)
        with self._session() as s:
            return list(s.scalars(query).all())

    def summaries(self, *, scenario_id: str | None = None) -> list[dict[str, Any]]:
        """One line of facts per run (no span bodies), oldest first."""
        query = (
            select(
                RunRow.run_id,
                RunRow.scenario_id,
                RunRow.workflow_version,
                RunRow.final_correct,
                RunRow.replay_of,
                RunRow.created_at,
                func.count(SpanRow.span_id),
            )
            .outerjoin(SpanRow, SpanRow.run_id == RunRow.run_id)
            .group_by(RunRow.run_id)
            .order_by(RunRow.created_at, RunRow.run_id)
        )
        if scenario_id is not None:
            query = query.where(RunRow.scenario_id == scenario_id)
        keys = (
            "run_id",
            "scenario_id",
            "workflow_version",
            "final_correct",
            "replay_of",
            "created_at",
            "spans",
        )
        with self._session() as s:
            return [dict(zip(keys, row, strict=True)) for row in s.execute(query).all()]

    def delete_run(self, run_id: str) -> None:
        """Delete a run and (by cascade) its spans."""
        with self._session() as s:
            row = s.get(RunRow, run_id)
            if row is None:
                raise KeyError(f"no stored run {run_id!r}")
            s.delete(row)

    def next_run_id(self, prefix: str) -> str:
        """`{prefix}-r1`, `-r2`, ...: the first id not yet stored."""
        n = 1
        while self.has_run(f"{prefix}-r{n}"):
            n += 1
        return f"{prefix}-r{n}"

    # ------------------------------------------------------------ spans

    def spans(
        self,
        *,
        run_id: str | None = None,
        node_id: str | None = None,
        kind: SpanKind | None = None,
        scenario_id: str | None = None,
    ) -> list[Span]:
        """Spans matching every given filter, in run order then recorded order.

        E.g. `spans(node_id="policy", scenario_id="clean-07")` is the Policy Agent's span
        in every stored run of that scenario.
        """
        query = (
            select(SpanRow.doc)
            .join(RunRow, RunRow.run_id == SpanRow.run_id)
            .order_by(RunRow.created_at, RunRow.run_id, SpanRow.seq)
        )
        if run_id is not None:
            query = query.where(SpanRow.run_id == run_id)
        if node_id is not None:
            query = query.where(SpanRow.node_id == node_id)
        if kind is not None:
            query = query.where(SpanRow.kind == kind)
        if scenario_id is not None:
            query = query.where(RunRow.scenario_id == scenario_id)
        with self._session() as s:
            docs = s.scalars(query).all()
        return [Span.model_validate_json(d) for d in docs]

    def get_span(self, run_id: str, span_id: str) -> Span:
        with self._session() as s:
            row = s.get(SpanRow, (run_id, span_id))
            if row is None:
                raise KeyError(f"no span {span_id!r} in run {run_id!r}")
            return Span.model_validate_json(row.doc)

    # ------------------------------------------------------------ scores

    def add_score(self, score: ExplainabilityScore) -> None:
        with self._session() as s:
            s.add(
                ScoreRow(
                    scope=score.scope,
                    run_id=score.run_id,
                    span_id=score.span_id,
                    agent=score.agent,
                    metric=score.metric,
                    judge=score.judge,
                    computed_at=_ts(score.computed_at),
                    doc=score.model_dump_json(),
                )
            )

    def scores(
        self,
        *,
        run_id: str | None = None,
        span_id: str | None = None,
        agent: str | None = None,
        metric: Metric | None = None,
    ) -> list[ExplainabilityScore]:
        query = select(ScoreRow.doc).order_by(ScoreRow.computed_at, ScoreRow.id)
        if run_id is not None:
            query = query.where(ScoreRow.run_id == run_id)
        if span_id is not None:
            query = query.where(ScoreRow.span_id == span_id)
        if agent is not None:
            query = query.where(ScoreRow.agent == agent)
        if metric is not None:
            query = query.where(ScoreRow.metric == metric)
        with self._session() as s:
            docs = s.scalars(query).all()
        return [ExplainabilityScore.model_validate_json(d) for d in docs]

    # ------------------------------------------------------------ diagnoses

    def save_diagnosis(self, diagnosis: Diagnosis, *, replace: bool = False) -> None:
        with self._session() as s:
            existing = s.get(DiagnosisRow, diagnosis.diagnosis_id)
            if existing is not None and not replace:
                raise StoreError(f"diagnosis {diagnosis.diagnosis_id!r} is already stored")
            s.merge(
                DiagnosisRow(
                    diagnosis_id=diagnosis.diagnosis_id,
                    run_id=diagnosis.run_id,
                    doc=diagnosis.model_dump_json(),
                )
            )

    def get_diagnosis(self, diagnosis_id: str) -> Diagnosis:
        with self._session() as s:
            row = s.get(DiagnosisRow, diagnosis_id)
            if row is None:
                raise KeyError(f"no stored diagnosis {diagnosis_id!r}")
            return Diagnosis.model_validate_json(row.doc)

    def diagnoses(self, *, run_id: str | None = None) -> list[Diagnosis]:
        query = select(DiagnosisRow.doc).order_by(DiagnosisRow.diagnosis_id)
        if run_id is not None:
            query = query.where(DiagnosisRow.run_id == run_id)
        with self._session() as s:
            docs = s.scalars(query).all()
        return [Diagnosis.model_validate_json(d) for d in docs]

    # ------------------------------------------------------------ replays

    def add_replay(self, replay: ReplayResult) -> None:
        with self._session() as s:
            s.add(
                ReplayRow(
                    original_run_id=replay.original_run_id,
                    replay_run_id=replay.replay_run_id,
                    diagnosis_id=replay.diagnosis_id,
                    doc=replay.model_dump_json(),
                )
            )

    def replays(
        self, *, original_run_id: str | None = None, diagnosis_id: str | None = None
    ) -> list[ReplayResult]:
        query = select(ReplayRow.doc).order_by(ReplayRow.id)
        if original_run_id is not None:
            query = query.where(ReplayRow.original_run_id == original_run_id)
        if diagnosis_id is not None:
            query = query.where(ReplayRow.diagnosis_id == diagnosis_id)
        with self._session() as s:
            docs = s.scalars(query).all()
        return [ReplayResult.model_validate_json(d) for d in docs]
