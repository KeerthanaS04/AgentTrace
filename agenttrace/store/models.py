"""SQLAlchemy tables for the trace store.

Each row keeps the record's exact Pydantic JSON in `doc` (the source of truth), so a
reload is byte-identical to what was saved. The other columns are copies of fields
used for querying (run, scenario, node, kind, metric), indexed where it matters.
"""

from __future__ import annotations

from sqlalchemy import Boolean, ForeignKey, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

# Bump when tables change; the store refuses to open a database with another version.
SCHEMA_VERSION = 1


class Base(DeclarativeBase):
    pass


class MetaRow(Base):
    __tablename__ = "meta"

    key: Mapped[str] = mapped_column(String, primary_key=True)
    value: Mapped[str] = mapped_column(String)


class RunRow(Base):
    __tablename__ = "runs"

    run_id: Mapped[str] = mapped_column(String, primary_key=True)
    scenario_id: Mapped[str | None] = mapped_column(String, index=True)
    workflow_version: Mapped[str] = mapped_column(String)
    final_correct: Mapped[bool | None] = mapped_column(Boolean)
    replay_of: Mapped[str | None] = mapped_column(String, index=True)
    created_at: Mapped[str] = mapped_column(String, index=True)  # ISO 8601 UTC
    # The Run without its spans (they live in `spans`).
    doc: Mapped[str] = mapped_column(Text)


class SpanRow(Base):
    __tablename__ = "spans"

    run_id: Mapped[str] = mapped_column(
        ForeignKey("runs.run_id", ondelete="CASCADE"), primary_key=True
    )
    span_id: Mapped[str] = mapped_column(String, primary_key=True)
    # Position in the run's span list, so spans reload in their recorded order.
    seq: Mapped[int] = mapped_column(Integer)
    node_id: Mapped[str] = mapped_column(String, index=True)
    kind: Mapped[str] = mapped_column(String, index=True)
    agent: Mapped[str] = mapped_column(String)
    parent_span_id: Mapped[str | None] = mapped_column(String)
    doc: Mapped[str] = mapped_column(Text)


class ScoreRow(Base):
    __tablename__ = "scores"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    scope: Mapped[str] = mapped_column(String)
    run_id: Mapped[str | None] = mapped_column(String, index=True)
    span_id: Mapped[str | None] = mapped_column(String, index=True)
    agent: Mapped[str | None] = mapped_column(String, index=True)
    metric: Mapped[str] = mapped_column(String, index=True)
    judge: Mapped[str | None] = mapped_column(String)
    computed_at: Mapped[str] = mapped_column(String)
    doc: Mapped[str] = mapped_column(Text)


class DiagnosisRow(Base):
    __tablename__ = "diagnoses"

    diagnosis_id: Mapped[str] = mapped_column(String, primary_key=True)
    run_id: Mapped[str] = mapped_column(String, index=True)
    doc: Mapped[str] = mapped_column(Text)


class ReplayRow(Base):
    __tablename__ = "replays"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    original_run_id: Mapped[str] = mapped_column(String, index=True)
    replay_run_id: Mapped[str] = mapped_column(String, index=True)
    diagnosis_id: Mapped[str | None] = mapped_column(String, index=True)
    doc: Mapped[str] = mapped_column(Text)
