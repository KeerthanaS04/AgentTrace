"""Shared base model and constrained types for all schemas."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Annotated

from pydantic import AfterValidator, AwareDatetime, BaseModel, ConfigDict, Field


class StrictModel(BaseModel):
    """Base for every schema: unknown fields are rejected and assignments are re-validated."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)


def _to_utc(value: datetime) -> datetime:
    return value.astimezone(UTC)


NonEmptyStr = Annotated[str, Field(min_length=1)]
Probability = Annotated[float, Field(ge=0.0, le=1.0)]
NonNegativeFloat = Annotated[float, Field(ge=0.0)]
NonNegativeInt = Annotated[int, Field(ge=0)]
# Timezone-aware timestamps, normalized to UTC. Naive datetimes are rejected.
UtcDatetime = Annotated[AwareDatetime, AfterValidator(_to_utc)]


def utc_now() -> datetime:
    return datetime.now(UTC)
