"""Request/response types and errors for the model gateway."""

from __future__ import annotations

from typing import Any, Generic, Literal, TypeVar

from pydantic import BaseModel, Field

from agenttrace.schemas.base import NonNegativeFloat, NonNegativeInt, StrictModel

ModelT = TypeVar("ModelT", bound=BaseModel)


class Message(StrictModel):
    role: Literal["system", "user", "assistant"]
    content: str


class CompletionResult(StrictModel):
    text: str
    role: str
    provider: str
    model: str
    params: dict[str, Any] = Field(default_factory=dict)
    tokens_in: NonNegativeInt = 0
    tokens_out: NonNegativeInt = 0
    # Hidden reasoning tokens, already included in tokens_out.
    reasoning_tokens: NonNegativeInt = 0
    finish_reason: str | None = None
    # Cost of this call. 0 on a cache hit (see `original_cost_usd`).
    cost_usd: NonNegativeFloat = 0.0
    original_cost_usd: NonNegativeFloat = 0.0
    cost_known: bool = True
    cache_hit: bool = False
    latency_s: NonNegativeFloat = 0.0
    cache_key: str = ""

    @property
    def truncated(self) -> bool:
        """The reply hit max_tokens (possibly before any visible text, if reasoning used it up)."""
        return self.finish_reason == "length"


class StructuredResult(BaseModel, Generic[ModelT]):
    """Outcome of a structured call. On parse failure `value` is None; the pipeline continues."""

    value: ModelT | None
    parse_failure: bool
    attempts: int
    errors: list[str] = Field(default_factory=list)
    completions: list[CompletionResult] = Field(default_factory=list)

    @property
    def cost_usd(self) -> float:
        return sum(c.cost_usd for c in self.completions)


class GatewayError(Exception):
    """Base class for gateway errors."""


class UnknownRoleError(GatewayError):
    pass


class UnverifiedModelError(GatewayError):
    """The role's model ID is still TODO_VERIFY. Confirm it from the provider dashboard."""


class ProviderUnavailableError(GatewayError):
    """The provider can't be used (e.g. its API key isn't set)."""


class ProviderRequestError(GatewayError):
    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code
