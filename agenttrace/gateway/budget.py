"""Budget guard: refuse model calls that would exceed per-run, per-provider, or total caps.

Spend is lifetime spend (seeded from the cost ledger on disk), so caps hold across
processes. Calls are checked *before* they are sent, using a worst-case estimate
(prompt size plus `max_tokens` of output); cache hits are always free and allowed.
"""

from __future__ import annotations

import logging
import threading
from collections import defaultdict
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass

from agenttrace.gateway.types import GatewayError

log = logging.getLogger(__name__)


class BudgetExceededError(GatewayError):
    pass


@dataclass
class RunBudget:
    run_id: str
    spent_usd: float = 0.0


@dataclass(frozen=True)
class ProviderLimit:
    budget_usd: float | None = None
    warn_usd: float | None = None


_current_run: ContextVar[RunBudget | None] = ContextVar("agenttrace_run_budget", default=None)


class BudgetGuard:
    def __init__(
        self,
        *,
        max_usd_total: float | None = None,
        max_usd_per_run: float | None = None,
        provider_limits: Mapping[str, ProviderLimit] | None = None,
        prior_spend: Mapping[str, float] | None = None,
    ) -> None:
        self.max_usd_total = max_usd_total
        self.max_usd_per_run = max_usd_per_run
        self.provider_limits = dict(provider_limits or {})
        self._spent: dict[str, float] = defaultdict(float, prior_spend or {})
        self._warned: set[str] = set()
        self._lock = threading.Lock()

    # ------------------------------------------------------------ run scope

    @contextmanager
    def run(self, run_id: str) -> Iterator[RunBudget]:
        """Attribute calls made inside this block to one run, capped by `max_usd_per_run`."""
        scope = RunBudget(run_id)
        token = _current_run.set(scope)
        try:
            yield scope
        finally:
            _current_run.reset(token)

    # ------------------------------------------------------------ checks

    def check(self, provider: str, estimate_usd: float) -> None:
        """Raise BudgetExceededError if a call costing `estimate_usd` would break a cap."""
        with self._lock:
            total = sum(self._spent.values())
            on_provider = self._spent[provider]
        if self.max_usd_total is not None and total + estimate_usd > self.max_usd_total:
            raise BudgetExceededError(
                f"total budget ${self.max_usd_total:.2f} would be exceeded "
                f"(spent ${total:.4f}, next call up to ${estimate_usd:.4f})"
            )
        limit = self.provider_limits.get(provider)
        if limit and limit.budget_usd is not None and on_provider + estimate_usd > limit.budget_usd:
            raise BudgetExceededError(
                f"{provider} budget ${limit.budget_usd:.2f} would be exceeded "
                f"(spent ${on_provider:.4f}, next call up to ${estimate_usd:.4f})"
            )
        scope = _current_run.get()
        if (
            scope is not None
            and self.max_usd_per_run is not None
            and scope.spent_usd + estimate_usd > self.max_usd_per_run
        ):
            raise BudgetExceededError(
                f"run {scope.run_id} budget ${self.max_usd_per_run:.2f} would be exceeded "
                f"(spent ${scope.spent_usd:.4f}, next call up to ${estimate_usd:.4f})"
            )

    def record(self, provider: str, cost_usd: float) -> None:
        with self._lock:
            self._spent[provider] += cost_usd
            on_provider = self._spent[provider]
        scope = _current_run.get()
        if scope is not None:
            scope.spent_usd += cost_usd
        limit = self.provider_limits.get(provider)
        if (
            limit
            and limit.warn_usd is not None
            and on_provider >= limit.warn_usd
            and provider not in self._warned
        ):
            self._warned.add(provider)
            log.warning(
                "%s spend $%.2f has crossed the warning level $%.2f (cap: %s)",
                provider,
                on_provider,
                limit.warn_usd,
                f"${limit.budget_usd:.2f}" if limit.budget_usd is not None else "none",
            )

    # ------------------------------------------------------------ reporting

    @property
    def spent_total(self) -> float:
        with self._lock:
            return sum(self._spent.values())

    def spent_by_provider(self) -> dict[str, float]:
        with self._lock:
            return dict(self._spent)
