"""Ping every configured role with a trivial prompt.

Roles whose model is TODO_VERIFY or whose provider has no API key are skipped,
not failed. Exit code is 1 only if a reachable role actually fails.

Usage: python -m agenttrace.gateway.smoke   (or `python tasks.py smoke`)
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Literal

from agenttrace.gateway.budget import BudgetExceededError
from agenttrace.gateway.gateway import Gateway
from agenttrace.gateway.types import (
    GatewayError,
    ProviderUnavailableError,
    UnverifiedModelError,
)

PROMPT = [{"role": "user", "content": "Reply with the single word: pong"}]
# Small, but with room for reasoning models to think before answering (reasoning
# tokens count toward max_tokens). The role's own reasoning_effort still applies.
PING_PARAMS = {"max_tokens": 256}


@dataclass(frozen=True)
class SmokeResult:
    role: str
    status: Literal["ok", "skipped", "failed"]
    detail: str
    cost_usd: float = 0.0
    latency_s: float = 0.0


def run_smoke(gateway: Gateway) -> list[SmokeResult]:
    results: list[SmokeResult] = []
    for role in gateway.config.roles:
        try:
            # Bypass the cache: a smoke test must actually reach the provider.
            res = gateway.complete(role, PROMPT, params=PING_PARAMS, tag="smoke", use_cache=False)
        except (UnverifiedModelError, ProviderUnavailableError, BudgetExceededError) as exc:
            results.append(SmokeResult(role, "skipped", str(exc)))
        except GatewayError as exc:
            results.append(SmokeResult(role, "failed", str(exc)))
        else:
            reply = res.text.strip().replace("\n", " ")[:40]
            if reply:
                status: Literal["ok", "failed"] = "ok"
                detail = f"{res.provider}/{res.model}: {reply!r}"
            else:
                # Reachable but useless: usually reasoning consumed the whole max_tokens.
                status = "failed"
                detail = (
                    f"{res.provider}/{res.model}: empty reply (finish={res.finish_reason}, "
                    f"{res.reasoning_tokens} reasoning tokens); "
                    "raise max_tokens or lower reasoning_effort"
                )
            results.append(
                SmokeResult(role, status, detail, cost_usd=res.cost_usd, latency_s=res.latency_s)
            )
    return results


def format_report(gateway: Gateway, results: list[SmokeResult]) -> str:
    lines = [f"{'role':<15} {'status':<8} {'latency':>8} {'cost':>10}  detail"]
    for r in results:
        reached = r.latency_s > 0 or r.cost_usd > 0
        latency = f"{r.latency_s:.2f}s" if reached else "-"
        cost = f"${r.cost_usd:.5f}" if reached else "-"
        lines.append(f"{r.role:<15} {r.status:<8} {latency:>8} {cost:>10}  {r.detail}")

    missing = gateway.config.missing_roles()
    if missing:
        lines.append(f"\nmissing required roles: {', '.join(missing)}")

    lines.append("\nlifetime spend:")
    spent = gateway.budget.spent_by_provider()
    for name, pc in gateway.config.providers.items():
        cap = f" / cap ${pc.budget_usd:.2f}" if pc.budget_usd is not None else ""
        lines.append(f"  {name:<12} ${spent.get(name, 0.0):.4f}{cap}")
    total_cap = gateway.budget.max_usd_total
    total_cap_str = f" / cap ${total_cap:.2f}" if total_cap is not None else ""
    lines.append(f"  {'total':<12} ${gateway.budget.spent_total:.4f}{total_cap_str}")

    counts = {s: sum(r.status == s for r in results) for s in ("ok", "skipped", "failed")}
    lines.append(f"\n{counts['ok']} ok, {counts['skipped']} skipped, {counts['failed']} failed")
    return "\n".join(lines)


def main() -> int:
    gateway = Gateway.from_env()
    try:
        results = run_smoke(gateway)
        print(format_report(gateway, results))
    finally:
        gateway.close()
    return 1 if any(r.status == "failed" for r in results) else 0


if __name__ == "__main__":
    sys.exit(main())
