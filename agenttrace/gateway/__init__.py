"""Model gateway: providers, cache, retry, cost accounting, structured output.

All LLM calls in AgentTrace go through `Gateway`; nothing else talks to a provider.
"""

from agenttrace.gateway.budget import BudgetExceededError, BudgetGuard, ProviderLimit
from agenttrace.gateway.cache import CompletionCache, content_hash
from agenttrace.gateway.config import GatewayConfig, load_config
from agenttrace.gateway.gateway import Gateway
from agenttrace.gateway.ledger import CostLedger, read_spend
from agenttrace.gateway.types import (
    CompletionResult,
    GatewayError,
    Message,
    ProviderRequestError,
    ProviderUnavailableError,
    StructuredResult,
    UnknownRoleError,
    UnverifiedModelError,
)

__all__ = [
    "BudgetExceededError",
    "BudgetGuard",
    "CompletionCache",
    "CompletionResult",
    "CostLedger",
    "Gateway",
    "GatewayConfig",
    "GatewayError",
    "Message",
    "ProviderLimit",
    "ProviderRequestError",
    "ProviderUnavailableError",
    "StructuredResult",
    "UnknownRoleError",
    "UnverifiedModelError",
    "content_hash",
    "load_config",
    "read_spend",
]
