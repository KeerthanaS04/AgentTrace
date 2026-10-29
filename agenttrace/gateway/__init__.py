"""Model gateway: providers, cache, retry, cost accounting, structured output.

All LLM calls in AgentTrace go through `Gateway`; nothing else talks to a provider.
"""

from agenttrace.gateway.cache import CompletionCache, content_hash
from agenttrace.gateway.config import GatewayConfig, load_config
from agenttrace.gateway.gateway import Gateway
from agenttrace.gateway.ledger import CostLedger
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
    "CompletionCache",
    "CompletionResult",
    "CostLedger",
    "Gateway",
    "GatewayConfig",
    "GatewayError",
    "Message",
    "ProviderRequestError",
    "ProviderUnavailableError",
    "StructuredResult",
    "UnknownRoleError",
    "UnverifiedModelError",
    "content_hash",
    "load_config",
]
