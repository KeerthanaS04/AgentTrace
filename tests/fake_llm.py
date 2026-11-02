"""Scripted fake model server for offline workflow tests.

Each support agent is recognized by its prompt's first line and answered with a fixed
JSON reply, so tests exercise the real gateway, agents, and tools without API keys.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx

from agenttrace.gateway import CompletionCache, Gateway, load_config

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "models.yaml"
EXPL = {"rationale": "r", "cited_evidence": [], "confidence": 0.9}

TICKET = {
    "ticket_id": "T-1",
    "customer_id": "C001",
    "subject": "Return a sweater",
    "body": "I want to return the wool sweater from order O1001 for a refund.",
}

Script = Callable[[str], dict[str, Any]]


def default_script(prompt: str) -> dict[str, Any]:
    if prompt.startswith("You are the Router"):
        return {**EXPL, "category": "refund", "order_id": "O1001"}
    if prompt.startswith("You are the Retriever"):
        return {**EXPL, "queries": ["return window refund apparel"]}
    if prompt.startswith("You are the Tool Agent"):
        if "(none yet)" in prompt:
            return {
                **EXPL,
                "calls": [
                    {"tool": "order_lookup", "args": {"order_id": "O1001"}},
                    {"tool": "account_status", "args": {"customer_id": "C001"}},
                ],
                "done": False,
            }
        if '"refund_api"' not in prompt.split("Tool results so far:")[1]:
            quote = {"order_id": "O1001", "amount": 48, "mode": "quote", "reason": "return"}
            return {**EXPL, "calls": [{"tool": "refund_api", "args": quote}], "done": True}
        return {**EXPL, "calls": [], "done": True}
    if prompt.startswith("You are the Policy Agent"):
        return {
            **EXPL,
            "decision": "approve",
            "refund_amount": 41.05,
            "escalate_to": None,
            "cited_doc_ids": ["KB-REF-001"],
            "justification": "Within 30 days; $6.95 return fee deducted.",
        }
    if prompt.startswith("You are the Response Writer"):
        return {**EXPL, "reply": "Approved: $41.05 back in 5-7 business days. Brightcart Support"}
    raise AssertionError(f"unexpected prompt: {prompt[:60]}")


def agent_of(prompt: str) -> str:
    """Which agent sent a prompt, e.g. "Policy Agent"."""
    return prompt.removeprefix("You are the ").split(" in a customer-support")[0]


def make_gateway(
    tmp_path: Path,
    script: Script = default_script,
    prompts: list[str] | None = None,
    calls: list[tuple[str, str]] | None = None,
    config_path: Path = CONFIG_PATH,
    usage: dict[str, int] | None = None,
) -> Gateway:
    """A real Gateway whose providers are replaced by the scripted server.

    `prompts` collects each agent prompt; `calls` collects (agent, model id) per request;
    `usage` is returned as each reply's token usage (default: none reported).
    """

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        # The agent's own prompt is the first user message; later ones are repair requests.
        prompt = next(m["content"] for m in body["messages"] if m["role"] == "user")
        if prompts is not None:
            prompts.append(prompt)
        if calls is not None:
            calls.append((agent_of(prompt), body["model"]))
        content = json.dumps(script(prompt))
        reply: dict[str, Any] = {"choices": [{"message": {"content": content}}]}
        if usage is not None:
            reply["usage"] = usage
        return httpx.Response(200, json=reply)

    config = load_config(config_path)
    return Gateway(
        config,
        cache=CompletionCache(tmp_path / "c.sqlite"),
        env={"NVIDIA_API_KEY": "k", "MODEL_API_KEY": "k", "OPENROUTER_API_KEY": "k"},
        transports={p: httpx.MockTransport(handler) for p in config.providers},
    )
