"""Rule 3: all LLM calls go through agenttrace/gateway.

No other module may import HTTP or LLM client libraries.
"""

from __future__ import annotations

import ast
from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[1] / "agenttrace"
FORBIDDEN = {"httpx", "requests", "openai", "anthropic", "aiohttp", "urllib3"}
# The API server (P10) and SDK exporter may need HTTP later; allow-list them explicitly then.
ALLOWED_DIRS = {PACKAGE / "gateway"}


def _imported_roots(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            roots.add(node.module.split(".")[0])
    return roots


def test_only_gateway_imports_http_clients() -> None:
    offenders = []
    for path in PACKAGE.rglob("*.py"):
        if any(path.is_relative_to(d) for d in ALLOWED_DIRS):
            continue
        bad = _imported_roots(path) & FORBIDDEN
        if bad:
            offenders.append(f"{path.relative_to(PACKAGE.parent)}: {sorted(bad)}")
    assert not offenders, "LLM/HTTP clients imported outside gateway/:\n" + "\n".join(offenders)
