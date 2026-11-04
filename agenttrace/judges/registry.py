"""Versioned judge prompts (P6-T1): `judges/prompts/<name>_<version>.md`.

    template, version = load_prompt("node_judge")          # the current version
    load_prompt("node_judge", "v1")                         # a pinned one

A released prompt never changes: `prompts/VERSIONS.json` records each version's SHA-256,
and `check_versions()` (run by the tests) fails if a file no longer matches. To change a
prompt, add `<name>_v2.md`, register it, and point CURRENT at it; old scores keep the
`prompt_version` they were made with, so they stay interpretable.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

PROMPT_DIR = Path(__file__).resolve().parent / "prompts"
VERSIONS_PATH = PROMPT_DIR / "VERSIONS.json"
CURRENT: dict[str, str] = {"node_judge": "v1", "escalation": "v1"}


def prompt_id(name: str, version: str) -> str:
    """The id stored on every score, e.g. "node-judge-v1"."""
    return f"{name.replace('_', '-')}-{version}"


def _sha(path: Path) -> str:
    # Normalize line endings so a checkout on Windows hashes the same.
    text = path.read_text(encoding="utf-8").replace("\r\n", "\n")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_prompt(name: str, version: str | None = None) -> tuple[str, str]:
    """(template text, prompt id) for a registered prompt version."""
    version = version or CURRENT[name]
    path = PROMPT_DIR / f"{name}_{version}.md"
    registered = json.loads(VERSIONS_PATH.read_text(encoding="utf-8"))
    key = f"{name}_{version}"
    if key not in registered:
        raise KeyError(f"prompt {key} is not registered in {VERSIONS_PATH.name}")
    if _sha(path) != registered[key]:
        raise ValueError(f"{path.name} changed after release: add a new version instead")
    return path.read_text(encoding="utf-8"), prompt_id(name, version)


def check_versions() -> list[str]:
    """Problems with the registry: unregistered files, edited releases, missing files."""
    registered = json.loads(VERSIONS_PATH.read_text(encoding="utf-8"))
    files = {p.stem: p for p in PROMPT_DIR.glob("*.md")}
    problems = [f"{stem}.md is not registered" for stem in sorted(set(files) - set(registered))]
    for key, sha in sorted(registered.items()):
        if key not in files:
            problems.append(f"{key}.md is registered but missing")
        elif _sha(files[key]) != sha:
            problems.append(f"{key}.md changed after release (sha mismatch)")
    return problems


def register_all() -> dict[str, str]:
    """Write the registry from the files on disk (use only when adding a new version)."""
    registry = {p.stem: _sha(p) for p in sorted(PROMPT_DIR.glob("*.md"))}
    VERSIONS_PATH.write_text(json.dumps(registry, indent=1) + "\n", encoding="utf-8")
    return registry


if __name__ == "__main__":
    print(json.dumps(register_all(), indent=1))
