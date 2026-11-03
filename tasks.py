"""Cross-platform task runner (stands in for `make` where it isn't installed).

Usage: python tasks.py <setup|test|lint|format|serve|eval|smoke|retrieval|toy|clean|verify> [args]
"""

from __future__ import annotations

import subprocess
import sys
from collections.abc import Callable


def _run(*cmd: str) -> int:
    print(f"$ {' '.join(cmd)}", flush=True)
    return subprocess.call(list(cmd))


def setup() -> int:
    rc = _run("uv", "sync")
    if rc == 0:
        rc = _run("uv", "run", "pre-commit", "install")
    return rc


def test() -> int:
    return _run("uv", "run", "pytest")


def lint() -> int:
    for cmd in (
        ("uv", "run", "ruff", "check", "."),
        ("uv", "run", "ruff", "format", "--check", "."),
        ("uv", "run", "mypy"),
    ):
        rc = _run(*cmd)
        if rc != 0:
            return rc
    return 0


def format_() -> int:
    rc = _run("uv", "run", "ruff", "check", "--fix", ".")
    return rc or _run("uv", "run", "ruff", "format", ".")


def _not_yet(task: str, phase: str) -> int:
    print(f"`{task}` is not available yet (arrives in {phase}).")
    return 1


def serve() -> int:
    return _not_yet("serve", "P10-T1")


def eval_() -> int:
    return _not_yet("eval", "P9-T7")


def smoke() -> int:
    return _run("uv", "run", "python", "-m", "agenttrace.gateway.smoke")


def retrieval() -> int:
    return _run("uv", "run", "python", "-m", "agenttrace.workflow.kb.report")


def toy() -> int:
    return _run("uv", "run", "python", "-m", "agenttrace.workflow.toy")


def clean(*args: str) -> int:
    return _run("uv", "run", "python", "-m", "agenttrace.workflow.clean_eval", *args)


def verify(*args: str) -> int:
    return _run("uv", "run", "python", "-m", "agenttrace.scenarios.verify", *args)


# Tasks that forward extra command-line arguments (e.g. `clean --profile bolt`).
WITH_ARGS: dict[str, Callable[..., int]] = {"clean": clean, "verify": verify}

TASKS: dict[str, Callable[..., int]] = {
    "setup": setup,
    "test": test,
    "lint": lint,
    "format": format_,
    "serve": serve,
    "eval": eval_,
    "smoke": smoke,
    "retrieval": retrieval,
    "toy": toy,
    **WITH_ARGS,
}


def main(argv: list[str]) -> int:
    if not argv or argv[0] not in TASKS or (len(argv) > 1 and argv[0] not in WITH_ARGS):
        print(f"usage: python tasks.py <{'|'.join(TASKS)}> [args for: {', '.join(WITH_ARGS)}]")
        return 2
    return TASKS[argv[0]](*argv[1:])


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
