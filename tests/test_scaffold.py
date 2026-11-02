import importlib

import pytest

from agenttrace import __version__, cli

SUBPACKAGES = [
    "schemas",
    "gateway",
    "tracing",
    "store",
    "graph",
    "workflow",
    "workflow.agents",
    "workflow.tools",
    "workflow.kb",
    "faults",
    "scenarios",
    "replay",
    "diagnosis",
    "explainability",
    "judges",
    "evaluation",
    "api",
]


@pytest.mark.parametrize("name", SUBPACKAGES)
def test_subpackage_imports(name: str) -> None:
    importlib.import_module(f"agenttrace.{name}")


def test_version() -> None:
    assert __version__ == "0.1.0"


def test_cli_rejects_unknown_command() -> None:
    with pytest.raises(SystemExit):
        cli.main(["nope"])


def test_cli_stub_returns_nonzero() -> None:
    assert cli.main(["diagnose"]) == 1  # `run` is real since P2-T3
