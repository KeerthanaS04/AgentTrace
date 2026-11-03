"""P4-T3: a test fails if any forbidden path to ground truth exists."""

from __future__ import annotations

from pathlib import Path

import pytest

from agenttrace.scenarios import isolation
from agenttrace.scenarios.isolation import PROTECTED, build_import_graph, find_violations

TRUTH = ("pkg.schemas.ground_truth", "pkg.scenarios")
ALLOWED = ("pkg.scenarios", "pkg.evaluation")
PROT = ("pkg.diagnosis", "pkg.explainability", "pkg.judges")


def make_pkg(tmp_path: Path, files: dict[str, str]) -> Path:
    """A throwaway package `pkg` with the protected packages and a truth module."""
    base = {
        "__init__.py": "",
        "schemas/__init__.py": "",
        "schemas/ground_truth.py": "ANSWERS = {}\n",
        "scenarios/__init__.py": "",
        "scenarios/ground_truth.py": "from pkg.schemas.ground_truth import ANSWERS\n",
        "diagnosis/__init__.py": "",
        "explainability/__init__.py": "",
        "judges/__init__.py": "",
        "evaluation/__init__.py": "",
        "util.py": "def helper():\n    return 1\n",
    }
    root = tmp_path / "pkg"
    for rel, text in {**base, **files}.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    return root


def check(tmp_path: Path, files: dict[str, str]) -> list[isolation.Violation]:
    root = make_pkg(tmp_path, files)
    return find_violations(root, "pkg", truth=TRUTH, allowed=ALLOWED, protected=PROT)


# ---------------------------------------------------------------- the done-when


def test_real_package_has_no_path_to_ground_truth() -> None:
    violations = find_violations()
    assert violations == [], "\n".join(str(v) for v in violations)


def test_protected_packages_exist_so_the_check_is_not_vacuous() -> None:
    modules = build_import_graph()
    assert all(p in modules for p in PROTECTED)


def test_clean_synthetic_package_passes(tmp_path: Path) -> None:
    assert check(tmp_path, {"diagnosis/rca.py": "from pkg.util import helper\n"}) == []


# ---------------------------------------------------------------- each kind of leak fails


@pytest.mark.parametrize(
    ("files", "module", "rule"),
    [
        (
            {"diagnosis/rca.py": "from pkg.schemas.ground_truth import ANSWERS\n"},
            "pkg.diagnosis.rca",
            "direct_import",
        ),
        (
            {"diagnosis/rca.py": "from ..schemas import ground_truth\n"},  # relative
            "pkg.diagnosis.rca",
            "direct_import",
        ),
        (
            {
                "judges/j.py": (
                    "import importlib\nM = importlib.import_module('pkg.scenarios.ground_truth')\n"
                )
            },
            "pkg.judges.j",
            "direct_import",
        ),
        (
            {
                "explainability/x.py": (
                    "def f():\n"
                    "    from pkg.scenarios import ground_truth\n"
                    "    return ground_truth\n"
                )
            },
            "pkg.explainability.x",
            "direct_import",
        ),
        (
            # The system under test is held to the rule too (the old Policy Agent leak).
            {
                "workflow/__init__.py": "",
                "workflow/policy.py": "from pkg.schemas.ground_truth import ANSWERS\n",
            },
            "pkg.workflow.policy",
            "direct_import",
        ),
        (
            {"diagnosis/rca.py": "PATH = 'data/scenarios/scenarios.json'\n"},
            "pkg.diagnosis.rca",
            "names_dataset",
        ),
    ],
    ids=["direct", "relative", "importlib", "inside-function", "workflow", "reads-file"],
)
def test_each_kind_of_leak_is_caught(
    tmp_path: Path, files: dict[str, str], module: str, rule: str
) -> None:
    found = check(tmp_path, files)
    assert any(v.module == module and v.rule == rule for v in found), found


def test_indirect_leak_through_a_helper_is_caught(tmp_path: Path) -> None:
    found = check(
        tmp_path,
        {
            "diagnosis/rca.py": "from pkg.util2 import load\n",
            "util2.py": (
                "from pkg.scenarios.ground_truth import ANSWERS\n\n"
                "def load():\n"
                "    return ANSWERS\n"
            ),
        },
    )
    rca = next(v for v in found if v.module == "pkg.diagnosis.rca")
    assert rca.rule == "reaches_truth"
    assert rca.detail.startswith("pkg.diagnosis.rca -> pkg.util2 -> pkg.scenarios")
    # The helper itself is flagged too: it is not part of the harness.
    assert any(v.module == "pkg.util2" and v.rule == "direct_import" for v in found)


def test_leak_through_a_harness_module_is_caught(tmp_path: Path) -> None:
    """Harness modules may see truth, but the debugger may not import them."""
    found = check(
        tmp_path,
        {
            "evaluation/score.py": "from pkg.scenarios.ground_truth import ANSWERS\n",
            "diagnosis/rca.py": "from pkg.evaluation.score import ANSWERS\n",
        },
    )
    assert [(v.module, v.rule) for v in found] == [("pkg.diagnosis.rca", "reaches_truth")]


def test_harness_may_use_truth_and_docstrings_may_name_it(tmp_path: Path) -> None:
    files = {
        "evaluation/score.py": "from pkg.scenarios.ground_truth import ANSWERS\n",
        "diagnosis/rca.py": '"""Never reads scenarios.json."""\n',
    }
    assert check(tmp_path, files) == []


def test_missing_protected_package_fails(tmp_path: Path) -> None:
    root = make_pkg(tmp_path, {})
    (root / "judges" / "__init__.py").unlink()
    (root / "judges").rmdir()
    found = find_violations(root, "pkg", truth=TRUTH, allowed=ALLOWED, protected=PROT)
    assert [(v.module, v.rule) for v in found] == [("pkg.judges", "missing_package")]


# ---------------------------------------------------------------- the access module and CLI


def test_ground_truth_module_serves_the_dataset() -> None:
    from agenttrace.scenarios.ground_truth import load_dataset, scenario, truth_for

    first = load_dataset()[0]
    assert scenario(first.scenario_id) == first
    assert truth_for(first.scenario_id) == first.ground_truth
    with pytest.raises(KeyError, match="no scenario"):
        truth_for("sc-999")


def test_cli_report(capsys: pytest.CaptureFixture[str]) -> None:
    assert isolation.main() == 0
    out = capsys.readouterr().out
    assert "violations             : none" in out and "agenttrace.diagnosis" in out
