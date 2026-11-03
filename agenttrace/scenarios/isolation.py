"""Ground-truth isolation check (P4-T3): who can reach the answers?

    uv run python -m agenttrace.scenarios.isolation       # report; exit 1 on any violation

Reads the package's source (nothing is imported or run) and builds its import graph:
`import x`, `from x import y`, relative imports, imports inside functions, and
`importlib.import_module("...")` / `__import__("...")` with a literal name. Then:

1. Only the harness (ALLOWED) may import a truth module (TRUTH) directly.
2. No module outside the harness may *reach* a truth module through any chain of
   imports. So diagnosis importing a helper that imports the dataset loader fails too.
3. Outside the harness, code may not name the dataset file ("scenarios.json"), so it
   cannot read the answers from disk either.

PROTECTED are the packages the check exists for (the debugger: diagnosis,
explainability, judges). They must exist, so the check can never pass vacuously.
"""

from __future__ import annotations

import ast
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

PACKAGE = "agenttrace"
PACKAGE_ROOT = Path(__file__).resolve().parents[1]

# Modules (and packages: every submodule too) that hold or hand out ground truth.
TRUTH = ("agenttrace.schemas.ground_truth", "agenttrace.scenarios")
# The evaluation harness: builds scenarios, injects faults, scores outcomes.
ALLOWED = (
    "agenttrace.scenarios",
    "agenttrace.faults",
    "agenttrace.evaluation",
    "agenttrace.workflow.clean_eval",
    "agenttrace.cli",
)
# The debugger: must never see the answers.
PROTECTED = ("agenttrace.diagnosis", "agenttrace.explainability", "agenttrace.judges")
DATASET_NAMES = ("scenarios.json",)


@dataclass(frozen=True)
class Violation:
    module: str
    rule: str  # "direct_import" | "reaches_truth" | "names_dataset"
    detail: str

    def __str__(self) -> str:
        return f"[{self.rule}] {self.module}: {self.detail}"


def _within(module: str, prefixes: Iterable[str]) -> bool:
    return any(module == p or module.startswith(p + ".") for p in prefixes)


def _module_name(path: Path, root: Path, package: str) -> str:
    parts = list(path.relative_to(root).with_suffix("").parts)
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join([package, *parts])


def _is_docstring(node: ast.AST, parent: ast.AST | None) -> bool:
    return isinstance(parent, ast.Expr) and isinstance(node, ast.Constant)


@dataclass
class _Module:
    name: str
    is_package: bool
    imports: set[str]
    dataset_mentions: list[str]


def _parse(path: Path, name: str, package: str) -> _Module:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    is_package = path.name == "__init__.py"
    base = name if is_package else name.rsplit(".", 1)[0]
    imports: set[str] = set()
    mentions: list[str] = []
    parents: dict[ast.AST, ast.AST] = {
        child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)
    }
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports |= {alias.name for alias in node.names}
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                anchor = base.split(".")
                anchor = anchor[: len(anchor) - (node.level - 1)]
                source = ".".join([*anchor, node.module] if node.module else anchor)
            else:
                source = node.module or ""
            imports.add(source)
            # `from pkg import mod` imports the submodule pkg.mod.
            imports |= {f"{source}.{alias.name}" for alias in node.names}
        elif isinstance(node, ast.Call) and node.args:
            func = node.func
            called = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
            first = node.args[0]
            if (
                called in ("import_module", "__import__")
                and isinstance(first, ast.Constant)
                and isinstance(first.value, str)
            ):
                imports.add(first.value)
        if (
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and not _is_docstring(node, parents.get(node))
            and any(n in node.value for n in DATASET_NAMES)
        ):
            mentions.append(node.value)
    return _Module(name, is_package, {i for i in imports if i.startswith(package)}, mentions)


def build_import_graph(root: Path = PACKAGE_ROOT, package: str = PACKAGE) -> dict[str, _Module]:
    modules: dict[str, _Module] = {}
    for path in sorted(root.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        name = _module_name(path, root, package)
        modules[name] = _parse(path, name, package)
    return modules


def _resolve(target: str, modules: dict[str, _Module]) -> list[str]:
    """The modules an import of `target` actually runs: it and its parent packages."""
    parts = target.split(".")
    return [m for m in (".".join(parts[:i]) for i in range(1, len(parts) + 1)) if m in modules]


def _chain_to_truth(start: str, modules: dict[str, _Module], truth: Iterable[str]) -> list[str]:
    """Shortest import chain from `start` to a truth module ([] if none)."""
    truth = tuple(truth)
    seen = {start}
    queue: list[list[str]] = [[start]]
    while queue:
        path = queue.pop(0)
        for target in sorted(modules[path[-1]].imports):
            for mod in _resolve(target, modules):
                if mod in seen:
                    continue
                seen.add(mod)
                if _within(mod, truth):
                    return [*path, mod]
                queue.append([*path, mod])
    return []


def find_violations(
    root: Path = PACKAGE_ROOT,
    package: str = PACKAGE,
    *,
    truth: Iterable[str] = TRUTH,
    allowed: Iterable[str] = ALLOWED,
    protected: Iterable[str] = PROTECTED,
) -> list[Violation]:
    truth, allowed = tuple(truth), tuple(allowed)
    modules = build_import_graph(root, package)
    found: list[Violation] = []
    for missing in (p for p in protected if p not in modules):
        found.append(Violation(missing, "missing_package", "protected package does not exist"))
    for name, mod in sorted(modules.items()):
        if _within(name, allowed) or _within(name, truth):
            continue
        direct = sorted(i for i in mod.imports if _within(i, truth))
        if direct:
            found.append(Violation(name, "direct_import", f"imports {', '.join(direct)}"))
            continue
        chain = _chain_to_truth(name, modules, truth)
        if chain:
            found.append(Violation(name, "reaches_truth", " -> ".join(chain)))
        for text in mod.dataset_mentions:
            found.append(Violation(name, "names_dataset", f"mentions {text!r}"))
    return found


def main() -> int:
    modules = build_import_graph()
    violations = find_violations()
    harness = sorted(m for m in modules if _within(m, ALLOWED) or _within(m, TRUTH))
    protected = sorted(m for m in modules if _within(m, PROTECTED))
    print(f"modules scanned        : {len(modules)}")
    print(f"harness (may see truth): {len(harness)}  ({', '.join(ALLOWED)})")
    print(f"protected modules      : {len(protected)}  ({', '.join(PROTECTED)})")
    if not violations:
        print(
            "violations             : none. No module outside the harness can reach ground truth."
        )
        return 0
    print(f"violations             : {len(violations)}")
    for v in violations:
        print(f"  {v}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
