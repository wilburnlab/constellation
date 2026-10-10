"""Import boundary of the viz layer.

The rule (see the project ``CLAUDE.md`` and ``constellation/viz/CLAUDE.md``):

- The viz **core** — everything under ``constellation/viz/`` outside
  ``modalities/`` — imports no domain module. It may import
  ``constellation.viz.*`` and ``constellation.core.*``.
- A **modality** package ``viz/modalities/<name>/`` may import from a
  domain module, but only a declared, pure-function surface:

  * the name is listed as ``(module, name)`` in that package's
    ``DOMAIN_IMPORTS``;
  * the import is a ``from <module> import <name>`` inside a function
    body, so ``import constellation.viz`` loads no domain module and
    works on a base install;
  * the imported object is a function or an exception class — no
    upstream state, no classes to instantiate or subclass.

This used to be a convention ("viz never imports a domain module") that
the code had quietly outgrown. It is checked here statically, by walking
the AST of every viz source file; ``tests/test_viz_base_install.py`` is
the runtime complement.
"""

from __future__ import annotations

import ast
import importlib
import inspect
from pathlib import Path

import constellation.viz

VIZ_ROOT = Path(constellation.viz.__file__).resolve().parent
MODALITIES_ROOT = VIZ_ROOT / "modalities"

#: ``constellation.<name>`` subpackages any viz file may import.
_ALWAYS_ALLOWED = {"viz", "core", "_version"}

#: The one sanctioned import of the CLI: the introspection endpoint walks
#: the production argparse tree to build the dashboard's forms.
_CLI_EXCEPTION = (
    Path("server/endpoints/cli_schema.py"),
    "constellation.cli.__main__",
    "_build_parser",
)


def _source_files() -> list[Path]:
    return sorted(
        p
        for p in VIZ_ROOT.rglob("*.py")
        if "node_modules" not in p.parts and "static" not in p.parts
    )


def _modality_of(path: Path) -> str | None:
    """Name of the modality package ``path`` belongs to, if any."""
    try:
        rel = path.relative_to(MODALITIES_ROOT)
    except ValueError:
        return None
    return rel.parts[0] if len(rel.parts) > 1 else None


def _declared(modality: str) -> set[tuple[str, str]]:
    """Read ``DOMAIN_IMPORTS`` from a modality's ``__init__`` without
    importing it, so a typo there cannot hide behind an import error."""
    init = MODALITIES_ROOT / modality / "__init__.py"
    for node in ast.parse(init.read_text()).body:
        targets = (
            node.targets
            if isinstance(node, ast.Assign)
            else [node.target]
            if isinstance(node, ast.AnnAssign)
            else []
        )
        if any(isinstance(t, ast.Name) and t.id == "DOMAIN_IMPORTS" for t in targets):
            value = ast.literal_eval(node.value)
            return {(str(m), str(n)) for m, n in value}
    return set()


def _resolve(node: ast.ImportFrom, path: Path) -> str:
    """Absolute dotted module for an ``ImportFrom``, resolving relative
    imports against the file's package."""
    if node.level == 0:
        return node.module or ""
    package = ("constellation.viz", *path.relative_to(VIZ_ROOT).parent.parts)
    base = package[: len(package) - (node.level - 1)]
    return ".".join((*base, node.module)) if node.module else ".".join(base)


def _imports(path: Path):
    """Yield ``(module, name | None, lineno, in_function, is_from)`` for
    every import of a ``constellation`` module in ``path``."""
    tree = ast.parse(path.read_text())
    in_function: dict[ast.AST, bool] = {tree: False}
    for parent in ast.walk(tree):
        inside = in_function.get(parent, False) or isinstance(
            parent, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)
        )
        for child in ast.iter_child_nodes(parent):
            in_function[child] = inside
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] == "constellation":
                    yield alias.name, None, node.lineno, in_function[node], False
        elif isinstance(node, ast.ImportFrom):
            module = _resolve(node, path)
            if module.split(".")[0] == "constellation":
                for alias in node.names:
                    yield module, alias.name, node.lineno, in_function[node], True


def _subpackage(module: str) -> str | None:
    parts = module.split(".")
    return parts[1] if len(parts) > 1 else None


def _violations() -> list[str]:
    problems: list[str] = []
    for path in _source_files():
        rel = path.relative_to(VIZ_ROOT)
        modality = _modality_of(path)
        declared = _declared(modality) if modality else set()
        for module, name, lineno, in_function, is_from in _imports(path):
            sub = _subpackage(module)
            # `import constellation` / `from constellation import __version__`
            if sub is None and name not in _all_subpackages():
                continue
            target = sub if sub is not None else name
            if target in _ALWAYS_ALLOWED:
                continue
            where = f"{rel}:{lineno}"
            if (rel, module, name) == _CLI_EXCEPTION:
                continue
            if modality is None:
                problems.append(
                    f"{where}: viz core imports {module}"
                    f"{'.' + name if name else ''}; only a modality package "
                    f"(viz/modalities/<name>/) may import a domain module"
                )
                continue
            if not is_from or name is None:
                problems.append(
                    f"{where}: use `from {module} import <name>` so the "
                    f"imported surface is explicit"
                )
                continue
            if not in_function:
                problems.append(
                    f"{where}: {module}.{name} is imported at module level; "
                    f"move it inside the function that uses it"
                )
            if (module, name) not in declared:
                problems.append(
                    f"{where}: ({module!r}, {name!r}) is not declared in "
                    f"viz/modalities/{modality}/__init__.py DOMAIN_IMPORTS"
                )
    return problems


def _all_subpackages() -> set[str]:
    root = VIZ_ROOT.parent
    return {p.name for p in root.iterdir() if (p / "__init__.py").exists()}


def _modalities() -> list[str]:
    return sorted(
        p.name
        for p in MODALITIES_ROOT.iterdir()
        if (p / "__init__.py").exists()
    )


def test_viz_imports_stay_inside_the_boundary() -> None:
    assert _violations() == []


def test_declared_domain_imports_are_used_and_pure() -> None:
    """Every ``DOMAIN_IMPORTS`` entry is actually imported by its modality
    (no stale grants), and names a function or an exception class."""
    for modality in _modalities():
        declared = _declared(modality)
        used = {
            (module, name)
            for path in _source_files()
            if _modality_of(path) == modality
            for module, name, *_ in _imports(path)
            if _subpackage(module) not in _ALWAYS_ALLOWED
        }
        assert declared <= used, (
            f"{modality}: declared but never imported: {sorted(declared - used)}"
        )
        for module, name in sorted(declared):
            obj = getattr(importlib.import_module(module), name)
            is_exception = isinstance(obj, type) and issubclass(obj, BaseException)
            assert inspect.isfunction(obj) or is_exception, (
                f"{modality}: {module}.{name} is a {type(obj).__name__}; the "
                f"boundary admits functions and exception classes only"
            )


def test_no_dynamic_imports_in_viz() -> None:
    """``importlib.import_module`` / ``__import__`` would walk straight
    past the static check above."""
    offenders: list[str] = []
    for path in _source_files():
        for node in ast.walk(ast.parse(path.read_text())):
            if not isinstance(node, ast.Call):
                continue
            callee = node.func
            name = getattr(callee, "attr", None) or getattr(callee, "id", None)
            if name in ("import_module", "__import__"):
                offenders.append(f"{path.relative_to(VIZ_ROOT)}:{node.lineno}")
    assert offenders == []


def test_the_check_catches_a_violation(tmp_path: Path, monkeypatch) -> None:
    """Guard the guard: a core file importing a domain module, and a
    modality importing an undeclared name at module level, are both
    reported."""
    fake_viz = tmp_path / "viz"
    (fake_viz / "server").mkdir(parents=True)
    (fake_viz / "modalities" / "toy").mkdir(parents=True)
    (fake_viz / "server" / "leak.py").write_text(
        "def f():\n    from constellation.sequencing.reference.handle import resolve\n"
    )
    (fake_viz / "modalities" / "toy" / "__init__.py").write_text(
        'DOMAIN_IMPORTS = (("constellation.massspec.peptide.mz", "precursor_mz"),)\n'
    )
    (fake_viz / "modalities" / "toy" / "kernel.py").write_text(
        "from constellation.massspec.peptide.mz import precursor_mz\n"
        "def g():\n    from constellation.massspec.peptide.ions import fragment_ladder\n"
    )
    module = importlib.import_module(__name__)
    monkeypatch.setattr(module, "VIZ_ROOT", fake_viz)
    monkeypatch.setattr(module, "MODALITIES_ROOT", fake_viz / "modalities")
    monkeypatch.setattr(module, "_all_subpackages", lambda: {"sequencing", "massspec", "viz", "core"})

    problems = _violations()
    assert any("server/leak.py:2: viz core imports" in p for p in problems)
    assert any("kernel.py:1" in p and "module level" in p for p in problems)
    assert any("kernel.py:3" in p and "not declared" in p for p in problems)
    assert len(problems) == 3
