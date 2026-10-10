"""Base-install import guard for the viz layer.

``constellation/cli/__main__.py`` imports ``constellation.viz.cli`` while
building the parser for *every* subcommand, and the release workflow runs
``python -m constellation.viz.frontend.build --pack`` from a base
``pip install -e .`` — no ``[viz]`` extras. CI installs ``.[viz,ms]``, so
nothing else exercises that path before a tag is pushed.

This test blocks the extras in a subprocess and checks that the modules
on that path still import, and that importing the viz layer does not
eagerly pull in a domain module (the viz → domain imports are all
function-level; see ``constellation/viz/CLAUDE.md``).
"""

from __future__ import annotations

import subprocess
import sys
import textwrap


# Everything the `[viz]` extra brings in that viz modules import directly.
_BLOCKED = ("fastapi", "pydantic", "starlette", "uvicorn", "datashader")

# Reached by `import constellation.viz`, by the CLI parser build, and by
# the release workflow's frontend build step.
_VIZ_MODULES = (
    "constellation.viz",
    "constellation.viz.cli",
    "constellation.viz.frontend.build",
)

_DOMAIN_PACKAGES = (
    "sequencing",
    "massspec",
    "codon",
    "structure",
    "nmr",
    "chromatography",
    "electrophoresis",
)


def test_viz_imports_without_viz_extras() -> None:
    script = textwrap.dedent(
        f"""
        import importlib
        import sys

        # `None` in sys.modules makes `import <name>` raise ImportError,
        # which is what a base install looks like to the importer.
        for name in {_BLOCKED!r}:
            sys.modules[name] = None

        for name in {_VIZ_MODULES!r}:
            importlib.import_module(name)

        domains = {_DOMAIN_PACKAGES!r}
        loaded = sorted(
            m for m in sys.modules
            if m.startswith("constellation.")
            and m.split(".")[1] in domains
        )
        if loaded:
            raise SystemExit(
                "viz import eagerly loaded domain modules: " + ", ".join(loaded)
            )

        # The dispatcher builds its parser (and so imports viz.cli) for
        # every command, `doctor` included.
        cli = importlib.import_module("constellation.cli.__main__")
        cli._build_parser()
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr or result.stdout
