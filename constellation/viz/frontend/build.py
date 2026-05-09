"""Frontend build helper.

Runs ``pnpm install`` (or ``npm install`` as fallback) followed by
``pnpm build`` against the source tree at
``constellation/viz/frontend/``. Output lands at
``constellation/viz/static/<entry>/`` per the multi-entry Vite config.

Usage::

    python -m constellation.viz.frontend.build [--entry genome] [--manager pnpm|npm]

The release wheel ships the prebuilt static/ tree; this helper exists
for developers iterating on the SPA + for source-tarball installs that
need to populate static/ before the server can serve `/`.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


_FRONTEND_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _FRONTEND_DIR.parents[3]


def _resolve_manager(preference: str | None) -> str:
    """Pick the JS package manager. `pnpm` preferred (lockfile shipped
    upstream); falls back to `npm` when pnpm isn't on PATH."""
    if preference:
        if shutil.which(preference) is None:
            raise FileNotFoundError(
                f"requested package manager `{preference}` not found on PATH"
            )
        return preference
    for candidate in ("pnpm", "npm"):
        if shutil.which(candidate):
            return candidate
    raise FileNotFoundError(
        "neither pnpm nor npm found on PATH — install one to build the viz frontend"
    )


def build(entry: str = "genome", manager: str | None = None) -> int:
    """Run install + build. Returns the build subprocess exit code."""
    pkg_manager = _resolve_manager(manager)
    install_args = [pkg_manager, "install"]
    build_args = [pkg_manager, "run", "build"]

    print(f"[constellation.viz.frontend] using {pkg_manager} in {_FRONTEND_DIR}")
    rc = subprocess.call(install_args, cwd=_FRONTEND_DIR)
    if rc != 0:
        print(f"  install failed (exit {rc})", file=sys.stderr)
        return rc
    env = {"CONSTELLATION_VIZ_ENTRY": entry}
    rc = subprocess.call(
        build_args,
        cwd=_FRONTEND_DIR,
        env={**__import__("os").environ, **env},
    )
    if rc == 0:
        out_dir = _FRONTEND_DIR.parent / "static" / entry
        print(f"  built bundle at {out_dir}")
    return rc


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="constellation.viz.frontend.build")
    parser.add_argument(
        "--entry",
        default="genome",
        help="entry point to build (default: genome)",
    )
    parser.add_argument(
        "--manager",
        choices=("pnpm", "npm"),
        help="JS package manager to use (default: prefer pnpm, fall back to npm)",
    )
    args = parser.parse_args(argv)
    return build(entry=args.entry, manager=args.manager)


if __name__ == "__main__":
    raise SystemExit(main())
