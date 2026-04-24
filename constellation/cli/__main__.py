"""`constellation <subcommand>` dispatcher + per-domain shims.

Usage:
    constellation --help
    constellation doctor

Domain subcommands (mzpeak, koina, pod5, structure) are wired as the
underlying modules are ported. Thin legacy binaries (``mzpeak``,
``koina-library``) forward to the same handlers via the ``main_*``
entry points declared in ``pyproject.toml``.
"""

from __future__ import annotations

import argparse
import sys
from typing import Callable

# Import adapter modules for their side-effect of registering ToolSpecs.
# Each adapter calls constellation.thirdparty.registry.register(...) at
# import time so the `doctor` subcommand sees every known tool.
from constellation.thirdparty import encyclopedia  # noqa: F401
from constellation.thirdparty.registry import registered, try_find


def _cmd_doctor(_args: argparse.Namespace) -> int:
    """Print a tool-status table."""
    tools = registered()
    if not tools:
        print("no third-party tools registered")
        return 0

    rows: list[tuple[str, str, str, str]] = []
    for spec in tools:
        handle = try_find(spec.name)
        if handle is None:
            status = "not found"
            where = f"(set {spec.env_var} or run {spec.install_script or '—'})"
            version = "—"
        else:
            status = "ok"
            where = f"{handle.source}: {handle.path}"
            version = handle.version or "—"
        rows.append((spec.name, status, version, where))

    name_w = max(len(r[0]) for r in rows + [("tool", "", "", "")])
    status_w = max(len(r[1]) for r in rows + [("", "status", "", "")])
    ver_w = max(len(r[2]) for r in rows + [("", "", "version", "")])

    header = f"{'tool':<{name_w}}  {'status':<{status_w}}  {'version':<{ver_w}}  location"
    print(header)
    print("-" * len(header))
    for name, status, version, where in rows:
        print(f"{name:<{name_w}}  {status:<{status_w}}  {version:<{ver_w}}  {where}")
    # exit nonzero if any tool is missing — lets CI gate on it
    return 0 if all(r[1] == "ok" for r in rows) else 1


def _cmd_not_wired(name: str) -> Callable[[argparse.Namespace], int]:
    def handler(_args: argparse.Namespace) -> int:
        print(
            f"subcommand `{name}` is scaffolded but not implemented yet. "
            "See the 90-day roadmap in CLAUDE.md.",
            file=sys.stderr,
        )
        return 2

    return handler


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="constellation")
    subs = parser.add_subparsers(dest="subcommand", required=True)

    p_doctor = subs.add_parser("doctor", help="Report third-party tool status")
    p_doctor.set_defaults(func=_cmd_doctor)

    # Placeholders so `--help` advertises the intended surface. Wire as
    # the underlying modules are ported.
    for name, summary in [
        ("mzpeak", "Convert raw MS files to Parquet-backed mzpeak (TODO)"),
        ("koina", "Build spectral libraries via Koina (TODO)"),
        ("pod5", "Ingest POD5 signal to Parquet (TODO)"),
        ("structure", "Prepare structures for MD (TODO)"),
    ]:
        p = subs.add_parser(name, help=summary)
        p.set_defaults(func=_cmd_not_wired(name))

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


# Thin legacy shims — `mzpeak`, `koina-library`, etc. Each forces the
# subcommand so the Cartographer muscle memory still works.
def main_mzpeak(argv: list[str] | None = None) -> int:
    raw = list(sys.argv[1:] if argv is None else argv)
    return main(["mzpeak", *raw])


def main_koina(argv: list[str] | None = None) -> int:
    raw = list(sys.argv[1:] if argv is None else argv)
    return main(["koina", *raw])


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
