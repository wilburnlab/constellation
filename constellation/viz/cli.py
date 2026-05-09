"""CLI parser + handler for ``constellation viz <subcommand>``.

Subcommands:

- ``constellation viz genome --session DIR`` — focused IGV-style genome
  browser on the session's parquet outputs. Boots a local FastAPI server
  via uvicorn and (unless ``--no-browser``) opens the browser at the
  served URL.

The dashboard subcommand (``constellation dashboard``) lands in PR 2.
PR 1 reserves the parser slot but does not wire it.
"""

from __future__ import annotations

import argparse
import socket
from pathlib import Path


def build_parser(subs: argparse._SubParsersAction) -> None:
    """Mount the ``viz`` subtree on the parent ``constellation`` parser."""
    p_viz = subs.add_parser(
        "viz",
        help="Open a focused visualization (genome browser, ...)",
    )
    viz_subs = p_viz.add_subparsers(dest="viz_subcommand", required=True)

    p_genome = viz_subs.add_parser(
        "genome",
        help="IGV-style genome browser over a session's parquet outputs",
    )
    p_genome.add_argument(
        "--session",
        required=True,
        type=Path,
        help=(
            "Path to a session root containing reference + alignment "
            "outputs. The directory may include `session.toml` for "
            "explicit layout, or follow the standard `genome/`, "
            "`annotation/`, `S2_align/`, `S2_cluster/` convention."
        ),
    )
    p_genome.add_argument(
        "--host",
        default="127.0.0.1",
        help="bind host (default 127.0.0.1 — localhost only)",
    )
    p_genome.add_argument(
        "--port",
        type=int,
        default=0,
        help=(
            "bind port (default 0 = pick a free ephemeral port). The "
            "URL is printed to stdout on startup."
        ),
    )
    p_genome.add_argument(
        "--no-browser",
        action="store_true",
        help="don't auto-open a browser (just print the URL)",
    )
    p_genome.set_defaults(func=cmd_viz_genome)


def cmd_viz_genome(args: argparse.Namespace) -> int:
    """Handler — lazy-imports uvicorn + the FastAPI app factory.

    Heavy imports stay inside the handler so the top-level
    `constellation` CLI doesn't pay for them on unrelated subcommands
    (matches the `_cmd_doctor` pattern for thirdparty discovery).
    """
    from constellation.viz.server.app import create_app
    from constellation.viz.server.session import Session

    try:
        session = Session.from_root(args.session)
    except FileNotFoundError as exc:
        print(f"error: {exc}")
        return 2

    port = args.port if args.port != 0 else _free_port(args.host)
    url = f"http://{args.host}:{port}"
    print(f"constellation viz genome — session: {session.label}")
    print(f"  root: {session.root}")
    print(f"  url:  {url}")

    if not args.no_browser:
        # webbrowser.open is best-effort across Linux / macOS / WSL —
        # WSL falls through to wslview / cmd.exe /c start when present.
        import webbrowser

        try:
            webbrowser.open(url)
        except Exception:  # noqa: BLE001 — non-fatal; we still print the URL
            pass

    import uvicorn

    app = create_app(session)
    config = uvicorn.Config(app, host=args.host, port=port, log_level="info")
    server = uvicorn.Server(config)
    server.run()
    return 0


def _free_port(host: str) -> int:
    """Bind a temporary socket to discover an available port. The
    returned number is reused immediately by uvicorn — there's a tiny
    race window in theory but it's practically irrelevant for a
    localhost dev tool."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        return sock.getsockname()[1]
