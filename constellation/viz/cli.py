"""CLI parser + handler for ``constellation viz <subcommand>``.

Subcommands:

- ``constellation viz genome --session DIR`` — focused IGV-style genome
  browser on the session's parquet outputs. Boots a local FastAPI server
  via uvicorn and (unless ``--no-browser``) opens the browser at the
  served URL.
- ``constellation viz install-frontend --from <tarball>`` — extract a
  prebuilt frontend bundle into the package's ``static/<entry>/`` dir.
  Used by source-checkout / HPC installs where the JS toolchain isn't
  available (or doesn't work — e.g. OSC's TLS proxy refuses npm). See
  ``constellation.viz.install`` for the extraction implementation.

The dashboard subcommand (``constellation dashboard``) lands in PR 2.
PR 1 reserves the parser slot but does not wire it.
"""

from __future__ import annotations

import argparse
import socket
import sys
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

    p_install = viz_subs.add_parser(
        "install-frontend",
        help=(
            "Extract a prebuilt frontend bundle into the package's "
            "static/ dir (for source-checkout / HPC installs that can't "
            "run pnpm/npm). Produce the tarball via "
            "`python -m constellation.viz.frontend.build --pack`."
        ),
    )
    p_install.add_argument(
        "--from",
        dest="from_path",
        required=True,
        type=Path,
        help=(
            "path to a local `constellation-viz-frontend-<version>"
            ".tar.gz` produced by the build helper's --pack flag. The "
            "adjacent <tarball>.sha256 sidecar is read for integrity "
            "verification (skip with --no-verify)."
        ),
    )
    p_install.add_argument(
        "--entry",
        default="genome",
        help="bundle entry to install (default: genome)",
    )
    p_install.add_argument(
        "--force",
        action="store_true",
        help=(
            "replace any existing bundle at static/<entry>/. Without "
            "this, install refuses to overwrite a non-empty target."
        ),
    )
    p_install.add_argument(
        "--no-verify",
        action="store_true",
        help=(
            "skip sha256 verification against the sidecar (warning "
            "printed to stderr)"
        ),
    )
    p_install.set_defaults(func=cmd_viz_install_frontend)


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
        # Defer the browser open until uvicorn has actually bound the
        # port — otherwise the auto-launched tab races the server and
        # shows "unable to connect" on first load. We poll the bind
        # state on a background thread and call `webbrowser.open` once
        # the server is ready (or give up after a short window).
        _open_browser_when_ready(args.host, port, url)

    import uvicorn

    app = create_app(session)
    config = uvicorn.Config(app, host=args.host, port=port, log_level="info")
    server = uvicorn.Server(config)
    server.run()
    return 0


def _open_browser_when_ready(host: str, port: int, url: str) -> None:
    """Open the URL in the user's browser once the server is listening.

    Polls the bind state on a background thread (one TCP connect per
    50ms, up to ~10s) and calls `webbrowser.open` exactly once when the
    handshake succeeds. Non-fatal: any failure is swallowed and the
    URL is left for the user to copy from the startup banner.
    """
    import socket
    import threading
    import time
    import webbrowser

    def _wait_then_open() -> None:
        deadline = time.monotonic() + 10.0
        connect_host = host if host != "0.0.0.0" else "127.0.0.1"
        while time.monotonic() < deadline:
            try:
                with socket.create_connection((connect_host, port), timeout=0.2):
                    break
            except OSError:
                time.sleep(0.05)
        else:
            return
        try:
            webbrowser.open(url)
        except Exception:  # noqa: BLE001 — non-fatal
            pass

    threading.Thread(target=_wait_then_open, daemon=True).start()


def cmd_viz_install_frontend(args: argparse.Namespace) -> int:
    """Handler for ``constellation viz install-frontend --from <tarball>``.

    Lazy-imports ``constellation.viz.install`` so the top-level CLI
    doesn't pay for the import on unrelated subcommands. Surface
    ``InstallError`` messages verbatim — they're already formatted for
    the user.
    """
    from constellation.viz.install import (
        InstallError,
        install_frontend_from_tarball,
    )

    try:
        result = install_frontend_from_tarball(
            local_path=args.from_path,
            entry=args.entry,
            force=args.force,
            verify=not args.no_verify,
        )
    except InstallError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    version = result.bundle_metadata.get("constellation_version") or "unknown"
    print(
        f"installed constellation viz frontend ({result.entry}, "
        f"version {version}) at {result.dest_dir}"
    )
    return 0


def _free_port(host: str) -> int:
    """Bind a temporary socket to discover an available port. The
    returned number is reused immediately by uvicorn — there's a tiny
    race window in theory but it's practically irrelevant for a
    localhost dev tool."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        return sock.getsockname()[1]
