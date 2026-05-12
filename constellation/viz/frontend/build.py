"""Frontend build helper.

Runs ``pnpm install`` (or ``npm install`` as fallback) followed by
``pnpm build`` against the source tree at
``constellation/viz/frontend/``. Output lands at
``constellation/viz/static/<entry>/`` per the multi-entry Vite config.

Usage::

    # build only — for local dev iteration
    python -m constellation.viz.frontend.build [--entry genome] [--manager pnpm|npm]

    # build + produce a shippable tarball + sha256 sidecar under
    # <repo_root>/dist/, suitable for `constellation viz install-frontend
    # --from <tarball>` on a machine without the JS toolchain
    python -m constellation.viz.frontend.build --pack

Release wheels (PR 1.6) ship the prebuilt ``static/`` tree directly.
This helper exists for two cases:

- Developers iterating on the frontend (`pnpm dev` is for hot-reload;
  ``build`` is for the production bundle).
- Building a release-style tarball on a workstation to ship to an HPC
  cluster where neither node nor open-network access is available.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import time
from collections.abc import Sequence
from pathlib import Path

import constellation as _constellation


_FRONTEND_DIR = Path(__file__).resolve().parent
_VIZ_DIR = _FRONTEND_DIR.parent  # constellation/viz/
# parents[2] = repo root (parents[0]=viz, parents[1]=constellation, parents[2]=repo).
# Latent bug from PR 1.5: this used parents[3], which silently put `dist/` one
# directory above the repo root because every test overrode `dist_dir`
# explicitly and never exercised the default. Surfaced when CI ran `--pack`
# without overrides — the tarball landed outside the workspace.
_REPO_ROOT = _FRONTEND_DIR.parents[2]
_DIST_DIR = _REPO_ROOT / "dist"
_BUNDLE_METADATA_NAME = "bundle.json"
_PACK_NAME_PREFIX = "constellation-viz-frontend"
_CHUNK_BYTES = 65_536


def _resolve_manager(preference: str | None) -> str:
    """Pick the JS package manager. ``pnpm`` preferred (lockfile shipped
    upstream); falls back to ``npm`` when pnpm isn't on PATH."""
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


def build(
    entry: str = "genome",
    manager: str | None = None,
    *,
    pack: bool = False,
) -> int:
    """Run install + build (+ optional pack).

    Returns the build subprocess exit code. When ``pack=True`` and the
    build succeeded, additionally produces a ``.tar.gz`` + ``.sha256``
    pair under ``<repo_root>/dist/`` — see ``pack_bundle``.
    """
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
        env={**os.environ, **env},
    )
    if rc != 0:
        return rc

    out_dir = _VIZ_DIR / "static" / entry
    print(f"  built bundle at {out_dir}")

    if pack:
        try:
            tarball, sidecar = pack_bundle(entry=entry)
        except FileNotFoundError as exc:
            print(f"  pack failed: {exc}", file=sys.stderr)
            return 1
        print(f"  packed tarball: {tarball}")
        print(f"  sha256 sidecar: {sidecar}")
    return rc


def pack_bundle(
    entry: str | Sequence[str] = "genome",
    *,
    static_root: Path | None = None,
    dist_dir: Path | None = None,
    version: str | None = None,
) -> tuple[Path, Path]:
    """Pack one or more built bundles into a single release tarball.

    The tarball has the layout documented in
    ``docs/plans/viz-and-dashboard.md``::

        constellation-viz-frontend-<version>/
        ├── bundle.json
        └── static/
            ├── <entry_a>/
            │   └── ...
            └── <entry_b>/   (when multiple entries are packed)
                └── ...

    Pass ``entry="genome"`` for the single-entry form (backwards-
    compatible) or ``entry=["genome", "dashboard"]`` to ship both
    bundles in one archive. The downstream installer's ``--entry``
    flag picks which subtree to extract; URL-fetch + sha256 paths are
    unchanged because there's still one tarball per release.

    ``version`` defaults to ``constellation.__version__``. Exposed as a
    public function so PR 1.6's release CI can call it directly without
    re-running ``pnpm build`` (the CI workflow runs build per entry,
    then packs once).

    Returns the (tarball, sidecar) Path pair. Raises ``FileNotFoundError``
    when any source ``static/<entry>/`` is missing (i.e. that entry's
    build didn't run).
    """
    entries: list[str] = [entry] if isinstance(entry, str) else list(entry)
    if not entries:
        raise ValueError("pack_bundle requires at least one entry")
    root = static_root or _VIZ_DIR / "static"
    for e in entries:
        source_dir = root / e
        if not source_dir.is_dir() or not any(source_dir.iterdir()):
            raise FileNotFoundError(
                f"no built bundle at {source_dir} — run the build first"
            )

    pkg_version = version or _constellation.__version__
    artifact_name = f"{_PACK_NAME_PREFIX}-{pkg_version}"
    out_dir = (dist_dir or _DIST_DIR).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    tarball = out_dir / f"{artifact_name}.tar.gz"
    sidecar = out_dir / f"{tarball.name}.sha256"

    metadata = {
        "constellation_version": pkg_version,
        # Single-entry tarballs keep the historical scalar "entry" key
        # for backwards compatibility with PR-1.5-era consumers. Multi-
        # entry tarballs use the list form "entries".
        **(
            {"entry": entries[0]}
            if len(entries) == 1
            else {"entries": entries}
        ),
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    metadata_bytes = (json.dumps(metadata, indent=2) + "\n").encode("utf-8")

    with tarfile.open(tarball, mode="w:gz") as tar:
        meta_info = tarfile.TarInfo(name=f"{artifact_name}/{_BUNDLE_METADATA_NAME}")
        meta_info.size = len(metadata_bytes)
        meta_info.mtime = int(time.time())
        tar.addfile(meta_info, fileobj=_BytesReader(metadata_bytes))

        for e in entries:
            # arcname rewrites <static_root>/<entry>/ → <artifact_name>/static/<entry>/
            tar.add(
                root / e,
                arcname=f"{artifact_name}/static/{e}",
                recursive=True,
            )

    digest = _sha256_of_file(tarball)
    sidecar.write_text(f"{digest}  {tarball.name}\n", encoding="utf-8")
    return tarball, sidecar


def _sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(_CHUNK_BYTES), b""):
            h.update(chunk)
    return h.hexdigest()


class _BytesReader:
    """Tiny BufferedIOBase-shaped wrapper around a `bytes` payload.

    `tarfile.addfile(info, fileobj=...)` reads via `.read(n)`; this is
    simpler than spinning up `io.BytesIO` and matches what the stdlib
    does internally for in-memory tar additions.
    """

    def __init__(self, payload: bytes) -> None:
        self._buf = payload
        self._pos = 0

    def read(self, n: int = -1) -> bytes:
        if n < 0 or n > len(self._buf) - self._pos:
            n = len(self._buf) - self._pos
        chunk = self._buf[self._pos : self._pos + n]
        self._pos += n
        return chunk


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="constellation.viz.frontend.build")
    parser.add_argument(
        "--entry",
        nargs="+",
        default=["genome"],
        metavar="NAME",
        help=(
            "entry point(s) to build. Default: 'genome'. Pass multiple "
            "entries to build them all (`--entry genome dashboard`); "
            "with --pack the result is a single tarball containing every "
            "built entry."
        ),
    )
    parser.add_argument(
        "--manager",
        choices=("pnpm", "npm"),
        help="JS package manager to use (default: prefer pnpm, fall back to npm)",
    )
    parser.add_argument(
        "--pack",
        action="store_true",
        help=(
            "after building, produce dist/constellation-viz-frontend-<version>"
            ".tar.gz + .sha256 sidecar containing every built entry for use "
            "with `constellation viz install-frontend --from <tarball>` on "
            "machines without a JS toolchain"
        ),
    )
    args = parser.parse_args(argv)

    # Build each requested entry; bail on first failure.
    for entry in args.entry:
        rc = build(entry=entry, manager=args.manager, pack=False)
        if rc != 0:
            return rc

    if args.pack:
        try:
            tarball, sidecar = pack_bundle(entry=args.entry)
        except FileNotFoundError as exc:
            print(f"  pack failed: {exc}", file=sys.stderr)
            return 1
        print(f"  packed tarball: {tarball}")
        print(f"  sha256 sidecar: {sidecar}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
