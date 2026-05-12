"""Install the prebuilt frontend bundle from a local tarball or release URL.

Companion to ``python -m constellation.viz.frontend.build --pack`` which
produces a bundle tarball + ``.sha256`` sidecar suitable for shipping to a
machine where the JS toolchain isn't available (HPC clusters, restricted
networks). Two install paths are supported:

* **Local tarball** (``install_frontend_from_tarball``) — the PR 1.5
  workflow. Build on a workstation → ``scp`` to the target → install.
* **GitHub Release URL** (``install_frontend_from_url`` — PR 1.6) — the
  default for source-checkout users. Fetches the release asset matching
  ``constellation.__version__`` (or an explicit ``--version``), caches
  it under ``~/.cache/constellation/viz-frontend/``, then dispatches
  into the same extract/verify pipeline as the local-tarball path.

Stdlib-only: ``tarfile``, ``hashlib``, ``shutil``, ``urllib.request``,
``pathlib``, ``json``. No third-party deps; runs under the base
``constellation`` install without the ``[viz]`` extras (the extras gate
``fastapi`` / ``uvicorn`` / ``datashader`` for the serve path, not the
install path).
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
import tarfile
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any


_FRONTEND_PKG_DIR = Path(__file__).resolve().parent
_DEFAULT_STATIC_ROOT = _FRONTEND_PKG_DIR / "static"

_BUNDLE_METADATA_NAME = "bundle.json"
_SHA256_SUFFIX = ".sha256"
_CHUNK_BYTES = 65_536

_PACK_NAME_PREFIX = "constellation-viz-frontend"
_DEFAULT_GITHUB_REPO = "wilburnlab/constellation"
_DEFAULT_CACHE_DIR = Path.home() / ".cache" / "constellation" / "viz-frontend"
_USER_AGENT = "constellation-viz-installer"
# URL host/scheme are split into constants so tests can monkeypatch them
# at a stdlib HTTP server (no live github.com round-trip in CI / dev).
_RELEASE_BASE_SCHEME = "https"
_RELEASE_BASE_HOST = "github.com"


class InstallError(RuntimeError):
    """Raised on install-frontend failure with a user-facing message.

    The CLI handler catches and exits non-zero; tests catch and assert
    the message text. The exception body is human-readable and may
    contain multiple lines (the CLI prints it verbatim, so it stays in
    one place — no special formatting required at the call sites).
    """


@dataclass(frozen=True)
class InstallResult:
    """What ``install_frontend_from_tarball`` returns on success.

    ``dest_dir`` is the populated ``<dest_root>/<entry>/`` directory.
    ``bundle_metadata`` is the parsed ``bundle.json`` payload from the
    tarball (empty dict when the tarball didn't ship one — older
    builds, hand-rolled tarballs).
    """

    dest_dir: Path
    entry: str
    bundle_metadata: dict[str, Any]


def install_frontend_from_tarball(
    *,
    local_path: Path | str,
    entry: str = "genome",
    force: bool = False,
    verify: bool = True,
    dest_root: Path | str | None = None,
) -> InstallResult:
    """Install a prebuilt frontend bundle from a local tarball.

    Parameters
    ----------
    local_path
        Path to the ``.tar.gz`` produced by ``--pack``. Must exist; the
        adjacent ``<local_path>.sha256`` sidecar is required when
        ``verify=True``.
    entry
        Vite entry name to install (default ``"genome"``). Must match
        the entry the tarball was packed for; the function looks for
        ``static/<entry>/`` inside the tarball.
    force
        When the destination directory is non-empty, refuse unless
        ``force=True`` (in which case the existing tree is removed
        before extraction).
    verify
        When ``True`` (default), require + check the ``.sha256`` sidecar
        next to the tarball. When ``False``, skip the check and print a
        warning to stderr.
    dest_root
        Override for the static-root directory. Defaults to
        ``constellation/viz/static/`` inside the installed package. Tests
        pass an explicit path to keep the working tree clean.

    Returns
    -------
    InstallResult
        On success. Raises ``InstallError`` on any failure (missing
        tarball, sha mismatch, target non-empty without ``force``, etc.).
    """
    tarball_path = Path(local_path).expanduser().resolve()
    if not tarball_path.is_file():
        raise InstallError(f"tarball not found: {tarball_path}")

    if verify:
        _verify_sha256(tarball_path)
    else:
        print(
            f"warning: --no-verify in effect; sha256 of "
            f"{tarball_path.name} NOT checked",
            file=sys.stderr,
        )

    dest_root_path = (
        Path(dest_root).expanduser().resolve()
        if dest_root is not None
        else _DEFAULT_STATIC_ROOT
    )
    dest_dir = dest_root_path / entry
    _check_clean_or_force(dest_dir, force=force)

    return _extract_tarball(
        tarball=tarball_path,
        dest_root=dest_root_path,
        dest_dir=dest_dir,
        entry=entry,
    )


def install_frontend_from_url(
    *,
    version: str,
    entry: str = "genome",
    force: bool = False,
    verify: bool = True,
    dest_root: Path | str | None = None,
    cache_dir: Path | str | None = None,
    github_owner_repo: str = _DEFAULT_GITHUB_REPO,
) -> InstallResult:
    """Install a prebuilt frontend bundle from a GitHub Release asset.

    Builds the URL ``https://github.com/<github_owner_repo>/releases
    /download/v<version>/constellation-viz-frontend-<version>.tar.gz``
    plus its ``.sha256`` sidecar. Both are downloaded to ``cache_dir``;
    on a cache hit (file present + sidecar digest matches), the network
    fetch of the tarball is skipped (the sidecar is always re-fetched —
    cheap, and lets a re-published release invalidate the cache).
    Then dispatches into ``install_frontend_from_tarball`` so the
    extract / verify / bundle-metadata pipeline is identical between
    install paths.

    Parameters
    ----------
    version
        PEP 440 version string matching a published release tag (the
        leading ``v`` is added automatically). Dev/local versions
        (containing ``.dev`` or ``+``) are refused before the network
        call with an actionable message — the matching release won't
        exist on GitHub.
    entry
        Vite entry name to install (default ``"genome"``). See
        ``install_frontend_from_tarball``.
    force
        Replace any existing install at ``<dest_root>/<entry>/``.
    verify
        Verify the downloaded tarball's sha256 against the sidecar.
        ``False`` skips the check and prints a warning to stderr —
        almost never what you want over the network.
    dest_root
        Override for the static-root directory. Defaults to
        ``constellation/viz/static/`` inside the installed package.
    cache_dir
        Override for the on-disk cache. Defaults to
        ``~/.cache/constellation/viz-frontend/``.
    github_owner_repo
        ``owner/repo`` slug to fetch from. Defaults to
        ``wilburnlab/constellation``; forks override here without a
        code change.

    Returns
    -------
    InstallResult
        On success. Raises ``InstallError`` on download failure (404,
        network error), sha mismatch, or any of the conditions
        ``install_frontend_from_tarball`` raises for.
    """
    if "+" in version or ".dev" in version:
        raise InstallError(
            f"cannot fetch release for dev version {version}\n"
            f"  pass --version X.Y.Z explicitly, or --from <local-tarball>\n"
            f"  to install a locally-built bundle."
        )

    cache = (
        Path(cache_dir).expanduser().resolve()
        if cache_dir is not None
        else _DEFAULT_CACHE_DIR
    )
    cache.mkdir(parents=True, exist_ok=True)

    artifact_name = f"{_PACK_NAME_PREFIX}-{version}"
    tarball_name = f"{artifact_name}.tar.gz"
    sidecar_name = f"{tarball_name}{_SHA256_SUFFIX}"
    cached_tarball = cache / tarball_name
    cached_sidecar = cache / sidecar_name

    base_url = (
        f"{_RELEASE_BASE_SCHEME}://{_RELEASE_BASE_HOST}/{github_owner_repo}"
        f"/releases/download/v{version}"
    )
    tarball_url = f"{base_url}/{tarball_name}"
    sidecar_url = f"{base_url}/{sidecar_name}"

    # Always re-fetch the sidecar — it's tiny, and lets a re-pushed
    # release invalidate the cache without a manual --force.
    print(f"fetching {sidecar_url}", file=sys.stderr)
    _download_to(sidecar_url, cached_sidecar, version=version)

    if cached_tarball.is_file():
        try:
            expected = _parse_sidecar(cached_sidecar, tarball_name)
            actual = _sha256_of_file(cached_tarball)
        except InstallError:
            expected = actual = ""
        if expected and expected.lower() == actual.lower():
            print(
                f"using cached {cached_tarball.name} ({len(cached_tarball.read_bytes()):,} bytes)",
                file=sys.stderr,
            )
        else:
            print(
                f"cached tarball stale, re-fetching {tarball_url}",
                file=sys.stderr,
            )
            _download_to(tarball_url, cached_tarball, version=version)
    else:
        print(f"fetching {tarball_url}", file=sys.stderr)
        _download_to(tarball_url, cached_tarball, version=version)

    return install_frontend_from_tarball(
        local_path=cached_tarball,
        entry=entry,
        force=force,
        verify=verify,
        dest_root=dest_root,
    )


def read_bundle_metadata(dest_dir: Path | str) -> dict[str, Any] | None:
    """Read the installed bundle's metadata, or return None.

    Used by ``constellation doctor`` to surface the installed bundle's
    version. Returns ``None`` when the directory is missing, empty, or
    has no parseable ``bundle.json``.
    """
    path = Path(dest_dir) / _BUNDLE_METADATA_NAME
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def list_tarball_entries(tarball_path: Path | str) -> list[str]:
    """Inspect a release tarball and return every entry it contains.

    Scans for ``<prefix>/static/<entry>/`` top-level directory members
    and returns the discovered entry names sorted. This is the source
    of truth for "install everything" — the CLI dispatches one
    ``install_frontend_from_tarball`` call per discovered entry when
    ``--entry`` is omitted. A single-entry tarball returns a 1-element
    list; multi-entry tarballs return all of them.

    Raises :class:`InstallError` when the tarball is unreadable or has
    no ``static/<entry>/`` directories (legacy / hand-rolled tarballs).
    """
    path = Path(tarball_path).expanduser().resolve()
    if not path.is_file():
        raise InstallError(f"tarball not found: {path}")
    try:
        with tarfile.open(path, mode="r:gz") as tar:
            members = tar.getmembers()
    except (tarfile.TarError, OSError) as exc:
        raise InstallError(f"failed to read tarball {path.name}: {exc}") from exc

    # Members look like '<prefix>/static/<entry>/...'. Pick out the
    # second-level segment after 'static/'.
    entries: set[str] = set()
    for member in members:
        parts = member.name.split("/")
        if len(parts) >= 3 and parts[1] == "static" and parts[2]:
            entries.add(parts[2])
    if not entries:
        raise InstallError(
            f"no static/<entry>/ directories found in {path.name}; "
            "is this a valid constellation-viz-frontend bundle?"
        )
    return sorted(entries)


# ----------------------------------------------------------------------
# Internals
# ----------------------------------------------------------------------


def _verify_sha256(tarball: Path) -> None:
    sidecar = tarball.with_name(tarball.name + _SHA256_SUFFIX)
    if not sidecar.exists():
        raise InstallError(
            f"sha256 sidecar not found: {sidecar}\n"
            f"  expected GNU coreutils format: '<hex>  {tarball.name}'\n"
            f"  pass --no-verify to install without checksum verification"
        )
    expected = _parse_sidecar(sidecar, tarball.name)
    actual = _sha256_of_file(tarball)
    if expected.lower() != actual.lower():
        raise InstallError(
            f"sha256 mismatch for {tarball.name}\n"
            f"  expected: {expected}\n"
            f"  actual:   {actual}\n"
            f"  tarball kept at: {tarball} (inspect manually)"
        )


def _parse_sidecar(sidecar: Path, expected_filename: str) -> str:
    """Parse a GNU coreutils sha256sum sidecar.

    Format: ``<64-hex-digits>  <filename>\\n`` (two spaces). We accept
    any single line with the digest first; the filename column is
    advisory (some tools emit relative paths or wildcards). Multi-line
    files prefer the line whose filename matches ``expected_filename``,
    falling back to the first valid line.
    """
    text = sidecar.read_text(encoding="utf-8")
    candidates: list[tuple[str, str]] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(maxsplit=1)
        if not parts:
            continue
        digest = parts[0]
        if len(digest) != 64 or not all(
            ch in "0123456789abcdefABCDEF" for ch in digest
        ):
            continue
        filename = parts[1].lstrip("*").strip() if len(parts) == 2 else ""
        candidates.append((digest, filename))
    if not candidates:
        raise InstallError(f"no valid sha256 lines in sidecar: {sidecar}")
    for digest, filename in candidates:
        if filename == expected_filename:
            return digest
    return candidates[0][0]


def _sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(_CHUNK_BYTES), b""):
            h.update(chunk)
    return h.hexdigest()


def _download_to(url: str, dest: Path, *, version: str) -> None:
    """Stream ``url`` to ``dest`` via stdlib ``urllib.request``.

    Writes to a sibling ``.partial`` file and atomically renames on
    success so a failed download never leaves a half-written file in
    the cache. Maps HTTPError 404 to a release-not-found InstallError
    pointing at the most likely fix (`--version` typo or pre-release
    tag missing); other URLErrors fall through with a `--from` escape
    hatch suggestion.
    """
    req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    partial = dest.with_name(dest.name + ".partial")
    try:
        with urllib.request.urlopen(req) as resp, partial.open("wb") as out:
            shutil.copyfileobj(resp, out, length=_CHUNK_BYTES)
    except urllib.error.HTTPError as exc:
        partial.unlink(missing_ok=True)
        if exc.code == 404:
            raise InstallError(
                f"release asset not found (HTTP 404): {url}\n"
                f"  the tag v{version} may not be published yet, or the\n"
                f"  asset is missing from the release. Verify at\n"
                f"  {_RELEASE_BASE_SCHEME}://{_RELEASE_BASE_HOST}/"
                f"{_DEFAULT_GITHUB_REPO}/releases/tag/v{version}\n"
                f"  or pass --from <local-tarball> for an offline install."
            ) from exc
        raise InstallError(
            f"download failed (HTTP {exc.code}): {url}\n  {exc.reason}"
        ) from exc
    except urllib.error.URLError as exc:
        partial.unlink(missing_ok=True)
        raise InstallError(
            f"network error fetching {url}\n"
            f"  {exc.reason}\n"
            f"  pass --from <local-tarball> for an offline install."
        ) from exc
    partial.replace(dest)


def _check_clean_or_force(dest_dir: Path, *, force: bool) -> None:
    if not dest_dir.exists():
        return
    has_contents = any(dest_dir.iterdir())
    if not has_contents:
        return
    if force:
        shutil.rmtree(dest_dir)
        return
    existing_version = _peek_existing_version(dest_dir)
    raise InstallError(
        f"destination not empty: {dest_dir}\n"
        f"  existing bundle: {existing_version or 'unknown version'}\n"
        f"  pass --force to replace, or remove the directory manually"
    )


def _peek_existing_version(dest_dir: Path) -> str | None:
    data = read_bundle_metadata(dest_dir)
    if not data:
        return None
    version = data.get("constellation_version")
    return str(version) if version else None


def _extract_tarball(
    *,
    tarball: Path,
    dest_root: Path,
    dest_dir: Path,
    entry: str,
) -> InstallResult:
    dest_root.mkdir(parents=True, exist_ok=True)
    dest_dir.mkdir(parents=True, exist_ok=True)

    static_prefix = f"static/{entry}/"
    metadata: dict[str, Any] = {}
    extracted_files = 0

    with tarfile.open(tarball, mode="r:gz") as tar:
        members = tar.getmembers()
        if not members:
            raise InstallError(f"tarball is empty: {tarball}")

        top_prefix = _detect_top_prefix(members)
        resolved_dest = dest_dir.resolve()

        for member in members:
            name = member.name
            if not name.startswith(top_prefix):
                continue
            rel = name[len(top_prefix) :]
            if rel == "":
                continue

            if rel == _BUNDLE_METADATA_NAME and member.isfile():
                payload = tar.extractfile(member)
                if payload is not None:
                    metadata = _parse_bundle_metadata(
                        payload.read(), source=tarball
                    )
                continue

            if not rel.startswith(static_prefix):
                # Anything else at top-level (README, license, etc.) is
                # silently ignored — keeps the install resilient to
                # extra artifacts the build helper might add later.
                continue

            sub_rel = rel[len(static_prefix) :]
            if sub_rel == "":
                continue
            target = dest_dir / sub_rel

            # Path-traversal guard: refuse anything that would land
            # outside `dest_dir` after resolution (defends against
            # tarballs with '..' or absolute paths).
            try:
                target.resolve().relative_to(resolved_dest)
            except ValueError as exc:
                raise InstallError(
                    f"refusing to extract member outside destination: {name}"
                ) from exc

            if member.issym() or member.islnk():
                raise InstallError(
                    f"refusing to extract link member: {name} "
                    f"(bundles should be plain files)"
                )
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            if not member.isfile():
                # Skip exotic entries (block devices, fifos, etc.).
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            payload = tar.extractfile(member)
            if payload is None:
                continue
            with target.open("wb") as out:
                shutil.copyfileobj(payload, out)
            extracted_files += 1

    if extracted_files == 0:
        raise InstallError(
            f"no files extracted from {tarball}; "
            f"expected entries under '{top_prefix}{static_prefix}'"
        )

    if metadata:
        (dest_dir / _BUNDLE_METADATA_NAME).write_text(
            json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
        )

    return InstallResult(
        dest_dir=dest_dir, entry=entry, bundle_metadata=metadata
    )


def _detect_top_prefix(members: list[tarfile.TarInfo]) -> str:
    """Find the single top-level directory prefix inside the tarball.

    Tarballs produced by ``--pack`` have exactly one top-level entry
    (``constellation-viz-frontend-<version>/``). Anything else is a
    structural error worth surfacing — we'd rather refuse than silently
    install half of a multi-bundle archive.
    """
    prefixes: set[str] = set()
    for member in members:
        name = member.name.strip("/")
        if not name:
            continue
        prefixes.add(name.split("/", 1)[0])
    if not prefixes:
        raise InstallError("tarball has no entries")
    if len(prefixes) > 1:
        raise InstallError(
            f"tarball has multiple top-level entries: {sorted(prefixes)}; "
            f"expected exactly one (e.g. constellation-viz-frontend-<version>/)"
        )
    return next(iter(prefixes)) + "/"


def _parse_bundle_metadata(payload: bytes, *, source: Path) -> dict[str, Any]:
    try:
        data = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise InstallError(
            f"bundle.json in {source} is not valid JSON: {exc}"
        ) from exc
    if not isinstance(data, dict):
        raise InstallError(f"bundle.json in {source} is not a JSON object")
    return data
