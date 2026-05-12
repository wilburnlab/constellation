"""URL-fetch install path — PR 1.6 unit tests.

Spins a stdlib ``http.server.ThreadingHTTPServer`` in a background
thread, serves real packed bundles + sidecars from a tmp directory, and
exercises ``install_frontend_from_url`` against it. The server stand-in
mimics the layout of the GitHub Releases download CDN
(``/<owner>/<repo>/releases/download/v<X.Y.Z>/<asset>``) so the
URL-construction code in ``install_frontend_from_url`` is on the live
codepath — only host + scheme differ (monkeypatched module-level).

No network access; everything runs against 127.0.0.1.
"""

from __future__ import annotations

import http.server
import socketserver
import threading
from pathlib import Path

import pytest

from constellation.viz.frontend.build import pack_bundle
from constellation.viz.install import (
    InstallError,
    install_frontend_from_url,
    read_bundle_metadata,
)


# ----------------------------------------------------------------------
# Fixture helpers
# ----------------------------------------------------------------------


def _write_fake_bundle(static_root: Path, entry: str = "genome") -> Path:
    entry_dir = static_root / entry
    entry_dir.mkdir(parents=True, exist_ok=True)
    (entry_dir / "index.genome.html").write_text(
        "<!doctype html><html><body><div id=\"app\"></div></body></html>\n",
        encoding="utf-8",
    )
    (entry_dir / "assets").mkdir(exist_ok=True)
    (entry_dir / "assets" / "main_genome-fakehash.js").write_text(
        "console.log('fixture');\n", encoding="utf-8"
    )
    return entry_dir


def _stage_release(
    *,
    server_root: Path,
    owner_repo: str,
    version: str,
    entry: str = "genome",
) -> tuple[Path, Path]:
    """Pack a fixture bundle into the path layout the test server serves.

    Mirrors GitHub's release-download URL: writes the tarball + sidecar
    to ``<server_root>/<owner_repo>/releases/download/v<version>/...``.
    """
    src_static = server_root / f"_src_{version}"
    _write_fake_bundle(src_static, entry=entry)
    rel_dir = (
        server_root / owner_repo / "releases" / "download" / f"v{version}"
    )
    rel_dir.mkdir(parents=True, exist_ok=True)
    tarball, sidecar = pack_bundle(
        entry=entry,
        static_root=src_static,
        dist_dir=rel_dir,
        version=version,
    )
    return tarball, sidecar


# ----------------------------------------------------------------------
# Test server fixture
# ----------------------------------------------------------------------


@pytest.fixture
def file_server(tmp_path: Path):
    """Yield (server_root, port) for a transient HTTP file server."""
    server_root = tmp_path / "srv"
    server_root.mkdir()

    handler_cls = type(
        "_QuietHandler",
        (http.server.SimpleHTTPRequestHandler,),
        {
            "log_message": lambda self, fmt, *args: None,
            "__init__": lambda self, *a, **kw: (
                http.server.SimpleHTTPRequestHandler.__init__(
                    self, *a, directory=str(server_root), **kw
                )
            ),
        },
    )

    httpd = socketserver.ThreadingTCPServer(("127.0.0.1", 0), handler_cls)
    port = httpd.server_address[1]
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield server_root, port
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=2.0)


@pytest.fixture
def patched_install(monkeypatch: pytest.MonkeyPatch, file_server):
    """Point install_frontend_from_url at the local test server.

    Patches the module-level scheme/host constants that
    ``install_frontend_from_url`` uses to build release URLs. Yields
    ``(server_root, port)`` for staging release assets.
    """
    server_root, port = file_server
    monkeypatch.setattr(
        "constellation.viz.install._RELEASE_BASE_SCHEME", "http"
    )
    monkeypatch.setattr(
        "constellation.viz.install._RELEASE_BASE_HOST", f"127.0.0.1:{port}"
    )
    yield server_root, port


# ----------------------------------------------------------------------
# Happy paths
# ----------------------------------------------------------------------


def test_url_fetch_installs_release_asset(
    patched_install, tmp_path: Path
) -> None:
    server_root, _ = patched_install
    owner_repo = "wilburnlab/constellation"
    _stage_release(
        server_root=server_root, owner_repo=owner_repo, version="0.0.1"
    )

    cache = tmp_path / "cache"
    dest_root = tmp_path / "install"
    result = install_frontend_from_url(
        version="0.0.1",
        dest_root=dest_root,
        cache_dir=cache,
    )
    assert result.entry == "genome"
    assert (dest_root / "genome" / "index.genome.html").is_file()
    meta = read_bundle_metadata(dest_root / "genome")
    assert meta is not None
    assert meta["constellation_version"] == "0.0.1"

    assert (cache / "constellation-viz-frontend-0.0.1.tar.gz").is_file()
    assert (cache / "constellation-viz-frontend-0.0.1.tar.gz.sha256").is_file()


def test_url_fetch_uses_cache_on_second_call(
    patched_install,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    server_root, _ = patched_install
    owner_repo = "wilburnlab/constellation"
    _stage_release(
        server_root=server_root, owner_repo=owner_repo, version="0.0.2"
    )

    from constellation.viz import install as install_mod

    real_dl = install_mod._download_to
    download_calls: list[str] = []

    def _track(url: str, dest, *, version):
        download_calls.append(url)
        return real_dl(url, dest, version=version)

    monkeypatch.setattr(install_mod, "_download_to", _track)

    cache = tmp_path / "cache"
    dest_root = tmp_path / "install"
    install_frontend_from_url(
        version="0.0.2", dest_root=dest_root, cache_dir=cache
    )
    first_calls = list(download_calls)
    assert any(c.endswith(".tar.gz") for c in first_calls)
    assert any(c.endswith(".sha256") for c in first_calls)

    # Second install with --force re-uses cached tarball; only sidecar
    # is re-fetched (it's tiny + lets a re-pushed release invalidate).
    download_calls.clear()
    install_frontend_from_url(
        version="0.0.2",
        dest_root=dest_root,
        cache_dir=cache,
        force=True,
    )
    assert any(c.endswith(".sha256") for c in download_calls)
    assert not any(c.endswith(".tar.gz") for c in download_calls), (
        "expected tarball to be served from cache, not re-fetched"
    )


# ----------------------------------------------------------------------
# Failure modes
# ----------------------------------------------------------------------


def test_url_fetch_404_raises_actionable_error(
    patched_install, tmp_path: Path
) -> None:
    # Don't stage anything — every fetch 404s.
    with pytest.raises(InstallError, match=r"release asset not found"):
        install_frontend_from_url(
            version="9.9.9",
            dest_root=tmp_path / "install",
            cache_dir=tmp_path / "cache",
        )


def test_url_fetch_refuses_dev_version_before_network(
    tmp_path: Path,
) -> None:
    """Dev/local versions can't have a matching release; fail fast with
    an actionable message before issuing any HTTP requests."""
    with pytest.raises(InstallError, match=r"dev version"):
        install_frontend_from_url(
            version="0.0.2.dev3+gabc1234",
            dest_root=tmp_path / "install",
            cache_dir=tmp_path / "cache",
        )
    with pytest.raises(InstallError, match=r"dev version"):
        install_frontend_from_url(
            version="0.0.2+local.tag",
            dest_root=tmp_path / "install",
            cache_dir=tmp_path / "cache",
        )


def test_url_fetch_sha_mismatch_propagates(
    patched_install, tmp_path: Path
) -> None:
    """A poisoned sidecar — sha256 doesn't match the served tarball —
    should surface the existing extract-side mismatch error."""
    server_root, _ = patched_install
    owner_repo = "wilburnlab/constellation"
    _, sidecar = _stage_release(
        server_root=server_root, owner_repo=owner_repo, version="0.0.3"
    )
    # Corrupt the sidecar — server still serves both, but the digest is wrong.
    poisoned = "0" * 64 + "  " + sidecar.name.removesuffix(".sha256") + "\n"
    sidecar.write_text(poisoned, encoding="utf-8")

    with pytest.raises(InstallError, match=r"sha256 mismatch"):
        install_frontend_from_url(
            version="0.0.3",
            dest_root=tmp_path / "install",
            cache_dir=tmp_path / "cache",
        )
