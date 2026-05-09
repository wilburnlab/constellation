"""FastAPI application factory for `constellation viz <subcommand>`.

`create_app(sessions, *, static_root=None)` returns a configured app with
the session/track endpoints mounted and (optionally) the built SPA
bundle served at `/static/<entry>/`. The CLI handler imports this and
hands the result to uvicorn.

Endpoints currently mounted in PR 1:

- `GET /                                    → 302 /static/genome/`
- `GET /api/health                          → version probe`
- `GET /api/sessions, /api/sessions/{id}/...`
- `GET /api/tracks, /api/tracks/{kind}/...`
- `GET /static/genome/...                   → built SPA bundle (if present)`

The dashboard PR adds `/api/commands`, `/api/cli/schema`, `/api/fs/list`
and the `/static/dashboard/` mount. None of those exist yet.
"""

from __future__ import annotations

from importlib import metadata
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

# Importing constellation.viz triggers the kernel modules' self-
# registration via @register_track. Without this the /api/tracks
# endpoints would return an empty list.
import constellation.viz  # noqa: F401
from constellation.viz.server.endpoints import sessions as sessions_ep
from constellation.viz.server.endpoints import tracks as tracks_ep
from constellation.viz.server.session import Session


# Resolve the bundled-static directory once at import time. It lives
# alongside this file at constellation/viz/static/<entry>/.
_VIZ_PKG = Path(__file__).resolve().parent.parent
_BUILT_STATIC_ROOT = _VIZ_PKG / "static"


def _package_version() -> str:
    try:
        return metadata.version("constellation")
    except metadata.PackageNotFoundError:
        return "0.0.0+local"


def create_app(
    sessions: dict[str, Session] | list[Session] | Session,
    *,
    static_root: Path | None = None,
    default_entry: str = "genome",
) -> FastAPI:
    """Build the viz FastAPI app.

    Parameters
    ----------
    sessions
        One Session, a list of Sessions, or a `{id: Session}` dict. The
        first form is the PR-1 single-session case; the dashboard
        plumbs the dict form once it lands.
    static_root
        Override for the built-bundle root. When `None`, falls back to
        `constellation/viz/static/`. The directory does not need to
        exist — when missing, the static-asset mount is skipped and the
        root path returns a JSON pointer at the API instead.
    default_entry
        Subdirectory under `static_root` whose `index.html` we redirect
        to from `/`. PR 1 ships only `"genome"`; PR 2 adds
        `"dashboard"` and the CLI dispatches to the right one.
    """
    sessions_dict = _normalize_sessions(sessions)
    app = FastAPI(
        title="Constellation viz",
        version=_package_version(),
        docs_url="/api/docs",
        redoc_url=None,
    )
    app.state.sessions = sessions_dict
    # Per-process cache of `(session_id, kind) → list[TrackBinding]`.
    # Discovery is fast but binding resolution is deterministic, so we
    # avoid recomputing on every /api/tracks/{kind}/data hit.
    app.state.track_bindings_cache = {}

    @app.get("/api/health")
    def _health() -> dict:
        return {
            "ok": True,
            "version": app.version,
            "sessions": list(sessions_dict.keys()),
        }

    app.include_router(sessions_ep.router)
    app.include_router(tracks_ep.router)

    # Static bundle: mount when present. The `genome/` and `dashboard/`
    # subdirectories are produced by `python -m constellation.viz.frontend.build`.
    root = (static_root or _BUILT_STATIC_ROOT).resolve()
    entry_dir = root / default_entry
    if entry_dir.is_dir():
        app.mount(
            f"/static/{default_entry}",
            StaticFiles(directory=str(entry_dir), html=True),
            name=f"static-{default_entry}",
        )

        @app.get("/", include_in_schema=False)
        def _root_redirect() -> RedirectResponse:
            return RedirectResponse(url=f"/static/{default_entry}/")
    else:
        @app.get("/", include_in_schema=False)
        def _root_no_bundle() -> dict:
            return {
                "ok": True,
                "message": (
                    "viz frontend bundle not built — run "
                    "`python -m constellation.viz.frontend.build` "
                    "or install a release wheel that ships the static/ tree"
                ),
                "expected_path": str(entry_dir),
                "api": "/api/health",
            }

    return app


def _normalize_sessions(
    sessions: dict[str, Session] | list[Session] | Session,
) -> dict[str, Session]:
    if isinstance(sessions, Session):
        return {sessions.session_id: sessions}
    if isinstance(sessions, list):
        out: dict[str, Session] = {}
        for s in sessions:
            if s.session_id in out:
                raise ValueError(f"duplicate session_id: {s.session_id}")
            out[s.session_id] = s
        return out
    if isinstance(sessions, dict):
        return dict(sessions)
    raise TypeError(
        f"sessions must be Session, list[Session], or dict[str, Session]; "
        f"got {type(sessions).__name__}"
    )
