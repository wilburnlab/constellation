"""Saved-session endpoints — CRUD over ``~/.constellation/sessions/``.

Saved sessions persist a reference handle + ordered list of data
sources so the dashboard can restore a configuration in one click.
These endpoints don't open the session — the form re-POSTs through
``/api/sessions/open`` after the user clicks ``Open`` so the
endpoint contract stays small.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel


router = APIRouter(prefix="/api/saved-sessions", tags=["saved-sessions"])


def _summarize(saved) -> dict[str, Any]:
    return {
        "slug": saved.slug,
        "label": saved.label,
        "reference_handle": saved.reference_handle,
        "n_sources": len(saved.sources),
        "saved_at": saved.saved_at,
        "last_viewed_locus": saved.last_viewed_locus,
    }


@router.get("")
def list_saved_endpoint() -> list[dict[str, Any]]:
    """Enumerate every saved session in the per-user cache."""
    from constellation.viz.sessions import list_saved

    return [_summarize(s) for s in list_saved()]


@router.get("/{slug}")
def get_saved(slug: str) -> dict[str, Any]:
    """Return one saved session's full payload (used by the form's
    ``Load saved session…`` prefill)."""
    from constellation.viz.sessions import read_saved

    try:
        saved = read_saved(slug)
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    payload = _summarize(saved)
    payload["sources"] = list(saved.sources)
    payload["track_layout"] = (
        list(saved.track_layout) if saved.track_layout else []
    )
    return payload


class SaveRequest(BaseModel):
    label: str
    reference_handle: str
    sources: list[dict[str, Any]]
    last_viewed_locus: dict[str, Any] | None = None
    track_layout: list[dict[str, Any]] | None = None
    slug: str | None = None


@router.post("", status_code=201)
def save_session_endpoint(body: SaveRequest) -> dict[str, Any]:
    """Persist a saved session. Pass an explicit ``slug`` to overwrite an
    existing entry; otherwise a fresh slug is derived."""
    from constellation.viz.sessions import write_saved

    try:
        saved = write_saved(
            label=body.label,
            reference_handle=body.reference_handle,
            sources=body.sources,
            last_viewed_locus=body.last_viewed_locus,
            track_layout=body.track_layout,
            slug=body.slug,
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    return _summarize(saved)


class LayoutPatchRequest(BaseModel):
    track_layout: list[dict[str, Any]]


@router.patch("/{slug}/layout")
def patch_saved_layout(slug: str, body: LayoutPatchRequest) -> dict[str, Any]:
    """Rewrite the ``[[track_layout]]`` block on an existing saved
    session, preserving everything else.

    Called by the client whenever the user changes per-binding
    visibility / order / height / collapsed state on a session that
    has a persisted slug. Returns the refreshed summary so the client
    can verify the write.
    """
    from constellation.viz.sessions import read_saved, write_saved

    try:
        existing = read_saved(slug)
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    try:
        saved = write_saved(
            label=existing.label,
            reference_handle=existing.reference_handle,
            sources=existing.sources,
            last_viewed_locus=existing.last_viewed_locus,
            track_layout=body.track_layout,
            saved_at=existing.saved_at,
            slug=existing.slug,
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    return _summarize(saved)


@router.delete("/{slug}", status_code=204)
def delete_saved_endpoint(slug: str) -> None:
    from constellation.viz.sessions import delete_saved

    if not delete_saved(slug):
        raise HTTPException(404, f"no saved session at slug {slug!r}")
    return None
