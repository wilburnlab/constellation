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
    return payload


class SaveRequest(BaseModel):
    label: str
    reference_handle: str
    sources: list[dict[str, Any]]
    last_viewed_locus: dict[str, Any] | None = None
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
            slug=body.slug,
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
