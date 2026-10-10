"""Saved-session endpoints — CRUD over ``~/.constellation/sessions/``.

A saved session persists one browser configuration — its modality, the
ordered list of data sources, and whatever anchors them (a reference
handle for the genome browser) — so the dashboard can restore it in one
click. These endpoints don't open the session — the form re-POSTs
through ``/api/sessions/open`` after the user clicks ``Open`` so the
endpoint contract stays small.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from fastapi import APIRouter, Body, HTTPException
from pydantic import BaseModel

from constellation.viz.modalities import DEFAULT_MODALITY, get_modality
from constellation.viz.server.validation import validate


router = APIRouter(prefix="/api/saved-sessions", tags=["saved-sessions"])


def _summarize(saved) -> dict[str, Any]:
    return {
        "slug": saved.slug,
        "modality": saved.modality,
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
    payload["options"] = dict(saved.options) if saved.options else {}
    return payload


@router.post("", status_code=201)
def save_session_endpoint(body: dict[str, Any] = Body(...)) -> dict[str, Any]:
    """Persist a saved session. Pass an explicit ``slug`` to overwrite an
    existing entry; otherwise a fresh slug is derived.

    ``body["modality"]`` (default ``"genome"``) selects the modality,
    whose ``save_request`` model the body is validated against — so the
    fields a configuration must carry (a ``reference_handle`` for the
    genome browser) are that modality's to require.
    """
    from constellation.viz.sessions import write_saved

    name = str(body.get("modality") or DEFAULT_MODALITY)
    try:
        modality = get_modality(name)
    except KeyError as exc:
        raise HTTPException(400, str(exc.args[0])) from exc
    fields = asdict(validate(modality.save_request, body, where="body"))

    try:
        saved = write_saved(
            modality=name,
            label=fields["label"],
            reference_handle=fields.get("reference_handle") or "",
            sources=fields["sources"],
            last_viewed_locus=fields.get("last_viewed_locus"),
            track_layout=fields.get("track_layout"),
            options=fields.get("options"),
            slug=fields.get("slug"),
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    return _summarize(saved)


class LayoutPatchRequest(BaseModel):
    track_layout: list[dict[str, Any]]
    options: dict[str, Any] | None = None


@router.patch("/{slug}/layout")
def patch_saved_layout(slug: str, body: LayoutPatchRequest) -> dict[str, Any]:
    """Rewrite the ``[[track_layout]]`` block on an existing saved
    session, preserving everything else.

    Called by the client whenever the user changes per-binding state
    (visibility / order / height / collapsed / per-track style / per-track
    filter) on a session that has a persisted slug, or toggles a
    browser-wide option. Returns the refreshed summary so the client
    can verify the write.
    """
    from constellation.viz.sessions import read_saved, write_saved

    try:
        existing = read_saved(slug)
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    # Options absent in the patch means "keep what's there"; pass an
    # empty dict explicitly to clear (the normalizer drops empty back
    # to None).
    options = (
        body.options
        if body.options is not None
        else (dict(existing.options) if existing.options else None)
    )
    try:
        # The file is rewritten whole: every field of ``existing`` has to
        # be passed back or it is lost.
        saved = write_saved(
            modality=existing.modality,
            label=existing.label,
            reference_handle=existing.reference_handle,
            sources=existing.sources,
            last_viewed_locus=existing.last_viewed_locus,
            track_layout=body.track_layout,
            options=options,
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
