"""Reference-cache enumeration endpoint.

Surfaces ``constellation reference list`` results to the dashboard, so
the genome-browser entry form can populate its reference dropdown
without re-implementing the cache walk in TypeScript. The handle
dropdown is the canonical first choice in the new entry flow.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException


router = APIRouter(prefix="/api/references", tags=["references"])


@router.get("")
def list_references() -> list[dict]:
    """Return every reference installed in the per-user cache.

    One row per ``(organism, release_slug)``; ``is_default`` reflects the
    ``defaults.toml`` entry for the organism so the frontend can star /
    pre-select the default.
    """
    from constellation.sequencing.reference.handle import (
        list_installed,
        read_defaults,
    )

    try:
        installed = list_installed()
        defaults = read_defaults()
    except OSError as exc:  # noqa: BLE001
        raise HTTPException(500, f"reference cache walk failed: {exc}") from exc

    out: list[dict] = []
    for entry in installed:
        out.append(
            {
                "handle": entry.handle,
                "organism": entry.organism,
                "release_slug": entry.release_slug,
                "source": entry.source,
                "release": entry.release,
                "path": str(entry.path),
                "assembly_accession": entry.assembly_accession,
                "assembly_name": entry.assembly_name,
                "annotation_release": entry.annotation_release,
                "fetched_at": entry.fetched_at,
                "size_bytes": entry.size_bytes,
                "scientific_name": entry.scientific_name,
                "is_default": entry.is_default(defaults),
            }
        )
    return out
