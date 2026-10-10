"""Genome-only endpoints.

The session and track endpoints in ``viz.server.endpoints`` are
modality-neutral. These three only make sense for a genome session:

- ``GET /api/references``                       → installed reference cache
- ``GET /api/sessions/{session_id}/contigs``    → contigs of the reference
- ``GET /api/sessions/{session_id}/search``     → annotation-feature search

``/api/references`` surfaces ``constellation reference list`` to the
dashboard, so the genome-browser entry form can populate its reference
dropdown without re-implementing the cache walk in TypeScript.

Imported only through the modality's ``routers()`` factory — it needs
fastapi, which ``import constellation.viz`` must not.
"""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as pa_ds
import pyarrow.parquet as pq
from fastapi import APIRouter, HTTPException, Query, Request

from constellation.viz.modalities.genome.session import GenomeSession


router = APIRouter(tags=["genome"])


def _genome_session(request: Request, session_id: str) -> GenomeSession:
    """Look up ``session_id`` and require it to be a genome session."""
    session = request.app.state.sessions.get(session_id)
    if session is None:
        raise HTTPException(404, f"unknown session_id: {session_id}")
    if getattr(session, "modality", None) != GenomeSession.modality:
        raise HTTPException(
            404, f"session {session_id} is not a genome session"
        )
    return session


@router.get("/api/references")
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


# ----------------------------------------------------------------------
# Contigs
# ----------------------------------------------------------------------


@router.get("/api/sessions/{session_id}/contigs")
def get_contigs(session_id: str, request: Request) -> list[dict]:
    """Return ``[{contig_id, name, length}]`` for the session's reference
    genome. The frontend uses this to populate the locus picker."""
    session = _genome_session(request, session_id)
    contigs_path = session.reference_genome / "contigs.parquet"
    if not contigs_path.exists():
        raise HTTPException(
            500,
            f"reference genome ParquetDir is incomplete: {contigs_path} missing",
        )
    table = pq.read_table(contigs_path, columns=["contig_id", "name", "length"])
    out: list[dict] = []
    for row in table.to_pylist():
        out.append(
            {
                "contig_id": row["contig_id"],
                "name": row["name"],
                "length": row["length"],
            }
        )
    return out


# ----------------------------------------------------------------------
# Feature search — the reference annotation plus each source's derived
# annotation
# ----------------------------------------------------------------------


@router.get("/api/sessions/{session_id}/search")
def search_features(
    session_id: str,
    request: Request,
    q: str = Query(default="", description="Substring to match (case-insensitive)."),
    limit: int = Query(default=50, ge=1, le=500),
) -> list[dict]:
    """Case-insensitive substring match against annotation features.

    Scans the curated ``reference_annotation/features.parquet`` plus
    each align source's ``derived_annotation/features.parquet`` when
    present; tags each hit with a ``source`` field so the client can
    show which annotation bundle produced it. A purely-numeric ``q``
    also matches ``feature_id`` exactly. Returns up to ``limit`` rows,
    reference hits first.
    """
    session = _genome_session(request, session_id)

    query = (q or "").strip()
    if not query:
        return []

    contig_name_by_id = _load_contig_name_map(session.reference_genome)
    if not contig_name_by_id:
        return []

    numeric_id: int | None = None
    try:
        numeric_id = int(query)
    except ValueError:
        numeric_id = None

    out: list[dict] = []
    # Reference annotation first (curated wins display order), then each
    # source's derived annotation in the order the user added them.
    ordered: list[tuple[Path | None, str]] = [
        (session.reference_annotation, "reference"),
    ]
    for src in session.sources:
        if src.derived_annotation is not None:
            ordered.append((src.derived_annotation, f"derived ({src.label})"))

    for annotation_dir, source_tag in ordered:
        if annotation_dir is None:
            continue
        if len(out) >= limit:
            break
        features_path = annotation_dir / "features.parquet"
        if not features_path.exists():
            continue
        remaining = limit - len(out)
        rows = _search_features_in_parquet(
            features_path=features_path,
            query=query,
            numeric_id=numeric_id,
            limit=remaining,
        )
        for row in rows:
            contig_name = contig_name_by_id.get(int(row["contig_id"]))
            if contig_name is None:
                continue
            out.append(
                {
                    "feature_id": int(row["feature_id"]),
                    "name": row["name"],
                    "type": row["type"],
                    "strand": row["strand"],
                    "contig_name": contig_name,
                    "start": int(row["start"]),
                    "end": int(row["end"]),
                    "source": source_tag,
                }
            )
            if len(out) >= limit:
                break
    return out


def _load_contig_name_map(genome_dir: Path) -> dict[int, str]:
    contigs_path = genome_dir / "contigs.parquet"
    if not contigs_path.exists():
        return {}
    table = pq.read_table(contigs_path, columns=["contig_id", "name"])
    ids = table.column("contig_id").to_pylist()
    names = table.column("name").to_pylist()
    return {int(i): str(n) for i, n in zip(ids, names) if i is not None and n is not None}


def _search_features_in_parquet(
    *,
    features_path: Path,
    query: str,
    numeric_id: int | None,
    limit: int,
) -> list[dict]:
    """Run the substring (+ optional feature_id) filter against one
    annotation parquet, returning at most ``limit`` rows as plain dicts."""
    dataset = pa_ds.dataset(str(features_path), format="parquet")
    name_match = pc.match_substring(pc.field("name"), query, ignore_case=True)
    predicate = name_match
    if numeric_id is not None:
        predicate = predicate | (
            pc.field("feature_id") == pa.scalar(numeric_id, pa.int64())
        )
    scanner = dataset.scanner(
        columns=["feature_id", "contig_id", "start", "end", "strand", "type", "name"],
        filter=predicate,
    )
    table = scanner.head(limit)
    if table.num_rows == 0:
        return []
    return table.to_pylist()
