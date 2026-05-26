"""Session model for the viz layer.

A ``Session`` binds one reference (resolved from the per-user reference
cache) to a list of "result" directories the user wants to overlay onto
that reference — outputs of ``constellation transcriptome align`` and/or
``transcriptome cluster``. The reference cache (``handle.py``) is the
canonical axis here; the genome browser dashboard always picks a
reference first and then adds zero or more sources keyed to it.

Sessions are constructed exclusively via :meth:`Session.open` (handle +
list-of-sources) or :meth:`Session.from_saved` (load a saved-session
TOML from ``~/.constellation/sessions/``). The legacy ``from_root`` /
directory-walk discovery / ``session.toml`` v1 reader were removed in
the reference-cache-first cutover — the dashboard's entry form is now
the sole on-ramp.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Iterable, Literal


# ----------------------------------------------------------------------
# Source artifact slot map — derived from the producing stage's manifest
# ----------------------------------------------------------------------


# Per-kind well-known filenames that map manifest ``outputs`` keys to the
# slots ``SessionSource`` exposes. Kernel ``discover()`` methods read
# these slots directly; missing slots resolve to ``None`` and kernels
# skip the corresponding binding.
_ALIGN_SLOT_KEYS: tuple[tuple[str, str, bool], ...] = (
    # (manifest_key, slot, is_dir)
    ("alignments", "alignments", True),
    ("alignment_blocks", "alignment_blocks", True),
    ("alignment_cs", "alignment_cs", True),
    ("coverage", "coverage", False),
    ("introns", "introns", False),
    ("derived_annotation", "derived_annotation", True),
    ("block_exon_assignments", "block_exon_assignments", False),
    ("exon_psi", "exon_psi", False),
)

_CLUSTER_SLOT_KEYS: tuple[tuple[str, str, bool], ...] = (
    ("clusters", "clusters", False),
    ("cluster_membership", "cluster_membership", False),
)


# Per-kind well-known relative filenames (fallback when the manifest's
# ``outputs`` dict doesn't carry the key — the current align/cluster CLI
# does populate them, but the fallback is cheap and keeps the loader
# robust against partial manifests).
_ALIGN_DEFAULT_PATHS: dict[str, str] = {
    "alignments": "alignments",
    "alignment_blocks": "alignment_blocks",
    "alignment_cs": "alignment_cs",
    "coverage": "coverage.parquet",
    "introns": "introns.parquet",
    "derived_annotation": "derived_annotation",
    "block_exon_assignments": "block_exon_assignments.parquet",
    "exon_psi": "exon_psi.parquet",
}

_CLUSTER_DEFAULT_PATHS: dict[str, str] = {
    "clusters": "clusters.parquet",
    "cluster_membership": "cluster_membership.parquet",
}


# ----------------------------------------------------------------------
# Dataclasses
# ----------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SessionSource:
    """One data source attached to a session.

    A source is the output directory of one ``transcriptome align`` or
    ``transcriptome cluster`` run. The per-stage artifact slots
    (``alignments``, ``coverage``, ``introns``, …) are resolved at
    construction time from the source's ``manifest.json`` and exposed as
    optional ``Path`` fields — kernels skip the slot when ``None``.
    """

    path: Path
    kind: Literal["align", "cluster"]
    label: str
    assembly_accession: str | None
    reference_handle: str | None

    # Per-kind resolved artifact paths. Kernels read these directly.
    alignments: Path | None = None
    alignment_blocks: Path | None = None
    alignment_cs: Path | None = None
    coverage: Path | None = None
    introns: Path | None = None
    derived_annotation: Path | None = None
    block_exon_assignments: Path | None = None
    exon_psi: Path | None = None
    clusters: Path | None = None
    cluster_membership: Path | None = None

    samples: tuple[str, ...] = ()

    @property
    def source_id(self) -> str:
        """Stable client-side identifier derived from ``(path, kind)``.

        Survives add/remove cycles so persisted per-binding layout state
        (visibility, display order, height) keyed by ``source_id``
        remains valid across session rebuilds.
        """
        payload = f"{self.path}|{self.kind}".encode("utf-8")
        digest = hashlib.blake2b(payload, digest_size=4).hexdigest()
        return f"src-{digest}"

    def slot_paths(self) -> dict[str, str | None]:
        """JSON-friendly slot view for the dashboard's session manifest."""
        return {
            "alignments": _stringify(self.alignments),
            "alignment_blocks": _stringify(self.alignment_blocks),
            "alignment_cs": _stringify(self.alignment_cs),
            "coverage": _stringify(self.coverage),
            "introns": _stringify(self.introns),
            "derived_annotation": _stringify(self.derived_annotation),
            "block_exon_assignments": _stringify(self.block_exon_assignments),
            "exon_psi": _stringify(self.exon_psi),
            "clusters": _stringify(self.clusters),
            "cluster_membership": _stringify(self.cluster_membership),
        }


@dataclass(frozen=True, slots=True)
class Session:
    """Resolved entry point for a viz server.

    A session is one reference plus zero-or-more attached sources. The
    ``warnings`` tuple carries assembly-mismatch notices (when a
    source's recorded ``assembly_accession`` differs from the chosen
    reference's) — the dashboard surfaces them as inline yellow text but
    does not block the open.
    """

    session_id: str
    label: str
    reference_handle: str
    reference_path: Path
    reference_genome: Path
    reference_annotation: Path | None
    assembly_accession: str | None
    sources: tuple[SessionSource, ...]
    warnings: tuple[str, ...] = ()
    saved_as: str | None = None
    extras: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def open(
        cls,
        *,
        reference_handle: str,
        sources: Iterable[dict[str, Any]],
        label: str | None = None,
        saved_as: str | None = None,
        cache_root: Path | None = None,
    ) -> "Session":
        """Build a Session from a reference handle and a list of sources.

        ``sources`` is an iterable of dicts shaped::

            {"path": str | Path, "kind": "align" | "cluster" | None,
             "label": str | None}

        ``kind`` is auto-detected from each source's ``manifest.json``
        when omitted. ``label`` defaults to the directory basename. Each
        source must have a schema-v2 manifest (``reference_handle``
        optional, ``assembly_accession`` may be ``None`` for escape-hatch
        runs); pre-v2 manifests raise ``ValueError`` with an actionable
        message.
        """
        from constellation.sequencing.reference.handle import (
            ReferenceNotInstalledError,
            parse_handle,
            read_meta_toml,
            resolve as resolve_handle,
        )

        handle = parse_handle(reference_handle)
        try:
            release_path = resolve_handle(handle, root=cache_root)
        except (ValueError, ReferenceNotInstalledError) as exc:
            raise ValueError(
                f"reference handle {reference_handle!r} could not be "
                f"resolved against the cache: {exc}"
            ) from exc
        meta = read_meta_toml(release_path) or {}
        # Canonical handle string from meta.toml when present — preserves
        # the fully-qualified form even when a bare organism slug was
        # passed and resolved via defaults.toml.
        canonical_handle = str(meta.get("handle") or handle)
        ref_assembly: str | None = (
            str(meta["assembly_accession"])
            if meta.get("assembly_accession") is not None
            else None
        )
        genome_dir = release_path / "genome"
        if not genome_dir.is_dir():
            raise ValueError(
                f"reference release at {release_path} is missing genome/ subdir"
            )
        annotation_dir = release_path / "annotation"
        annotation_path: Path | None = annotation_dir if annotation_dir.is_dir() else None

        built_sources: list[SessionSource] = []
        warnings: list[str] = []
        for entry in sources:
            built = _load_source(entry)
            built_sources.append(built)
            if (
                built.assembly_accession is not None
                and ref_assembly is not None
                and built.assembly_accession != ref_assembly
            ):
                warnings.append(
                    f"source {built.label!r} was produced against assembly "
                    f"{built.assembly_accession!r} but the chosen reference "
                    f"is {ref_assembly!r} — coordinates may not align"
                )

        resolved_label = label or release_path.name
        session_id = _derive_session_id(release_path, resolved_label)
        return cls(
            session_id=session_id,
            label=resolved_label,
            reference_handle=canonical_handle,
            reference_path=release_path,
            reference_genome=genome_dir,
            reference_annotation=annotation_path,
            assembly_accession=ref_assembly,
            sources=tuple(built_sources),
            warnings=tuple(warnings),
            saved_as=saved_as,
        )

    def with_sources(
        self,
        sources: Iterable[dict[str, Any]],
        *,
        cache_root: Path | None = None,
    ) -> "Session":
        """Rebuild this session with a new list of attached sources.

        Used by the runtime add/remove-source endpoints. Reuses
        ``reference_handle``, ``label``, and ``saved_as``; the returned
        Session's ``session_id`` matches ``self.session_id`` by
        construction (``_derive_session_id`` is deterministic over
        ``(reference_path, label)``), so the registry entry can be
        swapped in place without notifying the client.
        """
        return Session.open(
            reference_handle=self.reference_handle,
            sources=sources,
            label=self.label,
            saved_as=self.saved_as,
            cache_root=cache_root,
        )

    # ------------------------------------------------------------------
    # Inspection helpers (consumed by /api/sessions/{id}/manifest)
    # ------------------------------------------------------------------

    def stages_present(self) -> dict[str, bool]:
        """Return a small map of stage-name → present-bool for the UI."""
        has_align = any(s.kind == "align" for s in self.sources)
        has_cluster = any(s.kind == "cluster" for s in self.sources)
        return {
            "reference_genome": True,
            "reference_annotation": self.reference_annotation is not None,
            "alignments": any(s.alignments is not None for s in self.sources),
            "coverage": any(s.coverage is not None for s in self.sources),
            "introns": any(s.introns is not None for s in self.sources),
            "derived_annotation": any(
                s.derived_annotation is not None for s in self.sources
            ),
            "clusters": has_cluster,
            "has_align_sources": has_align,
            "has_cluster_sources": has_cluster,
        }

    def to_manifest(self) -> dict[str, Any]:
        """Serialize to JSON-friendly form for the sessions endpoint."""
        return {
            "session_id": self.session_id,
            "label": self.label,
            "reference": {
                "handle": self.reference_handle,
                "path": str(self.reference_path),
                "genome": str(self.reference_genome),
                "annotation": _stringify(self.reference_annotation),
                "assembly_accession": self.assembly_accession,
            },
            "sources": [
                {
                    "source_id": src.source_id,
                    "path": str(src.path),
                    "kind": src.kind,
                    "label": src.label,
                    "assembly_accession": src.assembly_accession,
                    "reference_handle": src.reference_handle,
                    "samples": list(src.samples),
                    "slots": src.slot_paths(),
                }
                for src in self.sources
            ],
            "stages_present": self.stages_present(),
            "warnings": list(self.warnings),
            "saved_as": self.saved_as,
            "extras": dict(self.extras),
        }


# ----------------------------------------------------------------------
# Source loading from manifest.json
# ----------------------------------------------------------------------


def _load_source(entry: dict[str, Any]) -> SessionSource:
    """Build a SessionSource from a ``{path, kind?, label?}`` dict.

    Reads the source's ``manifest.json`` (schema v4 required), assembles
    the per-kind slot map by joining the manifest's ``outputs`` against
    the well-known relative paths, and returns a frozen dataclass.
    """
    from constellation.sequencing.transcriptome.manifest import (
        read_manifest_dir,
    )

    raw_path = entry.get("path")
    if not raw_path:
        raise ValueError(f"source entry missing 'path': {entry!r}")
    source_path = Path(str(raw_path)).expanduser().resolve()
    if not source_path.is_dir():
        raise ValueError(f"source path is not a directory: {source_path}")

    manifest = read_manifest_dir(source_path)
    if manifest.kind == "demux":
        raise ValueError(
            f"source at {source_path} is a demux output; the genome "
            f"browser attaches `transcriptome align` or `transcriptome "
            f"cluster` output dirs — pass one of those instead"
        )
    kind_hint = entry.get("kind")
    if kind_hint is not None and kind_hint != manifest.kind:
        raise ValueError(
            f"source at {source_path} is kind={manifest.kind!r} but caller "
            f"asserted kind={kind_hint!r}"
        )
    label = str(entry.get("label") or source_path.name)
    samples = tuple(manifest.samples) if manifest.samples else ()

    slots: dict[str, Path | None] = {
        slot: None
        for slot, *_ in (
            *_ALIGN_SLOT_KEYS,
            *_CLUSTER_SLOT_KEYS,
        )
    }
    outputs = dict(manifest.outputs)
    schema = _ALIGN_SLOT_KEYS if manifest.kind == "align" else _CLUSTER_SLOT_KEYS
    defaults = _ALIGN_DEFAULT_PATHS if manifest.kind == "align" else _CLUSTER_DEFAULT_PATHS
    for manifest_key, slot, _is_dir in schema:
        rel = outputs.get(manifest_key) or defaults.get(manifest_key)
        if rel is None:
            continue
        resolved = _resolve_slot(source_path, rel)
        if resolved is not None:
            slots[slot] = resolved

    return SessionSource(
        path=source_path,
        kind=manifest.kind,
        label=label,
        assembly_accession=manifest.assembly_accession,
        reference_handle=manifest.reference_handle,
        samples=samples,
        **slots,
    )


def _resolve_slot(base: Path, rel: str) -> Path | None:
    """Resolve a manifest-recorded artifact path against the source dir.

    Manifests may record paths in any of three shapes:
      1. absolute (post-fix CLI writes these)
      2. relative to the source dir itself (``coverage.parquet``)
      3. relative to the cwd the producing CLI was run from
         (``<source-name>/coverage.parquet`` — the pre-fix shape, which
         is what manifests written before this fix carry)

    Try (1), then (2), then (3); return the first that resolves and
    exists on disk. ``None`` when none do — the kernel then skips that
    slot.
    """
    candidate = Path(rel)
    if candidate.is_absolute():
        return candidate if candidate.exists() else None
    relative_to_source = (base / candidate).resolve()
    if relative_to_source.exists():
        return relative_to_source
    # Fallback: the producing CLI wrote a path relative to its cwd, and
    # the rel path therefore starts with the source dir's own basename.
    # Resolving against the source dir's parent reproduces that cwd.
    relative_to_parent = (base.parent / candidate).resolve()
    if relative_to_parent.exists():
        return relative_to_parent
    return None


# ----------------------------------------------------------------------
# Saved-session shim
# ----------------------------------------------------------------------


def _from_saved_session(saved: Any, *, cache_root: Path | None = None) -> Session:
    """Construct a Session from a SavedSession dataclass.

    Centralized here so the saved-sessions endpoint, the standalone
    ``constellation viz genome --saved-session`` CLI, and any future
    callers all route through the same loader.
    """
    return Session.open(
        reference_handle=saved.reference_handle,
        sources=[
            {"path": src["path"], "kind": src.get("kind"), "label": src.get("label")}
            for src in saved.sources
        ],
        label=saved.label,
        saved_as=saved.slug,
        cache_root=cache_root,
    )


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _stringify(p: Path | None) -> str | None:
    return None if p is None else str(p)


def _derive_session_id(release_path: Path, label: str) -> str:
    """Stable short id for a (release_path, label) pair, URL-safe."""
    payload = f"{release_path}|{label}".encode("utf-8")
    digest = hashlib.blake2b(payload, digest_size=4).hexdigest()
    return f"{_slug(label)}-{digest}"


def _slug(s: str) -> str:
    out: list[str] = []
    for ch in s.lower():
        if ch.isalnum() or ch in "-_":
            out.append(ch)
        else:
            out.append("-")
    return "".join(out).strip("-") or "session"


__all__ = [
    "Session",
    "SessionSource",
]
