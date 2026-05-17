"""Session discovery for the viz layer.

A `Session` is the GUI's entry point: a directory containing one or more
pipeline-stage outputs the kernels can render. Discovery is layered:

1. **Explicit** — `<root>/session.toml`, when present, names what's
   available. This is the canonical, shareable artifact for a run.
2. **Implicit** — `Session.discover` walks `<root>` one level deep when no
   `session.toml` is present, classifying subdirs by directory-name
   heuristics (`S2_align/`, `S2_cluster/`, `genome/`, `annotation/`, ...).
3. **Ad-hoc** — `Session.from_paths` constructs a Session directly from
   caller-supplied paths, for the `--reference DIR --align-dir DIR` CLI
   shape that bypasses session-style layouts entirely.

The shape is intentionally flat: a `Session` carries one `Path | None`
per known artifact slot. Each kernel's `discover()` reads only the slots
it needs and returns the bindings it can render.
"""

from __future__ import annotations

import hashlib
import tomllib
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any


# ----------------------------------------------------------------------
# Stage-directory name heuristics
# ----------------------------------------------------------------------

# Substrings (case-insensitive) we consider strong evidence of a stage
# directory. Multiple matches are allowed — a real run typically has both
# S2_align and S2_cluster, plus a sibling `genome/` and `annotation/`.
_STAGE_S2_ALIGN = ("s2_align", "s2-align", "align")
_STAGE_S2_CLUSTER = ("s2_cluster", "s2-cluster", "cluster")
_STAGE_REFERENCE_GENOME = ("genome",)
_STAGE_REFERENCE_ANNOTATION = ("annotation",)
_STAGE_DERIVED_ANNOTATION = ("derived_annotation",)


def _matches(name: str, needles: tuple[str, ...]) -> bool:
    n = name.lower()
    return any(needle in n for needle in needles)


# ----------------------------------------------------------------------
# Session record
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class Session:
    """Resolved entry point for a viz server.

    `session_id` is short and stable for one root path (used as a URL
    component). `label` is the human-readable name shown in the UI.
    Path fields are absolute and `None` when the corresponding artifact
    is absent — kernels gate their `discover()` results on whichever
    slots they need.

    `samples` is the (possibly empty) list of sample identifiers
    available across the resolved stages. Empty means "either we don't
    know the sample partitioning, or the data is unlabeled" — kernels
    fall back to a single all-samples binding when this is empty.

    `extras` is the round-tripped `[extras]` table from `session.toml`,
    used for kernel-specific overrides (palettes, configured heights,
    etc.). The viz layer never mutates this; the dashboard later adds
    a "save view" affordance that writes back through a separate path.
    """

    session_id: str
    root: Path
    label: str
    schema_version: int = 1

    # Reference layer (one of these is required for the genome browser
    # to do anything useful — the kernel discovery rejects the session
    # otherwise).
    reference_genome: Path | None = None
    reference_annotation: Path | None = None

    # Per-stage outputs.
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
    extras: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_root(cls, root: Path | str, *, session_id: str | None = None) -> "Session":
        """Resolve a Session from a directory.

        Tries `<root>/session.toml` first; falls back to directory-walk
        discovery. Raises `FileNotFoundError` if `<root>` does not exist
        and `ValueError` if neither path produces any usable artifacts.
        """
        root_path = Path(root).expanduser().resolve()
        if not root_path.is_dir():
            raise FileNotFoundError(f"session root not found or not a dir: {root_path}")

        sid = session_id or _derive_session_id(root_path)
        toml_path = root_path / "session.toml"
        if toml_path.exists():
            return _load_session_toml(toml_path, root=root_path, session_id=sid)
        return _discover(root_path, session_id=sid)

    @classmethod
    def from_paths(
        cls,
        *,
        session_id: str,
        root: Path,
        label: str,
        reference_genome: Path | None = None,
        reference_annotation: Path | None = None,
        alignments: Path | None = None,
        alignment_blocks: Path | None = None,
        coverage: Path | None = None,
        introns: Path | None = None,
        derived_annotation: Path | None = None,
        clusters: Path | None = None,
        cluster_membership: Path | None = None,
        samples: tuple[str, ...] = (),
    ) -> "Session":
        """Construct a Session directly from explicit paths.

        Used by the `--reference DIR --align-dir DIR` CLI shape that
        bypasses session-style layouts. Each path is converted to an
        absolute path; `None` slots stay `None`.
        """

        def _abs(p: Path | None) -> Path | None:
            return Path(p).expanduser().resolve() if p is not None else None

        return cls(
            session_id=session_id,
            root=Path(root).expanduser().resolve(),
            label=label,
            reference_genome=_abs(reference_genome),
            reference_annotation=_abs(reference_annotation),
            alignments=_abs(alignments),
            alignment_blocks=_abs(alignment_blocks),
            coverage=_abs(coverage),
            introns=_abs(introns),
            derived_annotation=_abs(derived_annotation),
            clusters=_abs(clusters),
            cluster_membership=_abs(cluster_membership),
            samples=samples,
        )

    # ------------------------------------------------------------------
    # Inspection helpers (consumed by /api/sessions/{id}/manifest)
    # ------------------------------------------------------------------

    def stages_present(self) -> dict[str, bool]:
        """Return a small map of stage-name → present-bool for the UI."""
        return {
            "reference_genome": self.reference_genome is not None,
            "reference_annotation": self.reference_annotation is not None,
            "alignments": self.alignments is not None,
            "coverage": self.coverage is not None,
            "introns": self.introns is not None,
            "derived_annotation": self.derived_annotation is not None,
            "clusters": self.clusters is not None,
        }

    def to_manifest(self) -> dict[str, Any]:
        """Serialize the resolved Session to JSON-friendly form for
        `GET /api/sessions/{id}/manifest`."""

        def _rel(p: Path | None) -> str | None:
            if p is None:
                return None
            try:
                return str(p.relative_to(self.root))
            except ValueError:
                return str(p)

        return {
            "session_id": self.session_id,
            "schema_version": self.schema_version,
            "root": str(self.root),
            "label": self.label,
            "samples": list(self.samples),
            "stages_present": self.stages_present(),
            "paths": {
                "reference_genome": _rel(self.reference_genome),
                "reference_annotation": _rel(self.reference_annotation),
                "alignments": _rel(self.alignments),
                "alignment_blocks": _rel(self.alignment_blocks),
                "alignment_cs": _rel(self.alignment_cs),
                "coverage": _rel(self.coverage),
                "introns": _rel(self.introns),
                "derived_annotation": _rel(self.derived_annotation),
                "block_exon_assignments": _rel(self.block_exon_assignments),
                "exon_psi": _rel(self.exon_psi),
                "clusters": _rel(self.clusters),
                "cluster_membership": _rel(self.cluster_membership),
            },
            "extras": dict(self.extras),
        }


# ----------------------------------------------------------------------
# session.toml loader
# ----------------------------------------------------------------------


def _load_session_toml(path: Path, *, root: Path, session_id: str) -> Session:
    """Parse a `session.toml` file into a Session.

    Schema (v1)::

        schema_version = 1
        label = "..."

        [reference]
        handle = "homo_sapiens@ensembl-111"  # resolves via reference cache
        # genome = "<path>"                  # absolute or root-relative path
        # annotation = "<path>"              # — wins over `handle` when set

        [stages.s2_align]
        path = "S2_align"            # base path for relative resolution
        alignments = "alignments"    # all relative to `path`, optional
        alignment_blocks = "alignment_blocks"
        coverage = "coverage.parquet"
        introns = "introns.parquet"
        derived_annotation = "derived_annotation"
        samples = ["sample_1", ...]

        [stages.s2_cluster]
        path = "S2_cluster"
        clusters = "clusters.parquet"
        cluster_membership = "cluster_membership.parquet"

        [extras]
        # arbitrary kernel-specific overrides; not interpreted by the
        # session loader — pass through to kernels via `session.extras`.
    """
    raw = tomllib.loads(path.read_text(encoding="utf-8"))

    schema_version = int(raw.get("schema_version", 1))
    if schema_version != 1:
        raise ValueError(
            f"unsupported session.toml schema_version={schema_version} at {path}; "
            "this constellation supports v1"
        )
    label = str(raw.get("label") or root.name)

    def _resolve(rel: Any, *, base: Path = root) -> Path | None:
        if rel is None or rel == "":
            return None
        p = Path(str(rel)).expanduser()
        if not p.is_absolute():
            p = (base / p).resolve()
        else:
            p = p.resolve()
        return p

    ref = raw.get("reference", {}) or {}
    reference_genome = _resolve(ref.get("genome"))
    reference_annotation = _resolve(ref.get("annotation"))

    # Handle resolution — only applies when explicit genome/annotation
    # paths are absent. Explicit paths always win.
    handle_str = ref.get("handle")
    if handle_str and (reference_genome is None or reference_annotation is None):
        cache_genome, cache_annotation = _resolve_handle(str(handle_str), path=path)
        if reference_genome is None:
            reference_genome = cache_genome
        if reference_annotation is None:
            reference_annotation = cache_annotation

    stages = raw.get("stages", {}) or {}
    s2_align = stages.get("s2_align", {}) or {}
    s2_cluster = stages.get("s2_cluster", {}) or {}

    align_base = _resolve(s2_align.get("path", "S2_align")) or root
    alignments = _resolve(s2_align.get("alignments", "alignments"), base=align_base)
    alignment_blocks = _resolve(s2_align.get("alignment_blocks", "alignment_blocks"), base=align_base)
    alignment_cs = _resolve(s2_align.get("alignment_cs"), base=align_base)
    coverage = _resolve(s2_align.get("coverage", "coverage.parquet"), base=align_base)
    introns = _resolve(s2_align.get("introns", "introns.parquet"), base=align_base)
    derived_annotation = _resolve(s2_align.get("derived_annotation", "derived_annotation"), base=align_base)
    block_exon_assignments = _resolve(s2_align.get("block_exon_assignments", "block_exon_assignments.parquet"), base=align_base)
    exon_psi = _resolve(s2_align.get("exon_psi", "exon_psi.parquet"), base=align_base)
    samples = tuple(s2_align.get("samples", []))

    cluster_base = _resolve(s2_cluster.get("path", "S2_cluster")) or root
    clusters = _resolve(s2_cluster.get("clusters", "clusters.parquet"), base=cluster_base)
    cluster_membership = _resolve(s2_cluster.get("cluster_membership", "cluster_membership.parquet"), base=cluster_base)

    extras = dict(raw.get("extras", {}) or {})

    return _drop_missing(
        Session(
            session_id=session_id,
            root=root,
            label=label,
            schema_version=schema_version,
            reference_genome=reference_genome,
            reference_annotation=reference_annotation,
            alignments=alignments,
            alignment_blocks=alignment_blocks,
            alignment_cs=alignment_cs,
            coverage=coverage,
            introns=introns,
            derived_annotation=derived_annotation,
            block_exon_assignments=block_exon_assignments,
            exon_psi=exon_psi,
            clusters=clusters,
            cluster_membership=cluster_membership,
            samples=samples,
            extras=extras,
        )
    )


def _resolve_handle(handle_str: str, *, path: Path) -> tuple[Path | None, Path | None]:
    """Resolve a session.toml [reference] handle to (genome_dir, annotation_dir).

    Looks up the handle in the per-user reference cache via
    :mod:`constellation.sequencing.reference.handle`. Missing cache
    entries surface as a ``ValueError`` referencing the session.toml so
    the user knows where to look.
    """
    # Lazy import: keeps the viz dashboard cheap-to-load when no session
    # uses a handle. The handle module itself is stdlib-only.
    from constellation.sequencing.reference.handle import (
        ReferenceNotInstalledError,
        resolve as resolve_handle,
    )

    try:
        release_dir = resolve_handle(handle_str)
    except (ValueError, ReferenceNotInstalledError) as exc:
        raise ValueError(
            f"session.toml at {path} references handle {handle_str!r}, "
            f"but: {exc}"
        ) from exc
    genome_dir = release_dir / "genome"
    annotation_dir = release_dir / "annotation"
    return (
        genome_dir if genome_dir.is_dir() else None,
        annotation_dir if annotation_dir.is_dir() else None,
    )


# ----------------------------------------------------------------------
# Implicit directory-walk discovery
# ----------------------------------------------------------------------


def _discover(root: Path, *, session_id: str) -> Session:
    """Walk `root` one level deep, classifying directories as stages.

    Recognized layouts:

    - `<root>/genome/`          → reference_genome (ParquetDir)
    - `<root>/annotation/`      → reference_annotation (ParquetDir)
    - `<root>/<stage>/alignments/`
    - `<root>/<stage>/alignment_blocks/`
    - `<root>/<stage>/coverage.parquet`
    - `<root>/<stage>/introns.parquet`
    - `<root>/<stage>/derived_annotation/`
    - `<root>/<stage>/clusters.parquet`
    - `<root>/<stage>/cluster_membership.parquet`

    Where `<stage>` is any subdir whose name matches the corresponding
    heuristic (e.g. `S2_align`, `aligned`, `S2_cluster`). The first
    match wins for each artifact slot — a session with two `S2_align/`
    candidates will pick whichever the filesystem orders first. Use
    `session.toml` for reproducible disambiguation.
    """
    align_dir: Path | None = None
    cluster_dir: Path | None = None
    reference_genome: Path | None = None
    reference_annotation: Path | None = None

    for entry in sorted(root.iterdir()):
        if not entry.is_dir():
            continue
        name = entry.name
        if reference_genome is None and _matches(name, _STAGE_REFERENCE_GENOME):
            reference_genome = entry
        elif reference_annotation is None and _matches(
            name, _STAGE_REFERENCE_ANNOTATION
        ) and not _matches(name, _STAGE_DERIVED_ANNOTATION):
            reference_annotation = entry
        elif align_dir is None and _matches(name, _STAGE_S2_ALIGN):
            align_dir = entry
        elif cluster_dir is None and _matches(name, _STAGE_S2_CLUSTER):
            cluster_dir = entry

    def _exists(base: Path | None, name: str) -> Path | None:
        if base is None:
            return None
        candidate = base / name
        return candidate if candidate.exists() else None

    samples = _read_samples_from_align_manifest(align_dir)
    label = root.name

    return _drop_missing(
        Session(
            session_id=session_id,
            root=root,
            label=label,
            reference_genome=reference_genome,
            reference_annotation=reference_annotation,
            alignments=_exists(align_dir, "alignments"),
            alignment_blocks=_exists(align_dir, "alignment_blocks"),
            alignment_cs=_exists(align_dir, "alignment_cs"),
            coverage=_exists(align_dir, "coverage.parquet"),
            introns=_exists(align_dir, "introns.parquet"),
            derived_annotation=_exists(align_dir, "derived_annotation"),
            block_exon_assignments=_exists(align_dir, "block_exon_assignments.parquet"),
            exon_psi=_exists(align_dir, "exon_psi.parquet"),
            clusters=_exists(cluster_dir, "clusters.parquet"),
            cluster_membership=_exists(cluster_dir, "cluster_membership.parquet"),
            samples=samples,
        )
    )


def _read_samples_from_align_manifest(align_dir: Path | None) -> tuple[str, ...]:
    """Best-effort: read `<align_dir>/manifest.json` and pull the
    `samples` list. Missing or malformed → return empty tuple. Only
    pulls strings; non-string entries silently dropped."""
    if align_dir is None:
        return ()
    manifest_path = align_dir / "manifest.json"
    if not manifest_path.exists():
        return ()
    try:
        import json

        data = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return ()
    samples = data.get("samples")
    if not isinstance(samples, list):
        return ()
    return tuple(s for s in samples if isinstance(s, str))


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _derive_session_id(root: Path) -> str:
    """Stable short id for a path. Used as a URL component, so we keep
    it filesystem-safe. blake2b for speed; first 8 hex chars are plenty
    given the dashboard expects O(few) sessions per process."""
    digest = hashlib.blake2b(str(root).encode("utf-8"), digest_size=4).hexdigest()
    return f"{_slug(root.name)}-{digest}"


def _slug(s: str) -> str:
    out = []
    for ch in s.lower():
        if ch.isalnum():
            out.append(ch)
        elif ch in "-_":
            out.append(ch)
        else:
            out.append("-")
    return "".join(out).strip("-") or "session"


def _drop_missing(session: Session) -> Session:
    """Replace path slots whose target doesn't exist on disk with
    `None`. The discovery layer can't guarantee every slot it picked is
    valid (e.g. the user pointed `--session` at a partial run); kernels
    react to `None` by skipping their bindings. Idempotent."""
    updates: dict[str, Path | None] = {}
    for attr in (
        "reference_genome",
        "reference_annotation",
        "alignments",
        "alignment_blocks",
        "alignment_cs",
        "coverage",
        "introns",
        "derived_annotation",
        "block_exon_assignments",
        "exon_psi",
        "clusters",
        "cluster_membership",
    ):
        value: Path | None = getattr(session, attr)
        if value is not None and not value.exists():
            updates[attr] = None
    if not updates:
        return session
    return replace(session, **updates)
