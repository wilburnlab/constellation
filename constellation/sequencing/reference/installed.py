"""High-level ``Reference`` API over an installed reference-cache slot.

Read-only view onto ``~/.constellation/references/<organism>/<source>-<release>/``.
``Reference.open(handle)`` resolves a handle (qualified or bare organism)
to a cache directory and exposes:

  * Path views — ``genome_dir``, ``annotation_dir``, ``protein_fasta_path``,
    ``cdna_fasta_path``, ``meta_path``. These do NOT verify existence;
    callers gate via the boolean presence accessors below.
  * Presence flags — ``has_genome`` / ``has_annotation`` / ``has_proteome``
    / ``has_cdna``. Reads the v2 ``[contents]`` block in ``meta.toml``;
    falls back to on-disk file existence for v1 caches.
  * Provenance — ``organism`` / ``source`` / ``release`` /
    ``assembly_accession`` / ``annotation_release``.
  * Eager loaders — ``load_genome()``, ``load_annotation()``,
    ``gene_map()``. ``gene_map()`` resolves a ``protein_id ->
    gene_symbol`` map from the annotation parquet bundle, falling back
    to GFF3 / GBFF parsing if the bundle isn't present (used by callers
    that imported a raw GFF/GBFF via ``constellation reference import``).
  * ``require(*, proteome=False, annotation=False, genome=False,
    cdna=False)`` — raises ``ReferenceNotInstalledError`` with an
    actionable hint when a required artifact isn't installed.

The ``Reference`` API is read-only — it never fetches. If a caller asks
for an artifact that isn't installed, the error message points at the
``constellation reference fetch`` CLI verb. This keeps the resolver
mono-modal (the orchestrator + viz layer + future bridges all consume
``Reference`` objects) without dragging the fetch dependency chain
(catalog, taxonomy, urllib) into every call site.

Note: ``transcriptome align`` / ``transcriptome cluster`` currently
construct their genome + annotation paths via ``resolve(handle) /
"genome"`` directly (not via ``Reference.open()``). Migrating those
call sites to this API is a deferred follow-up — the API is designed
to support it (``Reference.load_genome()`` returns the same
``GenomeReference`` they already work with).
"""

from __future__ import annotations

import gzip
import json as _json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from constellation.sequencing.reference.handle import (
    Handle,
    ReferenceNotInstalledError,
    parse_handle,
    read_meta_toml,
    resolve,
)


__all__ = ["Reference"]


# ──────────────────────────────────────────────────────────────────────
# Reference
# ──────────────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class Reference:
    """Read-only view onto one installed reference-cache slot.

    Construct via ``Reference.open(handle)``. Direct construction
    requires a fully-qualified handle and an existing release directory
    — callers should prefer ``open()`` which does both.
    """

    handle: Handle
    release_dir: Path
    # Parsed meta.toml. Synthesised for v1 caches without a [contents]
    # block; always carries a 'contents' key after read_meta_toml().
    _meta: dict[str, Any] = field(default_factory=dict, repr=False)

    @classmethod
    def open(
        cls, handle: Handle | str, *, root: Path | None = None
    ) -> "Reference":
        """Resolve a handle to an installed cache slot.

        Accepts a fully-qualified handle, a bare organism slug, or a
        ``Handle`` dataclass. Bare organisms resolve via
        ``defaults.toml`` / the ``current/`` symlink / single-install
        precedence chain (see ``handle.resolve``).

        Raises ``ReferenceNotInstalledError`` when the handle does not
        resolve to an on-disk slot.
        """
        h = parse_handle(handle) if isinstance(handle, str) else handle
        release_dir = resolve(h, root=root)
        # Re-build a qualified Handle when ``open()`` was given a bare
        # organism — the resolved directory name is authoritative.
        if not h.is_qualified():
            slug = release_dir.name
            from constellation.sequencing.reference.handle import parse_release_slug
            h = parse_release_slug(slug, organism=h.organism)
        meta = read_meta_toml(release_dir) or {}
        return cls(handle=h, release_dir=release_dir, _meta=meta)

    # ── Path views ───────────────────────────────────────────────────
    @property
    def genome_dir(self) -> Path:
        return self.release_dir / "genome"

    @property
    def annotation_dir(self) -> Path:
        return self.release_dir / "annotation"

    @property
    def protein_fasta_path(self) -> Path:
        return self.release_dir / "protein.faa"

    @property
    def cdna_fasta_path(self) -> Path:
        return self.release_dir / "cdna.fna"

    @property
    def meta_path(self) -> Path:
        return self.release_dir / "meta.toml"

    # ── Presence flags ───────────────────────────────────────────────
    @property
    def has_genome(self) -> bool:
        return self._contents_flag("has_genome", self.genome_dir.is_dir())

    @property
    def has_annotation(self) -> bool:
        return self._contents_flag("has_annotation", self.annotation_dir.is_dir())

    @property
    def has_proteome(self) -> bool:
        return self._contents_flag(
            "has_proteome", self.protein_fasta_path.is_file()
        )

    @property
    def has_cdna(self) -> bool:
        return self._contents_flag("has_cdna", self.cdna_fasta_path.is_file())

    def _contents_flag(self, key: str, default: bool) -> bool:
        contents = self._meta.get("contents") or {}
        value = contents.get(key)
        if value is None:
            return default
        return bool(value)

    # ── Provenance ───────────────────────────────────────────────────
    @property
    def organism(self) -> str:
        return self.handle.organism

    @property
    def source(self) -> str:
        # handle is always qualified after open(); source is non-None.
        assert self.handle.source is not None
        return self.handle.source

    @property
    def release(self) -> str:
        assert self.handle.release is not None
        return self.handle.release

    @property
    def assembly_accession(self) -> str | None:
        return _str_or_none(self._meta, "assembly_accession")

    @property
    def annotation_release(self) -> str | None:
        return _str_or_none(self._meta, "annotation_release")

    # ── Eager loaders ────────────────────────────────────────────────
    def load_genome(self):
        """Load the cached ``GenomeReference``.

        Late-imports the loader to keep ``installed.py`` light — callers
        that only use path-views never pay the ``GenomeReference``
        construction cost.
        """
        self.require(genome=True)
        from constellation.sequencing.reference.io import load_genome_reference

        return load_genome_reference(self.genome_dir)

    def load_annotation(self):
        """Load the cached ``Annotation`` (FEATURE_TABLE parquet bundle)."""
        self.require(annotation=True)
        from constellation.sequencing.annotation.io import load_annotation

        return load_annotation(self.annotation_dir)

    def gene_map(self) -> dict[str, str]:
        """Return ``protein_id -> gene_symbol`` for this reference.

        Prefers the parquet annotation bundle (canonical form produced
        by ``constellation reference fetch``); falls back to legacy GFF3
        / GBFF parsing on the release directory when present. Used by
        the transcriptome-to-proteome orchestration to annotate
        reference protein headers with their gene symbol.
        """
        self.require(annotation=True)
        return _gene_map_from_dir(self.annotation_dir)

    # ── Requirements ─────────────────────────────────────────────────
    def require(
        self,
        *,
        proteome: bool = False,
        annotation: bool = False,
        genome: bool = False,
        cdna: bool = False,
    ) -> None:
        """Assert the requested artifacts are installed; raise with a hint if not.

        The error message names the missing artifact and includes the
        concrete CLI command the user should run.
        """
        missing: list[tuple[str, str]] = []
        if genome and not self.has_genome:
            missing.append(
                (
                    "genome",
                    f"constellation reference fetch "
                    f"--organism {self.organism} --source {self.source}",
                )
            )
        if annotation and not self.has_annotation:
            missing.append(
                (
                    "annotation",
                    f"constellation reference fetch "
                    f"--organism {self.organism} --source {self.source}",
                )
            )
        if proteome and not self.has_proteome:
            missing.append(
                (
                    "proteome",
                    f"constellation reference fetch "
                    f"--organism {self.organism} --source uniprot",
                )
            )
        if cdna and not self.has_cdna:
            missing.append(
                (
                    "cdna",
                    f"constellation reference fetch "
                    f"--organism {self.organism} --source {self.source}",
                )
            )
        if not missing:
            return
        kinds = ", ".join(k for k, _ in missing)
        hints = "\n  ".join(cmd for _, cmd in missing)
        raise ReferenceNotInstalledError(
            f"reference {self.handle} is missing required artifact(s): {kinds}. "
            f"Run:\n  {hints}"
        )


def _str_or_none(meta: dict[str, Any] | None, key: str) -> str | None:
    if not meta:
        return None
    value = meta.get(key)
    return str(value) if value is not None else None


# ──────────────────────────────────────────────────────────────────────
# Gene-map helpers (moved from constellation/transcriptome_to_proteome.py)
# ──────────────────────────────────────────────────────────────────────


_GFF_PROTEIN_ID_RE = re.compile(r"protein_id=([^;]+)")
_GFF_GENE_RE = re.compile(r"\bgene=([^;]+)")
_GBFF_QUALIFIER_RE = re.compile(r'^\s+/(\w+)=(.*)$')


def _gene_map_from_dir(annotation_dir: Path) -> dict[str, str]:
    """Resolve a ``protein_id -> gene_symbol`` map from an annotation
    location.

    Dispatches on what's on disk:
      * Constellation ParquetDir bundle (preferred) — read via
        :func:`load_annotation`, filter to CDS rows, decode
        ``attributes_json``.
      * Legacy raw GFF3 / GBFF file alongside (rare; only when the user
        ``reference import``-ed a non-parquet bundle).

    Used by :meth:`Reference.gene_map`. The path-based dispatcher in
    ``constellation/transcriptome_to_proteome.py`` (``read_reference_gene_map``)
    delegates to this helper for the parquet case and to the GFF3/GBFF
    helpers below for raw-file inputs.
    """
    if (annotation_dir / "features.parquet").is_file():
        return _gene_map_from_parquet(annotation_dir)
    # Look for sibling GFF3 / GBFF files (rare — only when the user
    # imported a raw annotation outside the parquet bundle format).
    for suffix in (".gff3", ".gff3.gz", ".gff", ".gff.gz"):
        candidate = annotation_dir / f"annotation{suffix}"
        if candidate.is_file():
            return _gene_map_from_gff3(candidate)
    for suffix in (".gbff", ".gbff.gz", ".gb", ".gb.gz", ".genbank", ".genbank.gz"):
        candidate = annotation_dir / f"annotation{suffix}"
        if candidate.is_file():
            return _gene_map_from_gbff(candidate)
    raise ValueError(
        f"no annotation file found at {annotation_dir!r}: expected a "
        f"Constellation parquet bundle (features.parquet + manifest.json), "
        f"or a sibling annotation.{{gff3,gbff}}[.gz]"
    )


def _gene_map_from_parquet(annotation_dir: Path) -> dict[str, str]:
    """Extract ``protein_id -> gene_symbol`` from a Constellation
    annotation ParquetDir bundle.

    Filters to CDS rows with Arrow compute and decodes per-row
    ``attributes_json`` to pull ``protein_id`` + ``gene`` keys. CDS
    attributes land in the JSON blob (not promoted to columns).
    """
    import pyarrow.compute as pc
    from constellation.sequencing.annotation.io import load_annotation

    annotation = load_annotation(annotation_dir)
    cds_mask = pc.equal(annotation.features.column("type"), "CDS")
    cds = annotation.features.filter(cds_mask)
    attrs_col = cds.column("attributes_json").to_pylist()

    out: dict[str, str] = {}
    for blob in attrs_col:
        if not blob:
            continue
        try:
            attrs = _json.loads(blob)
        except (TypeError, ValueError):
            continue
        protein_id = attrs.get("protein_id")
        gene = attrs.get("gene") or attrs.get("gene_name")
        if protein_id and gene:
            out.setdefault(str(protein_id), str(gene))
    return out


def _gene_map_from_gff3(path: Path) -> dict[str, str]:
    """Scan a GFF3 file for CDS records, extracting protein_id + gene."""
    out: dict[str, str] = {}
    with _open_text(path) as fh:
        for line in fh:
            if not line or line.startswith("#"):
                continue
            cols = line.rstrip("\n").split("\t")
            if len(cols) < 9:
                continue
            if cols[2] != "CDS":
                continue
            attrs = cols[8]
            m_pid = _GFF_PROTEIN_ID_RE.search(attrs)
            m_gene = _GFF_GENE_RE.search(attrs)
            if m_pid is None or m_gene is None:
                continue
            protein_id = m_pid.group(1).strip()
            gene = m_gene.group(1).strip()
            if protein_id and gene:
                out[protein_id] = gene
    return out


def _gene_map_from_gbff(path: Path) -> dict[str, str]:
    """Scan a GBFF file's CDS qualifier blocks for /protein_id + /gene.

    Walks the FEATURES section, opens a per-CDS qualifier dict on each
    ``CDS`` feature header, accumulates ``/protein_id="..."`` +
    ``/gene="..."`` lines, and flushes the pair on the next feature
    boundary or end-of-features marker.
    """
    out: dict[str, str] = {}
    in_features = False
    in_cds = False
    current: dict[str, str] = {}

    def _flush() -> None:
        nonlocal current
        pid = current.get("protein_id")
        gene = current.get("gene")
        if pid and gene:
            out[pid] = gene
        current = {}

    with _open_text(path) as fh:
        for raw in fh:
            line = raw.rstrip("\n")
            if line.startswith("FEATURES"):
                in_features = True
                continue
            if line.startswith("ORIGIN") or line.startswith("//"):
                if in_cds:
                    _flush()
                    in_cds = False
                in_features = False
                continue
            if not in_features:
                continue
            # Feature header: cols 5..21 carry the feature type.
            if line[:5] == "     " and line[5:6].strip():
                if in_cds:
                    _flush()
                head = line[5:21].strip()
                in_cds = head == "CDS"
                continue
            if not in_cds:
                continue
            m = _GBFF_QUALIFIER_RE.match(line)
            if m is None:
                continue
            key = m.group(1)
            val = m.group(2)
            if val.startswith('"') and val.endswith('"'):
                val = val[1:-1]
            if key in {"protein_id", "gene"}:
                current[key] = val.strip()
    if in_cds:
        _flush()
    return out


def _open_text(path: Path):
    """Open a FASTA / GBFF / GFF3 for text reading, transparently
    handling gzip."""
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("r", encoding="utf-8")
