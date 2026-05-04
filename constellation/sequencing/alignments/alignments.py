"""``Alignments`` container — ALIGNMENT_TABLE + ALIGNMENT_TAG_TABLE.

Per-acquisition mapped-read records. Validation on construction:
``alignment_id`` PK uniqueness; ``acquisition_id`` FK closure into a
supplied ``Acquisitions``; tag-table ``alignment_id`` references must
exist in the parent table.

``validate_against(reference)`` cross-checks that every ``ref_name``
appears in the supplied ``GenomeReference.contigs.name`` set.

The in-memory container exists for tests, ParquetDir round-trip, and
small Jupyter exploration. The pipeline never materialises a full
Alignments at scale (30–200M alignments per real transcriptomics run);
production paths stream partitioned shards via the fused worker in
:mod:`sequencing.quant.genome_count` and only ever materialise compact
``gene_assignments`` rows in memory.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

import pyarrow as pa
import pyarrow.compute as pc

from constellation.core.io.schemas import cast_to_schema
from constellation.sequencing.acquisitions import Acquisitions, validate_acquisitions
from constellation.sequencing.reference.reference import GenomeReference
from constellation.sequencing.schemas.alignment import (
    ALIGNMENT_TABLE,
    ALIGNMENT_TAG_TABLE,
)


# CIGAR ops that consume reference (advance ref_end) — not used here, but
# kept as a comment for the parser table below to make intent obvious:
#   M, =, X, D, N — consume reference
#   M, =, X, I, S — consume query (used for read_length except H)
#   H — consumes neither but counts toward "original read length"
#
# Per SAM v1 spec §1.4.6:
#   aligned_bp  = sum of M/=/X op lengths
#   read_length = sum of M/=/X/I/S/H op lengths
_ALIGNED_OPS = frozenset({"M", "=", "X"})
_READ_LENGTH_OPS = frozenset({"M", "=", "X", "I", "S", "H"})


@dataclass(frozen=True, slots=True)
class Alignments:
    """Bundles ``ALIGNMENT_TABLE`` with its ``ALIGNMENT_TAG_TABLE``.

    Tag-table reference: alignment_id ⊆ alignments.alignment_id. Tag
    table may be empty (most alignments only carry promoted columns).
    """

    alignments: pa.Table
    tags: pa.Table
    acquisitions: Acquisitions | None = None
    metadata_extras: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        alignments = self.alignments
        tags = self.tags
        # Tolerate empty Arrow tables that lack the schema's columns
        # (commonly from ``pa.Table.from_pylist([])``).
        if alignments.num_rows == 0 and alignments.num_columns == 0:
            alignments = ALIGNMENT_TABLE.empty_table()
        if tags.num_rows == 0 and tags.num_columns == 0:
            tags = ALIGNMENT_TAG_TABLE.empty_table()
        object.__setattr__(
            self, "alignments", cast_to_schema(alignments, ALIGNMENT_TABLE)
        )
        object.__setattr__(self, "tags", cast_to_schema(tags, ALIGNMENT_TAG_TABLE))
        self.validate()

    # ── validation ──────────────────────────────────────────────────
    def validate(self) -> None:
        """PK uniqueness on alignment_id + FK closure (tags →
        alignments, alignments → acquisitions if set)."""
        ids = self.alignments.column("alignment_id").to_pylist()
        if len(set(ids)) != len(ids):
            seen: set[int] = set()
            dups: list[int] = []
            for v in ids:
                if v in seen:
                    dups.append(v)
                else:
                    seen.add(v)
            raise ValueError(
                f"alignment_id contains duplicate values: "
                f"{dups[:5]}{'...' if len(dups) > 5 else ''}"
            )
        valid = set(ids)
        tag_ids = self.tags.column("alignment_id").to_pylist()
        missing = {t for t in tag_ids if t not in valid}
        if missing:
            sample = sorted(missing)[:5]
            raise ValueError(
                f"ALIGNMENT_TAG_TABLE references alignment_ids absent "
                f"from ALIGNMENT_TABLE: "
                f"{sample}{'...' if len(missing) > 5 else ''}"
            )
        if self.acquisitions is not None:
            validate_acquisitions(
                self.alignments, self.acquisitions, column="acquisition_id"
            )

    def validate_against(self, reference: GenomeReference) -> None:
        """Check that every ``ref_name`` appears in the reference's
        contig name list."""
        valid = set(reference.contigs.column("name").to_pylist())
        ref_names = set(self.alignments.column("ref_name").to_pylist())
        missing = ref_names - valid
        if missing:
            sample = sorted(missing)[:5]
            raise ValueError(
                f"Alignments reference ref_names absent from GenomeReference: "
                f"{sample}{'...' if len(missing) > 5 else ''}"
            )

    # ── views ───────────────────────────────────────────────────────
    @property
    def n_alignments(self) -> int:
        return self.alignments.num_rows

    @property
    def metadata(self) -> dict[str, Any]:
        return dict(self.metadata_extras)

    def primary(self) -> "Alignments":
        """Filter to primary (non-secondary, non-supplementary) alignments."""
        return self.filter(primary_only=True)

    def filter(
        self,
        *,
        min_length: int | None = None,
        min_aligned_fraction: float | None = None,
        min_mapq: int | None = None,
        primary_only: bool = False,
        allow_antisense: bool = False,
    ) -> "Alignments":
        """Return a new ``Alignments`` with filter predicates applied.

        Each active predicate runs as a separate stage so per-stage
        row-count diagnostics surface in ``metadata_extras['filter_steps']``
        as a list of ``{stage, kept, dropped}`` dicts. Strand handling:
        ``allow_antisense=False`` (default) keeps both strands here —
        antisense filtering is applied at gene-overlap time against the
        annotation, since strandedness only matters relative to a feature.

        ``min_aligned_fraction`` triggers a per-row CIGAR walk to compute
        ``aligned_bp / read_length`` per SAM v1 spec §1.4.6; for one-shot
        pipeline use this is acceptable. See module-level note on
        production paths.
        """
        del allow_antisense  # accepted for symmetry; applied in compute_gene_overlap
        table = self.alignments
        steps: list[dict[str, int]] = []

        def _record(stage: str, before: int, after_table: pa.Table) -> pa.Table:
            after = after_table.num_rows
            steps.append({"stage": stage, "kept": after, "dropped": before - after})
            return after_table

        if primary_only:
            before = table.num_rows
            mask = pc.and_(
                pc.invert(table.column("is_secondary")),
                pc.invert(table.column("is_supplementary")),
            )
            table = _record("primary_only", before, table.filter(mask))

        if min_length is not None:
            before = table.num_rows
            length = pc.subtract(table.column("ref_end"), table.column("ref_start"))
            mask = pc.greater_equal(length, int(min_length))
            table = _record(f"min_length>={min_length}", before, table.filter(mask))

        if min_mapq is not None:
            before = table.num_rows
            mask = pc.greater_equal(table.column("mapq"), int(min_mapq))
            table = _record(f"min_mapq>={min_mapq}", before, table.filter(mask))

        if min_aligned_fraction is not None:
            before = table.num_rows
            af = _aligned_fraction_array(table)
            # Null aligned_fraction (degenerate CIGAR) is dropped — fail-safe.
            mask = pc.and_(
                pc.is_valid(af),
                pc.greater_equal(af, float(min_aligned_fraction)),
            )
            table = _record(
                f"min_aligned_fraction>={min_aligned_fraction}",
                before,
                table.filter(mask),
            )

        kept_ids = set(table.column("alignment_id").to_pylist())
        if kept_ids != set(self.alignments.column("alignment_id").to_pylist()):
            tag_mask = pc.is_in(
                self.tags.column("alignment_id"), value_set=pa.array(sorted(kept_ids))
            )
            new_tags = self.tags.filter(tag_mask)
        else:
            new_tags = self.tags

        new_metadata = dict(self.metadata_extras)
        prior = list(new_metadata.get("filter_steps", []))
        prior.extend(steps)
        new_metadata["filter_steps"] = prior

        return replace(
            self,
            alignments=table,
            tags=new_tags,
            metadata_extras=new_metadata,
        )

    def with_metadata(self, extras: dict[str, Any]) -> "Alignments":
        merged = dict(self.metadata_extras)
        merged.update(extras)
        return replace(self, metadata_extras=merged)


# ──────────────────────────────────────────────────────────────────────
# Aligned-fraction helper — per-row CIGAR walk
# ──────────────────────────────────────────────────────────────────────


def _aligned_fraction_array(table: pa.Table) -> pa.Array:
    """Compute ``aligned_bp / read_length`` per row by walking the
    ``cigar_string`` column.

    Per SAM v1 spec §1.4.6:
        aligned_bp  = sum of M/=/X op lengths
        read_length = sum of M/=/X/I/S/H op lengths

    Returns a float32 Arrow array; rows with degenerate / unparseable
    CIGAR (empty, ``*``, or zero read_length) get null. Hand-rolled
    state machine — no regex; ~30 lines per the plan.
    """
    cigars = table.column("cigar_string").to_pylist()
    out: list[float | None] = []
    for cig in cigars:
        if cig is None or cig == "" or cig == "*":
            out.append(None)
            continue
        aligned = 0
        readlen = 0
        digits: list[str] = []
        valid = True
        for ch in cig:
            if ch.isdigit():
                digits.append(ch)
                continue
            if not digits:
                valid = False
                break
            length = int("".join(digits))
            digits.clear()
            if ch in _ALIGNED_OPS:
                aligned += length
                readlen += length
            elif ch in _READ_LENGTH_OPS:
                readlen += length
            elif ch in {"D", "N", "P"}:
                # consume reference / pad — neither aligned_bp nor read_length
                pass
            else:
                valid = False
                break
        if not valid or digits or readlen == 0:
            out.append(None)
            continue
        out.append(aligned / readlen)
    return pa.array(out, type=pa.float32())


__all__ = ["Alignments"]
