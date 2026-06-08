"""``Assembly`` container — de novo contigs, scaffolds, summary stats.

Distinct from :class:`GenomeReference` because an assembly carries
provenance Reference doesn't need: read-coverage per contig,
polishing-round counts, haplotype tags, scaffolding edges. Once an
assembly is finalized, ``Assembly.to_genome_reference()`` lifts it into
a ``GenomeReference`` so downstream alignment / quant code is uniform
across external-ref and de-novo-ref workflows. Annotation features
(genes, repeats, telomeres) are produced separately by downstream
BUSCO / RepeatMasker / telomere passes and live in their own
:class:`sequencing.annotation.Annotation` container that pairs with the
GenomeReference.

Tables:

    contigs       ASSEMBLY_CONTIG_TABLE — per-contig provenance
    sequences     SEQUENCE_TABLE        — same shape as GenomeReference
    scaffolds     SCAFFOLD_TABLE        — optional; ``None`` when no
                                          scaffolding has run
    stats         ASSEMBLY_STATS        — one-row summary (N50, BUSCO)

Mirrors the :class:`GenomeReference` template: cast on construction,
PK / FK validation, ParquetDir round-trip via :mod:`sequencing.assembly.io`.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

import pyarrow as pa

from constellation.core.io.schemas import cast_to_schema
from constellation.sequencing.reference.reference import GenomeReference
from constellation.sequencing.schemas.assembly import (
    ASSEMBLY_CONTIG_TABLE,
    ASSEMBLY_STATS,
    SCAFFOLD_TABLE,
)
from constellation.sequencing.schemas.reference import CONTIG_TABLE, SEQUENCE_TABLE


@dataclass(frozen=True, slots=True)
class Assembly:
    """De novo assembly product (draft, scaffolded, or polished).

    ``to_genome_reference()`` projects the contigs + sequences into a
    ``GenomeReference`` so the rest of the pipeline can treat
    external-ref and de-novo-ref workflows identically. Annotation
    features (BUSCO genes, RepeatMasker hits, telomere detections) are
    built separately and ride in a paired ``Annotation`` container.
    """

    contigs: pa.Table
    sequences: pa.Table
    scaffolds: pa.Table | None
    stats: pa.Table
    metadata_extras: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "contigs", cast_to_schema(self.contigs, ASSEMBLY_CONTIG_TABLE)
        )
        object.__setattr__(
            self, "sequences", cast_to_schema(self.sequences, SEQUENCE_TABLE)
        )
        if self.scaffolds is not None:
            object.__setattr__(
                self, "scaffolds", cast_to_schema(self.scaffolds, SCAFFOLD_TABLE)
            )
        object.__setattr__(self, "stats", cast_to_schema(self.stats, ASSEMBLY_STATS))
        self.validate()

    # ── construction ────────────────────────────────────────────────
    @classmethod
    def from_tables(
        cls,
        contigs: pa.Table,
        sequences: pa.Table,
        *,
        scaffolds: pa.Table | None = None,
        metadata_extras: dict[str, Any] | None = None,
    ) -> "Assembly":
        """Build an ``Assembly``, computing ``stats`` from the tables.

        The ergonomic constructor for the runners (and tests): callers
        supply contigs + sequences (+ optional scaffolds) and the
        ``ASSEMBLY_STATS`` row is derived via
        :func:`sequencing.assembly.stats._stats_from_tables`.
        """
        # Deferred import — stats.py only references Assembly under
        # TYPE_CHECKING, so importing it here avoids a runtime cycle.
        from constellation.sequencing.assembly.stats import _stats_from_tables

        contigs = cast_to_schema(contigs, ASSEMBLY_CONTIG_TABLE)
        sequences = cast_to_schema(sequences, SEQUENCE_TABLE)
        scaff = (
            cast_to_schema(scaffolds, SCAFFOLD_TABLE) if scaffolds is not None else None
        )
        stats = _stats_from_tables(contigs, sequences, scaff)
        return cls(
            contigs=contigs,
            sequences=sequences,
            scaffolds=scaff,
            stats=stats,
            metadata_extras=dict(metadata_extras or {}),
        )

    @classmethod
    def from_genome_reference(
        cls,
        genome: GenomeReference,
        *,
        scaffolds: pa.Table | None = None,
        haplotype: str | None = None,
        polish_rounds: int | None = None,
        provenance_json: str | None = None,
        metadata_extras: dict[str, Any] | None = None,
    ) -> "Assembly":
        """Build an ``Assembly`` from a ``GenomeReference`` (e.g. one read
        back from a scaffolded / polished FASTA), stamping scalar
        provenance onto every contig. Used by the scaffold + polish stages
        whose tool outputs are FASTA files."""
        contigs = cast_to_schema(genome.contigs, ASSEMBLY_CONTIG_TABLE)
        n = contigs.num_rows

        def _fill(table: pa.Table, name: str, value: Any, typ: pa.DataType) -> pa.Table:
            idx = table.schema.get_field_index(name)
            return table.set_column(idx, name, pa.array([value] * n, type=typ))

        if haplotype is not None:
            contigs = _fill(contigs, "haplotype", haplotype, pa.string())
        if polish_rounds is not None:
            contigs = _fill(contigs, "polish_rounds", polish_rounds, pa.int32())
        if provenance_json is not None:
            contigs = _fill(contigs, "provenance_json", provenance_json, pa.string())
        return cls.from_tables(
            contigs,
            genome.sequences,
            scaffolds=scaffolds,
            metadata_extras=metadata_extras,
        )

    # ── validation ──────────────────────────────────────────────────
    def validate(self) -> None:
        """PK uniqueness on ``contig_id`` + FK closure
        (``sequences.contig_id`` ⊆ ``contigs.contig_id``), one sequence
        per contig.

        ``scaffolds.contig_id`` is **not** FK-checked against ``contigs``:
        after scaffolding the container's contigs become the scaffold
        sequences, while ``SCAFFOLD_TABLE`` records how the *pre-scaffold*
        draft contigs composed them — a distinct id space kept as
        provenance.
        """
        _check_unique(self.contigs, "contig_id")
        contig_ids = set(self.contigs.column("contig_id").to_pylist())
        _check_fk(self.sequences, "contig_id", contig_ids, "ASSEMBLY_CONTIG_TABLE")
        seq_contigs = self.sequences.column("contig_id").to_pylist()
        if len(set(seq_contigs)) != len(seq_contigs):
            raise ValueError(
                "SEQUENCE_TABLE has multiple rows for the same contig_id; "
                "Assembly allows one sequence per contig"
            )

    # ── projection ──────────────────────────────────────────────────
    def to_genome_reference(self) -> GenomeReference:
        """Project this Assembly into a ``GenomeReference``.

        Keeps contig_id / name / length / circular (topology filled
        null); drops assembly-only provenance (read_coverage,
        polish_rounds, haplotype). Annotation features are NOT carried —
        those build downstream into a sibling ``Annotation`` container.
        This is what reference / draft FASTA materialization for RagTag /
        polish / dorado-aligner consumes.
        """
        contigs = cast_to_schema(self.contigs, CONTIG_TABLE)
        return GenomeReference(
            contigs=contigs,
            sequences=self.sequences,
            metadata_extras=dict(self.metadata_extras),
        )

    # ── views ───────────────────────────────────────────────────────
    @property
    def n_contigs(self) -> int:
        return self.contigs.num_rows

    @property
    def total_length(self) -> int:
        return int(sum(int(x) for x in self.contigs.column("length").to_pylist()))

    @property
    def metadata(self) -> dict[str, Any]:
        return dict(self.metadata_extras)

    def with_metadata(self, extras: dict[str, Any]) -> "Assembly":
        merged = dict(self.metadata_extras)
        merged.update(extras)
        return replace(self, metadata_extras=merged)


# ──────────────────────────────────────────────────────────────────────
# Validation helpers (sibling to reference.py — kept local to avoid
# importing module-private names across container modules).
# ──────────────────────────────────────────────────────────────────────


def _check_unique(table: pa.Table, column: str) -> None:
    values = table.column(column).to_pylist()
    if len(set(values)) != len(values):
        seen: set[Any] = set()
        dups: list[Any] = []
        for v in values:
            if v in seen:
                dups.append(v)
            else:
                seen.add(v)
        raise ValueError(
            f"{column} contains duplicate values: "
            f"{dups[:5]}{'...' if len(dups) > 5 else ''}"
        )


def _check_fk(
    table: pa.Table,
    column: str,
    valid_ids: set[Any],
    target_name: str,
) -> None:
    values = table.column(column).to_pylist()
    missing = {v for v in values if v is not None and v not in valid_ids}
    if missing:
        sample = sorted(missing)[:5]
        raise ValueError(
            f"{column} references ids not present in {target_name}: "
            f"{sample}{'...' if len(missing) > 5 else ''}"
        )


__all__ = ["Assembly"]
