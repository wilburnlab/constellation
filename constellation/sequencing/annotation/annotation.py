"""``Annotation`` container — GFF-shaped features keyed against a genome.

Validation on construction: cast features to ``FEATURE_TABLE`` schema;
PK uniqueness on ``feature_id``; FK closure on ``parent_id`` (must be
``null`` or appear in this table's ``feature_id`` column).

``validate_against(genome)`` cross-checks ``contig_id`` ⊆
``GenomeReference.contigs.contig_id`` so an Annotation can be cleanly
paired with its companion ``GenomeReference``.

Two origins flow into this container:

    External annotation     GFF3 → Annotation (paired with the
                            companion GenomeReference)
    De novo passes          BUSCO / RepeatMasker / telomere finders
                            produce FEATURE_TABLE rows that aggregate
                            into a fresh Annotation (added downstream)

Annotation is sample-agnostic; per-feature counts live in
``FEATURE_QUANT`` rows tagged ``feature_origin='gene_id'`` (or other
feature-type granularities).
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

import pyarrow as pa
import pyarrow.compute as pc

from constellation.core.io.schemas import cast_to_schema
from constellation.sequencing.reference.reference import GenomeReference
from constellation.sequencing.schemas.reference import FEATURE_TABLE


@dataclass(frozen=True, slots=True)
class Annotation:
    """Bundles a single ``FEATURE_TABLE`` Arrow table."""

    features: pa.Table
    metadata_extras: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "features", cast_to_schema(self.features, FEATURE_TABLE)
        )
        self.validate()

    # ── validation ──────────────────────────────────────────────────
    def validate(self) -> None:
        """PK uniqueness on ``feature_id`` plus parent-id closure
        (``parent_id`` is ``null`` or references another feature_id in
        the same table)."""
        feature_ids = self.features.column("feature_id").to_pylist()
        if len(set(feature_ids)) != len(feature_ids):
            seen: set[int] = set()
            dups: list[int] = []
            for v in feature_ids:
                if v in seen:
                    dups.append(v)
                else:
                    seen.add(v)
            raise ValueError(
                f"feature_id contains duplicate values: "
                f"{dups[:5]}{'...' if len(dups) > 5 else ''}"
            )
        valid = set(feature_ids)
        parents = self.features.column("parent_id").to_pylist()
        missing = {p for p in parents if p is not None and p not in valid}
        if missing:
            sample = sorted(missing)[:5]
            raise ValueError(
                f"parent_id references feature_ids not present in this "
                f"Annotation: {sample}{'...' if len(missing) > 5 else ''}"
            )

    def validate_against(self, genome: GenomeReference) -> None:
        """Cross-check that every feature's ``contig_id`` is present in
        the supplied ``GenomeReference``."""
        valid = set(genome.contigs.column("contig_id").to_pylist())
        contig_ids = self.features.column("contig_id").to_pylist()
        missing = {c for c in contig_ids if c not in valid}
        if missing:
            sample = sorted(missing)[:5]
            raise ValueError(
                f"Annotation references contig_ids absent from GenomeReference: "
                f"{sample}{'...' if len(missing) > 5 else ''}"
            )

    # ── views ───────────────────────────────────────────────────────
    @property
    def n_features(self) -> int:
        return self.features.num_rows

    @property
    def metadata(self) -> dict[str, Any]:
        return dict(self.metadata_extras)

    def features_on(self, contig_id: int) -> pa.Table:
        """Return all features on a given contig."""
        mask = pc.equal(self.features.column("contig_id"), int(contig_id))
        return self.features.filter(mask)

    def features_of_type(self, type: str) -> pa.Table:
        """Return all features of a given SO type (e.g. 'gene', 'mRNA')."""
        mask = pc.equal(self.features.column("type"), type)
        return self.features.filter(mask)

    def with_metadata(self, extras: dict[str, Any]) -> "Annotation":
        merged = dict(self.metadata_extras)
        merged.update(extras)
        return replace(self, metadata_extras=merged)


__all__ = ["Annotation"]
