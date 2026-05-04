"""``TranscriptReference`` container — spliced transcript records.

Holds one ``TRANSCRIPT_TABLE`` Arrow table with one row per transcript
(id + name + optional gene_id FK + spliced sequence + length + source).
Two origins:

    Direct FASTA import     A standalone transcript FASTA (Ensembl
                            cdna release, NCBI RefSeq mRNA dump, lab
                            cDNA panel) → ``read_fasta_transcriptome``
    Derived from annotation ``TranscriptReference.from_annotation(genome,
                            annotation)`` materialises spliced
                            transcripts by joining mRNA/exon features
                            against contig sequences (strand-aware,
                            uses ``core.sequence.nucleic.reverse_complement``)

A ``TranscriptReference`` is sample-agnostic and may be paired with an
``Annotation`` (for gene-level grouping) or stand alone for
transcriptome-only species. ``transcript_id`` namespacing is global
within a single TranscriptReference; use ``feature_origin='transcript_id'``
in ``FEATURE_QUANT`` to key per-transcript counts back here.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field, replace
from typing import Any

import pyarrow as pa
import pyarrow.compute as pc

from constellation.core.io.schemas import cast_to_schema
from constellation.core.sequence.nucleic import reverse_complement
from constellation.sequencing.annotation.annotation import Annotation
from constellation.sequencing.reference.reference import GenomeReference
from constellation.sequencing.schemas.reference import TRANSCRIPT_TABLE


@dataclass(frozen=True, slots=True)
class TranscriptReference:
    """Bundles a single ``TRANSCRIPT_TABLE`` Arrow table."""

    transcripts: pa.Table
    metadata_extras: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "transcripts", cast_to_schema(self.transcripts, TRANSCRIPT_TABLE)
        )
        self.validate()

    # ── validation ──────────────────────────────────────────────────
    def validate(self) -> None:
        """PK uniqueness on ``transcript_id``; ``length`` matches
        ``len(sequence)`` per row."""
        ids = self.transcripts.column("transcript_id").to_pylist()
        if len(set(ids)) != len(ids):
            seen: set[int] = set()
            dups: list[int] = []
            for v in ids:
                if v in seen:
                    dups.append(v)
                else:
                    seen.add(v)
            raise ValueError(
                f"transcript_id contains duplicate values: "
                f"{dups[:5]}{'...' if len(dups) > 5 else ''}"
            )
        sequences = self.transcripts.column("sequence").to_pylist()
        lengths = self.transcripts.column("length").to_pylist()
        bad = [
            (tid, len(seq), declared)
            for tid, seq, declared in zip(ids, sequences, lengths, strict=True)
            if len(seq) != declared
        ]
        if bad:
            sample = bad[:3]
            raise ValueError(
                f"length column does not match len(sequence) for "
                f"{len(bad)} transcript(s) (sample: {sample})"
            )

    def validate_against(self, annotation: Annotation) -> None:
        """Cross-check that every non-null ``gene_id`` is a valid
        feature_id of type 'gene' in the supplied Annotation."""
        gene_ids_in_self = self.transcripts.column("gene_id").to_pylist()
        non_null = {g for g in gene_ids_in_self if g is not None}
        if not non_null:
            return
        gene_features = annotation.features_of_type("gene")
        valid = set(gene_features.column("feature_id").to_pylist())
        missing = non_null - valid
        if missing:
            sample = sorted(missing)[:5]
            raise ValueError(
                f"TranscriptReference.gene_id references ids absent from "
                f"Annotation gene features: "
                f"{sample}{'...' if len(missing) > 5 else ''}"
            )

    # ── views ───────────────────────────────────────────────────────
    @property
    def n_transcripts(self) -> int:
        return self.transcripts.num_rows

    @property
    def metadata(self) -> dict[str, Any]:
        return dict(self.metadata_extras)

    def sequence_of(self, transcript_id: int) -> str:
        """Return the spliced nucleotide sequence for a transcript.

        Raises ``KeyError`` if ``transcript_id`` is not present.
        """
        ids = self.transcripts.column("transcript_id").to_pylist()
        try:
            idx = ids.index(int(transcript_id))
        except ValueError as exc:
            raise KeyError(
                f"transcript_id {transcript_id!r} not in TranscriptReference"
            ) from exc
        return self.transcripts.column("sequence")[idx].as_py()

    def with_metadata(self, extras: dict[str, Any]) -> "TranscriptReference":
        merged = dict(self.metadata_extras)
        merged.update(extras)
        return replace(self, metadata_extras=merged)

    # ── builders ────────────────────────────────────────────────────
    @classmethod
    def from_annotation(
        cls,
        genome: GenomeReference,
        annotation: Annotation,
        *,
        transcript_type: str = "mRNA",
        exon_type: str = "exon",
    ) -> "TranscriptReference":
        """Materialise spliced transcripts from a paired GenomeReference
        and Annotation.

        For each feature with ``type == transcript_type``, finds its
        child features with ``type == exon_type`` (via ``parent_id``),
        sorts them by genomic ``start``, concatenates the contig
        sequence over each exon span, and reverse-complements when the
        parent transcript's strand is ``'-'``. Transcripts without any
        exon children are skipped (with a warning recorded into
        ``metadata_extras``).

        ``transcript_id`` is reused from the parent feature's
        ``feature_id``; ``gene_id`` is resolved by walking
        ``parent_id`` upward to a feature of type ``'gene'`` (or left
        null if no such ancestor exists).
        """
        annotation.validate_against(genome)

        features = annotation.features

        # Build parent → children index for exon lookup.
        feature_ids = features.column("feature_id").to_pylist()
        parent_ids = features.column("parent_id").to_pylist()
        types = features.column("type").to_pylist()
        contig_ids = features.column("contig_id").to_pylist()
        starts = features.column("start").to_pylist()
        ends = features.column("end").to_pylist()
        strands = features.column("strand").to_pylist()
        names = features.column("name").to_pylist()

        children_by_parent: dict[int, list[int]] = defaultdict(list)
        type_by_id: dict[int, str] = {}
        parent_of: dict[int, int | None] = {}
        for i, fid in enumerate(feature_ids):
            type_by_id[fid] = types[i]
            parent_of[fid] = parent_ids[i]
            if parent_ids[i] is not None:
                children_by_parent[parent_ids[i]].append(i)

        # Cache contig sequences once.
        contig_sequence: dict[int, str] = {}
        for cid in {*contig_ids}:
            contig_sequence[cid] = genome.sequence_of(int(cid))

        rows: list[dict[str, Any]] = []
        skipped: list[int] = []
        for i, fid in enumerate(feature_ids):
            if types[i] != transcript_type:
                continue

            exon_indices = [
                j
                for j in children_by_parent.get(fid, ())
                if types[j] == exon_type
            ]
            if not exon_indices:
                skipped.append(fid)
                continue

            # Sort exons by genomic start.
            exon_indices.sort(key=lambda j: starts[j])

            seq_parts = [
                contig_sequence[contig_ids[j]][starts[j] : ends[j]]
                for j in exon_indices
            ]
            spliced = "".join(seq_parts)
            if strands[i] == "-":
                spliced = reverse_complement(spliced)

            # Resolve gene ancestor by walking parent chain.
            gene_id: int | None = None
            cursor = parent_of.get(fid)
            while cursor is not None:
                if type_by_id.get(cursor) == "gene":
                    gene_id = cursor
                    break
                cursor = parent_of.get(cursor)

            rows.append(
                {
                    "transcript_id": int(fid),
                    "name": names[i] if names[i] is not None else f"t{fid}",
                    "gene_id": gene_id,
                    "sequence": spliced,
                    "length": len(spliced),
                    "source": "derived_from_annotation",
                }
            )

        metadata: dict[str, Any] = {}
        if skipped:
            metadata["skipped_transcripts_without_exons"] = skipped

        if not rows:
            empty = TRANSCRIPT_TABLE.empty_table()
            return cls(transcripts=empty, metadata_extras=metadata)

        return cls(
            transcripts=pa.Table.from_pylist(rows, schema=TRANSCRIPT_TABLE),
            metadata_extras=metadata,
        )

    def filter_by_name(self, names: list[str]) -> "TranscriptReference":
        """Return a new TranscriptReference holding only transcripts
        whose ``name`` is in ``names``."""
        keep = pa.array(list(names), type=pa.string())
        mask = pc.is_in(self.transcripts.column("name"), value_set=keep)
        return replace(self, transcripts=self.transcripts.filter(mask))


__all__ = ["TranscriptReference"]
