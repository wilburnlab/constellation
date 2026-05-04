"""``GeneticTools`` container — curated common cloning / engineering parts.

The cRAP equivalent for genomics/transcriptomics: a small curated
catalogue of sequences that frequently appear in cell-culture
transcriptomes/genomes because they were deliberately introduced
(antibiotic resistance genes, fluorescent proteins, epitope tags, common
promoters/terminators, selection markers, common enzymes like Cas9 / Cre,
secretion signals, common cloning vector backbones). These don't belong
to the host organism's natural biology, but they are usually present.

This container is the reference-side companion to the S3 mmseqs2 /
genetic_tools pre-screen step: incoming ORFs / contigs are matched
against this DB before UniRef90 search, so known engineering elements
get a stable identifier instead of being absorbed into a UniRef hit
that's a poor description of the introduced part.

Schema: ``GENETIC_TOOL_TABLE`` (``tool_id`` PK, ``name``, ``category``,
``sequence_type``, ``sequence``, ``source``, ``source_url?``,
``references_json?``).

Curation lives at :file:`constellation/data/genetic_tools.json`,
regenerable via :file:`scripts/build-genetic-tools-json.py`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from importlib import resources
from typing import Any

import pyarrow as pa
import pyarrow.compute as pc

from constellation.core.io.schemas import cast_to_schema
from constellation.sequencing.schemas.reference import (
    GENETIC_TOOL_CATEGORIES,
    GENETIC_TOOL_SEQUENCE_TYPES,
    GENETIC_TOOL_TABLE,
)


@dataclass(frozen=True, slots=True)
class GeneticTools:
    """Bundles a single ``GENETIC_TOOL_TABLE`` Arrow table."""

    tools: pa.Table
    metadata_extras: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "tools", cast_to_schema(self.tools, GENETIC_TOOL_TABLE))
        self.validate()

    # ── validation ──────────────────────────────────────────────────
    def validate(self) -> None:
        """PK uniqueness on ``tool_id``; ``category`` and
        ``sequence_type`` belong to their respective vocabularies;
        ``sequence`` is non-empty."""
        ids = self.tools.column("tool_id").to_pylist()
        if len(set(ids)) != len(ids):
            seen: set[int] = set()
            dups: list[int] = []
            for v in ids:
                if v in seen:
                    dups.append(v)
                else:
                    seen.add(v)
            raise ValueError(
                f"tool_id contains duplicate values: "
                f"{dups[:5]}{'...' if len(dups) > 5 else ''}"
            )

        categories = self.tools.column("category").to_pylist()
        bad_cat = sorted({c for c in categories if c not in GENETIC_TOOL_CATEGORIES})
        if bad_cat:
            raise ValueError(
                f"unknown category values: {bad_cat[:5]}; "
                f"allowed: {sorted(GENETIC_TOOL_CATEGORIES)}"
            )

        seq_types = self.tools.column("sequence_type").to_pylist()
        bad_st = sorted({s for s in seq_types if s not in GENETIC_TOOL_SEQUENCE_TYPES})
        if bad_st:
            raise ValueError(
                f"unknown sequence_type values: {bad_st}; "
                f"allowed: {sorted(GENETIC_TOOL_SEQUENCE_TYPES)}"
            )

        sequences = self.tools.column("sequence").to_pylist()
        empty = [
            ids[i] for i, s in enumerate(sequences) if not isinstance(s, str) or not s
        ]
        if empty:
            raise ValueError(
                f"sequence is empty for tool_id(s): {empty[:5]}"
                f"{'...' if len(empty) > 5 else ''}"
            )

    # ── views ───────────────────────────────────────────────────────
    @property
    def n_tools(self) -> int:
        return self.tools.num_rows

    @property
    def metadata(self) -> dict[str, Any]:
        return dict(self.metadata_extras)

    def of_category(self, category: str) -> pa.Table:
        """Return all rows matching ``category``."""
        if category not in GENETIC_TOOL_CATEGORIES:
            raise KeyError(
                f"unknown category {category!r}; "
                f"allowed: {sorted(GENETIC_TOOL_CATEGORIES)}"
            )
        mask = pc.equal(self.tools.column("category"), category)
        return self.tools.filter(mask)

    def of_sequence_type(self, sequence_type: str) -> pa.Table:
        """Return all rows whose ``sequence_type`` matches."""
        if sequence_type not in GENETIC_TOOL_SEQUENCE_TYPES:
            raise KeyError(
                f"unknown sequence_type {sequence_type!r}; "
                f"allowed: {sorted(GENETIC_TOOL_SEQUENCE_TYPES)}"
            )
        mask = pc.equal(self.tools.column("sequence_type"), sequence_type)
        return self.tools.filter(mask)

    def to_fasta(self, *, sequence_type: str | None = None) -> str:
        """Render the bundled sequences as a multi-record FASTA string.

        If ``sequence_type`` is given (``'nucleotide'`` or ``'protein'``),
        only rows matching that type are emitted; else all rows are
        emitted (callers usually want the type-specific projection
        because mmseqs2 / minimap2 each take only one alphabet at a
        time).
        """
        rows: pa.Table
        if sequence_type is None:
            rows = self.tools
        else:
            rows = self.of_sequence_type(sequence_type)

        ids = rows.column("tool_id").to_pylist()
        names = rows.column("name").to_pylist()
        cats = rows.column("category").to_pylist()
        seqs = rows.column("sequence").to_pylist()
        chunks: list[str] = []
        for tid, name, cat, seq in zip(ids, names, cats, seqs, strict=True):
            chunks.append(f">tool:{tid} {name} category={cat}")
            # 60-character line wrap to match canonical FASTA style
            for i in range(0, len(seq), 60):
                chunks.append(seq[i : i + 60])
        return "\n".join(chunks) + ("\n" if chunks else "")

    def with_metadata(self, extras: dict[str, Any]) -> "GeneticTools":
        merged = dict(self.metadata_extras)
        merged.update(extras)
        return replace(self, metadata_extras=merged)


# ──────────────────────────────────────────────────────────────────────
# Default-bundle loader — reads constellation/data/genetic_tools.json
# ──────────────────────────────────────────────────────────────────────


def _load_bundled_json() -> dict[str, Any]:
    pkg = resources.files("constellation.data")
    path = pkg / "genetic_tools.json"
    return json.loads(path.read_text(encoding="utf-8"))


def load_default_genetic_tools() -> GeneticTools:
    """Load the bundled ``constellation/data/genetic_tools.json``.

    The JSON schema is a single object with two keys:
        ``meta``    free-form metadata (build date, source URLs, ...)
        ``tools``   list of dicts, each with the GENETIC_TOOL_TABLE
                    columns
    """
    payload = _load_bundled_json()
    rows = payload.get("tools", [])
    if not rows:
        table = GENETIC_TOOL_TABLE.empty_table()
    else:
        table = pa.Table.from_pylist(rows, schema=GENETIC_TOOL_TABLE)
    return GeneticTools(
        tools=table,
        metadata_extras=dict(payload.get("meta", {})),
    )


__all__ = [
    "GeneticTools",
    "load_default_genetic_tools",
]
