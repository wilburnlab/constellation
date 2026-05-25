"""Samples — first-class M:N resolver between acquisitions and barcodes.

A *sample* is the unit of biological identity downstream code keys on
(per-feature counts, per-feature TPM, differential abundance). Samples
do not always correspond 1:1 with acquisitions:

    1 sample × N flowcells   genomic ultra-HMW (>60× coverage for
                             eukaryote genomes; the lab routinely runs
                             multi-flowcell aggregations); barcode null

    M samples × 1 flowcell   multiplexed cDNA / direct-RNA — M barcodes
                             discriminate samples on a single flow cell

    M:N mixed                multi-flowcell + multiplexed in any
                             combination

The ``SAMPLE_ACQUISITION_EDGE`` table encodes the resolver as a tiny
Arrow table; ``Samples`` wraps both with PK uniqueness + FK closure.
``barcode_id`` may be null (= "use all reads from this acquisition")
to handle the genomic case without bifurcating the schema.

A ParquetDir Reader/Writer pair (``save_samples`` / ``load_samples``)
persists the container as a sibling artifact in the demux output dir
so downstream stages (``transcriptome align``, ``transcriptome
cluster``) read the resolved samples back without re-parsing the TSV.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from constellation.core.io.schemas import (
    cast_to_schema,
    register_schema,
)


# ──────────────────────────────────────────────────────────────────────
# Schemas
# ──────────────────────────────────────────────────────────────────────


SAMPLE_TABLE: pa.Schema = pa.schema(
    [
        pa.field("sample_id", pa.int64(), nullable=False),
        pa.field("sample_name", pa.string(), nullable=False),
        pa.field("description", pa.string(), nullable=True),
    ],
    metadata={b"schema_name": b"SampleTable"},
)


SAMPLE_ACQUISITION_EDGE: pa.Schema = pa.schema(
    [
        pa.field("sample_id", pa.int64(), nullable=False),
        pa.field("acquisition_id", pa.int64(), nullable=False),
        # null = "all reads from this acquisition" (no barcoding); int64
        # otherwise FKs into a barcode registry resolved at demux time.
        pa.field("barcode_id", pa.int64(), nullable=True),
    ],
    metadata={b"schema_name": b"SampleAcquisitionEdge"},
)


register_schema("SampleTable", SAMPLE_TABLE)
register_schema("SampleAcquisitionEdge", SAMPLE_ACQUISITION_EDGE)


# ──────────────────────────────────────────────────────────────────────
# Container
# ──────────────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class Samples:
    """Bundles a ``SAMPLE_TABLE`` with its ``SAMPLE_ACQUISITION_EDGE``.

    Validation on construction: ``sample_id`` PK uniqueness, edge-table
    cast, and edge-side FK closure (every ``edges.sample_id`` exists in
    ``samples.sample_id``). ``acquisition_id`` FK closure across an
    external :class:`Acquisitions` is left to ``validate_acquisitions``
    at the call site so the container itself doesn't need an
    ``Acquisitions`` reference.
    """

    samples: pa.Table
    edges: pa.Table

    def __post_init__(self) -> None:
        samples_cast = cast_to_schema(self.samples, SAMPLE_TABLE)
        edges_cast = cast_to_schema(self.edges, SAMPLE_ACQUISITION_EDGE)
        object.__setattr__(self, "samples", samples_cast)
        object.__setattr__(self, "edges", edges_cast)

        ids = samples_cast.column("sample_id").to_pylist()
        if len(set(ids)) != len(ids):
            duplicates = sorted({i for i in ids if ids.count(i) > 1})
            raise ValueError(
                f"sample_id contains duplicates: {duplicates[:5]}"
                f"{'...' if len(duplicates) > 5 else ''}"
            )

        # sample_name is the user-facing identity. SAMPLE_TABLE carries
        # one row per biological sample; multi-barcode / multi-flowcell
        # membership lives in SAMPLE_ACQUISITION_EDGE under a shared
        # sample_id. Two SAMPLE_TABLE rows with the same sample_name
        # would mean "two distinct samples happen to share a name" —
        # downstream code that uses sample_name as a column / file-name
        # key (gene matrix headers, fastq filenames) silently corrupts
        # in that state.
        names = samples_cast.column("sample_name").to_pylist()
        if len(set(names)) != len(names):
            counts = Counter(names)
            duplicates = sorted(n for n, c in counts.items() if c > 1)
            raise ValueError(
                f"sample_name appears on multiple sample rows: "
                f"{duplicates[:5]}{'...' if len(duplicates) > 5 else ''} "
                f"— each biological sample is one SAMPLE_TABLE row; "
                f"reads from the same sample via multiple barcodes or "
                f"flowcells should share that row's sample_id and "
                f"contribute additional SAMPLE_ACQUISITION_EDGE rows, "
                f"not additional sample rows"
            )

        known = set(ids)
        edge_sample_ids = edges_cast.column("sample_id").to_pylist()
        unknown = {sid for sid in edge_sample_ids if sid not in known}
        if unknown:
            sample = sorted(unknown)[:5]
            raise ValueError(
                f"edges reference unknown sample_ids: "
                f"{sample}{'...' if len(unknown) > 5 else ''}"
            )

    @classmethod
    def empty(cls) -> Samples:
        return cls(SAMPLE_TABLE.empty_table(), SAMPLE_ACQUISITION_EDGE.empty_table())

    @classmethod
    def from_records(
        cls,
        samples: Iterable[dict[str, object]],
        edges: Iterable[dict[str, object]],
    ) -> Samples:
        sample_rows = list(samples)
        edge_rows = list(edges)
        s_table = (
            pa.Table.from_pylist(sample_rows, schema=SAMPLE_TABLE)
            if sample_rows
            else SAMPLE_TABLE.empty_table()
        )
        e_table = (
            pa.Table.from_pylist(edge_rows, schema=SAMPLE_ACQUISITION_EDGE)
            if edge_rows
            else SAMPLE_ACQUISITION_EDGE.empty_table()
        )
        return cls(s_table, e_table)

    @property
    def ids(self) -> list[int]:
        return self.samples.column("sample_id").to_pylist()

    def acquisitions_for(self, sample_id: int) -> list[int]:
        """All acquisition_ids that contribute to ``sample_id``."""
        mask = pc.equal(self.edges.column("sample_id"), sample_id)
        filtered = self.edges.filter(mask)
        return filtered.column("acquisition_id").to_pylist()

    def samples_for(
        self,
        acquisition_id: int,
        barcode_id: int | None = None,
    ) -> list[int]:
        """All sample_ids that draw from ``(acquisition_id, barcode_id)``.

        If ``barcode_id`` is None, returns the sample(s) for un-barcoded
        rows (genomic). If barcode_id is supplied, returns the sample
        for that specific barcode within the acquisition.
        """
        acq_match = pc.equal(self.edges.column("acquisition_id"), acquisition_id)
        if barcode_id is None:
            bc_match = pc.is_null(self.edges.column("barcode_id"))
        else:
            bc_match = pc.equal(self.edges.column("barcode_id"), barcode_id)
        mask = pc.and_(acq_match, bc_match)
        return self.edges.filter(mask).column("sample_id").to_pylist()

    def __len__(self) -> int:
        return self.samples.num_rows


# ──────────────────────────────────────────────────────────────────────
# ParquetDir persistence
# ──────────────────────────────────────────────────────────────────────
#
# Mirrors ``constellation.sequencing.annotation.io``'s ParquetDir
# Reader/Writer. Demux writes ``<demux-dir>/samples/`` so align +
# cluster can ``load_samples`` it back without re-parsing the TSV.

_SAMPLES_SCHEMA_VERSION = 1
_SAMPLES_FORMAT_NAME = "parquet_dir"


def save_samples(samples: Samples, path: str | Path, **opts: Any) -> None:
    """Write a :class:`Samples` to a ParquetDir bundle at ``path``.

    Layout::

        <path>/samples.parquet
        <path>/edges.parquet
        <path>/manifest.json
    """
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    pq.write_table(samples.samples, p / "samples.parquet")
    pq.write_table(samples.edges, p / "edges.parquet")
    manifest = {
        "format": _SAMPLES_FORMAT_NAME,
        "tables": ["samples", "edges"],
        "container": "Samples",
        "schema_version": _SAMPLES_SCHEMA_VERSION,
    }
    (p / "manifest.json").write_text(json.dumps(manifest, indent=2))


def load_samples(path: str | Path, **opts: Any) -> Samples:
    """Read a :class:`Samples` from a ParquetDir bundle written by
    :func:`save_samples`."""
    p = Path(path)
    samples_table = pq.read_table(p / "samples.parquet")
    edges_table = pq.read_table(p / "edges.parquet")
    return Samples(samples=samples_table, edges=edges_table)


__all__ = [
    "SAMPLE_TABLE",
    "SAMPLE_ACQUISITION_EDGE",
    "Samples",
    "save_samples",
    "load_samples",
]
