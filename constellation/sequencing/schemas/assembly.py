"""Assembly schemas — de novo contigs, scaffolds, summary stats.

An ``Assembly`` is the in-progress representation of a de novo
genome / transcriptome assembly. It carries provenance the
``Reference`` shape doesn't need: which acquisitions fed in, polishing
history, gap maps, scaffolding edges. ``Assembly.to_reference()``
projects a finalized assembly to ``CONTIG_TABLE`` + ``SEQUENCE_TABLE``
+ ``FEATURE_TABLE`` so downstream alignment / quant code is uniform
across external-ref and de-novo-ref workflows.

Three tables shipped here; ``Assembly`` container in
:mod:`sequencing.assembly.assembly` bundles them.
"""

from __future__ import annotations

import pyarrow as pa

from constellation.core.io.schemas import register_schema


# ──────────────────────────────────────────────────────────────────────
# Per-contig assembly records
# ──────────────────────────────────────────────────────────────────────


ASSEMBLY_CONTIG_TABLE: pa.Schema = pa.schema(
    [
        pa.field("contig_id", pa.int64(), nullable=False),
        pa.field("name", pa.string(), nullable=False),
        pa.field("length", pa.int64(), nullable=False),
        # Read coverage from the original assembly (HiFiASM provides this)
        pa.field("read_coverage", pa.float32(), nullable=True),
        # Polishing rounds applied; null = unpolished
        pa.field("polish_rounds", pa.int32(), nullable=True),
        # 'primary' | 'haplotigs' (HiFiASM dual-haplotype output)
        pa.field("haplotype", pa.string(), nullable=True),
        pa.field("circular", pa.bool_(), nullable=True),
        # Free-form provenance (assembler version, command line, etc.)
        # JSON-encoded
        pa.field("provenance_json", pa.string(), nullable=True),
    ],
    metadata={b"schema_name": b"AssemblyContigTable"},
)


# ──────────────────────────────────────────────────────────────────────
# Scaffolds (RagTag, manual stitching)
# ──────────────────────────────────────────────────────────────────────


SCAFFOLD_TABLE: pa.Schema = pa.schema(
    [
        pa.field("scaffold_id", pa.int64(), nullable=False),
        pa.field("name", pa.string(), nullable=False),
        # Constituent contig in scaffold order (0-indexed within scaffold)
        pa.field("contig_id", pa.int64(), nullable=False),
        pa.field("position", pa.int32(), nullable=False),
        # Orientation of contig within scaffold ('+' | '-')
        pa.field("orientation", pa.string(), nullable=False),
        # Gap to next contig in scaffold (Ns inserted); -1 for terminal
        pa.field("gap_size", pa.int64(), nullable=True),
        # 'estimated' | 'measured' | 'unknown' (RagTag uses 'estimated')
        pa.field("gap_type", pa.string(), nullable=True),
    ],
    metadata={b"schema_name": b"ScaffoldTable"},
)


# ──────────────────────────────────────────────────────────────────────
# Summary stats (one row per assembly)
# ──────────────────────────────────────────────────────────────────────


ASSEMBLY_STATS: pa.Schema = pa.schema(
    [
        pa.field("assembly_id", pa.int64(), nullable=False),
        pa.field("n_contigs", pa.int32(), nullable=False),
        pa.field("n_scaffolds", pa.int32(), nullable=True),
        pa.field("total_length", pa.int64(), nullable=False),
        pa.field("n50", pa.int64(), nullable=False),
        pa.field("l50", pa.int32(), nullable=False),
        pa.field("n90", pa.int64(), nullable=True),
        pa.field("l90", pa.int32(), nullable=True),
        pa.field("largest_contig", pa.int64(), nullable=False),
        pa.field("gc_content", pa.float32(), nullable=True),
        # BUSCO completeness fractions (0–1); null if not run
        pa.field("busco_complete", pa.float32(), nullable=True),
        pa.field("busco_single", pa.float32(), nullable=True),
        pa.field("busco_duplicated", pa.float32(), nullable=True),
        pa.field("busco_fragmented", pa.float32(), nullable=True),
        pa.field("busco_missing", pa.float32(), nullable=True),
        pa.field("busco_lineage", pa.string(), nullable=True),
    ],
    metadata={b"schema_name": b"AssemblyStats"},
)


register_schema("AssemblyContigTable", ASSEMBLY_CONTIG_TABLE)
register_schema("ScaffoldTable", SCAFFOLD_TABLE)
register_schema("AssemblyStats", ASSEMBLY_STATS)


__all__ = [
    "ASSEMBLY_CONTIG_TABLE",
    "SCAFFOLD_TABLE",
    "ASSEMBLY_STATS",
]
