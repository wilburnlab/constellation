"""Reference schemas — contigs, sequences, features.

A ``Reference`` packages three tables: a ``CONTIG_TABLE`` (one row per
named contig), a ``SEQUENCE_TABLE`` (one row per contig holding the
literal nucleotide bytes — separated from CONTIG_TABLE so coordinate
queries don't pull megabytes), and a ``FEATURE_TABLE`` (GFF3-shaped
annotation features: genes, exons, repeats, telomeres, anything
positional on the reference).

Reference is the only sequencing container whose shape mirrors the
``massspec.library.Library`` template: PK/FK closure, ParquetDir
native form, Reader/Writer Protocols. Two origins flow into it:

    External reference     FASTA + GFF → Reference (contigs, sequences,
                           features all populated)
    De novo assembly       Assembly.to_reference() lifts a finished
                           assembly into the same shape so downstream
                           alignment / quant code is uniform

Coordinates are 0-based half-open throughout (matching BAM internal,
not GFF3's 1-based — readers convert at the boundary).
"""

from __future__ import annotations

import pyarrow as pa

from constellation.core.io.schemas import register_schema


# ──────────────────────────────────────────────────────────────────────
# Contigs
# ──────────────────────────────────────────────────────────────────────


CONTIG_TABLE: pa.Schema = pa.schema(
    [
        pa.field("contig_id", pa.int64(), nullable=False),
        # Display name as written in FASTA / used in BAM ref_name
        pa.field("name", pa.string(), nullable=False),
        pa.field("length", pa.int64(), nullable=False),
        # Optional: 'chromosome' | 'scaffold' | 'contig' | 'unplaced'
        pa.field("topology", pa.string(), nullable=True),
        # 'linear' | 'circular' (mtDNA, plastids)
        pa.field("circular", pa.bool_(), nullable=True),
    ],
    metadata={b"schema_name": b"ContigTable"},
)


# ──────────────────────────────────────────────────────────────────────
# Sequences (separated so coord queries don't pull bytes)
# ──────────────────────────────────────────────────────────────────────


SEQUENCE_TABLE: pa.Schema = pa.schema(
    [
        pa.field("contig_id", pa.int64(), nullable=False),
        # Literal nucleotide string ('A','C','G','T','N','U' for RNA
        # references); softmasking preserved as lowercase, hardmasking
        # as 'N'
        pa.field("sequence", pa.string(), nullable=False),
    ],
    metadata={b"schema_name": b"SequenceTable"},
)


# ──────────────────────────────────────────────────────────────────────
# Features (GFF3-shaped)
# ──────────────────────────────────────────────────────────────────────


FEATURE_TABLE: pa.Schema = pa.schema(
    [
        pa.field("feature_id", pa.int64(), nullable=False),
        pa.field("contig_id", pa.int64(), nullable=False),
        # 0-based half-open
        pa.field("start", pa.int64(), nullable=False),
        pa.field("end", pa.int64(), nullable=False),
        # '+' | '-' | '.' (strandless)
        pa.field("strand", pa.string(), nullable=False),
        # SO term: 'gene' | 'mRNA' | 'CDS' | 'exon' | 'repeat_region' |
        # 'telomere' | 'centromere' | ...
        pa.field("type", pa.string(), nullable=False),
        # Display name; may differ from feature_id (HGNC, Ensembl, custom)
        pa.field("name", pa.string(), nullable=True),
        # GFF3 Parent= attribute resolved to feature_id (mRNA→gene,
        # CDS→mRNA); null at the top of the hierarchy
        pa.field("parent_id", pa.int64(), nullable=True),
        # Source field from GFF3 (BUSCO, RepeatMasker, custom annotator)
        pa.field("source", pa.string(), nullable=True),
        pa.field("score", pa.float32(), nullable=True),
        # Phase for CDS features (0/1/2)
        pa.field("phase", pa.int32(), nullable=True),
        # GFF3 attribute column ("ID=g1;Name=foo") — JSON-encoded for
        # consistency with massspec.library_fragment_table conventions
        pa.field("attributes_json", pa.string(), nullable=True),
    ],
    metadata={b"schema_name": b"FeatureTable"},
)


# ──────────────────────────────────────────────────────────────────────
# Transcripts (TranscriptReference)
# ──────────────────────────────────────────────────────────────────────


TRANSCRIPT_TABLE: pa.Schema = pa.schema(
    [
        pa.field("transcript_id", pa.int64(), nullable=False),
        # Display name as written in transcript FASTA / used as
        # ref_name when minimap2 maps to transcriptome
        pa.field("name", pa.string(), nullable=False),
        # Optional FK to FEATURE_TABLE.feature_id of the parent gene
        # (only populated when a TranscriptReference was materialised
        # from a paired GenomeReference + Annotation; transcriptome-only
        # references leave this null and resolve later if a genome
        # arrives)
        pa.field("gene_id", pa.int64(), nullable=True),
        # Spliced transcript sequence (not the genomic span — exonic
        # bases concatenated with strand respected)
        pa.field("sequence", pa.string(), nullable=False),
        pa.field("length", pa.int32(), nullable=False),
        # Provenance: 'ensembl' | 'refseq' | 'fasta_import' |
        # 'derived_from_annotation'
        pa.field("source", pa.string(), nullable=True),
    ],
    metadata={b"schema_name": b"TranscriptTable"},
)


# ──────────────────────────────────────────────────────────────────────
# Genetic tools (GeneticTools — common AbR / FPs / tags / promoters /
# selection markers / common enzymes / vector backbones; the cRAP
# analog for genomics/transcriptomics)
# ──────────────────────────────────────────────────────────────────────


# Allowed categories. Not enforced at the schema layer; consumed by
# the GeneticTools container validator.
GENETIC_TOOL_CATEGORIES: frozenset[str] = frozenset(
    {
        "antibiotic_resistance",
        "fluorescent_protein",
        "epitope_tag",
        "promoter",
        "terminator",
        "selection_marker",
        "common_enzyme",
        "secretion_signal",
        "cloning_vector_backbone",
    }
)


GENETIC_TOOL_SEQUENCE_TYPES: frozenset[str] = frozenset({"nucleotide", "protein"})


GENETIC_TOOL_TABLE: pa.Schema = pa.schema(
    [
        pa.field("tool_id", pa.int64(), nullable=False),
        pa.field("name", pa.string(), nullable=False),
        # See GENETIC_TOOL_CATEGORIES
        pa.field("category", pa.string(), nullable=False),
        # 'nucleotide' | 'protein'
        pa.field("sequence_type", pa.string(), nullable=False),
        pa.field("sequence", pa.string(), nullable=False),
        # Where the sequence came from: 'fpbase' | 'ncbi' | 'addgene' |
        # 'univec' | 'manual_curation'
        pa.field("source", pa.string(), nullable=False),
        pa.field("source_url", pa.string(), nullable=True),
        # JSON-encoded list of citations / DOIs / supporting refs
        pa.field("references_json", pa.string(), nullable=True),
    ],
    metadata={b"schema_name": b"GeneticToolTable"},
)


register_schema("ContigTable", CONTIG_TABLE)
register_schema("SequenceTable", SEQUENCE_TABLE)
register_schema("FeatureTable", FEATURE_TABLE)
register_schema("TranscriptTable", TRANSCRIPT_TABLE)
register_schema("GeneticToolTable", GENETIC_TOOL_TABLE)


__all__ = [
    "CONTIG_TABLE",
    "SEQUENCE_TABLE",
    "FEATURE_TABLE",
    "TRANSCRIPT_TABLE",
    "GENETIC_TOOL_TABLE",
    "GENETIC_TOOL_CATEGORIES",
    "GENETIC_TOOL_SEQUENCE_TYPES",
]
