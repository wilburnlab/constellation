# Reference-guided clustering — strategic plan + Session 1 data classes

## Context

**The problem.** `constellation transcriptome demultiplex` (S1, shipped) hard-thresholds barcodes and predicts proteins by naive ORF translation. The resulting `feature_quant` matrix only captures ~1–2% of demuxed reads as protein-grouped — most reads get thrown away because naive ORF prediction can't tolerate the error structure of nanopore long reads (indel-heavy, non-uniform error model, possibly basecaller artifacts that we've already pinned a `do-not-fit` rule on per the scoring-domain note in CLAUDE.md). For organisms with assembled genomes/transcriptomes (Pichia, S. cerevisiae, the lab's main targets), reference-guided assignment can recover substantially more of the data.

**The strategy.** Bootstrap a reference layer that:
1. Imports genome, annotation, and (optionally) transcriptome data into Arrow-shaped containers (the same load-bearing pattern that `massspec.library.Library` follows).
2. Maps reads to a reference genome via `minimap2 -ax splice` (the well-validated path; gene-level locus assignment is deterministic from spliced alignments + GFF3 overlap, no EM needed).
3. Falls back to de novo k-mer-graph clustering for the unmapped fraction.
4. Defers transcriptome-mode mapping + isoform-resolving abundance estimation to a later session — when we get there, we roll our own EM-based quantifier on top of `core.optim` rather than wrapping `oarfish` / `NanoCount`. Reasoning: (a) the math is well-understood (Salmon/RSEM-style EM); (b) staying close to the alignment data lets us spot basecaller systematics that a black-box quantifier would mask, consistent with the scoring-domain rule already enshrined in CLAUDE.md; (c) we already have the optimization infrastructure shipped (`core.optim.LBFGSOptimizer` / `DifferentialEvolution` + `core.stats.Parametric`).
5. Annotates ORFs from clusters/transcripts via `mmseqs2` against UniRef + a Constellation-curated `genetic_tools` DB (the cRAP-equivalent the field is missing — common AbR genes / fluorescent proteins / epitope tags / promoters / common enzymes / vector backbones).

**The 3-session split** (user proposal, accepted with refinements):

- **S1 — data classes + import infrastructure** *(this session's deliverable)*. Reference container refactor (split into `GenomeReference` / `Annotation` / `TranscriptReference` per user decision), FASTA + GFF3 readers (hand-rolled, stdlib-only), ParquetDir round-trip, `genetic_tools` curation, `feature_origin` schema extension, `constellation reference` CLI surface (import + stdlib-HTTP fetch).
- **S2 — genome-guided alignment + de novo clustering**. `minimap2` genome-spliced wrapper, `Alignments` container completion, ENCODE-style read filters, hand-rolled gene-level overlap counter (no EM — read-to-best-overlap-gene is deterministic), de novo k-mer-graph clustering for the unmapped fraction. `constellation reference align` + `constellation transcriptome cluster` CLI. *No transcriptome-mode mapping, no isoform-EM in this session* — those defer until we have the gene-level path well-tested and a clean baseline to compare against.
- **S3 — protein layer**. `mmseqs2` wrapper, ORF→UniRef90 search, `genetic_tools` pre-screen, protein-tier `feature_quant` partition, the bridge into `massspec.library` (ORF → predicted peptides → spectral library entries).
- **Future session (post-S3) — transcriptome-mode + custom isoform-EM**. Adds transcriptome-mode `minimap2 -x map-ont` against `TranscriptReference`, a Constellation-native EM quantifier built on `core.optim` + `core.stats.Parametric` (Salmon/RSEM-style, with explicit length priors and stranded-read handling), `feature_origin='transcript_id'` partition. Stays close enough to the alignment math that basecaller-systematics diagnostics flow through naturally.

Sessions 2 and 3 get their own plan files. **This plan executes only Session 1.**

---

## Strategic decisions locked

These shape every file we touch this session:

1. **Reference is split into three sibling containers** — `GenomeReference` (contigs + sequences), `Annotation` (features, FK into a GenomeReference), `TranscriptReference` (transcript records: id + name + gene_id + sequence) — matching the heterogeneity of real-world references (full Ensembl-class genomes vs transcriptome-only species vs Pichia-style assembled-but-sparsely-annotated genomes). The existing single `Reference` stub gets renamed to `GenomeReference`; `Assembly.to_reference()` becomes `Assembly.to_genome_reference()`.
2. **Genetic tools ship as a curated bundle** (`constellation/data/genetic_tools.json`, ~150 sequences, regenerable). Sibling pattern to `unimod.json` / `proteases.json`. This is the cRAP-equivalent for genomics/transcriptomics — common AbR genes, fluorescent proteins, epitope tags, promoters, selection markers, common enzymes, vector backbones — the field doesn't have a community version, so Constellation contributes one. Schema lands S1; the mmseqs2 pre-filter lands S3.
3. **`feature_quant` is multi-tag**, not strict-precedence. Add a `feature_origin` column (`'gene_id'` | `'transcript_id'` | `'cluster_id'` | `'genetic_tool'` | `'protein_uniref'` | `'protein_hash'`). PK becomes `(feature_id, sample_id, engine, feature_origin)`. Within-partition sums are consistent counts; cross-partition sums are nonsense and a downstream consumer's responsibility to avoid. Closer to how `oarfish` / EM quantifiers expose multimapper weights.
4. **Coordinates: 0-based half-open internally** (matches BAM, the existing schema comment, and `pyranges` Arrow output). GFF3 (1-based inclusive) and BED (0-based half-open) conversions happen at the reader boundary.
5. **GFF3 parsing is a hand-rolled streaming parser** — no `pyranges`, no `gffutils`, no pandas. Reasoning: pyranges1 (the active fork) has moved GTF parsing to a `gtfreader` sub-dep, added a Rust backend (`ruranges`), and still pulls pandas; the dep stack is getting heavier and is at odds with the no-pandas invariant. GFF3 grammar is small (9 tab-separated columns + `key=value;…` attribute column + `##` directives + optional embedded `##FASTA` section), and a streaming parser building Arrow column-list arrays is ~250 lines. Same precedent as `readers/sam_bam.py`, which wraps `pysam` directly rather than going through pandas. BED format is even smaller (3–12 tabular columns, no nested attrs) and lands as a sibling reader.
6. **No new ABCs.** `GenomeReference`, `Annotation`, `TranscriptReference`, `EngineeringParts` are containers with the same `__post_init__` cast-and-validate pattern as `massspec.library.Library`. Reader/Writer Protocols mirror `massspec.library.io`.

---

## Session 1 deliverables

### A. Schema additions/changes

**[constellation/sequencing/schemas/reference.py](constellation/sequencing/schemas/reference.py)** — add two schemas alongside existing `CONTIG_TABLE` / `SEQUENCE_TABLE` / `FEATURE_TABLE`:

- `TRANSCRIPT_TABLE` — `(transcript_id int64 PK, name string, gene_id int64? FK, sequence string, length int32, source string?)`. Stand-alone table for `TranscriptReference`. `gene_id` is nullable so transcriptome-only references (no GenomeReference + Annotation companion) are first-class.
- `GENETIC_TOOL_TABLE` — `(tool_id int64 PK, name string, category string, sequence_type string, sequence string, source string, source_url string?, references_json string?)`. `category` is one of `{'antibiotic_resistance', 'fluorescent_protein', 'epitope_tag', 'promoter', 'terminator', 'selection_marker', 'common_enzyme', 'secretion_signal', 'cloning_vector_backbone'}`. `sequence_type` is `'nucleotide'` or `'protein'` — both ship in one table because the categories cross both (vector backbones are nucleotide; tags ship as protein).

Self-register via `register_schema(...)`.

**[constellation/sequencing/schemas/quant.py](constellation/sequencing/schemas/quant.py)** — extend `FEATURE_QUANT`:

- Add `feature_origin string nullable=False` column.
- Update PK comment from `(feature_id, sample_id)` to `(feature_id, sample_id, engine, feature_origin)`.
- `count` becomes `float64` (already is — confirm). Multimapper-weighted counts from `oarfish`/`NanoCount` are float, so the schema already accommodates.

### B. Container refactor + new containers

**Rename `Reference` → `GenomeReference`:**

- [constellation/sequencing/reference/reference.py](constellation/sequencing/reference/reference.py) — class rename. Drop `features` field (moves to `Annotation`). Fields are now `(contigs, sequences, metadata_extras)`. Implement `__post_init__` (cast-to-schema + `validate()`), `validate()` (PK uniqueness on `contig_id` + FK closure `SEQUENCE_TABLE.contig_id ⊆ CONTIG_TABLE.contig_id`), the typed-stub properties (`n_contigs`, `total_length`, `sequence_of(contig_id) -> str`).
- [constellation/sequencing/reference/io.py](constellation/sequencing/reference/io.py) — rename Protocols to `GenomeReferenceReader` / `GenomeReferenceWriter`, registry dicts to `GENOME_REFERENCE_READERS` / `GENOME_REFERENCE_WRITERS`. Implement `ParquetDirReader` / `ParquetDirWriter` (one parquet per table + `manifest.json`). Implement `save_genome_reference` / `load_genome_reference` dispatchers.
- [constellation/sequencing/assembly/assembly.py](constellation/sequencing/assembly/assembly.py) — `to_reference` → `to_genome_reference`; signature drops the `features` param (Annotation is its own container now — Assembly only produces a GenomeReference; downstream BUSCO/repeat passes build the companion Annotation in S2/S3).

**New: `Annotation` container** at [constellation/sequencing/annotation/annotation.py](constellation/sequencing/annotation/annotation.py):

```python
@dataclass(frozen=True, slots=True)
class Annotation:
    features: pa.Table  # FEATURE_TABLE
    metadata_extras: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # cast to schema + validate
    def validate(self) -> None:
        # PK uniqueness on feature_id; parent_id ⊆ feature_id ∪ {null}
    def validate_against(self, genome: GenomeReference) -> None:
        # contig_id ⊆ genome.contigs.contig_id
    def features_on(self, contig_id: int) -> pa.Table: ...
    def features_of_type(self, type: str) -> pa.Table: ...
```

Reader/Writer Protocols + ParquetDir live in [constellation/sequencing/annotation/io.py](constellation/sequencing/annotation/io.py) (new). Existing `annotation/{busco,repeats,telomeres,transcripts}.py` runner stubs remain — they produce `Annotation` objects in later phases.

**New: `TranscriptReference` container** at [constellation/sequencing/transcripts/transcripts.py](constellation/sequencing/transcripts/transcripts.py) (new directory):

```python
@dataclass(frozen=True, slots=True)
class TranscriptReference:
    transcripts: pa.Table  # TRANSCRIPT_TABLE
    metadata_extras: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None: ...
    def validate(self) -> None:
        # PK uniqueness on transcript_id
    def validate_against(self, annotation: Annotation) -> None:
        # transcripts.gene_id ⊆ annotation.features.feature_id where type='gene'
    @classmethod
    def from_annotation(
        cls, genome: GenomeReference, annotation: Annotation, *, level: str = 'mRNA'
    ) -> 'TranscriptReference':
        """Materialise transcripts by joining mRNA/exon features against contig sequence."""
    def sequence_of(self, transcript_id: int) -> str: ...
```

`from_annotation` is the workhorse — splices exon sequences for each `mRNA` (or `transcript`) row, respects strand (reverse-complement on `-`), uses `core.sequence.nucleic.reverse_complement` already in the codebase. This is the fast path for transcriptome-mode `minimap2` in S2 — call it once, write a transcript FASTA artifact.

Reader/Writer Protocols + ParquetDir at [constellation/sequencing/transcripts/io.py](constellation/sequencing/transcripts/io.py).

**New: `GeneticTools` container** at [constellation/sequencing/genetic_tools.py](constellation/sequencing/genetic_tools.py):

```python
@dataclass(frozen=True, slots=True)
class GeneticTools:
    tools: pa.Table  # GENETIC_TOOL_TABLE
    metadata_extras: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None: ...
    def validate(self) -> None: ...  # PK uniqueness, sequence_type vocab
    def of_category(self, category: str) -> pa.Table: ...
    def to_fasta(self, *, sequence_type: str = 'nucleotide') -> str: ...

def load_default_genetic_tools() -> GeneticTools:
    """Load the bundled constellation/data/genetic_tools.json."""
```

### C. External-format readers

**FASTA reader** at [constellation/sequencing/readers/fastx.py](constellation/sequencing/readers/fastx.py) — extend the existing module with a hand-rolled streaming parser, no `pysam` for FASTA. Reasoning: pysam.FastaFile's value is `.fai`-indexed random contig access on huge (human-scale) genomes; we're doing full one-pass reads into Arrow tables, so the index buys nothing. Same logic as the GFF3 decision — keep pysam scoped to where it genuinely earns its keep (SAM/BAM CIGAR parsing, virtual offsets, sorting/indexing).

Public surface:

- `_iter_fasta(path) -> Iterator[tuple[str, str, str]]` — internal generator yielding `(id, description, sequence)` per record. ~25 lines: gzip-aware (`.fa.gz` via stdlib `gzip.open`); strips line endings; splits header on first whitespace into `id` + `description`; concatenates sequence lines. Whitespace inside the sequence (tabs, line breaks) is stripped; non-ATCGN-U characters pass through (downstream consumers validate).
- `read_fasta_genome(path) -> GenomeReference` — yields `CONTIG_TABLE` (one row per record: `contig_id` sequential int64, `name` from FASTA id, `length`, `topology` left null, `circular` left null) + `SEQUENCE_TABLE` (the literal sequence string). Header description goes into `metadata_extras['contig_descriptions']` keyed by name. Soft-masking (lowercase) preserved per the schema comment.
- `read_fasta_transcriptome(path) -> TranscriptReference` — same parsing, different schema target. `transcript_id` is sequential int64; `name` is the FASTA id. `gene_id` left null by default; if Ensembl-style headers (`ENST… gene:ENSG…`) are detected, parse and stash the parent gene name in `metadata_extras['gene_name_by_transcript']` for later resolution against an Annotation in S2.
- `read_fasta_genetic_tools(path) -> GeneticTools` — same parser; caller supplies category via sidecar TSV (`name<TAB>category<TAB>sequence_type`) or kwargs (`*, category='cloning_vector_backbone', sequence_type='nucleotide'`). Not used by the bundled DB (built from JSON); lets users BYO FASTA + minimal metadata. Validates that `category` and `sequence_type` are in the allowed vocabularies.

The existing `readers/fastx.py` already imports `pysam` for FASTQ-as-input-reads use cases (read ingest path → `READ_TABLE`); that import stays — we're only declining to introduce pysam.FastaFile *additionally* for the new FASTA-as-reference path. If we later hit a human-class genome and need O(1) contig random access, a sibling `IndexedFastaReader` (pysam-backed) lands as a fast-path optimization without touching the streaming reader.

**GFF3 reader** at [constellation/sequencing/readers/gff.py](constellation/sequencing/readers/gff.py) — hand-rolled streaming parser, no third-party deps. Public surface:

- `read_gff3(path) -> Annotation` — ~250 lines, two-pass:
  - **Pass 1 (streaming):** open file (gzip-aware via stdlib `gzip.open` when path ends in `.gz`); skip blank lines and `#` comments; honour `##gff-version 3` and `##sequence-region` directives (collected into `metadata_extras`); break on `##FASTA` if present (the embedded FASTA companion is read separately into a sibling `GenomeReference` if the caller asks). For each data line, split on `\t`, parse the 9-column row into pre-allocated Python lists (one per output column) — start/end as int64 with `start - 1` to convert 1-based inclusive → 0-based half-open; strand kept as `'+' / '-' / '.'`; phase as int32 nullable; score as float32 nullable. Parse the attribute column: split on `;`, then `=`, URL-decode values via `urllib.parse.unquote`. Pull reserved keys (`ID`, `Name`, `Parent`, `Dbxref`, `Note`, `Alias`, `Target`) into dedicated working dicts; pack the rest into a per-row JSON string for `attributes_json`. Allocate a sequential `feature_id: int64` per row; build a `name (=ID attribute) → feature_id` dict during the pass.
  - **Pass 2:** walk the collected `Parent=` strings; resolve to `parent_id: int64` via the dict, leaving null on top-level features. Multi-parent (`Parent=g1,g2`) is rare in well-formed GFF3 but legal — handle by emitting a duplicate feature row per parent (ID stays unique by suffixing); the Spec allows this and pyranges does the same.
  - Convert column lists to `pa.array(...)` and then `pa.Table.from_arrays(...)`, cast to `FEATURE_TABLE`, hand to `Annotation(features=...)`.
- `read_gff3_streaming(path, *, chunk_size=100_000) -> Iterator[pa.RecordBatch]` — generator variant for huge GFF3s where the full table won't fit in memory. (Vertebrate GFF3 is ~50 MB gzipped, ~3M rows; non-streaming is fine on a typical workstation, streaming is here for VCF-companion or chunked workflows.) Probably skipped this session if scope tightens — call out as deferred if so.
- `write_gff3(annotation, path, *, genome=None) -> None` — emits a GFF3 with a top `##gff-version 3` line, `##sequence-region <name> 1 <length>` per contig (when `genome` is supplied), then features sorted by `(contig_id, start, end)` with 0-based half-open → 1-based inclusive on the way out. Useful when downstream tools (BUSCO, RepeatMasker outputs) flow through Annotation and we want to round-trip; can defer the writer to S2 if needed but it's small enough to ship now.

**BED reader** at [constellation/sequencing/readers/bed.py](constellation/sequencing/readers/bed.py) — small companion (~80 lines). `read_bed(path, *, columns: int | None = None) -> Annotation`. BED is already 0-based half-open so no coordinate conversion. Maps BED columns to `FEATURE_TABLE`: name (col 4) → `name`, score (col 5) → `score`, strand (col 6) → `strand`. BED entries don't have parent relations so `parent_id` is always null. `type` defaults to `"interval"` (configurable).

**Edge cases the parser handles** (and tests cover): comment-only files, empty attribute column (`.`), URL-encoded characters in attributes (`Name=foo%20bar` → `foo bar`), GFF3 with only directives (no data rows) → empty Annotation, embedded `##FASTA` section, gzipped input, mixed line endings, multi-parent features.

### D. Genetic-tools curation

**[constellation/data/genetic_tools.json](constellation/data/genetic_tools.json)** — bundle the curated DB. Build script at [scripts/build-genetic-tools-json.py](scripts/build-genetic-tools-json.py) hits FPbase REST API for fluorescent proteins (https://www.fpbase.org/api/proteins/), aggregates curated AbR genes / common tags / promoters from explicit sources (NCBI accessions hardcoded, with `source_url` per entry), writes `genetic_tools.json`. Vendored sources go under `constellation/data/_raw/genetic_tools/` (FPbase JSON snapshot, AbR FASTA, etc.) regenerable via the script. Initial target: ~150 entries covering AbR (8: amp/kan/cm/tet/spec/hyg/zeo/bla), FPs (~40 most common from FPbase including GFP/mCherry/mScarlet/mNeonGreen/etc.), epitope tags (~10: His/FLAG/HA/Myc/V5/Strep/SBP/AviTag/SUMO/MBP), promoters (~10: T7/CMV/EF1α/SV40/CAG/TEF1/GAL1/AOX1/lac/tac), selection markers (~5: URA3/LEU2/HIS3/TRP1/zeocin), common enzymes (~10: Cas9 variants/Cre/FLP/T4 ligase/T7 polymerase/reverse transcriptases), secretion signals (~5: α-factor/IgK/HSA), vector backbones (UniVec import — top ~50 most common).

The pre-screen filter itself (running `genetic_tools` mmseqs2 search before UniRef90) is S3 work; S1 just ships the data + container.

### E. CLI surface

[constellation/cli/__main__.py](constellation/cli/__main__.py) — add a new top-level `reference` subparser. **No new third-party tool deps** — fetch uses stdlib HTTP. Verbs:

- `constellation reference import --fasta <fa> [--gff3 <gff>] --output-dir <dir>` — the workhorse. Local-files-in, ParquetDir-bundles-out. Calls `read_fasta_genome` + `read_gff3` (when GFF3 supplied), validates closure (`Annotation.validate_against(genome)`), saves both as `<output-dir>/{genome,annotation}/`. Idempotent. This is what most users will actually invoke since labs typically download references manually once and reuse them across many runs.
- `constellation reference fetch <source>:<id> --output-dir <dir>` — thin convenience using stdlib `urllib.request`, no new tool deps. `<source>` is one of `ensembl`, `ensembl_genomes`, `refseq`. Examples: `ensembl_genomes:saccharomyces_cerevisiae`, `refseq:GCF_001708105.1`. The fetcher constructs deterministic URL patterns per source, downloads the genome FASTA and GFF3 directly from the published FTP/HTTPS endpoints, gunzips inline, and pipes through the same `read_fasta_genome` + `read_gff3` path as `import`. Source-specific URL builders live in [constellation/sequencing/reference/fetch.py](constellation/sequencing/reference/fetch.py) — small (~100 lines per source), no auth, no signed URLs, no NCBI Datasets CLI. If a user wants exotic NCBI accessions or Ensembl-non-standard accessions, they fall back to manual download + `constellation reference import`. The fetcher is a 90/10 convenience tool, not a comprehensive accession resolver.
- `constellation reference summary <ref-dir>` — prints contig count, total length, feature-type histogram, transcripts-derivable check (mRNA/exon present?). Read-only.
- `constellation reference validate <ref-dir>` — round-trips through `load_*` then `.validate()` + `.validate_against(...)` on companion containers; reports PK/FK violations cleanly.

Lazy-imports the heavy modules (Reference/Annotation/TranscriptReference) inside the handler function — the CLI startup stays light.

**Deferral note:** if the lab later hits an organism that doesn't follow Ensembl/RefSeq URL conventions (e.g. an Addgene-hosted plasmid map, a recent NCBI submission with signed-URL-only access), an NCBI Datasets CLI adapter at `constellation/thirdparty/datasets.py` lands then — same `ToolSpec` pattern as `minimap2.py`. Keeping it out of S1 means we don't introduce an 80MB Go-binary tool dep before there's a concrete use case driving it.

### F. Cross-cutting

- [constellation/sequencing/quant.py](constellation/sequencing/quant.py) (currently a partial-quant builder used by transcriptome S1) — minor refactor: when emitting `feature_quant` rows, stamp `feature_origin='protein_hash'` for the existing protein-clustering output. This is the only S1 code-path touching the new column — preserves S1 behaviour while threading the new schema through.
- [tests/](tests/) — new test files:
  - `test_reference_genome.py` — round-trip a small synthetic GenomeReference through ParquetDir; PK/FK validation; `validate_against` on a paired Annotation.
  - `test_reference_annotation.py` — GFF3 → Annotation parse + write round-trip; coord conversion (1-based GFF3 → 0-based internal → 1-based on write); URL-decoded attributes; gzipped input; multi-parent features; embedded `##FASTA` section split correctly; empty / comment-only / directive-only files; comparison vs a known-good Pichia GFF3 (compare row counts + a sampled feature's start/end vs IGV display).
  - `test_reference_transcripts.py` — `TranscriptReference.from_annotation` against a hand-built 3-gene synthetic GenomeReference; verify reverse-complement on `-` strand; verify multi-exon splicing.
  - `test_genetic_tools.py` — load the bundled JSON, verify category histogram, FASTA emission round-trips through SeqIO-style parsing.
  - `test_imports.py` — extend with new submodule imports (`sequencing.annotation`, `sequencing.transcripts`, `sequencing.genetic_tools`).

---

## Critical files to modify

| File | Action |
|---|---|
| [constellation/sequencing/schemas/reference.py](constellation/sequencing/schemas/reference.py) | Add `TRANSCRIPT_TABLE`, `ENGINEERING_PART_TABLE`. Existing schemas unchanged. |
| [constellation/sequencing/schemas/quant.py](constellation/sequencing/schemas/quant.py) | Add `feature_origin` column to `FEATURE_QUANT`. |
| [constellation/sequencing/reference/reference.py](constellation/sequencing/reference/reference.py) | Rename `Reference` → `GenomeReference`; drop `features`; ship `__post_init__` + `validate` + `n_contigs` + `sequence_of`. |
| [constellation/sequencing/reference/io.py](constellation/sequencing/reference/io.py) | Rename Protocols/registries; ship `ParquetDirReader/Writer` and `save_genome_reference/load_genome_reference`. |
| [constellation/sequencing/annotation/annotation.py](constellation/sequencing/annotation/annotation.py) | New file. `Annotation` container with cast-and-validate. |
| [constellation/sequencing/annotation/io.py](constellation/sequencing/annotation/io.py) | New file. Reader/Writer Protocols + ParquetDir. |
| [constellation/sequencing/transcripts/__init__.py](constellation/sequencing/transcripts/__init__.py) | New directory + `TranscriptReference`. |
| [constellation/sequencing/transcripts/transcripts.py](constellation/sequencing/transcripts/transcripts.py) | New file. Container + `from_annotation` materialiser. |
| [constellation/sequencing/transcripts/io.py](constellation/sequencing/transcripts/io.py) | New file. Reader/Writer Protocols + ParquetDir. |
| [constellation/sequencing/genetic_tools.py](constellation/sequencing/genetic_tools.py) | New file. `GeneticTools` container + `load_default_genetic_tools`. |
| [constellation/sequencing/readers/fastx.py](constellation/sequencing/readers/fastx.py) | Extend with `read_fasta_genome`, `read_fasta_transcriptome`, `read_fasta_genetic_tools`. |
| [constellation/sequencing/readers/gff.py](constellation/sequencing/readers/gff.py) | Hand-rolled streaming GFF3 parser — `read_gff3`, `write_gff3`, no third-party deps. |
| [constellation/sequencing/readers/bed.py](constellation/sequencing/readers/bed.py) | New file. Hand-rolled BED reader → `Annotation`. |
| [constellation/sequencing/assembly/assembly.py](constellation/sequencing/assembly/assembly.py) | Rename `to_reference` → `to_genome_reference`. |
| [constellation/sequencing/quant.py](constellation/sequencing/quant.py) | Stamp `feature_origin='protein_hash'` on existing partial-quant output. |
| [constellation/cli/__main__.py](constellation/cli/__main__.py) | Add `reference {import, fetch, summary, validate}` subparser. |
| [constellation/sequencing/reference/fetch.py](constellation/sequencing/reference/fetch.py) | New file. Stdlib-HTTP fetchers for Ensembl / Ensembl Genomes / RefSeq URL conventions. No third-party tool dep. |
| [constellation/data/genetic_tools.json](constellation/data/genetic_tools.json) | New bundled DB. |
| [scripts/build-genetic-tools-json.py](scripts/build-genetic-tools-json.py) | New file. Idempotent builder mirroring `build-unimod-json.py` style. |
| [environment.yml](environment.yml) | No new deps required — GFF3, FASTA, and BED readers are all hand-rolled (stdlib only). |
| [CLAUDE.md](CLAUDE.md) | Update module index to reflect split containers + multi-tag `feature_quant`; add note that `Reference` references in earlier text now mean `GenomeReference`; note hand-rolled GFF3/BED parsers (no pyranges/gffutils). |

## Existing utilities to reuse

- `core.sequence.nucleic.reverse_complement` — for `TranscriptReference.from_annotation` strand handling.
- `core.io.schemas.register_schema` — auto-discovery of new schemas.
- `massspec.library.Library.__post_init__` pattern — copy the cast-and-validate idiom verbatim into the four new containers (saves arguing about edge cases the Library design already settled).
- `massspec.library.io` Protocols + registry — the template for all four new Reader/Writer registries.
- `sequencing.parallel.run_batched` + `sequencing.progress.ProgressCallback` — not exercised this session but the FASTA/GFF3 readers should be progress-callback-aware (single-stage, but the Protocol cost is zero).
- `core.sequence.proforma.format_proforma` — for protein engineering parts, store sequences as ProForma 2.0 strings if mods are involved (most won't be — flat AA sequence is fine).
- Stdlib only for the new FASTA + GFF3 + BED readers — `gzip.open` for `.gz`, `urllib.parse.unquote` for GFF3 attribute decoding, `json` for `attributes_json` packing. Pysam stays scoped to `readers/sam_bam.py` where it genuinely earns its keep (CIGAR / virtual offsets / sorting); FASTA-as-reference parsing is hand-rolled.

## Verification

End-to-end smoke test (run after implementation):

```bash
# 1. Library imports cleanly
pytest tests/test_imports.py

# 2. New container tests pass
pytest tests/test_reference_genome.py tests/test_reference_annotation.py \
       tests/test_reference_transcripts.py tests/test_genetic_tools.py

# 3. Genetic tools JSON loads + summary makes sense
python -c "from constellation.sequencing.genetic_tools import load_default_genetic_tools; \
            gt = load_default_genetic_tools(); \
            print(gt.tools.group_by('category').aggregate([('tool_id','count')]).to_pylist())"

# 4a. Local-file import path (the most common workflow)
constellation reference import --fasta tests/data/pichia/GCF_001708105.1.fna.gz \
    --gff3 tests/data/pichia/GCF_001708105.1.gff.gz \
    --output-dir /tmp/pichia_ref

# 4b. Stdlib-HTTP fetch path — Pichia is a small + non-vertebrate test target
constellation reference fetch refseq:GCF_001708105.1 --output-dir /tmp/pichia_fetched

# 4c. Summary + validate work on either output
constellation reference summary /tmp/pichia_ref
constellation reference validate /tmp/pichia_ref

# 5. Manual sanity: load + materialize transcripts
python -c "
from pathlib import Path
from constellation.sequencing.reference.io import load_genome_reference
from constellation.sequencing.annotation.io import load_annotation
from constellation.sequencing.transcripts.transcripts import TranscriptReference
g = load_genome_reference(Path('/tmp/pichia_ref/genome'))
a = load_annotation(Path('/tmp/pichia_ref/annotation'))
a.validate_against(g)
t = TranscriptReference.from_annotation(g, a)
print(f'{g.n_contigs} contigs, {a.n_features} features, {len(t.transcripts)} transcripts')
"

# 6. Existing transcriptome demux still works (regression check)
pytest tests/test_transcriptome_demux.py -q

# 7. Lint
ruff check constellation/ tests/ scripts/build-engineering-parts-json.py
```

Success criteria:
- All four new containers round-trip ParquetDir lossless.
- GFF3 → Annotation → `write_gff3` produces a byte-similar output (modulo attribute key ordering) and IGV-displayed coordinates match the original.
- `TranscriptReference.from_annotation` reproduces Ensembl/NCBI-published transcript sequences for ≥3 sampled transcripts (positive-strand single-exon, positive-strand multi-exon, negative-strand multi-exon — handpicked from Pichia annotation).
- `feature_quant` from existing transcriptome demux still validates; new `feature_origin` column populated as `'protein_hash'` on every row.
- `genetic_tools.json` loads, contains ≥150 entries spanning all 9 categories, FASTA round-trips.
- Documentation in CLAUDE.md reflects the split-Reference architecture and multi-tag `feature_quant`.

---

## Sketch of S2 + S3 (separate plan files)

**S2 — genome-guided alignment + de novo clustering** (~1 session). Scope deliberately narrow: deliver **two independent pipelines** that the user picks between based on whether a reference is available, never mixed. Transcriptome-mode mapping and isoform-resolution defer to a later session.

**Mode A — Reference-guided (genome) gene-level counts.** Single pipeline: `minimap2 -ax splice -uf --cs=long --secondary=no` → `Alignments` (validated against `GenomeReference`) → ENCODE-style filter (`min_length=300, min_aligned_fraction=0.7, sense_only=True, primary_only=True`) → hand-rolled gene-level overlap counter. For each filtered primary alignment, intersect with `Annotation.features_of_type('gene')` via vectorized `pa.compute` interval-overlap, assign to the highest-overlap gene with a strand check, drop reads with no qualifying overlap (default threshold: ≥0.5 of aligned span within one gene). Emits `feature_quant` rows tagged `feature_origin='gene_id'` only. Reads that fail to align or fail the threshold are recorded in QC stats (per-sample mapping rate, mean alignment fraction, % primary, sense vs antisense breakdown) but **not assigned a cluster_id and not propagated into feature_quant** — a read either contributes to the reference-guided result or it doesn't. CLI: `constellation reference align --reads <bam> --genome <ref> --annotation <ann> --output-dir <dir>`.

**Mode B — De novo clustering.** Independent pipeline that does not consult a reference. Initial implementation reuses the user-sketched k-mer-similarity approach: build a read–read graph keyed on shared minimizers (or a kmer-Jaccard sketch), with edge weights modulated by per-read abundance evidence, then consolidate into clusters via connected-components / community-detection on that graph. mmseqs2 `easy-linclust` is a candidate baseline for comparison rather than the final implementation — the user has a clearer vision for an abundance-weighted minimizer graph that handles nanopore error structure better than linclust's identity threshold. Hand-labelled fixtures from `transcriptome.simulator` drive the tuning per the user-feedback-memory rule on synthetic-fixtures-first. Emits `feature_quant` rows tagged `feature_origin='cluster_id'` only. CLI: `constellation transcriptome cluster --reads <bam> --output-dir <dir>`.

**Why we keep the modes separate.** Mixing reference-guided and de novo within one pipeline imposes inconsistent priors on the resulting count matrix — reads near the threshold of acceptable overlap are statistically different from reads with no overlap at all, and shoving them into a single `feature_quant` table conflates two distinct statistical models of "what feature did this read measure?". Clean separation lets users compare modes downstream by running both pipelines on the same input and joining the outputs externally — that's an analysis choice, not a pipeline behaviour.

*Per-sample stratification* (both modes). Counting runs grouped by `sample_id` — joining `Alignments` (Mode A) or the cluster-membership table (Mode B) against the existing `READ_DEMUX_TABLE` on `read_id` to recover the sample assignment, then group_by-summing per `(feature_id, sample_id)`. ProcessPool fan-out via the existing `parallel.run_batched`. Output is a partitioned-parquet `feature_quant/` dataset, queryable from R / Python / DuckDB without a single-file aggregation step (matches the architectural invariant about partitioned-parquet handoffs from CLAUDE.md).

*Downstream tie-in.* The `feature_quant` parquet is the canonical output. Users who want DESeq2/edgeR analysis read the parquet directly via `arrow::read_parquet` in R or pivot to the gene × sample (or cluster × sample) matrix DESeq2 expects. We do not ship a Salmon-style `quant.sf` writer in this session — that's a future-session deliverable when the EM transcriptome workflow lands and Salmon-format compatibility actually matters.

Synthetic ground-truth fixtures from `transcriptome.simulator` test both pipelines end-to-end.

**S3 — protein layer** (~1 session). mmseqs2 wrapper with `easy-search` against UniRef90 + the bundled `genetic_tools` DB. Pre-screen ORFs against `genetic_tools` before UniRef90 (cRAP-equivalent — known-construct filter). New `feature_origin='protein_uniref'` and `'genetic_tool'` partitions in `feature_quant`. Wire the bridge into `massspec.library` — predicted ORFs → cleavage → `Peptidoform` records → entries in a sample-specific `Library`, completing the transcriptome→proteome connection that's currently a stub at `constellation/transcriptome_to_proteome.py`.

**Future session — transcriptome-mode + custom isoform-EM.** Adds `minimap2 -x map-ont -N 100 -p 1.0` against `TranscriptReference` (the data class lands in S1 but this is the first session that exercises it for alignment). The EM quantifier is a Constellation-native implementation: a `TranscriptAbundance(Parametric)` model parameterised by per-transcript log-abundance with a length-aware emission distribution; fit via `core.optim.LBFGSOptimizer` against the read→transcript multi-mapper assignment matrix. Stranded-read handling, 3′-bias correction, and read-length priors all drop in as model components rather than CLI flags, matching the `core.stats` design. Output: `feature_origin='transcript_id'` rows in `feature_quant`. The Salmon-style `quant.sf` writer lands here for R-ecosystem compatibility — strictly a derived view of the canonical Arrow `feature_quant`.
