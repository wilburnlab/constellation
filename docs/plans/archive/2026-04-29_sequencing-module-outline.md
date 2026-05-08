# `constellation.sequencing` — module outline & stub session

## Context

The user wants to pivot Constellation development from `core` + `massspec` (largely shipped) into the sequencing modality, which absorbs **NanoporeAnalysis** as a clean rewrite. The lab's current and near-future workflows span:

- **Genomic DNA** (immediate): POD5 → Dorado (3–4 days/flow-cell on A100) → multi-flowcell aggregation (>60× for large eukaryotes) → HiFiASM → polish (Dorado + minimap2) → RagTag scaffold → BUSCO + repeat/telomere annotation.
- **Transcriptomic cDNA** (immediate): same basecalling → custom 3-step segmented demux (polyA scan → 3' PCR adapter + barcode → 5' adapter + transcript) using `edlib` → ORF prediction → mmseqs-style kmer clustering with abundance-weighted consensus → reference-free transcriptome assembly → predicted proteins feed back into `massspec` spectral libraries.
- **Direct RNA / modifications** (research): direct-current analysis, custom physics-based basecaller models trained on in-house synthetic libraries.

This session ships **documented stubs only** — class signatures + dataclass field lists + Arrow schema column definitions + docstrings + `NotImplementedError` bodies — laying the architectural foundation for sequential implementation in future sessions. **No new dependencies land in `pyproject.toml` or `environment.yml`**; each phase declares its own deps. Imports of not-yet-required external libraries (`pysam`, `mappy`, etc.) are deferred to inside function bodies so module-level imports stay green. Existing import smoke tests must still pass; new smoke tests cover the new tree.

## Design overview (architectural calls)

Validated against [CLAUDE.md](../../projects/constellation/CLAUDE.md), the [massspec template](../../projects/constellation/constellation/massspec/), and [core.io patterns](../../projects/constellation/constellation/core/io/) by a Plan agent. Key decisions:

1. **No Library/Quant/Search trio.** MS's clean theoretical/empirical/scoring split doesn't map onto sequencing — for de novo work, the "library" *is* the experimental output. Instead: `acquisitions + samples + reads + alignments + reference + assembly + quant + annotation`, where:
   - `Reference` is the only container that mirrors `Library`'s shape (PK/FK closure, ParquetDir native form, Reader/Writer Protocols).
   - `Assembly` is its own container; `Assembly.to_reference()` lifts a finished assembly into a `Reference` so external-ref and de-novo-ref workflows share downstream code paths.
   - `Quant` keys per-feature counts/coverage/TPM on `(feature_id, sample_id)`.
   - `Alignments` is its own peer container (per-acquisition, FK to `Reference`).
   - No `Search` analog — alignments are observations, not detection scores.

2. **`Samples` is first-class.** A sample = `(acquisition_ids, barcode_id?)`. M:N edge table `SAMPLE_ACQUISITION_EDGE` handles single-sample multi-flowcell (genomic) AND multi-sample single-flowcell (multiplexed transcriptomic) without bifurcating the schema. Lives at `sequencing/samples.py`, sibling to `acquisitions.py`.

3. **Pure-functional core, `Project` orchestrator on top.** Verbs like `bam_dir_to_assembly(bam_paths)` work standalone. `Project.open(path)` reads `manifest.toml`, knows the `pod5/ → bam/ → assembly/ → annotation/ → transcriptome/` layout, and is the only layer that mutates filesystem state. Mirrors the `Library`-doesn't-touch-disk / `save_library` pattern.

4. **`DoradoRunner` is one class with method-per-subcommand.** Returns `RunHandle` objects (not blocking by default) so 3–4-day basecalls support detached / `--follow` / `--resume`. `DoradoModel` dataclass parses `"sup@v5.0+5mC,5hmC"` shorthand.

5. **`align/` modules are named by verb, not backend.** `pairwise.py` (edlib + parasail behind a unified API), `map.py` (minimap2/mappy), `locate.py` (the substring/edge-distance utilities demux needs). Backend is a kwarg.

6. **Generic `ragged_trace_1d` schema lands in `core.io`** — POD5 raw signal and MS scan tables (m/z + intensity blobs, when MS scan readers ship) share the same shape: variable-length 1D array per row with rich row-level metadata. POD5 reads are 1k–5M+ samples each (~10¹³ rows if flattened to `Trace1D`); MS scans are 10⁴–10⁵ peaks each. Both clear the "≥2 modalities" bar for `core.io.schemas` membership. Lifted out of sequencing-specific scope into a generic constructor — see "Lift to core.io" section below. POD5's `RAW_SIGNAL_TABLE` is then a thin instantiation of it; future massspec scan readers consume the same primitive.

7. **Demux output is `READ_SEGMENT_TABLE`** (one row per *located segment* — adapter, polyA, barcode, transcript) so chimeras/concatemers are first-class. Derived view `READ_DEMUX_TABLE` collapses to one row per resolved segment for downstream clustering.

8. **`Adapter` and `Barcode` are independent registry-grade objects.** `LibraryConstruct` composes them via `tuple[Segment, ...]` (AdapterSlot, BarcodeSlot, PolyASlot, UMISlot, TranscriptSlot) so different ONT/SMARTer-derived chemistries are encodable.

9. **CLI commands live in `constellation.cli`, not `sequencing/cli/`.** Match the massspec pattern. New verbs (`basecall`, `demux`, `assemble`, `annotate`) are placeholders alongside the existing `pod5` placeholder in [cli/__main__.py](../../projects/constellation/constellation/cli/__main__.py).

10. **Cross-modality workflow ships as a top-level script for now.** `constellation/transcriptome_to_proteome.py` is the only such workflow today, so it lives at top-level per CLAUDE.md's "≥2 workflows before a folder" rule. **Open question, not decided this session:** when a second cross-modality pipeline lands (e.g. genome-to-spectral-library, structure-to-spectral-prediction), we'll revisit how to organize these — likely a dedicated subpackage, but the name is undecided. `bridges/` and `pipelines/` are both ruled out (CLAUDE.md vetoes them and the user dislikes "bridges" specifically). Naming candidates worth weighing later — drawing on the project's astronomical / cartographic / weaving theme — include `asterism/`, `confluence/`, `weft/`, `synthesis/`, `composer/`, `lattice/`. **Defer until we have two workflows in flight to compare.**

11. **`signal/` houses POD5 processing AND future custom basecaller models.** Physics-based translocation/k-mer-current models are `core.stats.Parametric`-shaped — stub a `signal/models/` ABC now, even with no implementations, so the research direction is documented.

## Module layout

```
constellation/sequencing/
├── __init__.py                       # re-exports Acquisitions, Samples, Reference, Project, key schemas
├── acquisitions.py                   # SEQUENCING_ACQUISITION_TABLE + Acquisitions container
├── samples.py                        # SAMPLE_TABLE + SAMPLE_ACQUISITION_EDGE + Samples container
├── schemas/
│   ├── __init__.py                   # re-exports + self-registers all schemas with core.io
│   ├── reads.py                      # READ_TABLE (basecalled), QUALITY semantics
│   ├── alignment.py                  # ALIGNMENT_TABLE (BAM-canonical) + ALIGNMENT_TAG_TABLE long-form
│   ├── signal.py                     # RAW_SIGNAL_TABLE (POD5 raw signal, var-length list<int16>)
│   ├── reference.py                  # CONTIG_TABLE, FEATURE_TABLE, SEQUENCE_TABLE
│   ├── assembly.py                   # ASSEMBLY_CONTIG_TABLE, SCAFFOLD_TABLE, ASSEMBLY_STATS
│   ├── quant.py                      # FEATURE_QUANT, COVERAGE_TABLE
│   └── transcriptome.py              # READ_SEGMENT_TABLE, READ_DEMUX_TABLE, TRANSCRIPT_CLUSTER_TABLE
├── quality/
│   ├── __init__.py
│   └── phred.py                      # offset-33 ASCII codec, QC summaries
├── reference/                        # external-ref or assembled-ref container
│   ├── __init__.py
│   ├── reference.py                  # Reference container (PK/FK, ParquetDir native form)
│   └── io.py                         # ReferenceReader/Writer Protocols + registries + ParquetDir
├── assembly/                         # de novo assembly + polish + scaffold
│   ├── __init__.py
│   ├── assembly.py                   # Assembly container; Assembly.to_reference()
│   ├── io.py                         # AssemblyReader/Writer Protocols + ParquetDir
│   ├── hifiasm.py                    # HiFiASM subprocess wrapper (uses thirdparty/hifiasm)
│   ├── polish.py                     # Dorado-based polishing (uses minimap2)
│   ├── ragtag.py                     # RagTag scaffolding wrapper
│   └── stats.py                      # N50, contig stats, BUSCO completeness
├── annotation/
│   ├── __init__.py
│   ├── busco.py
│   ├── repeats.py                    # repeat element finding (RepeatMasker/EDTA TBD)
│   ├── telomeres.py                  # telomere/centromere search
│   └── transcripts.py                # transcript-to-genome mapping
├── alignments/                       # per-acquisition alignment records
│   ├── __init__.py
│   ├── alignments.py                 # Alignments container; FK to Reference + Acquisitions
│   └── io.py                         # ParquetDir + BAM/SAM round-trip via sequencing/io/sam_bam.py
├── readers/                          # RawReader subclasses, self-register
│   ├── __init__.py
│   ├── pod5.py                       # POD5 → RAW_SIGNAL_TABLE (uses pod5 PyPI)
│   ├── fastx.py                      # FASTA/FASTQ (gzipped + plain) → READ_TABLE
│   ├── sam_bam.py                    # SAM/BAM → READ_TABLE + ALIGNMENT_TABLE (uses pysam)
│   ├── paf.py                        # PAF → ALIGNMENT_TABLE (no separate PAF schema)
│   └── gff.py                        # GFF3/GTF → FEATURE_TABLE
├── io/                               # cross-tier file-format adapters
│   ├── __init__.py
│   └── sam_bam.py                    # BAM produces (Reads, Alignments) tuple; self-registers with both registries
├── basecall/
│   ├── __init__.py
│   ├── dorado.py                     # DoradoRunner: subprocess wrapper, RunHandle, model resolution
│   └── models.py                     # DoradoModel dataclass; "sup@v5.0+5mC" shorthand parser
├── signal/                           # POD5 squiggle processing + future custom basecaller models
│   ├── __init__.py
│   ├── normalize.py                  # signal scaling (mad-norm, scale/offset application)
│   ├── segment.py                    # squiggle segmentation (deferred)
│   └── models/
│       ├── __init__.py               # BASECALLER_MODEL registry (stub)
│       └── physical.py               # Physics-based basecaller ABC (deferred research stub)
├── align/
│   ├── __init__.py
│   ├── pairwise.py                   # pairwise_align(query, ref, *, backend='edlib'|'parasail')
│   ├── map.py                        # minimap2 / mappy mapping
│   └── locate.py                     # error-tolerant substring search (anchors demux)
├── transcriptome/
│   ├── __init__.py
│   ├── adapters.py                   # Adapter, Barcode, LibraryConstruct registries
│   ├── demux.py                      # locate_segments(reads, construct) -> READ_SEGMENT_TABLE
│   ├── orf.py                        # thin wrapper over core.sequence.nucleic.find_orfs → ORF_TABLE
│   ├── cluster.py                    # mmseqs-style kmer + abundance-weighted consensus clustering
│   ├── consensus.py                  # consensus transcript building from clusters
│   └── network.py                    # read-similarity → core.graph.Network → gene/allele/exon graph
├── modifications/
│   ├── __init__.py
│   └── basemod.py                    # m6A / m5C / ψ calling from MM/ML BAM tags or signal squiggles (deferred)
└── projects/
    ├── __init__.py
    ├── layout.py                     # Project directory schema constants
    └── manifest.py                   # Project class; manifest.toml R/W; verbs delegating to pure layer
```

Plus:

- **`constellation/transcriptome_to_proteome.py`** — top-level cross-modality script (stub). Placement is pragmatic-for-now (one workflow, follows CLAUDE.md's `≥2 workflows` rule), not a permanent architectural commitment — see decision #10 above.
- **`constellation/thirdparty/{dorado,hifiasm,minimap2,samtools,ragtag,busco}.py`** — `ToolSpec` adapters.
- **`scripts/install-{dorado,hifiasm,ragtag,busco}.sh`** — hash-pinned installers for tools not in conda (or where conda isn't preferred). `minimap2` and `samtools` are bioconda-installable; declare via `environment.yml` with `path_bin="minimap2"` / `path_bin="samtools"` in their adapters and skip the install script.
- **CLI placeholders** in [cli/__main__.py](../../projects/constellation/constellation/cli/__main__.py): `basecall`, `demux`, `assemble`, `polish`, `scaffold`, `annotate`, `cluster`, `project`. Each raises "scaffolded but not implemented" today (matching the existing `pod5` placeholder).

## What ships this session vs. future sessions

**Ships this session (real code, not just stubs):**
- `core.io.schemas.ragged_trace_1d(...)` — the full constructor, ~20 lines. Tiny enough that stubbing it would be silly when downstream schemas reference it. Tests cover the shape contract.
- All Arrow schema *definitions* (column names, dtypes, PK/FK declarations) — schemas are data, not behavior, so "stub" doesn't really apply. They land as their final shape, parameterized via `ragged_trace_1d` where applicable.
- All `__init__.py` re-export wiring + schema self-registration calls into `core.io.schemas.register_schema`.
- `DoradoModel` dataclass with the `parse("sup@v5.0+5mC")` shorthand parser fully implemented (small; used by every Phase 3 stub signature).

**Stubs this session (signatures + `NotImplementedError` bodies):**
- All container classes (`Acquisitions`, `Samples`, `Reference`, `Assembly`, `Alignments`, `Quant`, `Project`) — class definitions, dataclass field lists, method signatures with type hints + docstrings. Method bodies raise `NotImplementedError("Pending Phase N — see plan in-our-development-of-fuzzy-quilt.md")`.
- All Reader/Writer Protocol declarations (`ReferenceReader`, `AssemblyReader`, etc.) and registry dicts (empty at import time; populated as readers ship).
- All third-party tool wrapper classes (`DoradoRunner`, HiFiAsm/RagTag/BUSCO wrappers) — method signatures only.
- All `thirdparty/{dorado,hifiasm,minimap2,samtools,ragtag,busco}.py` — `ToolSpec` registration + `_probe_version` callable signature; the install-script paths are stub paths that don't yet exist on disk.
- All transcriptome-pipeline functions (`locate_segments`, `cluster_reads`, etc.) — signatures + docstrings.
- `constellation/transcriptome_to_proteome.py` — top-level workflow stub.
- New CLI placeholders (`basecall`, `demux`, `assemble`, `polish`, `scaffold`, `annotate`, `cluster`, `project`) wired into `cli/__main__.py` raising "scaffolded but not implemented" alongside the existing `pod5` placeholder.

**Out of scope this session:**
- `scripts/install-{dorado,hifiasm,ragtag,busco}.sh` — install scripts are not written; the `ToolSpec` adapters merely *reference* them as paths to be created in the relevant phase. Writing 6 installer scripts is its own session.
- Any new `pyproject.toml` or `environment.yml` declarations — declared per-phase as discussed.
- Any tests beyond import-smoke + `ragged_trace_1d` shape — full unit tests land per-phase.

## Per-module stub specification

The following sketches what each scaffold file contains. Each function/method body is `raise NotImplementedError("Pending Phase N — see plan")` with phase numbers from the implementation roadmap. Each module's `__init__.py` re-exports the public surface so downstream imports don't depend on file split.

### `acquisitions.py` ([template: massspec/acquisitions.py](../../projects/constellation/constellation/massspec/acquisitions.py))

```python
SEQUENCING_ACQUISITION_TABLE = pa.schema([
    ("acquisition_id", pa.int64()),       # PK
    ("source_path", pa.string()),         # POD5 dir or BAM file
    ("source_kind", pa.string()),         # 'pod5_dir' | 'bam' | 'fastq'
    ("acquisition_datetime", pa.timestamp("us", tz="UTC")),  # nullable
    ("instrument_id", pa.string()),       # nullable; e.g. "PromethION-PC24-001"
    ("flow_cell_id", pa.string()),        # nullable; e.g. "FAQ12345"
    ("flow_cell_type", pa.string()),      # nullable; e.g. "FLO-PRO114M" (R10.4.1)
    ("sample_kit", pa.string()),          # nullable; e.g. "SQK-LSK114"
    ("basecaller_model", pa.string()),    # nullable; e.g. "dna_r10.4.1_e8.2_sup@v5.0.0"
    ("experiment_type", pa.string()),     # nullable; 'genomic_dna' | 'cdna' | 'drna'
])

class Acquisitions:
    """Run-level provenance container. Mirrors massspec.acquisitions semantics."""
    def __init__(self, table: pa.Table): ...
    @property
    def ids(self) -> pa.Array: ...
    def to_parquet(self, path: Path) -> None: ...
    @classmethod
    def from_parquet(cls, path: Path) -> "Acquisitions": ...

def validate_acquisitions(table, acquisitions, *, column="acquisition_id", nullable=False): ...
```

### `samples.py`

```python
SAMPLE_TABLE = pa.schema([
    ("sample_id", pa.int64()),           # PK
    ("sample_name", pa.string()),
    ("description", pa.string()),        # nullable
])

SAMPLE_ACQUISITION_EDGE = pa.schema([
    ("sample_id", pa.int64()),           # FK
    ("acquisition_id", pa.int64()),      # FK
    ("barcode_id", pa.int64()),          # nullable; null = use all reads
])

class Samples:
    """M:N samples-to-acquisitions resolver. Handles both:
       - 1 sample × N flowcells (genomic ultra-HMW, no barcode)
       - M samples × 1 flowcell (multiplexed transcriptomic, M barcodes)
       - mixed multi-flowcell multiplexed designs.
    """
    def __init__(self, samples: pa.Table, edges: pa.Table): ...
    def acquisitions_for(self, sample_id: int) -> list[int]: ...
    def samples_for(self, acquisition_id: int, barcode_id: int | None = None) -> list[int]: ...
```

### `schemas/reads.py`

```python
READ_TABLE = pa.schema([
    ("read_id", pa.string()),            # PK; ONT UUID
    ("acquisition_id", pa.int64()),      # FK
    ("sequence", pa.string()),
    ("quality", pa.string()),            # offset-33 Phred
    ("length", pa.int32()),
    ("mean_quality", pa.float32()),      # nullable; precomputed Q-score
    ("channel", pa.int32()),             # nullable
    ("start_time_s", pa.float64()),      # nullable
    ("duration_s", pa.float32()),        # nullable
])
```

### `schemas/alignment.py`

Per Plan agent guidance: canonical columns + most-queried numeric tags inline; long-form `ALIGNMENT_TAG_TABLE` for the rare-tag tail. CIGAR stays as string (BAM-native) with a `cigar_to_ops()` helper for queryability — no exploded ops table at default.

```python
ALIGNMENT_TABLE = pa.schema([
    ("alignment_id", pa.int64()),        # PK
    ("read_id", pa.string()),            # FK
    ("acquisition_id", pa.int64()),      # FK
    ("ref_name", pa.string()),
    ("ref_start", pa.int64()),           # 0-based
    ("ref_end", pa.int64()),
    ("strand", pa.string()),             # '+' | '-'
    ("mapq", pa.int32()),
    ("flag", pa.int32()),
    ("cigar_string", pa.string()),
    ("nm_tag", pa.int32()),              # edit distance; nullable
    ("as_tag", pa.float32()),            # alignment score; nullable
    ("read_group", pa.string()),         # nullable
    ("is_secondary", pa.bool_()),
    ("is_supplementary", pa.bool_()),
])

ALIGNMENT_TAG_TABLE = pa.schema([
    ("alignment_id", pa.int64()),
    ("tag", pa.string()),                # 2-char BAM tag
    ("type", pa.string()),               # 'i' | 'f' | 'Z' | 'B'...
    ("value", pa.string()),              # serialized; cast on read
])

def cigar_to_ops(cigar_string: str) -> pa.Array: ...  # list<struct{op, length}>
```

### Lift to `core.io.schemas` — `ragged_trace_1d` constructor

This is the only `core.io` change in this session — a generic schema constructor for variable-length 1D arrays per row, addressing the gap between `TRACE_1D` (flat row-per-sample, fine for short HPLC/CE traces) and `spectra_matrix_2d(n)` (fixed-size list, fine for regular-grid DAD matrices). Neither covers the ragged case where each row holds a 1D blob of irregular length.

```python
# constellation/core/io/schemas.py — new function

def ragged_trace_1d(
    value_dtype: pa.DataType,
    *,
    value_name: str = "y",
    x_name: str | None = None,                                  # None → x is implicit (derivable)
    extra_columns: Sequence[tuple[str, pa.DataType]] = (),      # per-row metadata
) -> pa.Schema:
    """Canonical schema for variable-length 1D arrays per row.

    Use cases:
    - POD5 raw signal: value_name='signal', value_dtype=int16, x implicit
      (derive `t` from `start_sample / sampling_rate_hz`)
    - MS scan blob (when ms-scan readers ship): value_name='intensity',
      value_dtype=float64, x_name='mz', x_dtype=float64
    - Future NMR FIDs, CE squiggles, anywhere a row holds an irregular 1D blob

    Distinct from TRACE_1D (row-per-sample; fine for short benchtop traces but
    explodes to ~10^13 rows for one POD5 flow cell) and spectra_matrix_2d(n)
    (fixed-size list; only works on regular-grid bins like DAD wavelength
    matrices). The ≥2-modality test for core.io.schemas membership is met
    cleanly: sequencing + mass spectrometry both need this exact shape.
    """
    cols = [(name, dtype) for name, dtype in extra_columns]
    if x_name is not None:
        cols.append((x_name, pa.list_(pa.float64())))
    cols.append((value_name, pa.list_(value_dtype)))
    return pa.schema(cols)
```

Tensor bridges (added alongside `trace_to_tensor` / `spectra_to_tensor`):

```python
def ragged_to_tensors(
    table: pa.Table,
    *,
    value_column: str = "y",
    x_column: str | None = None,
) -> tuple[list[torch.Tensor], list[torch.Tensor] | None]:
    """Project ragged-list columns to a list of per-row tensors.
    Returns (values, xs?) where each list element is the row's 1D blob.
    Each row may have a different length — caller pads/batches as needed.
    """
    ...
```

### `sequencing/schemas/signal.py`

```python
from constellation.core.io.schemas import ragged_trace_1d

RAW_SIGNAL_TABLE = ragged_trace_1d(
    value_dtype=pa.int16(),                         # raw ADC counts
    value_name="signal",
    x_name=None,                                    # t = start_sample / sampling_rate_hz; implicit
    extra_columns=(
        ("read_id", pa.string()),                   # PK
        ("acquisition_id", pa.int64()),             # FK
        ("channel", pa.int32()),
        ("well", pa.int32()),
        ("start_sample", pa.int64()),
        ("sampling_rate_hz", pa.float32()),
        ("scale", pa.float32()),                    # int16 → pA
        ("offset", pa.float32()),
    ),
)
```

A future `massspec/schemas.py` MS_SCAN_TABLE (when ms-scan readers ship) will instantiate the same constructor with `value_dtype=pa.float64(), value_name="intensity", x_name="mz"` plus its own metadata columns. Schemas remain modality-distinct; the *shape primitive* is shared.

### `schemas/reference.py` + `schemas/assembly.py` + `schemas/quant.py` + `schemas/transcriptome.py`

Schemas declared with column lists; full PK/FK semantics documented in the container modules ([reference/reference.py](../../projects/constellation/constellation/sequencing/reference/reference.py), [assembly/assembly.py](../../projects/constellation/constellation/sequencing/assembly/assembly.py)).

Notable schemas:

- `CONTIG_TABLE` — `(contig_id PK, name, length, sequence_id FK)`
- `FEATURE_TABLE` — GFF3-shaped: `(feature_id PK, contig_id FK, start, end, strand, type, name, parent_id, attributes_json)`
- `ASSEMBLY_STATS` — `(assembly_id, n_contigs, total_length, n50, l50, gc_content, busco_complete, busco_fragmented, busco_missing)`
- `READ_SEGMENT_TABLE` — `(read_id, segment_index, segment_kind, start, end, score, barcode_id?, orientation)` — one row per located segment
- `READ_DEMUX_TABLE` — derived: `(read_id, sample_id, transcript_start, transcript_end, score, is_chimera)`
- `TRANSCRIPT_CLUSTER_TABLE` — `(cluster_id, representative_read_id, n_reads, consensus_sequence, predicted_orf, predicted_protein)`
- `FEATURE_QUANT` — `(feature_id FK, sample_id FK, count, tpm, coverage_mean, coverage_median)`

### `basecall/dorado.py`

```python
@dataclass
class RunHandle:
    pid: int
    output_path: Path
    stdout_log: Path
    stderr_log: Path
    pid_file: Path
    def wait(self, *, timeout=None) -> int: ...
    def poll(self) -> "RunStatus": ...
    def tail_progress(self) -> Iterator["DoradoProgressEvent"]: ...
    def cancel(self, *, sig=signal.SIGTERM) -> None: ...

class DoradoRunner:
    """Thin subprocess wrapper around the `dorado` binary.
    Resolved via constellation.thirdparty.find('dorado').
    All long-running methods return RunHandle; caller chooses block/follow/detach.
    """
    def basecaller(self, model: DoradoModel, pod5_paths: list[Path], output: Path,
                   *, modified_bases: list[str] = (), device: str = "cuda:0",
                   resume: bool = True, detach: bool = False) -> RunHandle: ...
    def duplex(self, ...) -> RunHandle: ...
    def aligner(self, reference: Path, bam: Path, output: Path) -> RunHandle: ...
    def demux(self, ...) -> RunHandle: ...   # ONT's built-in demux; we'll use ours instead
    def summary(self, ...) -> RunHandle: ...
    def fetch_model(self, model: DoradoModel) -> Path: ...   # `dorado download`
```

### `basecall/models.py`

```python
@dataclass(frozen=True)
class DoradoModel:
    family: Literal["sup", "hac", "fast"]
    version: str                          # "v5.0.0"
    chemistry: str = "dna_r10.4.1_e8.2"   # or "rna004_130bps"
    mods: tuple[str, ...] = ()            # ('5mC', '5hmC')

    @classmethod
    def parse(cls, shorthand: str) -> "DoradoModel":
        """Parse 'sup@v5.0.0+5mC,5hmC' → DoradoModel."""
        ...
    def model_name(self) -> str:
        """Render to the canonical Dorado model directory name."""
        ...
```

### `transcriptome/adapters.py`

```python
@dataclass(frozen=True)
class Adapter:
    name: str
    sequence: str
    kind: Literal["5p", "3p", "polyA", "umi"]

@dataclass(frozen=True)
class Barcode:
    name: str
    sequence: str
    kit: str

class Segment(Protocol): ...
class AdapterSlot(Segment): ...
class BarcodeSlot(Segment):
    barcodes: list[Barcode]
class PolyASlot(Segment):
    min_length: int
class UMISlot(Segment):
    length: int
class TranscriptSlot(Segment): ...

@dataclass(frozen=True)
class LibraryConstruct:
    """Composable description of a sequencing library's read structure.
    Examples ship for the lab's in-house SMARTer-derived chemistry plus
    standard ONT kits (PCS111, PCB111, RCB114).
    """
    name: str
    layout: tuple[Segment, ...]           # 5' to 3'
    allow_reverse_complement: bool = True
```

### `transcriptome/demux.py`

```python
def locate_segments(
    reads: pa.Table,                      # READ_TABLE
    construct: LibraryConstruct,
    *,
    polyA_anchor_kmer: str = "AAAA",
    polyA_max_edge_distance: int = 1,
    adapter_max_distance: int = 1,
    barcode_min_score: float = 0.7,
) -> pa.Table:                            # READ_SEGMENT_TABLE
    """User's 3-step algorithm: anchor on polyA, locate 3' adapter + barcode,
    locate 5' adapter, isolate transcript window. Per-stage edlib edit-distance
    constraints; chimeras emerge as multiple segment-runs per read.
    Lifted from NanoporeAnalysis as a clean rewrite — segmented approach beats
    Dorado's full-primer Smith-Waterman because high homopolymer error in oligo-dT
    distorts the score (empirical mean A≈26 against expected A=20).
    """
    ...

def resolve_demux(segments: pa.Table) -> pa.Table:
    """READ_SEGMENT_TABLE → READ_DEMUX_TABLE: assign sample_id per resolved
    transcript window; flag chimeras via `is_chimera`.
    """
    ...
```

### `thirdparty/dorado.py`

```python
DORADO_VERSION = "0.8.3"   # update as needed; only requirement is "supports latest released models"

register(ToolSpec(
    name="dorado",
    env_var="CONSTELLATION_DORADO_HOME",
    artifact=None,                          # binary on PATH (conda) OR HOME-relative
    path_bin="dorado",                      # primary discovery: shutil.which('dorado')
    install_script="scripts/install-dorado.sh",  # fallback for non-conda installs
    version_probe=_probe_version,           # `dorado --version`
))
```

`scripts/install-dorado.sh` follows the encyclopedia.sh pattern: hash-pinned tarball download from ONT's CDN, sha256 verify, atomic symlink swap into `third_party/dorado/<version>/`.

**License note:** Constellation itself is Apache 2.0. Dorado is distributed independently under the [Oxford Nanopore Technologies PLC. Public License Version 1.0](https://github.com/nanoporetech/dorado/blob/release-v1.4/LICENCE.txt) — an MPL-derived public license **restricted to Research Purposes only** (§1.10, §2.1, §2.3(b)). The two licenses don't conflict — wrapping a more-restrictively-licensed tool from an Apache 2.0 project is fine — but Dorado's terms cascade to anyone invoking the wrapper regardless of how Constellation itself is licensed. Key implications:

- **Permitted for the lab's use case.** §1.10 defines "Research Purposes" as "use for internal research… not intended for or directed towards commercial advantages or monetary compensation; provided, however, that monetary compensation does not include sponsored research [or] research funded by grants." Academic labs and grant-funded research labs are explicitly covered.
- **Wrapping via subprocess is fine** — that's "use" of Covered Software, permitted for Research Purposes.
- **We do NOT bundle Dorado binaries.** The install script downloads from ONT's CDN at install time, never redistributes. Downstream users get a download script, not a binary.
- **Anyone invoking the Dorado wrapper independently accepts ONT's Research Purposes restriction.** Constellation's Apache 2.0 grant doesn't extend to the wrapped tool. The `DoradoRunner` module docstring should call this out so a user landing on the file from go-to-definition understands the constraint without leaving their editor.
- **Phase 3 checklist** — before merging the Dorado wrapper, add a `LICENSE.md` (or similar) at `constellation/sequencing/basecall/` summarizing the constraint and linking to the upstream LICENCE.txt; mirror the gist in `DoradoRunner.__doc__`.

### Other thirdparty adapters

- `thirdparty/hifiasm.py` — `path_bin="hifiasm"`; install script for source build (no bioconda for latest versions of HiFiASM at all platforms).
- `thirdparty/minimap2.py` — `path_bin="minimap2"`; bioconda-declared in `environment.yml`; no install script needed.
- `thirdparty/samtools.py` — `path_bin="samtools"`; bioconda-declared.
- `thirdparty/ragtag.py` — `path_bin="ragtag.py"`; bioconda or install script.
- `thirdparty/busco.py` — `path_bin="busco"`; bioconda; needs lineage-data download separately (defer to Phase 9).
- `thirdparty/mmseqs2.py` — `path_bin="mmseqs"`; already in bioconda env.

## Dependencies — declared per-phase, not this session

The stubs ship with **no new dependencies**. Each implementation phase adds what it needs:

- **Phase 2** (readers): `pysam>=0.22` to `[sequencing]` extra, `pysam` to `environment.yml` bioconda block. (`pod5`, `edlib`, `parasail` already declared.)
- **Phase 3** (Dorado): `dorado` install script + `environment-cuda.yml` overlay.
- **Phase 4** (align): `mappy>=2.28` to `[sequencing]` extra, `minimap2` to `environment.yml` bioconda.
- **Phase 7+** (assembly + downstream): `samtools`, `hifiasm`, `ragtag`, `busco` to `environment.yml` bioconda as their phases land.

Every stub function for a not-yet-imported dep wraps the import inside the function body so module import doesn't fail (`def parse_bam(...): import pysam; ...`). This keeps `pytest tests/test_imports.py` green this session without polluting the env.

### `tests/test_imports.py` updates

Add import smoke for every new subpackage. The existing `constellation.sequencing` and `constellation.sequencing.readers` lines stay; add lines for `acquisitions`, `samples`, `schemas`, `quality`, `reference`, `assembly`, `annotation`, `alignments`, `io`, `basecall`, `signal`, `align`, `transcriptome`, `modifications`, `projects`. Also add `constellation.transcriptome_to_proteome` import.

## Implementation roadmap (future sessions, in order)

1. **Foundation — `core.io.schemas.ragged_trace_1d` + `acquisitions` + `samples` + `schemas/*` + `Reference` container.** First task: add `ragged_trace_1d` constructor + `ragged_to_tensors` bridge to `core.io.schemas` (modifies `core` — verify no regressions in existing `Trace1D`/`spectra_matrix_2d` consumers). Then sequencing schemas self-register with `core.io.schemas`. Containers ship PK/FK validation + ParquetDir round-trip. ~1–2 sessions.
2. **Reader/Writer Protocols + `quality/phred` + `readers/{fastx, sam_bam, pod5}`.** Each reader self-registers with the suffix/modality registry. Phred is the small prereq for FASTQ. `readers/sam_bam.py` produces `(Reads, Alignments)` tuple via the cross-tier adapter pattern. ~1–2 sessions.
3. **`thirdparty/dorado` + `scripts/install-dorado.sh` + `basecall/dorado.py` + `basecall/models.py`.** EULA review checkpoint. RunHandle + detach/follow modes. ~1 session.
4. **`align/{pairwise, map, locate}`.** edlib + parasail + mappy backends behind verb-named functions. Prereq for demux and polish. ~1 session.
5. **`transcriptome/{adapters, demux}`.** LibraryConstruct registry, segmented demux, READ_SEGMENT_TABLE → READ_DEMUX_TABLE. ~1 session.
6. **`transcriptome/{orf, cluster, consensus, network}`.** ORF wrapper over `core.sequence.nucleic`; mmseqs-style kmer + abundance-weighted consensus; read-similarity → `core.graph.Network`. ~1–2 sessions.
7. **`assembly/{hifiasm, stats}` + `thirdparty/hifiasm`.** ~1 session.
8. **`assembly/{polish, ragtag}` + `thirdparty/{minimap2, samtools, ragtag}`.** Polish loop uses Dorado + minimap2; samtools for BAM ops. ~1 session.
9. **`annotation/{busco, repeats, telomeres, transcripts}` + `thirdparty/busco`.** BUSCO lineage data management. ~1–2 sessions.
10. **`Quant` container + per-feature coverage/TPM.** ~1 session.
11. **`projects/{layout, manifest}`.** Project orchestrator on top of the pure-functional core. ~1 session.
12. **`constellation/transcriptome_to_proteome.py`.** Cross-modality bridge; predicted proteins from transcriptome assembly feed into `massspec.library`. ~1 session.
13. **`signal/*` + `modifications/*`.** Squiggle normalization + segmentation; physics-based basecaller-model ABC; modified-base calling from MM/ML BAM tags or signal. Research area. ~ongoing.

## Critical files to read before implementing each phase

- Phase 1 templates: [massspec/acquisitions.py](../../projects/constellation/constellation/massspec/acquisitions.py), [massspec/library/library.py](../../projects/constellation/constellation/massspec/library/library.py), [massspec/library/io.py](../../projects/constellation/constellation/massspec/library/io.py), [core/io/schemas.py](../../projects/constellation/constellation/core/io/schemas.py).
- Phase 2 templates: [core/io/readers.py](../../projects/constellation/constellation/core/io/readers.py), [core/io/bundle.py](../../projects/constellation/constellation/core/io/bundle.py), [chromatography/readers/agilent_dx.py](../../projects/constellation/constellation/chromatography/readers/) (when shipped) for the bench-instrument reader pattern.
- Phase 3 templates: [thirdparty/encyclopedia.py](../../projects/constellation/constellation/thirdparty/encyclopedia.py), [thirdparty/registry.py](../../projects/constellation/constellation/thirdparty/registry.py), [scripts/install-encyclopedia.sh](../../projects/constellation/scripts/install-encyclopedia.sh).
- Phase 5 prior art: NanoporeAnalysis at `~/projects/NanoporeAnalysis` (clean rewrite, no vendored source — read the existing edlib-based demux as reference, port the algorithm idiomatically).

## Verification

This session ships stubs only — verification is correspondingly limited:

1. **Import smoke test** — `pytest tests/test_imports.py` passes with every new subpackage imported. Already wired for `constellation.sequencing` and `constellation.sequencing.readers`; extend to cover the full new tree.
2. **`ruff check .`** — passes; no unused-import warnings.
3. **`constellation doctor`** — runs successfully, lists Dorado / HiFiASM / minimap2 / samtools / RagTag / BUSCO as either `ok` (where conda has installed them) or `not found (set $CONSTELLATION_<TOOL>_HOME or run scripts/install-<tool>.sh)`. `encyclopedia` continues to resolve as before.
4. **Schema self-registration** — `from constellation.core.io.schemas import REGISTRY; REGISTRY["READ_TABLE"]` returns the registered schema.
5. **Stub functions raise `NotImplementedError`** — every stub explicitly raises with a phase-tagged message (e.g. `"Pending Phase 5 — see plan in-our-development-of-fuzzy-quilt.md"`), so accidentally calling an unimplemented function fails loudly rather than silently no-oping.

End-to-end pipeline tests land per-phase as each module ships.
