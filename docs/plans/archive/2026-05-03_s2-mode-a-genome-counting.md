# S2 Mode A — genome-guided alignment + filter + gene-level counting

## Context

S1 (`c77d5c9`) shipped the reference data classes (`GenomeReference`, `Annotation`, `TranscriptReference`, `GeneticTools`), hand-rolled FASTA/GFF3/BED parsers, and `constellation reference {import,fetch,summary,validate}` CLI. The transcriptome demux pipeline emits naive-ORF protein clusters that cover only ~1–2% of demuxed reads — most of the biology is invisible without a real reference-driven counter.

This session promotes S1's reference layer into actual gene-level abundance estimates. Reads that have already been demuxed (`READ_DEMUX_TABLE` from S1) get aligned to a `GenomeReference`, filtered on alignment quality, and counted per `(gene_id, sample_id)`.

**Explicitly out of scope this session** — locked in by the user:
- Mode B (de novo minimizer-graph clustering). Defer to a later session; the prior plan's Section E is shelved.
- Transcriptome-mode mapping + isoform-resolving EM quantifier.
- mmseqs2 protein-search annotation.
- HiFiASM wrapper for de novo assembly (separate session).
- "Learn exons from per-base coverage" — future ambition once we have piled-up reads to look at; not S2.

## Strategic decisions locked

Carried from `for-our-next-session-spliced-tarjan.md` and refined in this session's Plan-agent review:

1. **Filtering is a discrete step between alignment and counting.** `Alignments.filter(...) -> Alignments` returns a new container with per-stage row counts in `metadata_extras['filter_steps']`. The counter never makes filtering decisions.
2. **Filter defaults are starting points.** `min_aligned_fraction=0.85` (tighter than ENCODE's published 0.7 floor — full-length cDNA from clean libraries should clear 0.85; cite ENCODE 0.7 as published baseline in the docstring). Per-call kwargs override. CLI exposes all knobs.
3. **Two-layer minimap2 architecture: generic runner + use-case orchestrators.** The low-level `minimap2_run` subprocess wrapper accepts an arbitrary `args: tuple[str, ...]` — knows nothing about splice, presets, or use case. It is the reusable building block. `map_to_genome` is a thin use-case orchestrator on top that hard-codes the splice flag set (`-ax splice -uf --cs=long --secondary=no`) for forward-stranded full-length cDNA → genome (matches `cdna_wilburn_v1`); `extra_minimap2_args` passes through. Future `map_assembly` (asm5/10/20), `map_dna_to_genome` (genomic DNA `-ax map-ont`), or PAF-output verbs all compose `minimap2_run` directly with their own arg lists — they do not subclass or wrap `map_to_genome`.
4. **No EM. Gene-level overlap counting is deterministic** — assign each filtered primary alignment to the highest-overlap gene; no-qualifying-overlap reads drop to QC stats.
5. **`Alignments.validate_against(genome)` checks ref_name ⊆ genome.contigs.name.** Annotation FK closure into the same genome is verified separately at S1's `Annotation.validate_against(genome)` boundary.

5a. **CLI verb is `constellation transcriptome align`**, not `constellation reference align`. The operation is a transcriptomics analysis (gene-level abundance estimation from RNA reads) that consumes a reference; it is not a reference-management op like `import`/`fetch`/`summary`/`validate`. Future Mode B (de novo clustering) shares the same verb — mode dispatched on presence of `--reference`. Without `--reference`: Mode B (deferred this session, errors with a clear "not yet implemented" message). With `--reference`: Mode A.

5b. **Primary input is `--demux-dir <s1-output>`**, not separate `--reads` + `--read-demux`. The S1 demux output already contains `read_demux/` (for read_id → sample_id) + `manifest.json` (input BAM paths + acquisition map). Chaining `transcriptome demultiplex` → `transcriptome align` becomes literally `--output-dir s1_out/` → `--demux-dir s1_out/`. The align step pulls original BAM paths from the demux manifest's `input_files` field and feeds them into minimap2.

5c. **`--reference <ref-root>` resolves both genome + annotation from a single directory** (the layout produced by `constellation reference import`/`fetch`: `<ref-root>/genome/` + `<ref-root>/annotation/`). Single argument; no separate `--genome` / `--annotation` flags. Users with non-standard layouts symlink.
6. **`map_reads` stub is deleted.** The two-backend mappy/minimap2 dispatcher in [align/map.py](constellation/sequencing/align/map.py) goes away. Mappy is rejected (user-confirmed slower); the only function in the file becomes `map_to_genome`. `map_assembly` keeps a TODO comment flagging the same future cleanup.
7. **BAM → ALIGNMENT_TABLE decode lives in `readers/sam_bam.py` as a free function, NOT on `BamReader`.** The existing `BamReader.read()` produces READ_TABLE (one row per primary read, skipping secondaries). The new `read_bam_alignments` is a separate decode pass that emits ALIGNMENT_TABLE (one row per alignment record, including secondaries/supplementaries) — different cardinality, different artefact. `io/sam_bam.py::read_bam` orchestrates the cross-tier wrap into `Alignments`.
8. **CIGAR parsing for `aligned_fraction` is per-row Python, not cached on the table.** `__post_init__` stays a pure cast+validate (matches `GenomeReference` precedent). A private `_aligned_fraction_array(table)` helper walks `cigar_string` row-by-row only when `Alignments.filter(min_aligned_fraction=...)` is invoked. ~3 s on 5M rows is acceptable for one-shot pipeline runs; vectorisation via `cigar_to_ops` is a follow-up if profiling demands it.
9. **Genome FASTA is always materialised from the `GenomeReference` ParquetDir.** No "FASTA-or-ref-dir" dual path. The CLI resolves `<ref-root>/genome/` from the single `--reference` flag (per decision 5c). `output_dir/genome.fa` + `output_dir/genome.mmi` cached + reused on mtime check.
10. **Parallelism budget goes on a fused decode+filter+overlap worker.** N-way fan-out via `_bam_alignment_chunks` + `run_batched`; each worker decodes its BAM chunk, applies the filter predicates, runs the gene-overlap join against a broadcast gene set, and emits per-chunk shards. Mirrors S1's `_fused_chunk_worker` pattern (decode+demux+ORF+partial-quant in one worker pass).
11. **Pipeline NEVER materialises the full `Alignments` container in memory.** Realistic transcriptomics scale is 30–200M alignments per run (mouse / Arabidopsis run = ~200M; lab Pichia run = ~30M). At ~200B/row that's 6–40 GB for the alignment table alone — too large to round-trip through a single in-memory Arrow table in a resolve stage. Architecture: alignments live as partitioned parquet on disk; filter + overlap-join run per-chunk inside the fused worker; the resolve stage hash-joins compact `(read_id, gene_id)` shards against `read_demux/` and aggregates to bounded-tiny FEATURE_QUANT. The in-memory `Alignments` container + `Alignments.filter` + `count_reads_per_gene` exist as sugar for **tests and Jupyter exploration only** — the pipeline does not call them.
12. **Default tag-table materialisation is empty.** `read_bam_alignments(path, *, tags_to_keep=())` extracts only promoted columns (`nm_tag`, `as_tag`, `read_group`); long-tail tags (`MD`, `SA`, `MM`/`ML`) are emitted into `ALIGNMENT_TAG_TABLE` only when explicitly listed. v1 pipeline does not request them.

## Deliverables

### A. `Alignments` container completion

[constellation/sequencing/alignments/alignments.py](constellation/sequencing/alignments/alignments.py) — replace the stub body. Pattern matches `GenomeReference`/`Annotation` exactly:

- `__post_init__`: cast `alignments` → `ALIGNMENT_TABLE`, `tags` → `ALIGNMENT_TAG_TABLE` via `cast_to_schema`, then `validate()`. No CIGAR derived columns at this stage.
- `validate()`: PK uniqueness on `alignment_id`; FK closure (`tags.alignment_id ⊆ alignments.alignment_id`); when `acquisitions` is set, `alignments.acquisition_id ⊆ acquisitions.acquisitions.acquisition_id`.
- `validate_against(genome: GenomeReference)`: distinct `ref_name` ⊆ `genome.contigs.name`.
- `n_alignments` property; `primary()` (filter `is_secondary == False AND is_supplementary == False`).
- `filter(*, min_length=None, min_aligned_fraction=None, min_mapq=None, primary_only=False, sense_only=False, allow_antisense=False) -> Alignments` — vectorised `pa.compute` predicates. The aligned-fraction branch only fires when `min_aligned_fraction is not None`. Per-stage row counts accumulate in `new_metadata['filter_steps']` as a list of `{stage, kept, dropped}` dicts so the audit surfaces cleanly in `summary.json` later. Returns a new `Alignments` instance (frozen dataclass — same construction pattern).
- Private helper `_aligned_fraction_array(table) -> pa.Array(float32)`: walks `cigar_string` rows in Python, sums M/=/X for aligned_bp and M/=/X/I/S/H for read_length per SAM v1 spec §1.4.6, returns `aligned_bp / read_length` (or null on degenerate rows). Hand-rolled state machine — ~30 lines, no regex. One-line citation to SAMv1.pdf in the docstring.

### B. ALIGNMENT_TABLE decoder + ParquetDir round-trip

**`read_bam_alignments`** — new free function in [constellation/sequencing/readers/sam_bam.py](constellation/sequencing/readers/sam_bam.py):

```python
def read_bam_alignments(
    path: Path,
    *,
    contig_name_to_id: dict[str, int],
    acquisition_id: int = 0,
    tags_to_keep: tuple[str, ...] = (),
    threads: int = 1,
) -> tuple[pa.Table, pa.Table]:
    """Decode all alignment records (incl. secondary/supplementary)
    into (ALIGNMENT_TABLE, ALIGNMENT_TAG_TABLE)."""
```

- Mirrors `_bam_record_to_row` (line 245-314 of `readers/sam_bam.py`) but does NOT skip secondaries (the existing READ_TABLE iterator skips them at `_iter_records`; the alignment iterator must not).
- Promoted-column extraction: `nm_tag` from `NM:i`, `as_tag` from `AS:i`, `read_group` from `RG:Z`.
- `tags_to_keep` allowlist gates rows into `ALIGNMENT_TAG_TABLE` long-format `(alignment_id, tag, type, value)` tuples. Empty default = no tag-table rows.
- Companion chunk-worker `read_bam_alignments_chunk(path, *, vo_start, n_records, ...)` for parallel ingest, mirroring `read_bam_chunk` (line 476-506). NOTE: `n_records` (not `n_primary` — we count all records).
- Companion chunker `_bam_alignment_chunks(path, *, chunk_size=100_000, threads=1)`: like `_bam_record_chunks` but counts every record toward the chunk budget, not just primaries.

**Cross-tier orchestrator** — [constellation/sequencing/io/sam_bam.py](constellation/sequencing/io/sam_bam.py):

```python
def read_bam(
    path: Path,
    *,
    genome: GenomeReference,
    acquisition_id: int = 0,
    tags_to_keep: tuple[str, ...] = (),
    threads: int = 1,
) -> Alignments:
```

In-memory single-shot decoder for **tests and small Jupyter use only** — drops the `tuple[Reads, Alignments]` shape from the stub (`Reads` is itself deferred and the dual-output is wrong). Builds `contig_name_to_id` from `genome.contigs`, calls `read_bam_alignments`, wraps in `Alignments`, validates against `genome`. The existing `_BamAlignmentsReader` adapter resolves cleanly off this. **Not used by the pipeline** — at 30–200M alignments this would OOM.

**Fused worker for the pipeline** — [constellation/sequencing/quant/genome_count.py](constellation/sequencing/quant/genome_count.py) hosts the worker function (kept colocated with the kernels it calls; mirrors S1's `transcriptome.stages._fused_chunk_worker` placement):

```python
def fused_decode_filter_overlap_worker(
    chunk_spec: tuple[str, int, int],   # (bam_path, vo_start, n_records)
    *,
    contig_name_to_id: dict[str, int],
    gene_set_bytes: bytes,              # serialised (Arrow IPC) gene_set, broadcast to workers
    filter_kwargs: dict[str, Any],
    overlap_kwargs: dict[str, Any],
    acquisition_id: int = 0,
) -> dict[str, pa.Table]:
    """Per-chunk fused worker. Pickleable (top-level). Output keys:
    'alignments' (filtered), 'alignment_tags' (filtered), 'gene_assignments'
    ((alignment_id, read_id, gene_id, overlap_fraction)), 'stats'."""
```

Internally:
1. `read_bam_alignments_chunk(bam_path, vo_start=..., n_records=...)` → `(chunk_alignments, chunk_tags)`.
2. `apply_filter_predicates(chunk_alignments, **filter_kwargs)` → filtered + audit.
3. Subset `chunk_tags` to surviving `alignment_id`s.
4. `compute_gene_overlap(filtered, gene_set, **overlap_kwargs)` → assignments table.
5. Stats dict: `{decoded, after_filter, after_overlap}` for the manifest.

`run_batched` in the CLI handler pickles this function across N workers; each worker writes its own shards to `alignments/`, `alignment_tags/`, `gene_assignments/` partition dirs. Worker-local `alignment_id = (worker_idx << 32) | local_idx` int64 packing — deterministic, debuggable, no second-pass renumber.

**Gene set broadcast.** `annotation.features_of_type('gene')` is bounded-small (~5–50k rows). Serialise once at the parent via `pa.ipc.serialize`, pass the bytes through `worker_kwargs` so each worker deserialises locally. Avoids per-chunk recomputation and stays well under the IPC pickle size limit.

**ParquetDir round-trip** — [constellation/sequencing/alignments/io.py](constellation/sequencing/alignments/io.py):

Ship `ParquetDirReader.read` / `ParquetDirWriter.write` following the `GenomeReference.io` precedent: `alignments.parquet` + `tags.parquet` + `manifest.json` (acquisition info, format version). `save_alignments` / `load_alignments` dispatchers route by extension or explicit `format=` kwarg. **Note:** `load_alignments` materialises the full table into memory — appropriate for ParquetDir-as-cache use cases (small alignment subsets, test fixtures), NOT for opening the pipeline's `alignments/` partition at full scale. For pipeline-scale access, callers open `pa.dataset.dataset(alignments/)` directly.

### C. minimap2 wrapper — generic runner + genome-mode orchestrator

Two-layer split. The generic runner stays use-case-agnostic so future `map_assembly`, `map_dna_to_genome`, and PAF-output verbs all compose it with their own arg lists.

**Layer 1: generic minimap2 subprocess wrapper.** New file [constellation/sequencing/align/minimap2.py](constellation/sequencing/align/minimap2.py):

```python
def minimap2_run(
    target: Path,                    # FASTA or pre-built .mmi
    queries: Sequence[Path],         # FASTQ/FASTA/BAM input files
    *,
    output_path: Path,               # output SAM/BAM/PAF
    args: tuple[str, ...] = (),      # full minimap2 arg list (caller-controlled, no defaults)
    threads: int = 8,
    progress_cb: ProgressCallback | None = None,
) -> Path:
    """Run minimap2 (subprocess) with caller-supplied args. Returns
    output_path. Knows nothing about splice/preset/use-case — that's
    the orchestrator layer's job."""

def minimap2_build_index(
    fasta: Path,
    mmi_path: Path,
    *,
    threads: int = 8,
    extra_args: tuple[str, ...] = (),
) -> Path:
    """Build a .mmi index via `minimap2 -d`. Returns mmi_path. Skips
    rebuild if mmi exists and is newer than fasta."""
```

Notes:
- `minimap2_run` always inserts `-t {threads}` at the front of the constructed command line. The caller's `args` tuple is appended verbatim — no defaults, no overrides.
- Output format inferred from `output_path` extension (`.sam` / `.bam` / `.paf`); when `.bam`, the runner pipes minimap2's SAM stdout through `samtools view -b` (no sort). Sort/index is a separate concern.
- `progress_cb` emits `stage_start` at subprocess launch + `stage_done` on exit. Per-line minimap2 stderr parsing for fine-grained progress is out of scope.
- Locates `minimap2` via `constellation.thirdparty.minimap2.try_find` ToolSpec; raises a clear error with a bioconda install hint if missing.

**Layer 2: genome-mode use-case orchestrator.** [constellation/sequencing/align/map.py](constellation/sequencing/align/map.py) — DELETE the `map_reads` stub. Add `map_assembly` TODO comment. Ship:

```python
def map_to_genome(
    reads_paths: Sequence[Path],
    genome: GenomeReference,
    *,
    output_dir: Path,
    threads: int = 8,
    extra_minimap2_args: tuple[str, ...] = (),
    progress_cb: ProgressCallback | None = None,
) -> Path:
    """Splice-aware full-length cDNA → genome alignment via minimap2.
    Returns the path to the sorted, indexed output BAM."""
```

Pipeline:
1. Materialise FASTA cache: `output_dir/genome.fa` from `genome.contigs` + `genome.sequence_of(contig_id)`. Skip rebuild if file already exists and `genome.contigs.num_rows` matches a stamped `output_dir/.genome_meta.json`.
2. Build `.mmi` index via `minimap2_build_index(fasta, output_dir/'genome.mmi', threads=threads)`.
3. Construct genome-mode arg list locally — this is the only place these flags live:
   ```python
   genome_args = ('-ax', 'splice', '-uf', '--cs=long', '--secondary=no') + extra_minimap2_args
   ```
4. For each input file, call `minimap2_run(target=mmi, queries=[reads_path], output_path=output_dir/bam/aligned.{idx}.bam, args=genome_args, threads=threads)`.
5. Sort + index via `samtools` subprocess (`samtools sort -@ {threads} -o aligned.bam <unsorted>` then `samtools index`). Multi-input: `samtools cat | samtools sort` — sorts maintain coordinate order across the merge. Inline subprocess calls in `map_to_genome` for now; if a second samtools-sort caller appears, lift into a `samtools_sort_index` helper at that point.
6. Emit progress events through `progress_cb` at the per-input granularity.

Returns the final sorted, indexed BAM path; the CLI handler then calls `read_bam(bam_path, genome=...)` to lift to `Alignments`.

### D. Gene-level overlap kernels + count function

New package [constellation/sequencing/quant/](constellation/sequencing/quant/) (sibling to `sequencing/transcriptome/`). Three pieces, one entry point each, factored so the per-chunk fused worker and the resolve-stage aggregator share implementations.

**Shared kernels** — pure pa.Table-in-Table-out, no I/O. [constellation/sequencing/quant/_kernels.py](constellation/sequencing/quant/_kernels.py):

```python
def apply_filter_predicates(
    alignments_table: pa.Table,
    *,
    min_length: int | None = None,
    min_aligned_fraction: float | None = None,
    min_mapq: int | None = None,
    primary_only: bool = False,
) -> tuple[pa.Table, list[dict[str, int]]]:
    """Vectorised pa.compute predicates; returns (filtered, audit).
    Computes aligned_fraction lazily via _aligned_fraction_array
    only when min_aligned_fraction is non-None."""

def compute_gene_overlap(
    alignments_chunk: pa.Table,        # filtered alignments
    gene_set: pa.Table,                # annotation.features_of_type('gene'), sorted (contig_id, start)
    *,
    min_overlap_fraction: float = 0.5,
    allow_antisense: bool = False,
) -> pa.Table:
    """Sort+searchsorted sweep-join. Per-chunk: sort alignments by
    (contig_id, ref_start), per-contig partition use np.searchsorted
    against gene_set's start array to find candidate-gene window,
    compute overlap_bp = max(0, min(end_a, end_g) - max(start_a, start_g)),
    pick best overlap per alignment, drop below threshold.
    Returns: (alignment_id, read_id, gene_id, overlap_fraction)."""
```

`Alignments.filter` calls `apply_filter_predicates`. The fused decode worker (Deliverable B) calls both kernels per-chunk.

**Count function** — ONE entry point. [constellation/sequencing/quant/genome_count.py](constellation/sequencing/quant/genome_count.py):

```python
def count_reads_per_gene(
    gene_assignments: pa.Table,        # (alignment_id, read_id, gene_id, overlap_fraction)
    read_demux: pa.Table,              # (read_id, sample_id, ...)
    samples: Samples,
    *,
    engine: str = "constellation_overlap",
) -> tuple[pa.Table, dict[str, int]]:
    """Hash-join gene_assignments × read_demux on read_id, group_by-sum
    to FEATURE_QUANT. Both inputs are pa.Tables — caller materialises
    partitioned datasets via .to_table() before invocation."""
```

Implementation:
1. Hash-join `gene_assignments` against `read_demux[read_id, sample_id]` on `read_id`. Drop rows where `sample_id` is null.
2. `group_by([gene_id, sample_id]).aggregate([(read_id, 'count')])` → count column.
3. Project to `FEATURE_QUANT` (`feature_id=gene_id`, `feature_origin='gene_id'`, `engine`, count→float64; TPM/CPM/coverage_* null this session) via the column-mapping pattern from [transcriptome/quant.py::to_feature_quant](constellation/sequencing/transcriptome/quant.py).
4. Stats dict: `{'gene_assignments_in', 'reads_with_sample', 'reads_without_sample', 'unique_(gene,sample)_pairs', 'total_count'}` for the manifest.

**How the two scales hit the same function:**
- Tests / Jupyter — construct small `gene_assignments` + `read_demux` Arrow tables directly (or run `compute_gene_overlap(small_alignments, gene_set)` to get assignments first); pass to `count_reads_per_gene`.
- Pipeline — CLI handler does `gene_assignments = pa.dataset.dataset(output_dir/'gene_assignments').to_table()` and `read_demux = pa.dataset.dataset(demux_dir/'read_demux').to_table()` at the resolve step, then passes the materialised tables. Memory budget: ~3 GB (assignments) + ~10 GB (read_demux) at 200M-read scale; fits comfortably on workstation RAM, trivially on cluster nodes. If we ever cross 1B reads, an Arrow-`Dataset.join`-based streaming variant lifts the materialisation — single-function refactor at that point, not a new public API.

The "streaming pipeline rule" (decision #11) still holds: the **alignment table** is never materialised whole. `gene_assignments` and `read_demux` ARE materialised whole at resolve time — that's deliberate and bounded.

### E. CLI: `constellation transcriptome align`

[constellation/cli/__main__.py](constellation/cli/__main__.py) — promote the existing `transcriptome cluster` placeholder slot logic; add an `align` verb to the `transcriptome` subparser. Mirror `_cmd_transcriptome_demultiplex` (line 399-477) for layout.

```
constellation transcriptome align \
    --demux-dir <s1-output-dir> \
    --reference <ref-root> \
    --samples <tsv> \
    --output-dir <dir> \
    [--threads 8] \
    [--min-length 300] [--min-aligned-fraction 0.85] [--min-mapq 0] \
    [--allow-antisense] \
    [--min-overlap-fraction 0.5] \
    [--resume] [--progress]
```

Mode dispatch: presence of `--reference` selects Mode A. Absent → error: `"Mode B (de novo clustering) not yet implemented in this release; pass --reference <ref-root> for reference-guided gene counting."` Future Mode B work plugs into the same handler under the absent-reference branch.

Handler flow (`_cmd_transcriptome_align`) — three stages, fused middle:

1. Lazy-import heavy modules.
2. Validate `--demux-dir` exists and contains `manifest.json` + `read_demux/` + `_SUCCESS`. Read `demux_manifest = json.load(<demux-dir>/manifest.json)`.
3. Mode dispatch: if `--reference` is None, error with the Mode B message above and exit 2.
4. Resolve genome + annotation paths from `--reference`: `genome_dir = <ref-root>/genome`, `annotation_dir = <ref-root>/annotation`. Load `genome = load_genome_reference(genome_dir)`, `annotation = load_annotation(annotation_dir)`. Call `annotation.validate_against(genome)`.
5. Resolve input BAM paths: `reads_paths = [Path(p) for p in demux_manifest["input_files"]]`. Verify each exists; error with a clear message if any have moved (suggest re-running demux or symlinking the BAMs back).
6. Parse `--samples` TSV via existing `_parse_samples_tsv` helper. (Documented future work: persist `samples/` ParquetDir under demux output so `--samples` becomes optional. For v1 the user re-passes the same TSV they used at demux time.)
7. Refuse to start if `output_dir/_SUCCESS` exists unless `--resume` was passed.
8. **Stage 1 (`align`):** `map_to_genome(reads_paths, genome, output_dir=output_dir, threads=args.threads, progress_cb=cb)` → sorted indexed BAM at `output_dir/bam/aligned.bam`. `_SUCCESS` marker at `output_dir/bam/_SUCCESS`.
9. **Stage 2 (`decode_filter_overlap`):** N-way parallel fused worker via `run_batched` over `_bam_alignment_chunks(bam_path)`. Each worker calls `fused_decode_filter_overlap_worker` (Deliverable B) and writes three shards: `alignments/part-NNNNN.parquet` (filtered), `alignment_tags/part-NNNNN.parquet`, `gene_assignments/part-NNNNN.parquet`. Per-stage row counts (`decoded`, `after_filter`, `after_overlap`) aggregate at the parent for the manifest. `_SUCCESS` markers per shard dir.
10. **Stage 3 (`count`):** Materialise `gene_assignments = pa.dataset.dataset(output_dir/'gene_assignments').to_table()` + `read_demux = pa.dataset.dataset(<demux-dir>/'read_demux').to_table()`, call `count_reads_per_gene(gene_assignments, read_demux, samples)` → FEATURE_QUANT table + stats dict. Write `output_dir/feature_quant/feature_origin=gene_id/part-00000.parquet` + the bounded-tiny aggregate `output_dir/feature_quant.parquet`. At ~50k genes × ~24 samples the aggregate is ~1.2M rows — single-file is fine.
11. Write `output_dir/manifest.json` with sections `inputs` (demux_dir, reference root, BAM paths from demux manifest) / `parameters` (filter thresholds, threads) / `stages` (per-stage row counts: `decoded`, `after_filter`, `after_overlap`, `reads_with_sample`, `reads_without_sample`) / `outputs`. Single top-level json.
12. Final `_SUCCESS` marker at `output_dir/_SUCCESS`.

Output layout:

```
<output-dir>/
  bam/
    aligned.bam
    aligned.bam.bai
    _SUCCESS
  alignments/
    part-NNNNN.parquet              # post-filter, partitioned (NEVER materialised whole)
    _SUCCESS
  alignment_tags/
    part-NNNNN.parquet
    _SUCCESS
  gene_assignments/
    part-NNNNN.parquet              # (alignment_id, read_id, gene_id, overlap_fraction)
    _SUCCESS
  feature_quant/
    feature_origin=gene_id/
      part-00000.parquet
    _SUCCESS
  feature_quant.parquet             # bounded-tiny aggregate (~1.2M rows worst case)
  genome.fa                         # cached materialisation
  genome.mmi
  .genome_meta.json
  manifest.json
  _SUCCESS
```

### F. Cross-cutting

- [constellation/sequencing/quant/__init__.py](constellation/sequencing/quant/__init__.py) — new package; exports `count_reads_per_gene`, the kernels (`apply_filter_predicates`, `compute_gene_overlap`), and `fused_decode_filter_overlap_worker`. Future EM quantifier and any genome-aligned counter siblings live here. Distinct from `sequencing.transcriptome.quant`, which stays the demux-pipeline-internal protein-quant builder.
- [CLAUDE.md](CLAUDE.md) — module index entry for `constellation.sequencing` updated to reflect S2 Mode A deliverables; new `sequencing.quant` package callout; document the **streaming pipeline rule**: realistic transcriptomics scale is 30–200M alignments per run, so the pipeline never materialises a full `Alignments` container — fused decode+filter+overlap worker writes partitioned shards, resolve stage hash-joins compact `gene_assignments/` against `read_demux/`. The in-memory `Alignments` / `Alignments.filter` / `count_reads_per_gene` API exists for tests + Jupyter only. Document the **CLI namespace rule**: gene-counting is a transcriptomics analysis, lives at `constellation transcriptome align` (not `reference align`); mode dispatched on `--reference`; Mode B (de novo clustering) shares the same verb when it lands. Clarify Mode A is shipped, Mode B deferred (depends on pre-assigned gene groupings, scope-coupled with the future "learn exons from coverage" ambition). Reaffirm the locked-in scope: subprocess-only minimap2; no mappy. Document the two-layer minimap2 split (generic `minimap2_run` + use-case orchestrators). Document that `transcriptome align` chains off `transcriptome demultiplex` via `--demux-dir` (no separate `--reads`/`--read-demux` flags).
- **Memory budget for the resolve stage:** `gene_assignments` at 200M assignments × 16 B/row ≈ 3 GB; `read_demux` at 200M reads × ~50 B/row ≈ 10 GB; the hash-join builds the smaller side (`gene_assignments`) and streams the larger. Fits comfortably on workstation RAM, comfortable on cluster nodes. Document this in the CLI handler docstring so future maintainers know the boundary.

## Critical files

| File | Action |
|---|---|
| [constellation/sequencing/alignments/alignments.py](constellation/sequencing/alignments/alignments.py) | Implement `__post_init__` / `validate` / `validate_against` / `n_alignments` / `primary` / `filter` + `_aligned_fraction_array` helper. |
| [constellation/sequencing/alignments/io.py](constellation/sequencing/alignments/io.py) | Ship `ParquetDirReader/Writer` + `save/load_alignments` dispatchers. |
| [constellation/sequencing/readers/sam_bam.py](constellation/sequencing/readers/sam_bam.py) | Add `read_bam_alignments` + `_bam_alignment_chunks` + `read_bam_alignments_chunk`. |
| [constellation/sequencing/io/sam_bam.py](constellation/sequencing/io/sam_bam.py) | Implement `read_bam(path, *, genome, ...) -> Alignments` (drop the dual-output `(Reads, Alignments)` stub shape). |
| [constellation/sequencing/align/minimap2.py](constellation/sequencing/align/minimap2.py) | New: generic `minimap2_run` + `minimap2_build_index` subprocess wrappers (use-case-agnostic, reusable for future verbs). |
| [constellation/sequencing/align/map.py](constellation/sequencing/align/map.py) | Delete `map_reads` stub. Ship `map_to_genome` as a thin orchestrator that composes `minimap2_run` + samtools sort/index. Leave TODO on `map_assembly`. |
| [constellation/sequencing/quant/__init__.py](constellation/sequencing/quant/__init__.py) | New package. |
| [constellation/sequencing/quant/_kernels.py](constellation/sequencing/quant/_kernels.py) | New: `apply_filter_predicates` + `compute_gene_overlap` shared kernels. |
| [constellation/sequencing/quant/genome_count.py](constellation/sequencing/quant/genome_count.py) | New: `count_reads_per_gene` (single entry, pa.Table inputs) + `fused_decode_filter_overlap_worker`. |
| [constellation/cli/__main__.py](constellation/cli/__main__.py) | Add `transcriptome align` subcommand + `_cmd_transcriptome_align` handler (mode-dispatched on `--reference` presence; Mode A only this session, Mode B errors with not-yet-implemented). |
| [CLAUDE.md](CLAUDE.md) | Module-index update; document S2 Mode A scope + scale ceiling + Mode B deferred. |
| [tests/test_alignments_container.py](tests/test_alignments_container.py) | New: hand-crafted 5–10 alignments, PK/FK validation, filter step semantics, ParquetDir round-trip. |
| [tests/test_io_sam_bam.py](tests/test_io_sam_bam.py) | New: BAM → Alignments round-trip via small fixture BAM. |
| [tests/test_align_map.py](tests/test_align_map.py) | New: `pytest.skip` if minimap2 missing; tiny synthetic genome FASTA + simulator-derived FASTQ → BAM. |
| [tests/test_quant_genome_count.py](tests/test_quant_genome_count.py) | New: hand-crafted ~3 contigs × 5 genes × 30 alignments fixture (alignments overlapping two genes, antisense, no-overlap, gene-boundary cases), known-counts assertions. |

## Existing utilities to reuse

- [GenomeReference.sequence_of](constellation/sequencing/reference/reference.py) + `genome.contigs` — for the FASTA materialisation cache and for building `contig_name_to_id` lookups.
- [Annotation.features_of_type](constellation/sequencing/annotation/annotation.py), `features_on(contig_id)` — gene-overlap candidate set.
- [Samples](constellation/sequencing/samples.py), [_parse_samples_tsv + _resolve_input_paths](constellation/cli/__main__.py) — CLI argument resolution.
- [READ_DEMUX_TABLE](constellation/sequencing/schemas/transcriptome.py) — opened as `pa.dataset.dataset(read_demux/)` from S1 demux output for `read_id → sample_id` resolution.
- [parallel.run_batched](constellation/sequencing/parallel.py) — N-way fan-out for the BAM-decode stage. Worker functions stay top-level for `ProcessPoolExecutor` pickling.
- [progress.{ProgressCallback,StreamProgress,NullProgress}](constellation/sequencing/progress.py) — `--progress` binding.
- [readers/sam_bam._bam_record_chunks, read_bam_chunk, _bam_record_to_row](constellation/sequencing/readers/sam_bam.py) — pattern + helpers (`_bam_record_to_row` is the row-decode reference; the new alignment-decode helper mirrors its tag-extraction shape).
- [transcriptome/quant.to_feature_quant](constellation/sequencing/transcriptome/quant.py) — column-mapping pattern for FEATURE_QUANT projection.
- [thirdparty.minimap2.try_find](constellation/thirdparty/minimap2.py) + [thirdparty.samtools.try_find](constellation/thirdparty/samtools.py) — already-registered ToolSpecs with version probes.
- [core.io.schemas.cast_to_schema](constellation/core/io/schemas.py) — for the `__post_init__` cast.

## Verification

```bash
# Unit tests
pytest tests/test_alignments_container.py tests/test_quant_genome_count.py \
       tests/test_io_sam_bam.py
pytest tests/test_align_map.py     # skips if minimap2 missing

# Existing S1 regression — must stay green
pytest tests/test_sequencing_transcriptome_demultiplex_parity.py -q

# Lint
ruff check constellation/ tests/

# End-to-end on a hand-crafted fixture (built inside tests/conftest.py at runtime):
#   tests/data/sim/genome.fa + annotation.gff3 + reads.bam + samples.tsv
# (1) Build the reference layer
constellation reference import \
    --fasta tests/data/sim/genome.fa \
    --gff3 tests/data/sim/annotation.gff3 \
    --output-dir /tmp/sim_ref
# (2) Demultiplex (S1, already shipped) — produces /tmp/sim_demux/
constellation transcriptome demultiplex \
    --reads tests/data/sim/reads.bam \
    --library-design cdna_wilburn_v1 \
    --samples tests/data/sim/samples.tsv \
    --output-dir /tmp/sim_demux \
    --threads 4
# (3) Align + count (THIS session) — chains directly off the demux dir
constellation transcriptome align \
    --demux-dir /tmp/sim_demux \
    --reference /tmp/sim_ref \
    --samples tests/data/sim/samples.tsv \
    --output-dir /tmp/sim_align \
    --threads 4 --progress
# Expected: feature_quant/feature_origin=gene_id/part-00000.parquet + manifest.json
#           with per-stage filter pass-through; clean exit
# Mode B negative test:
constellation transcriptome align \
    --demux-dir /tmp/sim_demux --samples tests/data/sim/samples.tsv \
    --output-dir /tmp/sim_align_modeB --threads 4
# Expected: exit 2; stderr message naming Mode B as not yet implemented
```

Success criteria:

- `Alignments.filter` row counts match expected per-filter-combination on the hand-crafted fixture. `metadata_extras['filter_steps']` carries one audit entry per active filter.
- `compute_gene_overlap` recovers correct `(alignment_id → gene_id, overlap_fraction)` on a hand-crafted alignments × gene-set fixture covering: alignments-overlapping-two-genes (best-overlap winner), antisense (drops without `allow_antisense`, kept with), alignments-with-no-overlap (drop), gene-boundary cases (alignment fully inside / partially inside / spanning).
- `count_reads_per_gene` recovers known `(gene_id, sample_id) → count` mapping at 100% per-cell parity given a hand-crafted `gene_assignments` + `read_demux` table pair (composes `compute_gene_overlap` output → `count_reads_per_gene` for the test, exercises the same path the pipeline uses).
- Fused worker on the small fixture writes the expected `alignments/`, `alignment_tags/`, `gene_assignments/` shards; per-chunk stats sum to the full counts.
- `--resume` short-circuits cleanly: re-running with `--resume` after a complete pipeline finishes ≤1 s and produces identical outputs.
- Refusing-to-overwrite contract: re-running without `--resume` against a finished `output_dir/` errors with a clear message.
- `manifest.json` contains: `inputs`, `parameters`, per-stage `stages.{align,decode_filter_overlap,count}` row counts (`decoded`, `after_filter`, `after_overlap`, `reads_with_sample`, `reads_without_sample`), output paths.
- `pytest tests/test_imports.py` still passes (the new `quant` package gets added to the smoke list).
- CLAUDE.md reflects: shipped Mode A + streaming-pipeline rule + memory-budget note + two-layer minimap2 architecture + Mode B/HiFiASM/coverage-learning deferred-to-later-session line.
