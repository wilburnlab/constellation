# Plan: Port NanoporeAnalysis transcriptomics pipeline → `constellation.sequencing` CLI

## CLI shape

The user-facing verbs are namespaced under `constellation transcriptome`. **`demultiplex`** is the per-read characterization stage — it folds segmentation, primer/polyA location, status classification, sample assignment by barcode, artifact tagging, and ORF prediction into one logical step, since these all treat each read as an isolated observation. The verb is named after the user-facing outcome (barcode-driven sample assignment) even though the algorithm does more — primer location and ORF prediction go hand-in-hand with barcode ID and don't merit separate verbs. Downstream verbs (`cluster`, etc.) operate on the aggregated demultiplexed reads.

```
constellation transcriptome demultiplex \
    --sam <path> --construct cdna_wilburn_v1 --samples <samples.tsv> \
    --output-dir <dir> [--threads N] [--scoring hard|probabilistic] \
    [--detect-artifacts none|standard] [--no-quantify] [--resume]
# Outputs: reads.parquet, read_segments.parquet, read_demux.parquet, orfs.parquet,
#          feature_quant.parquet, proteins.fasta, protein_counts.tsv

constellation transcriptome cluster \
    --demultiplex-dir <dir> --output-dir <dir> \
    [--mmseqs|--python-fallback] [--recover-orphans]
# Outputs: clusters.parquet, consensus.parquet, soft_assignments.parquet (S4 only)
```

Quantification (count matrix + protein FASTA) is the tail of `demultiplex` and `cluster` — produced by default; opt out with `--no-quantify`. No standalone `quantify` verb in S1; reserved for future when re-quantification of edited inputs is a real use case.

The flat `demux` and `cluster` placeholders in [constellation/cli/__main__.py](constellation/cli/__main__.py) get replaced by the `transcriptome` sub-parser. The other flat verbs (`basecall`, `assemble`, `polish`, `scaffold`, `annotate` for genome annotation, `project`) stay at top level — they're not transcriptomics-specific.

## Context

NanoporeAnalysis is the lab's existing nanopore RNA-seq processing tool. Its transcriptomics workflow runs as a Nextflow pipeline that splits SAMs into 100k-read batches, demultiplexes the lab's TSO-based PCR-cDNA construct via segmented edlib (5' SSP / polyA / 3' primer / barcode), predicts ORFs per transcript, then aggregates into a Protein × Sample count matrix. The lab uses this for routine analysis today; outputs have been hand-tuned and are trusted.

We are porting it into `constellation.sequencing` as the **first real Constellation CLI workflow**. Departures:

- **No Nextflow.** Replaced with `concurrent.futures.ProcessPoolExecutor`. Critically, **no physical SAM shards on disk** — Nextflow's `batch_sam.py` exists because Nextflow channels need file-shaped inputs. We don't have that constraint, so we stream from disk and batch in memory. Per-stage outputs (parquet) persist; per-batch intermediate state does not.
- **`Scorer` Protocol seam** — hard-threshold mode (parity baseline) and a probabilistic mode (calibratable, statistically principled per CLAUDE.md invariant #1) live behind the same surface.
- **Read-classification beyond "Full length cDNA"** — Complete + 3'Only + 5'Only + MissingBarcode + Unknown + Complex (NanoporeAnalysis statuses) plus new Palindromic / TerminalDropout / Truncated as artifact detection lands. 3'-only reads still feed sample assignment; 5'-only reads feed clustered-consensus recovery.
- **mmseqs-style abundance-weighted clustering** for transcript consensus (NanoporeAnalysis has this on its TODO list but unimplemented).
- **GUI-future-compatible.** Pure-functional verbs `(table) -> table`, parquet-on-disk between stages, structured `ProgressCallback` Protocol — CLI binds to tqdm today, Qt frontend can bind to a signal later.

**Out of scope for this plan (deferred):** Dorado / POD5 / basecall (lab basecalls externally; deferred to a separate work-stream). `Reference` / `Alignments` containers (transcriptomics doesn't need them). Direct-RNA library construct. Modified-base calling.

**Test data** — `~/WilburnLab/Coding/Constellation/test_nanopore_data/pod5_cDNA/Constellation_test_data/`:
- `PAQ63465_pass_*.pod5` (34 MB; raw POD5 — source for everything below; reserved for future POD5 work-stream, unused in this port)
- `test_simplex.sam` (7.6 MB) + `test_simplex.bam` (3 MB) — clean simplex Dorado basecalls. **S1 parity input: `test_simplex.sam`.**
- `test_simplex_wMoves.bam` (4.4 MB) — simplex with `--emit-moves` (per-base move map aligning bases to signal positions). Reserved for future signal-aware basecaller / mod-base work; unused in S1-S4.
- `test_duplex.bam` (3.1 MB) — duplex basecalls (paired complementary strands). **S3 artifact-detection input.**
- `test_simplex_0_fixed1.parquet` — NanoporeAnalysis's per-read output for `test_simplex.sam` with `_fixed1` patches applied (filter `-1` no-match sentinels before picking best UMI, strict `<` ORF cutoff at min_length=30). **This is the parity baseline.**
- `qJS00x_fixed1_proteins.fasta` — protein FASTA from the `_fixed1` rerun.
- `qJS00x_fixed1_protein-counts.tab` — Protein × Sample count matrix (12 active barcodes BC01–BC12) from the `_fixed1` rerun.
- The original (non-`_fixed`) parquet is left in `processed_reads/` for reference but **NOT** used for parity. It has the `-1`-sentinel-leaks-into-best-match bug (any read with no panel match within ed≤2 gets assigned to whichever barcode is first in dict order — typically BC01 — with `score=-1`). The original BC01 count of 212 is inflated by ~101 phantom no-match reads; `_fixed1`'s BC01=111 is the real population. The intermediate `*_fixed*` (without `1`) baseline is also NOT used — it collapsed all 12 barcodes onto BC01/BC02.

**Parity baseline** — supplied directly by the lab in `Constellation_test_data/` as the `_fixed1` triple. No NanoporeAnalysis pin or cache step needed — the parity test diffs `constellation transcriptome demultiplex` output against `test_simplex_0_fixed1.parquet` + `qJS00x_fixed1_proteins.fasta` + `qJS00x_fixed1_protein-counts.tab` from this directory. Test fixtures under [tests/data/sequencing/](tests/data/sequencing/) symlink or copy these (decision: symlink during dev, copy small artifacts on CI to keep test-data hermetic).

**Parity-replication notes for the HardThresholdScorer:**
- Barcode scoring: align `RC(barcode)` against `putative_umi` with `mode=HW, k=2`. **Filter out `editDistance=-1` no-match sentinels** before picking a winner. If the filtered set is empty, return no assignment (`None`). On the surviving matches, pick the smallest edit distance; tie-breaking by panel order is acceptable (stable sort).
- ORF cutoff: `min_aa_length=30` with strict `<` semantics — matches `core.sequence.nucleic.best_orf` defaults.
- Verified against `_fixed1` parquet across all 12 active barcodes: top `putative_umi` per label exactly matches `RC(panel[label])` at ed=0 for 8/12; the other 4 (BC05/06/11/12, all starting with 'A') show consistent 1-nt left-trim from polyA homopolymer overrun — still ed=1 matches, harmless.

**Sample panel for tests** — `cdna_wilburn_v1.json` ships all 24 barcodes (BC1001–BC1024). The S1 test fixture's `samples.tsv` enables BC01–BC12 (the 12 used in this run, per the original parquet's column shape). Output count matrix has 12 sample columns matching the original baseline.

## Architectural decisions adopted

These were the load-bearing calls; recording them so future sessions don't relitigate.

| Decision | Choice |
|---|---|
| Number of sessions | 4 (parallelization in S1; Dorado dropped) |
| Status enum extension | Schema columns added in S1; populated by S2/S3 over time |
| `READ_DEMUX_TABLE` schema bump | S1 only: `+ status (string)`, `+ is_fragment (bool)`, `+ artifact (string)` — additive, all populated as NanoporeAnalysis-parity values in S1 |
| `PolyASlot.max_length` | Add `max_length: int \| None = None` field (additive dataclass change) |
| Hard mode default | `--scoring hard` is default through all four sessions; flip requires explicit lab signoff |
| `parallel.py` location | [constellation/sequencing/parallel.py](constellation/sequencing/parallel.py) (sequencing-domain-scoped, not `core` until ≥2 modalities use it) |
| Status representation | Arrow string column matching NanoporeAnalysis values; Python-side `ReadStatus(StrEnum)` for type safety |
| Per-stage parquet shape | Partitioned dataset (`part-NNNN.parquet` + `_SUCCESS`) for per-read tables; singleton parquet for aggregated outputs |

---

## Session 1 — Hard-threshold parity port + multiprocessing CLI

**Complexity:** XL. The load-bearing session — everything downstream gates on its parquet outputs being byte-correct. Folds in the parallelization layer because "no Nextflow" is a stated directive and `process_sam.py`-style per-batch processing is exactly what `ProcessPoolExecutor` does.

**Goal:** End-to-end runnable `constellation transcriptome demultiplex` that ingests `test_simplex.sam`, segments + classifies + demultiplexes + ORF-predicts each read against the lab's `cdna_wilburn_v1` panel under hard-threshold scoring, optionally fans out across N workers, and writes a Protein × Sample count matrix equivalent to NanoporeAnalysis's output.

### New files

- [constellation/data/sequencing/cdna_wilburn_v1.json](constellation/data/sequencing/cdna_wilburn_v1.json) — primer panel as JSON (5' SSP, 3' primer template, 24 barcodes BC1001-BC1024). Sequences from `~/projects/NanoporeAnalysis/transcriptome_nf/scripts/primers.txt`. Lab-scoped, version-stamped name — future lab variants ship as `cdna_wilburn_v2`, etc.; ONT-stock kits ship as siblings (`cdna_ont_pcb111_24.json`, `cdna_ont_pcb114_96.json`) when implemented.
- [scripts/build-sequencing-primers-json.py](scripts/build-sequencing-primers-json.py) — regenerable builder per CLAUDE.md JSON-data convention.
- [constellation/sequencing/transcriptome/panels.py](constellation/sequencing/transcriptome/panels.py) — concrete `CDNA_WILBURN_V1: LibraryConstruct` + `load_panel(name)` registry.
- [constellation/sequencing/transcriptome/scoring.py](constellation/sequencing/transcriptome/scoring.py) — `Scorer` Protocol + `HardThresholdScorer` shipped; `ProbabilisticScorer` stubbed.
- [constellation/sequencing/transcriptome/classify.py](constellation/sequencing/transcriptome/classify.py) — `ReadStatus(StrEnum)` matching NanoporeAnalysis values + `classify_read(verdicts, *, fragment_threshold)`.
- [constellation/sequencing/transcriptome/quant.py](constellation/sequencing/transcriptome/quant.py) — `build_protein_count_matrix(orfs, demux, samples)` returns `(FEATURE_TABLE, FEATURE_QUANT, fasta_records)`.
- [constellation/sequencing/parallel.py](constellation/sequencing/parallel.py) — `run_batched(fn, batches, *, n_workers, output_dir, progress_cb) -> Path`. Consumes an iterator of `pa.Table` chunks (typically from `SamReader.iter_batches`); each worker computes its chunk's output and **writes its own `part-NNNN.parquet` shard directly into `output_dir/`** via `pq.write_table`. Returns the directory path. Downstream stages consume via `pa.dataset.dataset(output_dir)`, which streams shards lazily and supports out-of-core scans. This is the default — not a perf knob. Two cascading wins: (a) workers don't pickle Arrow tables back to the parent (each shard is written where it was computed), and (b) stage-level `--resume` becomes a directory-existence + `_SUCCESS` marker check, with worker-level partial failure recoverable by re-running just the missing shard indices.
- [constellation/sequencing/progress.py](constellation/sequencing/progress.py) — `ProgressCallback` Protocol; CLI binds to tqdm.
- [constellation/sequencing/transcriptome/stages.py](constellation/sequencing/transcriptome/stages.py) — canonical stage DAG (`sam_ingest → demux → orf → quant`). Each per-read stage writes its output as a **partitioned parquet dataset** (`<output-dir>/<stage>/part-NNNN.parquet`) with a `_SUCCESS` marker on completion; the aggregation tail (feature_quant, proteins.fasta, protein_counts.tsv) writes singletons. **Checkpoint/resume:** `--resume` short-circuits any stage whose `_SUCCESS` marker exists; if shards exist but `_SUCCESS` is missing (interrupted run), worker `n` re-runs only if `part-NNNN.parquet` is absent — partial recovery without restarting the whole stage.

### Existing stubs → shipped

- [constellation/sequencing/acquisitions.py](constellation/sequencing/acquisitions.py) — validators-only mode: PK uniqueness, cast-on-construct, `from_records`, `empty()`, `ids`, `__len__`.
- [constellation/sequencing/samples.py](constellation/sequencing/samples.py) — same shape; `samples_for(acquisition_id, barcode_id) -> list[int]`.
- [constellation/sequencing/readers/sam_bam.py](constellation/sequencing/readers/sam_bam.py) — `SamReader` (RawReader subclass, suffix-keyed `.sam`, emits `READ_TABLE`). Two methods:
  - `SamReader.read(path) -> pa.Table` — eager full read (small inputs / fixtures).
  - `SamReader.iter_batches(path, batch_size) -> Iterator[pa.Table]` — streaming read, yields one `pa.Table` per `batch_size` records via a single pass over the file. This is the parallelization-feeding path; nothing materializes on disk between SAM ingest and the worker shard. Pure-Python text scan (no pysam yet).
  - `BamReader` (binary) stays stubbed — S1 input is unaligned Dorado SAMs.
  - The forward-looking `sequencing/io/sam_bam.py` scaffold is **not touched in S1**. See "Risks / decisions deferred" for the `<domain>/io/` vs `<domain>/readers/` split — it's a real question for sequencing but not on this port's path.
- [constellation/sequencing/align/locate.py](constellation/sequencing/align/locate.py) — `locate_substring(query, ref, *, max_distance) -> LocateMatch | None` via edlib HW.
- [constellation/sequencing/align/pairwise.py](constellation/sequencing/align/pairwise.py) — `pairwise_align(query, ref, *, backend='edlib', mode='infix', max_distance=None)` shipped with edlib backend; parasail deferred.
- [constellation/sequencing/transcriptome/demux.py](constellation/sequencing/transcriptome/demux.py) — `locate_segments(reads_table, construct, scorer, *, progress_cb=None)` and `resolve_demux(segments_table, samples)`. Internal `_annotate_strand` mirrors NanoporeAnalysis's forward-and-RC processing with the strand-resolution rule (prefer "Complete"; fall back to "Complex" if both have annotations but neither is Complete; default to forward if both Unknown).
- [constellation/sequencing/transcriptome/orf.py](constellation/sequencing/transcriptome/orf.py) — `predict_orfs(demux, reads, *, codon_table=STANDARD, min_aa_length=30, eligible_statuses=("Complete","3' Only"))`. Wraps `core.sequence.nucleic.best_orf`.

### Schema additions (additive, Arrow-cast-compatible)

In [constellation/sequencing/schemas/transcriptome.py](constellation/sequencing/schemas/transcriptome.py):
- `READ_DEMUX_TABLE` gains `status` (string), `is_fragment` (bool), `artifact` (string, default `"none"`).

In [constellation/sequencing/transcriptome/adapters.py](constellation/sequencing/transcriptome/adapters.py):
- `PolyASlot.max_length: int | None = None` field added.

### CLI

In [constellation/cli/__main__.py](constellation/cli/__main__.py): introduce the `transcriptome` sub-parser; remove the flat `demux` and `cluster` placeholders.

```
constellation transcriptome demultiplex \
    --sam <path> \
    --construct cdna_wilburn_v1 \
    --samples <samples.tsv> \
    --output-dir <dir> \
    [--scoring hard|probabilistic]    # probabilistic stubbed in S1
    [--detect-artifacts none|standard] # standard stubbed in S1; default 'none' in S1
    [--threads N]                      # 1 = single-threaded
    [--batch-size 100000]              # NanoporeAnalysis default
    [--resume]                         # short-circuit completed stages
    [--no-quantify]                    # skip count-matrix + FASTA tail
    [--progress]
```

Outputs under `<dir>/`:
- Per-read partitioned datasets: `reads/`, `read_segments/`, `read_demux/`, `orfs/` (each containing `part-NNNN.parquet` + `_SUCCESS`).
- Aggregated singletons: `feature_quant.parquet`, `proteins.fasta`, `protein_counts.tsv`.
- `manifest.json` — per-stage timing, input hashes, shard counts, scoring config, NanoporeAnalysis pin SHA. Reproducibility audit lives here.

### Tests

- [tests/test_sequencing_acquisitions.py](tests/test_sequencing_acquisitions.py), [tests/test_sequencing_samples.py](tests/test_sequencing_samples.py) — PK uniqueness, cast-on-construct, FK closure raises on dangling IDs.
- [tests/test_sequencing_readers_sam.py](tests/test_sequencing_readers_sam.py) — read `test_simplex.sam`; assert schema match, row count vs `wc -l`, first-read sequence/quality round-trip.
- [tests/test_sequencing_align_locate.py](tests/test_sequencing_align_locate.py) — synthetic-string unit tests; ground-truth against direct edlib calls.
- [tests/test_sequencing_transcriptome_classify.py](tests/test_sequencing_transcriptome_classify.py) — table-driven enumeration of (5p, 3p, BC, transcript-len) → expected status; cross-check against NanoporeAnalysis truth table.
- [tests/test_sequencing_parallel.py](tests/test_sequencing_parallel.py) — `run_batched` row-count invariant under different worker counts (n=1,2,4); identical-output invariant (concatenated dataset equal up to row order regardless of worker count); stage-level checkpoint short-circuit (`_SUCCESS` marker present → no work); shard-level partial recovery (delete a single `part-NNNN.parquet`, rerun, only that shard regenerates).
- [tests/test_sequencing_readers_sam_streaming.py](tests/test_sequencing_readers_sam_streaming.py) — `iter_batches` produces row-count-equivalent output to eager `read()`; correctly chunks at `batch_size` boundaries; doesn't allocate the full table in memory (peak-RSS sanity check via `tracemalloc`).
- [tests/test_sequencing_transcriptome_demultiplex_parity.py](tests/test_sequencing_transcriptome_demultiplex_parity.py) — **parity gate, `@pytest.mark.slow`.** Loads `test_simplex.sam` through `constellation transcriptome demultiplex --scoring hard --detect-artifacts none`; loads supplied NanoporeAnalysis baseline (`processed_reads/test_simplex_0.parquet` + `qJS00x_proteins.fasta` + `qJS00x_protein-counts.tab`) from [tests/data/sequencing/](tests/data/sequencing/). Asserts 100% per-read `sample_id` agreement and 100% `status` agreement. Floating-point quality-mean drift ≤0.1% allowed.
- [tests/test_sequencing_transcriptome_quant.py](tests/test_sequencing_transcriptome_quant.py) — synthetic 5-read fixture (exact count assertions) + full-data parity (sorted `protein_counts.tsv` row-by-row equality).
- [tests/test_sequencing_cli_transcriptome_demultiplex.py](tests/test_sequencing_cli_transcriptome_demultiplex.py) — `subprocess` invocation; assert all per-read dataset directories have `_SUCCESS` markers and load via `pa.dataset.dataset(...)`; assert aggregated singletons (`feature_quant.parquet`, `proteins.fasta`, `protein_counts.tsv`, `manifest.json`) exist and load.
- [tests/test_sequencing_cli_transcriptome_demultiplex_resume.py](tests/test_sequencing_cli_transcriptome_demultiplex_resume.py) — kill mid-pipeline, restart with `--resume`, byte-identical output.
- [tests/test_imports.py](tests/test_imports.py) — add new submodules.

### Validation gate

1. Parity gate (`test_simplex.sam`): 100% per-read `sample_id` agreement, 100% `status` agreement (mapped from NanoporeAnalysis's `cDNA status`), sorted `protein_counts.tsv` element-wise equal to NanoporeAnalysis baseline.
2. `proteins.fasta` set-equal in `(sequence, count)`.
3. `--threads 4` produces byte-identical output to `--threads 1`; wall-clock ≥2.5× speedup.
4. `--resume` after partial run produces byte-identical output to clean run.
5. `ruff check` + full `pytest` green.

### Deferred to later sessions

- `ProbabilisticScorer` body (Protocol surface ships now).
- `Palindromic` / `TerminalDropout` / `Truncated` status values (column shipped, populated `"none"`).
- mmseqs-style clustering / consensus / 5'-only-read recovery.
- POD5 / Dorado / BAM reader.

### Dependencies

None — foundation session.

---

## Session 2 — Probabilistic scoring + extended classification populated

**Complexity:** M.

**Goal:** Statistical scoring beneath the same demux pipeline, calibratable from per-segment edit-distance distributions. Hard mode remains default; probabilistic is opt-in. 3'-only reads gain explicit sample-assignment validation. Schema is unchanged from S1 — only the column *values* extend.

### New files / modifications

- [constellation/core/stats/distributions.py](constellation/core/stats/distributions.py) — distribution selection is empirical and gated on a pre-session exploration step (see below). `core.stats` already ships 9 distributions (NormalDistribution, StudentT, GeneralizedNormal, Beta, Gamma, LogNormal, Poisson, Multinomial, Dirichlet); plausible new additions for this work are Binomial (count of mismatches over fixed-length adapter/barcode), BetaBinomial (overdispersion when quality variance is high), NegativeBinomial / Geometric (polyA length-of-run distribution), or just an EmpiricalHistogram nonparametric. Add only what the exploration justifies — `core.stats` will accumulate distributions over time as needs arise across the wider package, so this session shouldn't preempt that growth.

  **Pre-session exploration step (S2 prologue):** before committing to a distribution per segment kind, run the S1 hard-mode demux on a *larger* dataset (lab-provided; specific dataset TBD when S2 starts — the supplied `test_simplex.sam` is too small for fit-quality assessment), extract the per-segment edit-distance histograms (5' SSP / 3' primer / each barcode position / polyA length), and visually + statistically compare candidate fits. Pick the simplest distribution whose KS-stat is acceptable; document the fit-quality numbers in [docs/sequencing/probabilistic_scoring_calibration.md](docs/sequencing/probabilistic_scoring_calibration.md) so future re-calibrations have a reproducible baseline.

- **Synthetic-read test fixture (S2 prologue, before scorer code)** — build a small hand-labeled fixture (~30-50 reads) covering every classification branch + each edge case the S1 hard-mode work surfaced: polyA at the very end of the read (empty post_polyA → `parse_adapter3_seq` empty-subject bug); polyA at the very start (empty pre_polyA → SSP empty-subject bug); both orientations producing non-Unknown annotations (Complex tie-break); reverse orientation lands as `"Unknown Fragment"` while forward has a real annotation (Unknown-Fragment string-equality bug); `Complete Fragment` (all features detected, transcript < min_transcript_length); barcode-RC starts with A (polyA boundary trim → ed=1 instead of ed=0); reads with no SSP but valid 3'-primer + barcode (3' Only); reads with valid SSP but no 3'-primer (5' Only); reads where 2+ barcodes match within ed≤2 (panel-spacing edge case). Expected `(status, sample_id, has_orf)` per read is hand-curated. The fixture lives at `tests/data/sequencing/synthetic_reads/` and drives both `HardThresholdScorer` regression tests AND the `ProbabilisticScorer` development. **Rationale:** the S1 hard-mode work surfaced 3 distinct bugs in NanoporeAnalysis (Unknown-Fragment string-equality, `-1` sentinel leak, empty-subject "match"); a per-read regression fixture with hand-labeled expected outcomes would have caught all three at write time. We pay this debt once at the start of S2 so the probabilistic scorer ships with a solid behavioral spec, and so any future scorer (S3 artifact-aware, future signal-aware) inherits the same battery of edge-case tests.

- [constellation/sequencing/transcriptome/scoring.py](constellation/sequencing/transcriptome/scoring.py) — `ProbabilisticScorer` body shipped:
  - Per-segment `Parametric` model (per-adapter, per-barcode-position, per-polyA-length); the chosen distribution lives behind a uniform `Parametric` interface so swapping (e.g. Binomial → BetaBinomial) is a one-line config change without touching the scorer surface.
  - `score_*` methods return log-odds against a uniform-mismatch null derived from per-read mean Phred-Q.
  - `calibrate(reads_table, demux_table) -> ProbabilisticScorer` fits per-segment models from a prior hard-threshold run.
  - JSON serialization so `--calibration scorer.json` round-trips.
- [constellation/sequencing/transcriptome/classify.py](constellation/sequencing/transcriptome/classify.py) — `ReadStatus` enum gains `Truncated` (transcript ends without 3' adapter despite polyA — different from "3' Only" which also lacks polyA).
- CLI: `constellation transcriptome demultiplex --scoring probabilistic --calibration scorer.json` + new subcommand `constellation transcriptome calibrate --from <hard-mode-dir> --output scorer.json`.

### Tests

- `tests/test_core_stats_<dist>.py` (filename depends on which distributions land) — for any newly-added distribution: parameter recovery, log-prob sanity, vmap-compat (per CLAUDE.md `Parametric` requirements).
- [tests/test_sequencing_transcriptome_scoring_probabilistic.py](tests/test_sequencing_transcriptome_scoring_probabilistic.py) — synthetic fixtures with known generative model; `ProbabilisticScorer.calibrate` recovers parameters within 5%. Distribution-agnostic: parameterizes over whatever scorer config the empirical fit lands on.
- [tests/test_sequencing_transcriptome_demultiplex_parity.py](tests/test_sequencing_transcriptome_demultiplex_parity.py) — extend: `--scoring probabilistic` post-calibration agrees with hard mode on `sample_id` for ≥98% of reads (drift threshold tunable on first run); hard mode still 100% parity.
- [tests/test_sequencing_three_prime_only_assignment.py](tests/test_sequencing_three_prime_only_assignment.py) — explicit gate: `ThreePrimeOnly` reads receive `sample_id` (per directive #3 — barcode is present, sample assignment is well-defined).

### Validation gate

- Hard parity unchanged.
- Probabilistic mode end-to-end on `test_simplex.sam` post-calibration; `Sample_ID` delta vs hard ≤2%.
- Calibrated scorer round-trips through JSON; replay produces byte-identical output.

### Dependencies

Session 1 (Scorer Protocol seam, parity baseline, status enum scaffold).

---

## Session 3 — Palindrome / duplex artifact detection

**Complexity:** L.

**Goal:** Detect read-level artifacts the lab observes in nanopore data that don't appear in PacBio (palindromes, terminal dropout, multi-polyA chimeras). Use `test_duplex.bam` as the validation set since duplex basecalling is where these artifacts most often manifest. Populates the `artifact` column shipped in S1.

**Format note:** the duplex test fixture is a BAM (Dorado's natural output), not a SAM. S3 also ships `BamReader.read()` / `iter_batches()` via `pysam.AlignmentFile` (`pysam` already in `[sequencing]` extras). Same `READ_TABLE` output as `SamReader`; populates `channel`, `start_time_s`, `duration_s` from BAM tags (`ch:i`, `st:Z`, `du:f`) where present.

### New files

- [constellation/sequencing/transcriptome/artifacts.py](constellation/sequencing/transcriptome/artifacts.py) — pure detection functions:
  - `detect_palindrome(seq, *, k=15, identity=0.85)` — k-mer self-similarity scan via `core.sequence.ops.kmerize` + hashtable. Flags reads where 5' half is near-perfect RC of 3' half.
  - `detect_chimera_via_polyA(seq, polyA_matches)` — flags multi-polyA reads. NanoporeAnalysis silently picks the last polyA; we keep that behavior but tag `artifact="chimera"`.
  - `detect_terminal_dropout(seq, qual, *, window=50, threshold_q=7)` — flags terminal Q-score collapse (basecaller-fade artifact).

### Modifications

- [constellation/sequencing/transcriptome/demux.py](constellation/sequencing/transcriptome/demux.py) — `locate_segments` consults `artifacts.*` and writes `artifact` column. New status values `Palindromic` and `TerminalDropout` populate when artifacts dominate.
- [constellation/sequencing/transcriptome/classify.py](constellation/sequencing/transcriptome/classify.py) — `ReadStatus` gains `Palindromic`, `TerminalDropout`. Classification rule: if artifact is set, status reflects the artifact; otherwise the S1/S2 logic applies.

### Tests

- [tests/test_sequencing_transcriptome_artifacts.py](tests/test_sequencing_transcriptome_artifacts.py) — synthetic palindromes (`seq + reverse_complement(seq)`); synthetic chimeras; synthetic terminal dropouts.
- [tests/test_sequencing_readers_bam.py](tests/test_sequencing_readers_bam.py) — `BamReader` round-trip on `test_simplex.bam` (assert row-count + first-read sequence equality with `test_simplex.sam` via the SAM reader).
- [tests/test_sequencing_transcriptome_duplex_validation.py](tests/test_sequencing_transcriptome_duplex_validation.py) — `@pytest.mark.slow`, runs on `test_duplex.bam`. First run sets the regression baseline (count of `Palindromic` reads); subsequent runs assert non-zero detection and lower-bound the rate. Compares S1-style demux (artifact detection off) vs artifact-aware demux: artifact-aware should reduce `Complex` count and increase explained-status fraction.
- [tests/test_sequencing_transcriptome_demultiplex_parity.py](tests/test_sequencing_transcriptome_demultiplex_parity.py) — extend: `--detect-artifacts none` (now the parity flag) continues to preserve S1's 100% parity on `test_simplex.sam`. With `--detect-artifacts standard`, parity is no longer expected — the new column populates.

### Validation gate

- `test_duplex.bam` produces non-zero `Palindromic` reads; the count is documented as the regression baseline in [tests/data/duplex_artifact_baseline.json](tests/data/duplex_artifact_baseline.json).
- Hard parity on `test_simplex.sam` still passes when artifacts are disabled.

### Deferred

- Probabilistic palindrome model (currently threshold-based; lift to `core.stats` later when calibration data exists).

### Dependencies

Sessions 1, 2 (uses status enum, scorer protocol, demux output).

---

## Session 4 — Clustering + consensus + barcode-less-read recovery

**Complexity:** L. **Architecture deliberately undecided at plan time.** The user has additional design docs that will inform this session; mmseqs-as-primary is *not* committed. Naming for the recovery module is also provisional.

**Goal:** Move beyond per-read protein counting. Group reads that represent the same underlying transcript, build per-group consensus, predict ORFs from consensus (which gets us better error correction than per-read ORFs), and use the cluster structure to rescue reads that lack a barcode but whose transcript identity is recoverable from sequence similarity to clustered reads of known sample provenance. This closes the "5'-only reads can feed assembly/error correction, abundance priors can recover misassigned reads" loop the user described.

### Provisional file slots (algorithm TBD)

- `constellation/sequencing/transcriptome/cluster.py` — clustering verb. Backend(s) and clustering-criterion (sequence identity? k-mer Jaccard? minimizer-graph? abundance-weighted? something else from the user's design docs?) deferred. Likely shape: a top-level verb that returns a cluster-membership table; mechanics behind it open.
- `constellation/sequencing/transcriptome/consensus.py` — consensus-building verb. Per-cluster sequence rollup → consensus → ORF prediction → `TRANSCRIPT_CLUSTER_TABLE`. Likely uses `pairwise_align` from S1 + `core.sequence.nucleic.best_orf`, but the consensus-rule (majority? weighted? POA-graph?) is part of the deferred design.
- `constellation/sequencing/transcriptome/recovery.py` — *name provisional* — recovers reads that demux dropped or under-classified (barcode-less but transcript-identifiable; chimeras with one usable segment; etc.). Whether this is one module or several, and what it's named, settles when the design lands.
- `constellation/sequencing/transcriptome/quant.py` extension — once clusters exist, the count matrix is naturally cluster-keyed (proteins-per-cluster, not proteins-per-read), and orphan-read soft-assignments contribute fractional counts.
- CLI: `constellation transcriptome cluster --demultiplex-dir <dir> --output-dir <dir> [...]`. Tail-quantification runs by default; flags TBD with the design.

### Tests (sketches; will firm up with the design)

- Synthetic-cluster recovery on a small fabricated multi-transcript fixture (ground-truth partition known).
- Synthetic-consensus recovery: clusters with a known true consensus → recovered consensus identity ≥99%.
- Synthetic recovery test: hide a barcode from a `Complete` read so it becomes barcode-less; assert recovery routes it back to its true sample.
- Parity-preservation: extend [tests/test_sequencing_transcriptome_demultiplex_parity.py](tests/test_sequencing_transcriptome_demultiplex_parity.py) — chained `demultiplex` → `cluster --recover-orphans=false` (or whatever the flag-name lands as) → count matrix per protein remains within ±2% of S1 baseline.

### Validation gate

- Per-test gates above pass.
- S1 parity preserved when the cluster stage is omitted (don't regress the parity validation when this session ships).
- Cluster-keyed quant numerically agrees with per-read quant on `test_simplex.sam` at default settings (clustering shouldn't lose quantification fidelity).
- Recovery improves total-read-utilization ratio on `test_simplex.sam` (more reads contribute to the count matrix); first-run number becomes the regression baseline.

### Dependencies

Sessions 1, 2, 3 (uses demultiplex + status + artifact outputs). Session 4 also depends on a **design discussion with the user before implementation starts** — they have docs to share. Plan-only commitment now: file slots and verb shape (`cluster` + consensus + recovery); algorithm choice deferred.

---

## Critical files for execution

The five most load-bearing files for session 1; the parity port hinges on getting them right:

- [constellation/sequencing/transcriptome/demux.py](constellation/sequencing/transcriptome/demux.py)
- [constellation/sequencing/transcriptome/scoring.py](constellation/sequencing/transcriptome/scoring.py) — new
- [constellation/sequencing/transcriptome/panels.py](constellation/sequencing/transcriptome/panels.py) — new
- [constellation/sequencing/readers/sam_bam.py](constellation/sequencing/readers/sam_bam.py)
- [constellation/cli/__main__.py](constellation/cli/__main__.py)

Plus the parity-baseline reference: [~/projects/NanoporeAnalysis/NanoporeAnalysis/align.py](../../projects/NanoporeAnalysis/NanoporeAnalysis/align.py) — the algorithm we must match byte-for-byte in hard mode.

## Verification approach (how to test end-to-end)

1. **Per-session pytest** — every session's tests must pass before its session ships. `@pytest.mark.slow` parity / duplex tests run locally; CI runs unit subset.
2. **Parity gate (S1+):** `pytest tests/test_sequencing_transcriptome_demux_parity.py -m slow` against cached NanoporeAnalysis baseline.
3. **CLI smoke (S1+):** `constellation transcriptome demultiplex --sam ~/WilburnLab/Coding/Constellation/test_nanopore_data/pod5_cDNA/Constellation_test_data/test_simplex.sam --construct cdna_wilburn_v1 --samples <fixture> --output-dir /tmp/transcriptome-test --threads 4` produces all expected output files in <2 minutes; `protein_counts.tsv` matches the supplied `qJS00x_protein-counts.tab` baseline.
4. **Duplex artifact gate (S3+):** `pytest tests/test_sequencing_transcriptome_duplex_validation.py -m slow` non-zero `Palindromic` count.
5. **Hand audit (each session):** run `constellation transcriptome demultiplex` against `test_simplex.sam`; sanity-check `read_demux.parquet` in a notebook against expectations on a sample of reads.

## Risks / decisions deferred

- **NanoporeAnalysis baseline source** — supplied directly by the lab in `Constellation_test_data/` (per-read parquet + protein FASTA + counts TSV from a clean upstream run). No pin file or cache regeneration needed. If the lab re-runs upstream against new test data, drop in the new baseline files and bump a fixture-version comment in the parity test.
- **Probabilistic mode default-flip** — `--scoring hard` stays default through all four sessions. Promotion to default requires a separate PR with explicit lab signoff and a paired regression-results delta document.
- **Direct-RNA construct (`ONT_RCB114`)** — separate `LibraryConstruct` instance to be added when the lab runs direct-RNA. Not in this plan; trivially additive when the panel JSON exists.
- **Dorado / POD5 work-stream** — entirely out of scope for these four sessions. Spin into a separate plan when needed.
- **`Reference` / `Alignments` containers** — remain stubbed. Ship in a separate session when assembly / alignment work begins.
- **`sequencing/io/` vs `sequencing/readers/` split** — borrowed from `massspec` where the split is real (encyclopedia `.dlib`/`.elib` produces a `(Library, Quant?, Search?)` tuple — three distinct containers from one SQLite file, doesn't fit a single-schema RawReader). In sequencing today there is no analogous case: aligned BAM produces `(READ_TABLE, ALIGNMENT_TABLE)` but both end up in the same `Alignments` container, so it's multi-table not cross-tier — better expressed as a reader returning a tuple. Recommendation: leave the existing `sequencing/io/sam_bam.py` stub alone (don't touch in this port); revisit the directory when a real cross-tier format appears, or delete it if none does. Not on the transcriptomics-port critical path either way.
- **Move-table BAM as future signal-aware hook** — `test_simplex_wMoves.bam` is shipped in the test data and contains per-base move maps aligning called bases back to POD5 signal positions. Not used in S1-S4. Reserved for: (a) future modified-base calling from BAM tags, (b) future physics-based basecaller validation, (c) signal-quality-aware demux scoring (use per-base signal confidence instead of just Phred). Worth threading through `BamReader` so the move-table tag is preserved when present, even if S1-S4 don't consume it.
- **Performance on large inputs** — `test_simplex.sam` is 7 MB; production runs are 100s of GB. The streaming + worker-writes-own-shard design handles the test scale comfortably and should hold for typical lab runs. The remaining perf hook for GB-scale inputs is replacing pickle-over-pipe of the *input* batch with worker-side file-reread via `(path, byte_offset_start, byte_offset_end)` work units, so the parent thread does only an offset-scan pass and never holds full record batches. Profile in session 1 wrap-up to decide whether it's needed.
