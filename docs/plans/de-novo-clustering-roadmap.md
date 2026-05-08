# De novo transcriptome clustering — sequencing the path through validation infrastructure

## Context

S2 Mode A (`transcriptome align`) ships gene-level counting via genome-guided alignment, and on real mouse-testis cDNA the top hits track expected sperm markers (PRM1/PRM2, ODF2, CRISP, SPACA17). The next major step is the de novo transcriptome clustering algorithm specified in `~/WilburnLab/Nanopore/NanoporeAnalysis/clustering_roadmap_C260316.md` — a kmer-minimizer + abundance-weighted greedy set-cover replacement for mmseqs2 `linclust` that scales to PromethION-size datasets and avoids the overclustering / opacity problems the roadmap calls out.

Before turning the de novo algorithm loose on real data, we need a way to know whether a given pair of reads is "the same transcript" — and the only reliable answer comes from looking at their genome-guided alignments (same set of exon blocks, same splice junctions, same span). Right now `map_to_genome` produces an indexed BAM with `--cs=long` preserved, but downstream we only project to `gene_assignments` (one row per (read, gene)) and from there to FEATURE_QUANT. The CIGAR + cs:long stream is dropped on the floor.

This plan sequences the work in four phases so the validation substrate lands before — and independent of — the de novo algorithm. Each phase produces a useful end-user artifact on its own:

- **Phase 1** materialises three intermediates from the existing BAM stream (per-read exon blocks, aggregated splice junctions, pile-up) as power-user toggles, off by default.
- **Phase 2** uses Phase 1's per-read fingerprints to ship a genome-guided "clustering" baseline that is itself useful (collapses reads from the same isoform without bothering with kmers) AND becomes the ground truth for Phase 3.
- **Phase 3** ships the roadmap's Layer 0 + Layer 1 (dereplication + error-removal clustering + consensus) as torch-first reference impls.
- **Phase 4** cross-validates Phase 3 against Phase 2 on datasets where both can run, and quantifies de novo agreement to give us confidence before handing this to users.

Layers 2 (transcript-variant grouping) and 3 (gene-structure inference) from the roadmap are deferred to a follow-up plan; they need the empirical feedback Phase 4 generates to make their parameter choices on solid ground.

The same pile-up + alignment-block infrastructure also serves the Mode-B-adjacent workflows the user flagged: re-aligning reads to a freshly-assembled genome to check assembly uniformity-of-depth, hunt for assembly artifacts, and identify sex-determination loci by comparing male vs female depth. So Phase 1 lands once and pays for itself across multiple downstream features.

---

## Phase 1 — Genome-guided alignment intermediates (power-user toggles)

Goal: lift the data already implicit in the sorted BAM into addressable Arrow tables, behind off-by-default CLI flags. None of these change the default `transcriptome align` cost path.

### 1.1 New schema: `ALIGNMENT_BLOCK_TABLE`

One row per CIGAR-derived alignment block (a "block" = a contiguous M/=/X run between N or large I/D operations — i.e. one exon's worth of alignment).

```
alignment_id   int64   FK→ALIGNMENT_TABLE
block_index    int32
contig_id      int64   FK→CONTIG_TABLE (resolved from ref_name)
ref_start      int64   0-based half-open
ref_end        int64
query_start    int32   0-based half-open on the trimmed transcript window
query_end      int32
n_match        int32   from cs:long if available, else from CIGAR M
n_mismatch     int32   from cs:long; nullable when only CIGAR is present
n_insert       int32   query bases inserted relative to ref within this block
n_delete       int32   short deletions consumed within this block
```

Place in `constellation/sequencing/schemas/alignment.py` next to `ALIGNMENT_TABLE`. Self-register via `core.io.schemas.register_schema`.

### 1.2 New schema: `SPLICE_JUNCTION_TABLE`

Aggregated per (contig, donor, acceptor, strand). Produced by reduction over `ALIGNMENT_BLOCK_TABLE`.

**Why aggregate introns rather than exons.** Splice sites are *steep* — most reads spliced through the same intron produce near-identical (donor, acceptor) coordinates because the spliceosome is dinucleotide-anchored on canonical GT...AG. They're not strictly *discrete*: cryptic splice-site usage (the spliceosome picking a wrong-but-canonical GT/AG nearby) does happen and produces small excursions on otherwise-fixed coordinates, and that's a feature we want to preserve in the data — not collapse — so we can spot anomalous excursion rates as a basecaller / library-prep / spliceosome-fidelity diagnostic. Exons in long-read RNA-seq have one steep side (the splice-site end) and one genuinely fuzzy side on terminal exons (TSS / SSP-priming on the first exon, polyA on the last), so cross-read exon aggregation requires fuzzy matching on those terminal boundaries. Aggregating on the steeper primitive keeps the reduction exact and the table small (O(50k) rows for a mammalian genome). The Phase 2 fingerprint's `--intron-quantum-bp` knob (default 10) absorbs *small* cryptic excursions into the same fingerprint bucket while leaving larger excursions to surface as real cluster-discordance signal in Phase 4. The per-read exon view already lives in `ALIGNMENT_BLOCK_TABLE`; the cross-read junction view here provides the canonical key for "do these reads describe the same internal splicing topology" (Phase 2's fingerprint hash IS the canonicalised junction sequence). Junction aggregation also matches every existing long-read RNA-seq toolchain (minimap2 SJ, STAR SJ.out.tab, HISAT2/TopHat), so downstream interop is one-step. **Cross-read exon aggregation** (a `CANDIDATE_EXON_TABLE`) becomes useful for genuine exon discovery in un-annotated genomes — derived as the intron-complement of the junction table plus pile-up edge detection on terminal exons. That's Layer 3 / gene-structure inference, deferred per the plan; both Phase 1 inputs (junctions + pile-up) are already in hand when we get there.

```
junction_id        int64   PK
contig_id          int64   FK
donor_pos          int64   intron 5' (= prev_block.ref_end)
acceptor_pos       int64   intron 3' (= next_block.ref_start)
strand             string  '+' | '-' | '?' (when not splice-motif-disambiguated)
read_count         int32   number of distinct read_ids supporting this junction
motif              string  e.g. 'GT-AG', 'GC-AG', 'AT-AC', 'other' — derived
                            from genome at donor[..+2] / acceptor[-2..]
annotated          bool    true iff matches a FEATURE_TABLE intron
                            (donor_pos, acceptor_pos) within tol_bp (default 0)
gene_id            int64   nullable; populated when intersect with annotation
```

### 1.3 Pile-up — populate the existing `COVERAGE_TABLE`

Schema is already shipped (`/constellation/sequencing/schemas/quant.py`). It currently has no producer because S2 Mode A was scoped tight to "demux → align → gene counts" and pile-up landed in the schema layer (alongside the assembly schemas) in anticipation of downstream work that never wired through. The CIGAR string preserved in ALIGNMENT_TABLE has always made pile-up *derivable* from the shipped output — Phase 1.6 just builds the producer. RLE-compressed (one row per interval of constant depth) keeps the on-disk size manageable on mammalian-scale genomes with long-read data.

**Default-on vs default-off.** The plan currently defaults all `--emit-*` flags to off so existing `transcriptome align` invocations have unchanged wall-clock + output size. The argument for default-on: pile-up + junctions are cheap incremental cost over an already-running align stage, and surface high-value diagnostics (depth uniformity, basecaller-drift junction motifs, sex-det loci candidates) for free on every run. The argument for default-off: changing default cost / output footprint of a shipped command without explicit buy-in is wrong, especially for users on cluster nodes with disk quotas. Plan defaults off; flag polarity flips trivially if you prefer the other call.

### 1.4 New schema: `READ_FINGERPRINT_TABLE`

Per-read, primary-alignment-only, derived from `ALIGNMENT_BLOCK_TABLE`. The cluster key for Phase 2.

```
read_id            string  FK
contig_id          int64
strand             string
n_blocks           int32
span_start         int64   first block.ref_start
span_end           int64   last block.ref_end
fingerprint_hash   uint64  hash of (contig_id, strand, [(donor,acceptor)...])
                            quantised at qbp=10 (configurable). Quantisation
                            absorbs SSP-noise on intron boundaries while
                            keeping splicing topology intact.
junction_signature string  human-readable form: 'chrN:donor1-acceptor1,...'
                            for diagnostics; not the cluster key
```

### 1.5 New module: `constellation/sequencing/align/cigar.py`

Pure-Python parsers operating per-alignment (per-record cost is tiny; vectorisation buys nothing here). Returns Arrow tables, not pandas / numpy.

- `parse_cigar_blocks(cigar_string: str, ref_start: int, query_start: int = 0) -> list[Block]` — splits on N (and on I/D longer than `intron_min_bp`, default 25). Returns block records.
- `parse_cs_long(cs: str) -> CsRunStream` — yields (op, length, ref_consumed, query_consumed, n_match, n_mismatch). Used to populate `n_match`/`n_mismatch` on each block.
- `blocks_to_junctions(blocks)` — adjacent-block reduction into junction tuples.

Pyarrow is the canonical output format; do not vendor numpy or pandas at this seam.

### 1.6 New module: `constellation/sequencing/quant/coverage.py`

`build_pileup(alignment_blocks: pa.Table, *, contigs: pa.Table, samples: Samples | None = None) -> pa.Table`

Sweep-line over sorted block intervals, RLE-encoded depth output matching `COVERAGE_TABLE`. Per-sample stratification optional via the `samples` arg (joins gene_assignments → sample_id like `count_reads_per_gene` already does). When stratification is on, emits one (contig, sample) panel per sample; when off, sums across samples.

Memory budget at 200M reads ≈ 600M blocks × ~40 B ≈ ~25 GB if materialised; therefore `build_pileup` accepts a `pa.dataset.Dataset` and streams via `to_batches(...)` rather than `to_table()`. Output stays bounded (≤ contig-length rows per contig in the worst case).

### 1.7 New module: `constellation/sequencing/quant/junctions.py`

`aggregate_junctions(alignment_blocks: pa.Table, genome: GenomeReference, annotation: Annotation | None = None) -> pa.Table` — the SPLICE_JUNCTION_TABLE producer. Annotation arg is optional so the same code runs against assembled genomes without an annotation pass yet.

### 1.8 New module: `constellation/sequencing/transcriptome/fingerprints.py`

`compute_read_fingerprints(alignment_blocks: pa.Table, *, intron_quantum_bp: int = 10) -> pa.Table` — produces `READ_FINGERPRINT_TABLE`. Fingerprint hash is `xxhash64` over the canonical tuple `(contig_id, strand, ((donor//q)*q, (acceptor//q)*q for each junction))` — Python's `hashlib.blake2b` is acceptable if we want to stay inside the stdlib. Primary alignments only; secondary/supplementary excluded by `is_secondary == False & is_supplementary == False`.

### 1.9 CLI surface — `transcriptome align` flags

All three intermediates are off by default. On a typical user's run, S2 stays exactly as fast as it is today.

```
--emit-blocks       Write alignment_blocks/ partitioned dataset.
                    Required prereq for --emit-pileup, --emit-junctions,
                    --emit-fingerprints, and `transcriptome cluster
                    --mode genome-guided`.
--emit-pileup       Write coverage/ partitioned dataset (one parquet per
                    contig). Implies --emit-blocks.
--emit-junctions    Write junctions.parquet. Implies --emit-blocks.
--emit-fingerprints Write read_fingerprints.parquet. Implies --emit-blocks.
--intron-min-bp INT      Default 25; threshold above which CIGAR D/I gets
                         promoted to a block break.
--intron-quantum-bp INT  Default 10; fingerprint quantisation.
```

Wire the block extraction inside `fused_decode_filter_overlap_worker` (gated on `--emit-blocks`), so block-level data flows out of the same stream the gene-overlap pass walks. Aggregations (junctions, fingerprints, pile-up) run in the resolve stage as Arrow-dataset reductions over `alignment_blocks/`.

---

## Phase 2 — Genome-guided clustering as a validation tool

Goal: ship a working `transcriptome cluster --mode genome-guided` that consumes Phase 1's outputs and emits a `TRANSCRIPT_CLUSTER_TABLE`. This is itself a useful end-user feature (people who have a genome already get cleanly clustered isoforms today), and it generates the ground-truth labels Phase 4 measures de novo against.

### 2.1 Algorithm

For each (contig, strand) partition of `READ_FINGERPRINT_TABLE`:

1. Group rows by `fingerprint_hash`. Each group is a candidate cluster — these are reads whose intron chains agree exactly (modulo the quantum). The cluster key already absorbs SSP variation on UTRs because UTRs sit outside the junction signature.
2. Optional span-coherence filter: drop reads whose 5' end deviates from the cluster median by > `--max-5p-drift` bp (default 25) and whose 3' end deviates by > `--max-3p-drift` bp (default 75). Strand is already in the fingerprint, so 5' and 3' map cleanly onto `span_start` / `span_end` per orientation. Defaults sized from qualitative lab observations (5' TSS fidelity ≈ ~3 nt modal, ≤ ~20 nt outliers; 3' polyA termini scatter wider, up to ~50 nt). The asymmetry matters because conflating the two would either over-filter at the 3' end or under-filter at the 5' end and let alt-TSS isoforms slip into the wrong cluster. A future refinement could replace `--max-3p-drift` with a polyadenylation-signal-anchored window (AAUAAA position ± N), but that's overkill until Phase 4 gives us a concrete reason.
3. Within each cluster, dereplicate the trimmed transcript-window sequences (Layer-0-style hash) and pick the most-abundant unique sequence as the representative. This keeps the abundance prior consistent with Phase 3 and matches the Bayesian argument in roadmap §5.2 — abundance is a stronger signal of correctness than basecaller-emitted quality (per the user's own analysis, Dorado quality scores are a weak prior on per-read accuracy, and the ~0.5% of reads with non-trivial error are still ~50k reads on a 10M dataset). Tiebreak by mean `dorado_quality` only when all sequences in the cluster are singletons; record per-cluster total abundance.

4. **Optional `--build-consensus`.** When set, build a per-cluster consensus on the genome coordinate frame: walk the cluster's alignment blocks position-by-position against the genome, take per-position majority vote weighted by the Layer-0 abundance from step 3, and emit consensus on the cluster representative's sequence. Both this path and the Phase 3 final-stage consensus call into the shared primitive at `constellation/sequencing/align/consensus.py` — same per-position weighted-PWM kernel, different alignment substrate (genome-anchored here, centroid-anchored in Phase 3). Off by default to keep Phase 2 fast for users who just want fingerprint-grouped representatives; on when users want Phase-3-quality cluster sequences without paying the de novo clustering cost.
5. Optionally merge cluster pairs whose junction sets differ only by missing-junction noise (i.e. fingerprint A's junctions are a subset of fingerprint B's, and the reads in A are short enough that the missing junctions plausibly fall outside the alignment). This is a deferred refinement; ship clusters without merging in v1 and let Phase 4 tell us how much it matters.

### 2.2 New module: `constellation/sequencing/transcriptome/cluster_genome.py`

```python
def cluster_by_fingerprint(
    fingerprints: pa.Table,
    reads: pa.Table,
    *,
    alignment_blocks: pa.Table | None = None,    # required if build_consensus=True
    genome: GenomeReference | None = None,       # required if build_consensus=True
    max_5p_drift: int = 25,
    max_3p_drift: int = 75,
    min_cluster_size: int = 1,
    build_consensus: bool = False,
) -> pa.Table:  # TRANSCRIPT_CLUSTER_TABLE
```

Singleton clusters are kept by default (`min_cluster_size=1`) because at this layer a singleton with a well-defined intron chain is just as informative as a deep cluster — it's an isoform observed once. The de novo Phase 3 will be more aggressive about dropping singletons.

### 2.3 CLI dispatch

`transcriptome cluster --mode genome-guided` becomes the first non-stub branch of the cluster verb. It requires an align output dir produced with `--emit-fingerprints`. Detection: check for `read_fingerprints.parquet` in the align output dir; if missing, exit 2 with a clear "rerun align with --emit-fingerprints" message.

### 2.4 Auxiliary outputs

- `clusters.parquet` — TRANSCRIPT_CLUSTER_TABLE
- `cluster_membership.parquet` — long-form (read_id, cluster_id, score)
- `cluster_summary.tsv` — power-user diagnostic, off by default. n_clusters, cluster-size distribution histogram, n singletons, fraction of input reads clustered.
- `cluster.fa` — cluster representatives as FASTA (the highest-quality member's sequence). For genome-guided mode, cluster representatives are real read sequences, not consensus — that comes in Phase 3 / 4.

---

## Phase 3 — De novo Layer 0 + Layer 1 (torch-first)

Goal: implement the kmer-minimizer + abundance-weighted greedy clustering specified in the roadmap (sections 4–5), substituting torch where the roadmap proposes numpy. Plain re-port of the algorithm; the architectural decisions are already settled in the roadmap.

### 3.1 New subpackage: `constellation/sequencing/transcriptome/cluster_denovo/`

Split into modules now, not after the fact. The roadmap's "single flat `cluster.py`" rationale (avoid premature decomposition) was for a small package; Constellation already has a much higher complexity budget and the per-stage tests will benefit from the split.

```
cluster_denovo/
    __init__.py
    dereplicate.py      Layer 0: hash-based exact dereplication
    minimizers.py       canonical rolling hash + minimizer extraction (torch);
                        sorted index construction
    candidates.py       linear scan of sorted index, diagonal-filtered
                        candidate-pair generation, abundance-weighted
                        centroid ordering
    verify.py           edlib HW alignment batch wrapper around align/pairwise.py
    greedy.py           greedy set-cover clustering
    pipeline.py         orchestrator: cluster_transcripts(reads_dir, ...) ->
                        TRANSCRIPT_CLUSTER_TABLE.
                        Calls into align/consensus.py for the final
                        weighted-majority-vote step (centroid-anchored).
```

The consensus primitive itself lives at `constellation/sequencing/align/consensus.py` rather than inside `cluster_denovo/` — it's an alignment-derived primitive (convert a stack of aligned bases into a consensus call via per-position weighted majority vote on a torch.scatter_add_ PWM) that Phase 2's `--build-consensus` path also calls, and that future assembly-polishing / Mode-B-adjacent workflows will call. Keeping it in `align/` next to `pairwise.py` / `cigar.py` matches the natural information flow and avoids cross-sibling imports.

Existing `transcriptome/cluster.py` and `transcriptome/consensus.py` stubs become thin re-exports of the new public APIs (so the public name `cluster_reads` keeps working — `transcriptome/consensus.py::build_consensus` re-exports `align/consensus.py::build_consensus`) or get replaced outright. Pick the re-export form; less churn for any existing in-flight callers.

### 3.2 Torch-first decisions

Per the project invariant "torch-first numerics; numpy only at I/O boundaries":

- Sequence encoding: ASCII bytes → `torch.uint8` (A=0 C=1 G=2 T=3, anything else → 4 = no-op kmer break) via `torch.frombuffer`. Zero-copy from Python `bytes`.
- Rolling polynomial hash: `torch.cumsum` + multiply on a precomputed `base ** k` lookup; canonical via element-wise `min(forward, revcomp)`. Vectorisable across all reads in a batch.
- Minimizer window-min: `torch.nn.functional.max_pool1d` on negated hash sequence (torch lacks `min_pool1d`). Output is `(N, w_count)`; argmin per window via the same trick.
- Sorted index: stack `(mini_hash, seq_id, position)` as a `(K, 3)` int64 tensor, `torch.argsort` on column 0, gather. Lives in CPU RAM at scale (~6–12 GB for 10M reads; CUDA possible later but not the target).
- Candidate accumulation: roadmap says dict of pair → diagonal list. Torch translation is a `(P, 3)` int64 tensor of (seq_i, seq_j, diag) triples, then `torch.unique` + `torch.bincount` for diagonal counting. Tighter memory + parallelisable.
- PWM accumulation: `torch.scatter_add_` on a `(L_consensus, 5)` float64 tensor, weighted by abundance. Embarrassingly parallel across clusters via `concurrent.futures.ProcessPoolExecutor` or `torch.multiprocessing`.

### 3.3 Edlib boundary

`align/pairwise.py` already wraps edlib. `verify.py` uses `pairwise_align(query, ref, mode='HW')` per the existing API; if HW mode isn't yet exposed there, add it now (one-line forward to the underlying `edlib.align(..., mode='HW')`). NW mode for consensus alignment may need adding too. Verification cost is ProcessPoolExecutor-parallel as the roadmap specifies; the worker function lives at module top level for picklability (same pattern as `transcriptome.stages._fused_chunk_worker`).

### 3.4 Iterative refinement

Implement but default to `max_rounds=1`, with `--max-cluster-rounds N` as the CLI knob. Roadmap section 5.6 says abundance-weighted centroid selection should converge in one pass; we measure how often that's true in Phase 4.

### 3.5 Acceleration follow-up

After Phase 3 ships and runs end-to-end on a 2M-read synthetic dataset, profile. If `_extract_minimizers` or candidate accumulation dominates, add a numba JIT path with identical signature. Adding numba to `environment.yml` is a separate PR, motivated by profile data, not assumed up-front.

### 3.6 CLI dispatch

`transcriptome cluster --mode de-novo` becomes the second branch. Inputs: an S1 demux output dir (reads come from there). Outputs: same shape as Phase 2 (`clusters.parquet`, `cluster_membership.parquet`, `cluster.fa` — but here `cluster.fa` carries consensus sequences, not raw reads).

---

## Phase 4 — Cross-validation harness

Goal: when a genome reference is available, run both clustering modes on the same demuxed input and measure agreement. Power-user feature; off by default.

### 4.1 New module: `constellation/sequencing/transcriptome/cluster_validate.py`

```python
def compare_clusterings(
    genome_guided: pa.Table,   # cluster_membership from Phase 2
    de_novo: pa.Table,         # cluster_membership from Phase 3
) -> dict:
```

Returns a dict with:

- `n_reads_both`, `n_reads_only_genome_guided`, `n_reads_only_de_novo`
- **Adjusted Rand Index** + **Normalised Mutual Information** between the two cluster assignments (treat genome-guided as the reference labelling)
- **Per-cluster confusion matrix** (top-K most-discordant cluster pairs)
- **Splitting rate** (fraction of genome-guided clusters that are split across multiple de-novo clusters) — diagnostic for over-clustering
- **Merging rate** (fraction of de-novo clusters that span multiple genome-guided clusters) — diagnostic for under-clustering
- **Per-cluster consensus accuracy** — for the largest N de-novo clusters, edlib-align the consensus against the genome span of the corresponding genome-guided cluster's representative; report distribution of edit distances. This is the "did we recover the actual transcript sequence" check.

Use `core.graph.Network.connected_components` for the splitting / merging analysis (build a bipartite graph between cluster sets, look at component sizes).

### 4.2 CLI dispatch

`transcriptome cluster --mode validate` runs Phase 2 + Phase 3 + comparison. Single output dir with subdirs for each mode + a top-level `validation_report.json` and `validation_summary.tsv`.

### 4.3 Synthetic-data harness

Constellation already ships `transcriptome.simulator` for stress-testing the demux scorer. Extend with `simulate_transcript_panel(n_isoforms, depth_per_isoform, error_rate)` so Phase 4 can run on truly-known ground truth (not just genome-guided "ground truth", which still depends on alignment choices). The simulator output is the same SAM + ground-truth parquet shape, so the same `transcriptome align` + `transcriptome cluster` pipeline runs end-to-end.

---

## Critical files to be modified or created

**Schemas (extend / add):**
- [constellation/sequencing/schemas/alignment.py](constellation/sequencing/schemas/alignment.py) — add `ALIGNMENT_BLOCK_TABLE`, `SPLICE_JUNCTION_TABLE`, `READ_FINGERPRINT_TABLE`.

**New modules:**
- [constellation/sequencing/align/cigar.py](constellation/sequencing/align/cigar.py) — CIGAR + cs:long parsers.
- [constellation/sequencing/align/consensus.py](constellation/sequencing/align/consensus.py) — weighted-majority-vote PWM consensus primitive (torch.scatter_add_); shared between Phase 2 `--build-consensus` and Phase 3 de-novo final stage; positioned for future assembly-polishing reuse.
- [constellation/sequencing/quant/coverage.py](constellation/sequencing/quant/coverage.py) — pile-up / RLE depth producer.
- [constellation/sequencing/quant/junctions.py](constellation/sequencing/quant/junctions.py) — splice-junction aggregator.
- [constellation/sequencing/transcriptome/fingerprints.py](constellation/sequencing/transcriptome/fingerprints.py) — exon-block fingerprint hashing.
- [constellation/sequencing/transcriptome/cluster_genome.py](constellation/sequencing/transcriptome/cluster_genome.py) — Phase 2 genome-guided clustering.
- [constellation/sequencing/transcriptome/cluster_denovo/](constellation/sequencing/transcriptome/cluster_denovo/) — subpackage; five modules per §3.1 (dereplicate/minimizers/candidates/verify/greedy + pipeline). Calls into `align/consensus.py` for the final consensus step.
- [constellation/sequencing/transcriptome/cluster_validate.py](constellation/sequencing/transcriptome/cluster_validate.py) — Phase 4 comparison.

**Modified:**
- [constellation/sequencing/quant/genome_count.py](constellation/sequencing/quant/genome_count.py) — `fused_decode_filter_overlap_worker` gains gated `--emit-blocks` path that calls `parse_cigar_blocks` + `parse_cs_long` on each surviving alignment, returning a fifth shard `alignment_blocks`. No-op when the flag is off.
- [constellation/sequencing/transcriptome/cluster.py](constellation/sequencing/transcriptome/cluster.py), [transcriptome/consensus.py](constellation/sequencing/transcriptome/consensus.py) — stub modules become re-exports of `cluster_denovo.pipeline.cluster_transcripts` / `cluster_denovo.consensus.build_consensus`.
- [constellation/cli/__main__.py](constellation/cli/__main__.py) — add Phase 1 emit flags to `_cmd_transcriptome_align`; replace `_cmd_not_wired("transcriptome cluster")` with a real `_cmd_transcriptome_cluster` that dispatches on `--mode {genome-guided, de-novo, validate}`.
- [constellation/sequencing/align/pairwise.py](constellation/sequencing/align/pairwise.py) — add NW mode if not already present (consensus alignment needs it).
- [constellation/sequencing/transcriptome/simulator.py](constellation/sequencing/transcriptome/simulator.py) — add `simulate_transcript_panel` for Phase 4.
- [CLAUDE.md](CLAUDE.md) — extend the `constellation.sequencing` row of the module index with the Phase 1–4 surface; add an entry to the conventions section explaining the cluster-mode dispatcher pattern.

**Reused (no changes):**
- `core.sequence.ops.kmerize`, `sliding_window`, `hamming_distance` — Phase 3 dereplication / fallback minimizer impl.
- `core.graph.Network.connected_components` — Phase 4 splitting/merging analysis; potentially Phase 3 cluster-merge step in iterative refinement.
- `align/pairwise.py::pairwise_align` — Phase 3 candidate verification + consensus alignment.
- `align/locate.py::locate_substring` — already used in demux; adapter for Phase 3 high-frequency-minimizer suppression if we go that route.
- `core.io.schemas.register_schema` + `pack_metadata` / `unpack_metadata` — every new schema self-registers and round-trips namespaced metadata (`x.transcriptome.<key>`).
- `sequencing.parallel.run_batched` — Phase 3 verify + consensus stages reuse the existing fused-worker / partitioned-shard / `_SUCCESS` / `--resume` infrastructure.
- `sequencing.progress.ProgressCallback` — every Phase 1–4 stage takes the same `progress_cb` arg; CLI wires `StreamProgress` under `--progress` like the demux + align verbs already do.

---

## Verification

**Phase 1:**
- Unit test: hand-crafted CIGAR (`100M50N100M`) → 2 blocks with the expected (ref_start, ref_end). cs:long with one mismatch → `n_match=99, n_mismatch=1` on that block.
- Round-trip test: take a known minimap2 BAM (existing fixture in `tests/`), run `--emit-blocks --emit-junctions --emit-fingerprints`, verify (a) every block's (query_end - query_start) sums per alignment to the read length minus soft-clip, (b) every junction between adjacent blocks has the right donor/acceptor.
- End-to-end: `pytest tests/test_transcriptome_align.py` extended with `--emit-*` invocations on the existing fixture; verify output schemas + non-zero row counts.
- Smoke: actually run `constellation transcriptome align --emit-pileup --emit-junctions --emit-fingerprints` against the lab's mouse-testis demux output, eyeball the top junctions vs Ensembl's annotation for sanity (top junctions on Prm1 / Prm2 / Odf2 should match annotated introns).

**Phase 2:**
- Unit test: hand-crafted fingerprint table with known cluster structure → `cluster_by_fingerprint` produces the expected groupings, span filter drops the planted UTR-noise read.
- End-to-end: run Phase 2 against the real mouse-testis dataset; spot-check that clusters for known-multi-isoform genes (Odf2, Crisp) recover the expected number of distinct fingerprint groups, and that single-isoform genes (Prm1) collapse to one or two clusters.

**Phase 3:**
- Unit tests for each of dereplicate / minimizers / candidates / verify / greedy / consensus per the test list in roadmap §8 ("Testing strategy"), translated to torch tensors.
- Synthetic-data integration: generate 100k reads from 50 known transcripts via `simulate_transcript_panel`, run de novo, verify n_clusters ≈ 50 ± 10% with within-cluster edit distance ≪ between-cluster edit distance.
- Memory check: minimizer-index size on a 1M-read input matches the §5.1 calculation (~600 MB at 80 minimizers/seq).

**Phase 4:**
- On the 100k-read synthetic dataset, ARI(genome-guided, de-novo) > 0.9; consensus-vs-genome edit distance for top-100 clusters < 1% per base.
- On the real mouse-testis dataset, examine the discordant cluster pairs by hand. The point is not 100% agreement (allelic variation will produce real discordances) — it's understanding the failure modes.

**Per-phase runtime targets** (16-core workstation, 10M reads):
- Phase 1 emit-blocks adds ≤ 10% to align stage; emit-pileup adds ≤ 20% on top of that. Junctions + fingerprints aggregations < 2 min.
- Phase 2 < 5 min on top of an align run that already has `--emit-fingerprints`.
- Phase 3 targets per roadmap §8: full Layer 0+1 < 90 min on 10M reads; revisit after profiling.
- Phase 4 < (Phase 2 + Phase 3) + 5 min for the comparison.

---

## Deferred (next plan)

- **Roadmap Layer 2** (transcript-variant grouping into families): connected-component vs centroid-based choice; relaxed-identity reclustering. Decide based on Phase 4's empirical splitting-rate distribution.
- **Roadmap Layer 3** (gene-structure inference): MSA + block decomposition + coverage-based exon/intron boundary detection. Naturally builds on the pile-up + junction infrastructure Phase 1 ships.
- **Re-mapping reads to assemblies** (`map_assembly` orchestrator + `assembly remap` CLI verb): same Phase 1 pile-up / coverage producer applies, but the assembly side has no annotation, so the junction-annotation match column becomes null. Phase 1 already accounts for this.
- **Sex-determination loci diagnostic**: comparing male vs female pile-up tracks. Trivial once the per-sample pile-up panels from Phase 1.6 exist; ship as a `sequencing.diagnostics` verb when there's a concrete dataset to test on.
- **Numba JIT acceleration** of Phase 3 hot loops, contingent on profile data.
