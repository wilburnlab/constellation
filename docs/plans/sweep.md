# The sweep: matching queries against an acquisition in one pass

Status: P0–P5 built on `feat/sweep` (2026-11-03); P6 measurements below; P7 (the two-pass gate
at proteome scale, faster query compilation) waits on those numbers.

## Why

Constellation could compare spectra and extract XICs, but had no engine that matches a set of
hypotheses against a whole acquisition efficiently. The first need is purified or enriched protein
digests, tens to hundreds of peptides and increasingly DIA. For these, FDR calibration has nothing
to stand on, and the question is whether there is a clear elution peak with co-eluting precursor
isotopes and fragments. The same engine has to grow toward proteome scale.

The PI's framing (2026-11-03):
- The engine is one pass over the scans with an active set of queries.
- A query is matched in the scans inside its retention-time interval, and for fragments the
  isolation windows its precursor falls in. No interval means a global XIC.
- Fragment-index-style pruning is a way to avoid pointless work: a peptide with fewer than a couple
  of fragments is not worth scoring.
- Enumerating the species a purified protein can produce is the query generator's job, not the
  engine's.

## What it is

`constellation/massspec/sweep/`:

| Module | Holds |
|---|---|
| `queries` | `compile_queries(XIC_TARGET_TABLE, levels=…)` → `QueryPlan`: per-query arrays (RT interval, precursor m/z, assigned scan) and ions in compressed-row form, isotopes for MS1 and fragments for MS2, in output order; `IonSettings` (isotope offsets may be negative, for labelled peptides) |
| `schedule` | which queries a scan is matched against: the active set per block (RT overlap), scans grouped by isolation window, and `cell_eligible` (exact RT and assigned scan per cell) |
| `kernel` | one group of scans against one vector of ions, two ways: `match_cells` (ions → peaks, `core.signal`'s kernels on padded chunks) and `match_peaks` (peaks → ions sorted by m/z, then the exact predicate); identical results, chosen per group by `match_group` |
| `rules` | `Nearest`, `AllWithin`, `SumWithin` |
| `evidence` | `EvidenceGate`, the map's counts (`count_evidence`) and the reduce's verdict (`decide`) |
| `claims` | peaks matched by more than one (query, ion) cell, and the claims table |
| `reduce` | `Ledger` (the scans read), `Matches`, `QueryBatch`, the `Reducer` protocol (the scoring hook), `QuerySummaryReducer` |
| `engine` | `SweepSettings`, `sweep()`, `SweepResult`; in memory, or a map/reduce on disk over scan-range shards with worker processes |
| `schemas` | the plan, ledger, match, evidence and claim hand-off schemas, and the terminal summary and claims tables |

`bundle.iter_scan_blocks` streams a bundle as blocks of whole scans (with a scan range read from row-group
statistics), and `bundle.scan_range_shards` cuts it into ranges from the footers alone.
`quant.chromatogram.extract_xic` is the XIC entry point on top: any levels, a tolerance per level, a rule,
the gate, claims, a summary and workers. `extract_xic_scan_major` (Mode A) is one level of it, and
`massspec chromatogram extract` uses it.

## Decisions

- **Scans in order outside, the match direction chosen inside.**
  - The outer loop streams scans in stored order: one sequential read, bounded memory, every hypothesis for
    a scan present together (claims now, mixture fits later), and it works for profile points.
  - Inside, a group of scans that share their queries (one DIA window in a block, or every MS1 scan of it)
    is matched ions → peaks when its ions are fewer than `crossover` × its peaks per scan, and peaks → ions
    otherwise. `crossover` is 0.25, measured (below): peaks → ions is never slower above that, and 2–11×
    faster at proteome DIA ratios. The peak-major kernel takes candidates in a window widened by one part per million of the
    tolerance and then applies the exact test, including `nearest_within_tolerance`'s left-neighbour tie
    and duplicate-m/z handling, so the two give the same rows bit for bit (tested on random blocks with
    ties, duplicates and edge peaks).
  - Query-major over the mass-sorted index (Mode B) stays for random access.
- **The active set per block.** It is a vectorized interval-overlap test over all queries, with no
  enter/leave events. Within a block, scans are grouped by exact (float64) isolation bounds; each group
  shares one query vector. A run-wide sort by window is not done, because it would mean re-sorting the
  run.
- **The peak-major index is per group, not a two-level bucket layout.** The plan named ScribeTwo/Sage's
  two-level layout. Sorting each group's active ions by m/z gives the same property (no per-scan
  rebuild for DIA) with exact comparisons at window edges, and its build is amortized over the block's
  visits of the window. The bucket layout stays the option if measurements show the index build
  dominating.
- **Map/reduce, not finalizing a query when it leaves the active set.** A global XIC never leaves; a
  query's trace crosses shards; claims are per scan. So shards (contiguous scan ranges cut from the
  footers) write their matches sorted by query, and the reduce takes contiguous ranges of queries sized
  by match counts. Determinism: shard and bucket boundaries depend on the data alone, sums run in m/z
  order within a scan, every sort has a total key, and workers use one torch thread. Output is
  byte-identical for any worker count, block size, shard size or kernel (tested).
- **Memory is bounded on both paths.** A map shard writes its matches (and evidence and claims) out in
  sorted runs of `run_rows` (4M by default) instead of holding the shard; a worker peaks near 3 GB. A sweep
  run in memory stops holding past `memory_rows` matches (8M) and carries on on disk: what it holds becomes
  the first run of a single shard (and is freed once written), and the rest of the scans follow, inline, in
  a temporary directory that lives as long as its result, so outputs are read part by part. `memory_rows`
  was 32M until the timing table below showed a held match costing 200–500 bytes at peak, not the 57 of
  its arrays (sorting copies them, and the reducers build their tables beside them): one process reached
  7–11 GB. The first measurement session ended when WSL's OOM killer took
  a process holding 29.4 GB, every match of 139,067 targets across the wide-window run, on a 31 GB VM;
  neither guard existed then. `scripts/bench-sweep.py time` also kills a configuration whose processes
  together pass `--max-rss-gb`. The second session ended in a host crash (Windows Kernel-Power 41) with
  each run at about 3.4 GB and no OOM, which points at the machine rather than the sweep. The timing table
  below was then run at a lighter load (at most 4 workers and 4 torch threads, 30 s between
  configurations, the 20 GB guard) and finished.
- **Isolation gating.** The test is `lower <= precursor_mz <= upper` on each scan's float64 bounds, via
  two `searchsorted` calls over the active queries sorted by precursor m/z. A scan without bounds, or a
  query without a precursor m/z, passes. Float32 window ids are for reporting only.
- **Zero-filling** uses the scans that have peaks at the level, as Mode A did, through the same
  schedule the map used (`QueryBatch.eligible`). The ledger also holds peakless scans for a scorer that
  wants every scan.
- **Two behaviour changes from Mode A, both fixes.**
  - A target with a sequence and a charge but no precursor m/z has one computed. Mode A matched such
    targets' fragments in every window.
  - `assigned_scans_only` restricts fragments only. At MS1 it emptied the output, while the CLI said it
    was ignored there.

  Everything else is bit-identical to Mode A, held to by a frozen copy of the old implementation
  (`tests/_xic_scan_major_oracle.py`) over planted DIA, staggered-DIA and DDA runs and every option. That
  includes NaN, not null, on unmatched values and MS1 isolation bounds, and a NaN RT bound gating nothing
  (Mode A bisected on it).
- **The gate counts bonds, not ions.** A scan's evidence for a query is the number of distinct
  (ion type, position) pairs with an ion within tolerance of one of the scan's `top_peaks` most intense
  peaks, capped by the distinct peaks that matched them. A query passes with a run of `min_consecutive`
  such scans, each with at least `min_fragments`, within `max_gap_s`. Counting runs in the map, the
  verdict in the reduce. Gated reducers see only passing queries, with their whole traces.
- **Per-level tolerances** go in the trace's schema metadata as `x.massspec.extraction_tolerance.ms<L>`
  when a trace spans levels. The unqualified key appears when one tolerance was used, so single-level
  output is unchanged, and Counter reads the MS1 key first.

## Measurements (2026-11-03, this machine: 12 cores, WSL2; inputs copied into scratch)

Runs:
- **Purified:** BSA and ¹³C/¹⁵N lysin, ion-trap DIA, 100 windows of 8 m/z, 60 min; 1,160 MS1 and 116,000
  MS2 scans, 84M peaks.
- **U2OS GPF 900–1000:** 49 windows; 1,637 MS1 and 80,213 MS2 scans, 109M peaks.
- **U2OS wide window:** 350–1200, 8 m/z; 2,242 MS1 and 237,652 MS2 scans, 294M peaks.

Targets:
- **Purified:** the UniProt BSA (P02769) and lysin (P04552) entries digested by `precursors_from_fasta`
  (trypsin, 2 missed cleavages, charges 2–4, variable Met oxidation and Cys carbamidomethylation), lysin
  as `<13C><15N>`. 1,745 targets and 442 reversed decoys; the labelled peptides have none.
- **U2OS:** the chromatogram library's precursors at their library RT, and reversed decoys
  (`scripts/bench-sweep.py targets`).

**Same rows as Mode A on real data.** On the GPF run's first 30,000 scans (38.5M peak rows), the frozen
target-by-target extractor and the sweep gave identical tables:
- MS1 and MS2, 518 targets at ±120 s;
- MS1 and MS2, 180 targets as global XICs.

**Where a sweep spends its time.** The whole GPF run, 518 targets, both levels, ±120 s, one process: 9.8 s.
- 3.5 s reading and decoding the bundle;
- 2.0 s matching;
- 1.7 s building and sorting the trace.

The old extractor took 32 s on 35 % of the run, with the whole run's table held in memory. On the
purified run, 30 BSA queries as global XICs take 4.5 s and 1 GB.

**The gate at 0.35 Da.** Pass rates, targets / reversed decoys, by evidence peaks per scan, bonds per
scan (rows) and scans in a row (columns), on the purified run:

| top peaks | bonds | 1 scan | 2 | 3 | 5 |
|---|---|---|---|---|---|
| all | 2–8 | 91 % / 87 % | same | same | same |
| 150 | 8 | 83 / 74 | 67 / 53 | 50 / 39 | 29 / 19 |
| 50 | 4 | 80 / 71 | 60 / 49 | 45 / 35 | 26 / 19 |
| 50 | 6 | 30 / 17 | 18 / 6 | 13 / 2 | 8 / 1 |
| 50 | 8 | 13 / 1 | 9 / 0 | 7 / 0 | 5 / 0 |
| 20 | 4 | 25 / 14 | 13 / 6 | 10 / 2 | 5 / 1 |
| 20 | 5 | 12 / 3 | 7 / 1 | 6 / 0 | 3 / 0 |

- **With every peak as evidence the count saturates.** About 710 peaks over about 1,300 Th put a peak
  within 0.35 Th of a given m/z nearly half the time, so most of a peptide's bonds match by chance in
  every scan. The 9 % of targets that never pass have no MS2 scan they could be matched in.
- **Most targets are digest species that are not in the sample**, so the decoy rate is the measure;
  13 % of targets against 2 % of decoys is a gate that keeps what is there.
- **The defaults are the 50 / 6 / 3 cell.** In the full sweep they pass 230 targets and 11 of 442 decoys.
  What passes is the digest's known BSA peptides, often in long runs: KVPQVSTPTLVEVSR 66 scans,
  RHPYFYAPELLYYANK 67, LGEYGFQNALIVR 40. Their carbamidomethyl-C forms pass too (the sample was
  alkylated), and so do 32 labelled lysin peptides (TPVVGDYTR, VQIIAGFDR, SWHYVEPK, YMQTHWANYMLWINK, …).
- **The purified run is one ion-trap run.** On high-resolution fragments chance matching is far lower and
  these defaults are strict; re-measure there before relying on them.
- **Cost.** The gated sweep of the 2,187 purified queries as global XICs takes 83 s and 10 GB in memory,
  because every nearest match of every query in every scan is held. Counting with one-dimensional keys
  halved it from 177 s. On disk (`workers > 1`) memory is bounded by a shard.

**Timing and memory.** Nearest rule, Mode A's tolerances, inputs on the local disk. Seconds of sweep
(compile apart), and the peak resident memory of all the run's processes together.

| run | queries | ions (MS1 + MS2) | levels, RT | matches | workers | cells | peaks | memory |
|---|---|---|---|---|---|---|---|---|
| purified | 2,187 | 155,540 | MS2, global | 51.4M | 1 | 174 | 151 | 7.4 GB |
| | | | | | 2 | 90 | 91 | 6.4 GB |
| | | | | | 4 | 83 | 63 | 11.8 GB |
| GPF 900–1000 | 1,719 | 20,628 + 119,676 | both, global | 129.5M | 1 | 394 | 360 | 7.5 GB |
| | | | | | 4 | 178 | 180 | 12.2 GB |
| GPF 900–1000 | 6,435 | 77,220 + 447,314 | both, ±120 s | 20.4M | 1 | 44 | 44 | 10.6 GB |
| | | | | | 4 | 32 | 32 | 11.7 GB |
| wide window | 139,067 | 1.67M + 6.84M | both, ±120 s | 476.2M | 4 | 545 | 541 | 15.8 GB |

- **The kernel matters less than the rows.** Peaks → ions was never slower, but whole sweeps move by at
  most 25 %, because matching is a minority of the time. At proteome scale the three kernel settings
  are within 3 % of each other, at 476M matches. The time follows the matches written, which is what
  the two-pass gate is for.
- **The crossover, from the kernels alone** (one group of 4 or 16 scans, random peaks, Nearest):
  peaks → ions is 1.0–1.5× faster from 0.25 to 1 ion per peak, 1.4–2.1× at 2–4 for 0.35 Da, and 2–11×
  at 2–17 for 20 ppm. Below a quarter of an ion per peak cells → peaks is up to 1.2× faster, at under
  0.6 ms a group either way. `crossover` went from a guessed 4 to 0.25. With 4, `auto` had been choosing
  cells → peaks on the purified run (180 s, against 151 s forced to peaks).
- **Workers.** Four sweep a whole run 2.0–2.8× faster than one (purified 151 → 63 s, GPF 360 → 180 s),
  and a short RT-gated sweep 1.4×. Each worker peaks near 3 GB, so four use 12–16 GB together. The CLI's
  default went from the cores up to 8 (about 25 GB) to the cores up to 4. The library default stays one
  process.
- **One process held too much.** With `memory_rows` at 32M one process reached 7–11 GB, and the
  purified run's move to disk kept the 32M matches it had held for the rest of the sweep. After both
  fixes:

  | run, one process | before | after |
  |---|---|---|
  | purified, MS2 global (`auto`) | 180 s, 7.4 GB | 125 s, 3.0 GB |
  | GPF ±120 s, both levels | 44 s, 10.6 GB (in memory) | 54 s, 3.0 GB (moves to disk at 8M) |

- **Compiling the queries took longer than sweeping them.** The 139,067 wide-window targets took
  1,677 s to compile against a 545 s sweep, nearly all of it MS1 isotope envelopes. Each peptide's
  envelope was built once per charge in `charge_range`, and each element's polynomial power (FFT
  repeated squaring) once per peptide. `peptide.envelope.neutral_envelope` + `envelope_mz` now build
  a peptide's envelope once and put it at each charge, and `core.chem.isotopes` caches each
  (element, count) power. The plan is bit-identical, NaN abundances included, on real targets (the
  purified and GPF sets, natural and with negative isotope offsets), and tests hold both to the
  uncached computation. 180 s and 1.7 GB now, 9.3×; the GPF set 92 → 9 s.

## Still to do

- P7, only if the measurements call for it:
  - the two-pass gate at proteome scale (evidence only, then traces for the survivors). The timing table
    calls for it: at 139k queries the sweep's time follows the 476M matches it writes;
  - query compilation without Python per target. Two steps are done, both exact: ProForma parsing
    (2,187 purified-run targets took 10 s, nearly all of it lark's Earley parser at about 10 ms a string;
    `parse_proforma` now reads the simple form — labels, residues with UNIMOD accessions, signed masses
    or plain names, terminal mods, a charge — with a regular expression into identical objects, about
    200 times faster, and leaves the rest to the grammar, `tests/test_proforma_fast_path.py`), and the
    isotope envelopes above. What is left, 1.3 ms a target at proteome scale, is the per-target loop
    itself;
  - the bucket index layout, if a profile shows the per-group index build mattering. Not measured on
    its own yet; the kernel is a minority of a sweep's time.
- The purified-sample scorer (elution peaks, co-elution evidence, a verdict rule) as a gated reducer,
  and output as an experiment directory the browser reads: the next plan.
- Profile-points rule; isolation gating by the whole isotope envelope; mixture fits on a scan's
  competing queries.
