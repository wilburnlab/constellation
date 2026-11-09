# The sweep: matching queries against an acquisition in one pass

Status: P0–P6 built on `feat/sweep` (2026-11-03; vocabulary renamed 2026-11-04). Plumbing for a scorer
(2026-11-07, `feat/query-plumbing`): channels from one generator with abundances (a library's
predictions, or a uniform series) and top-N, MS1 at each target's own charge, the reducer's cell view
and scan context, reducers of several outputs, sweeping a scan or RT range, and the review's sweep
defects ("Plumbing for a scorer" below). The sweep read precursor-major (2026-11-09,
`feat/precursor-major-sweep`): a bundle's points also stored by isolation window, a unit of queries
read and reduced at a time with nothing shuffled, the two-pass gate (P7) inside a unit, and the match
path's sorts and nearest rule made cheaper ("Precursor-major" below); for a DIA search this replaces
the map/reduce below.

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

## Vocabulary

These words have one meaning each in the sweep's code, schemas and documentation (PI, 2026-11-04).

| term | meaning |
|---|---|
| point | One stored (m/z, intensity) value of a scan, in any bundle. A *profile point* is a sample on the instrument's m/z grid; a bundle's points are whatever it stores (`bundle.point_paths`). |
| peak | A centroid, made from profile points by the instrument, Thermo's centroider or Constellation's. A peak list's points are peaks; a profile point is never called a peak. |
| query | One hypothesis being matched: a precursor (modified sequence, charge, precursor m/z) with its retention-time interval, compiled from one extraction target. |
| channel | One m/z at which a query expects signal, with what it is: in MS1 isotope k of the precursor at charge z, in MS2 a fragment ion (ion type, number, charge, loss). Counter uses the word the same way. |
| cell | One channel in one scan: an entry of a query's scans × channels table, the place a match can occur. Counter's `(S, C)` cells are the same thing. |
| (scan, query) pair | One query in one scan, standing for all its cells there; what RT and isolation eligibility is decided for (`schedule.pair_eligible`). |
| match | A cell and the point or points its rule assigns it: `nearest` gives at most one, `all` every point within tolerance, `sum` one aggregated value. |
| candidate | A (point, channel) pair within tolerance, before the rule decides; only the point-driven kernel makes them. |
| window group | The scans of one block that share an isolation window (or every MS1 scan of the block), and so the same channels. |
| channel-driven, point-driven | The two kernels, named for which list drives the loop (below). |

"Ion" appears only in mass spectrometry's own compounds (fragment ion, y ion, ion type, precursor ion)
and for counted ions (ion counts, ions per point), never as a count of expected values. The evidence
gate says peaks on purpose: it is defined on a scan's most intense centroids.

Before 2026-11-04 the code and the P0–P6 commits said *ion* for channel, *cell-major* (`kernel="cells"`)
for channel-driven, *peak-major* (`kernel="peaks"`) for point-driven and `peak_claims` for
`point_claims`; the timing table below was measured under those names. The sweep's schemas went to
version 2 with the rename.

## What it is

`constellation/massspec/sweep/`:

| Module | Holds |
|---|---|
| `queries` | `compile_queries(XIC_TARGET_TABLE, levels=…, channels=ChannelSettings(…))` → `QueryPlan`: per-query arrays (RT interval, precursor m/z, assigned scan) and `Channels` in compressed-row form, isotopes for MS1 and fragments for MS2, in output order (isotope offsets may be negative, for labelled peptides) |
| `schedule` | which queries a scan is matched against: the active set per block (RT overlap), window groups, and `pair_eligible` (exact RT and assigned scan per (scan, query) pair) |
| `kernel` | one window group against its channels, two ways: `match_by_channel` (channel-driven: `core.signal`'s kernels on padded chunks of points) and `match_by_point` (point-driven: every point looked up in the channels sorted by m/z, then the exact predicate); identical results, chosen per group by `match_group` |
| `rules` | `Nearest`, `AllWithin`, `SumWithin` |
| `evidence` | `EvidenceGate`, the map's counts (`count_evidence`) and the reduce's verdict (`decide`) |
| `claims` | points matched by more than one cell, and the claims table (`point_claims`) |
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
  - Inside, a window group is matched channel-driven when its channels are fewer than `crossover` × its
    points per scan, and point-driven otherwise. `crossover` is 0.25, measured on centroids (below):
    point-driven is never slower above that, and 2–11× faster at proteome DIA scale. On profile points
    the deciding count for `nearest` is different (below). The point-driven kernel takes candidates in a
    window widened by one part per million of the tolerance and then applies the exact test, including
    `nearest_within_tolerance`'s left-neighbour tie and duplicate-m/z handling, so the two give the same
    rows bit for bit (tested on random blocks with ties, duplicates and edge points).
  - Channel-driven matching is the definition: it calls `core.signal`'s kernels as Mode A did, and the
    point-driven kernel is held to it. A new rule is written channel-driven first (a channel's tolerance
    bounds are a contiguous slice of the scan's points); a point-driven form follows only if measured
    worth it.
  - Query-major over the mass-sorted index (Mode B) stays for random access.
- **The active set per block.** It is a vectorized interval-overlap test over all queries, with no
  enter/leave events. Within a block, scans are grouped by exact (float64) isolation bounds; each group
  shares one query vector. A run-wide sort by window is not done, because it would mean re-sorting the
  run.
- **The point-driven index is per window group, not a two-level bucket layout.** The plan named
  ScribeTwo/Sage's two-level layout. Sorting each group's channels by m/z gives the same property (no per-scan
  rebuild for DIA) with exact comparisons at window edges, and its build is amortized over the block's
  visits of the window. The bucket layout stays the option if measurements show the index build
  dominating.
- **Map/reduce, not finalizing a query when it leaves the active set.** (Superseded for DIA search on
  2026-11-09, "Precursor-major" below: the reason given here, that a query's trace crosses shards "in
  any parallel scheme", holds for scan-range shards, not for DIA MS2, where a window over the whole
  gradient holds every scan a query can be matched in; the cost of the shuffle at proteome scale was
  about 16 bytes on disk per match, some 100 GB for one GPF fraction against the human proteome.)
  A global XIC never leaves; a query's trace crosses shards; claims are per scan. So shards (contiguous scan ranges cut from the
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
- **Zero-filling** uses the scans that have points at the level, as Mode A did, through the same
  schedule the map used (`QueryBatch.eligible`). The ledger also holds scans without points for a scorer
  that wants every scan.
- **Two behaviour changes from Mode A, both fixes.**
  - A target with a sequence and a charge but no precursor m/z has one computed. Mode A matched such
    targets' fragments in every window.
  - `assigned_scans_only` restricts fragments only. At MS1 it emptied the output, while the CLI said it
    was ignored there.

  Everything else is bit-identical to Mode A, held to by a frozen copy of the old implementation
  (`tests/_xic_scan_major_oracle.py`) over planted DIA, staggered-DIA and DDA runs and every option. That
  includes NaN, not null, on unmatched values and MS1 isolation bounds, and a NaN RT bound gating nothing
  (Mode A bisected on it).
- **The gate counts distinct fragment ions, not channels.** A scan's evidence for a query is the
  number of distinct (ion type, position) pairs, an ion at any charge or with any loss counting once,
  with a channel within tolerance of one of the scan's `top_peaks` most intense peaks, capped by the
  distinct peaks that matched them. Complementary b and y ions of one bond count twice: they sit at
  different m/z, so each is a match of its own (these words said "bonds" until 2026-11-07; the
  behaviour was always this). A query passes with a run of `min_consecutive`
  such scans, each with at least `min_fragments`, within `max_gap_s`. Counting runs in the map, the
  verdict in the reduce. Gated reducers see only passing queries, with their whole traces.
- **Per-level tolerances** go in the trace's schema metadata as `x.massspec.extraction_tolerance.ms<L>`
  when a trace spans levels. The unqualified key appears when one tolerance was used, so single-level
  output is unchanged, and Counter reads the MS1 key first. `extract_xic` stamps them on every output
  itself (`SweepResult.metadata`), so a trace written from a notebook carries them as the CLI's does.
- **The reduce builds only what is eligible, and its buckets are cut by what it builds.** Eligible
  (scan, query) pairs come from per-query ranges over each window group's scans in time order, never
  a queries × scans product filtered afterwards; the summary takes their counts without building
  them, and the gate builds pairs only for queries with evidence (a query without any decides the same
  on none). Buckets on disk weigh each query by its matches, plus its pairs when a gate walks them and
  every eligible cell when a reducer zero-fills (`Reducer.zero_fills`), so unmatched queries no longer
  pile into one bucket.
- **Review of the branch (2026-11-04), nine defects fixed**, each with a test that fails without it:
  the queries × scans product above (139,067 unmatched targets over 2,242 scans asked for 311M
  entries); evidence keys packed with a `max + 1` stride, which collided once isotope offsets went
  negative, so the same scans passed in separate blocks and failed in one; `query * 16 + charge`,
  which gave a charge-17 query's evidence to the next query; `n_shared_matches` counting a `sum`
  match once per point it summed; a sweep on disk of an acquisition with no points failing; the Python
  API leaving the tolerances out of its outputs, so Counter clamped to 20 ppm instead of a trace's 8;
  `--use-index` returning success while ignoring the gate, `--only-passing`, `--mz-axis`, `--claims`,
  `--summary` and `--work-dir` (now refused); `--ms1-tolerance 0` read as 20 ppm through `or` (and
  `--min-consecutive-scans 0` as 3; it is now refused, as is a negative tolerance); and
  `bench-sweep.py gate` reporting a spilled sweep's grid bucket by bucket. The gate measurements below
  are whole: every row of the grid they come from counts all 1,745 targets and 442 decoys.

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

| run | queries | channels (MS1 + MS2) | levels, RT | matches | workers | channel-driven | point-driven | memory |
|---|---|---|---|---|---|---|---|---|
| purified | 2,187 | 155,540 | MS2, global | 51.4M | 1 | 174 | 151 | 7.4 GB |
| | | | | | 2 | 90 | 91 | 6.4 GB |
| | | | | | 4 | 83 | 63 | 11.8 GB |
| GPF 900–1000 | 1,719 | 20,628 + 119,676 | both, global | 129.5M | 1 | 394 | 360 | 7.5 GB |
| | | | | | 4 | 178 | 180 | 12.2 GB |
| GPF 900–1000 | 6,435 | 77,220 + 447,314 | both, ±120 s | 20.4M | 1 | 44 | 44 | 10.6 GB |
| | | | | | 4 | 32 | 32 | 11.7 GB |
| wide window | 139,067 | 1.67M + 6.84M | both, ±120 s | 476.2M | 4 | 545 | 541 | 15.8 GB |

- **The kernel matters less than the rows.** Point-driven was never slower, but whole sweeps move by at
  most 25 %, because matching is a minority of the time. At proteome scale the three kernel settings
  are within 3 % of each other, at 476M matches. The time follows the matches written, which is what
  the two-pass gate is for.
- **The crossover, from the kernels alone** (one group of 4 or 16 scans, random peaks, Nearest):
  point-driven is 1.0–1.5× faster from 0.25 to 1 channel per point, 1.4–2.1× at 2–4 for 0.35 Da, and
  2–11× at 2–17 for 20 ppm. Below a quarter of a channel per point channel-driven is up to 1.2× faster,
  at under 0.6 ms a group either way. `crossover` went from a guessed 4 to 0.25. With 4, `auto` had been
  choosing channel-driven on the purified run (180 s, against 151 s forced point-driven).
- **On centroids, few channels do not make channel-driven cheap.** It copies a group's scans into a
  padded matrix before searching, a pass over every point, so at 0.02–0.1 channels per point both kernels
  take 0.3–0.5 ms a group.
- **On profile points channel-driven wins for `nearest`** (2026-11-04). The purified run was acquired
  in profile; 402 of its scans (all 100 windows), read from the `.raw` both as Thermo's centroids and as
  the stored profile points, the 2,187 queries, 0.35 Da, one thread, the kernel alone:

  | points | per scan | channels per point | rule | channel-driven | point-driven |
  |---|---|---|---|---|---|
  | Thermo's centroids | 753 | 1.9 | nearest | 100 ms | 68 ms |
  | | | | all | 123 ms | 56 ms |
  | profile points | 19,500 | 0.07 | nearest | 279 ms | 677 ms |
  | | | | all | 700 ms | 667 ms |

  A profile peak spans many grid points, so about ten lie within tolerance of each channel; point-driven
  collects all ten as candidates, sorts them and `nearest` keeps one, while channel-driven's one binary
  search per cell lands on the nearest directly. `all` wants every candidate, and the two tie. `auto`
  chose right on both, but only because these profile scans sit below the crossover: for `nearest` the
  deciding count is the points within tolerance of a channel, not channels per point, and at proteome
  scale on profile points (over 0.25 channels per point) `auto` would choose point-driven, 3.9× slower at
  0.5 in a synthetic test. Point-driven also holds a group's candidates at once, where channel-driven
  works in chunks of about 2M elements. Nothing sweeps profile points yet; the choice is to be settled
  with the profile rule.
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

## Plumbing for a scorer (2026-11-07, `feat/query-plumbing`)

Built before any scoring, because a scorer is a reducer and would have inherited what it reads.
Scoring, FDR and retention-time calibration are the next plans.

### What a reducer gets

- **The cell view.** `QueryBatch.cells(level, fill=True, queries=None)` gives every cell a query
  could have matched: its channels in each scan it was eligible for. Each cell has one row per match,
  plus one row (`matched=False`, intensity 0) for an eligible cell with none. Rows are in output order:
  query, scan, channel, point. Stored on each row: the ledger row, the channel, the point and its m/z
  and intensity. Derived from them: the channel's theoretical m/z, abundance, identity and evidence
  key, the error signed observed − theoretical in Da and ppm, and the scan's number, time, TIC and
  injection time. `Cells.table()` is `SWEEP_CELL_TABLE`. Under the `all` rule a cell keeps every
  point it was given; the private join the trace reducer used kept only the last, and the trace
  reducer is now built on the cell view, its output unchanged byte for byte.
- **Scan context.** The ledger carries each scan's TIC and injection time from the bundle's scan
  metadata (NaN where there is none, and for a table of points).
- **Several outputs.** A reducer either declares `outputs` (name → schema) and returns a mapping, or
  keeps `name` + `schema` and returns one table. On disk each output is `reduce/<name>/`. Every output
  is held to its schema by name at reduce time: a missing or extra column is refused, naming the
  reducer. Until now a disk write bound columns by position, so two float columns returned in
  another order were written swapped. Two reducers may not share an output name.
- **The same plan on disk as in memory.** A plan on disk (a spilled sweep, or workers) was rebuilt
  from the query table alone: the targets lost every column but the query's, and `ChannelSettings`
  went back to their defaults. `QueryPlan.write` / `read` now keep the whole target table, the
  settings and the abundance source's record.
- **Part of a run.** `sweep(..., scan_range=(first, last))` or `rt_range=(t0, t1)`, inclusive. Only
  the scans read are eligible, so zero-filled traces, the gate and the summary cover them alone, and
  every output records the range (`x.massspec.sweep.scan_range`). A general feature (PI); the
  calibration pass is expected to use calibrant queries rather than a range.

### Channels: one generator, one abundance convention

- **`massspec.peptide.channels`** makes the channels for every consumer: the sweep's queries, the
  indexed extraction (Mode B), `spectra.annotate` (the browser's ladders, annotation and trace
  targets) and the consensus basis. Each used to build its own, with its own defaults. Isotopes are
  matched at MS1 and fragment ions at MS2 and above; a per-level mapping is where MS3 (reporter ions,
  say) would go. Counter keeps its own, frozen for its revision.
- **Abundance is a channel's share of its group's whole modelled distribution**, computed before any
  selection. A group is a (query, charge) for isotopes and the query for fragments. 0 means modelled and
  predicted absent; NaN means not modelled. An isotope's abundance is its share of the full envelope,
  so `n_isotopes` no longer changes it.
- **Fragment sources.** `UNMODELLED` (NaN, the default); `Uniform`, the classic equal-intensity b/y
  series of pre-Prosit scoring, with optional weights per ion type, charge and loss; and
  `library.fragment_abundance.LibraryFragments`, a predicted or empirical library's intensities:
  - joined exactly on (precursor, ion type, bond, charge, loss);
  - 0 for an ion inside the coverage the predictor recorded (`x.massspec.prediction.fragment_coverage`)
    that it did not report, NaN outside it;
  - streamed from a library directory, keeping only the precursors compiled;
  - a loss the model covers gets a channel at every bond: the model, not the loss rules, says where
    its losses are;
  - a row's m/z must be the ladder's (1e-9 relative for our own libraries, 5 ppm otherwise), and an
    intact ion inside the settings that matches no channel is refused, since the library and the query
    then describe different molecules.
- **Top-N** (`top_isotopes`, `top_fragments`) keeps a group's N most abundant channels in the group's
  own order, and never changes an abundance. The top isotopes of a heavy peptide need not include
  M+0. Top fragments need a source that ranks them, so they are refused under `UNMODELLED` or
  `Uniform` (the browser's "longest y ions first" stays a display rule in `spectra.annotate`).
- **MS1 at each target's own charge by default** (PI). `charge_range` asks for every target at each
  charge in a range, the evidence for an analyte across its charges, which is what Counter wants. The
  old default, (1, 4) for every target, replaced a target's own charge: a 5+ precursor had no MS1 at
  5+, and a peptide's 2+ and 3+ precursors shared their channels. A target with no charge falls back
  to 1–4.
- **Times on a model's clock.** `XIC_TARGET_TABLE` v3 carries `rt_library` and `rt_scale`.
  `rt_center` is set only on the run's own clock (`observed`, or unstated), and an `rt_window` around
  model-scale times is refused until calibration lands.

### Defects fixed (from the 2026-11-06 review)

The plan on disk and outputs by name (above); NaN retention times (one scan without a time made a
block's bounds NaN and left every query of its level inactive for the block; now `nanmin` / `nanmax`,
and a scan without a time is never eligible); an empty isotope selection or a fragment charge ceiling
below 1 refused at `ChannelSettings`; a table of points too large for memory refused ("write it as a
bundle"); the gate's wording (it counts distinct fragment ions, not bonds; above); and the two costs
below.

### Measurements

- **Compiling queries is vectorized.** Distinct sequences are parsed once, envelopes and ladders
  built once, and nothing is done per target in Python. Envelopes come many at a time from one
  Fourier transform (`core.chem.isotopes.isotope_distributions`), with masses identical to the
  single-peptide function's to the bit: Python 3.12's `sum` is compensated, and the batched mass
  repeats that sum. On 209,952 targets of 1,000 human proteins (`bench-sweep.py compile`):

  | Levels | Before | After |
  |---|---|---|
  | MS1 | 174 s | 4.8 s |
  | MS2 | 8.9 s | 7.4 s |
  | both | — | 9.4 s |

- **The reduce reads its own row groups.** Every bucket of a sweep on disk decoded whole 2¹⁸-row
  groups of every sorted run, growing with the square of the matches. Runs are now written in
  groups of `run_row_group_rows` (2¹⁵), and a bucket reads only the groups whose query statistics
  overlap it. On a test sweep of 7 runs and 6 buckets, 1.1× the matches are decoded, against 5.8×.
- **Eligibility is indexed once.** Grouping scans by isolation window at every call was windows ×
  scans, quadratic on DDA. A `WindowIndex` is built once per ledger and level, and eligibility is one
  vectorized search over it with exact integer keys. On a 100,000-scan DDA ledger, 5.5 s a call
  becomes 0.35 s (0.17 s once built), with the same pairs.
- **Real data: the full ladder against the top 10 predicted fragments.** The purified BSA/lysin run
  (bovine serum albumin and *Haliotis* egg lysin; CID, ion trap, 0.35 Da) against its search space as JChronologer predicts it: 3,512 precursors,
  1,756 targets and as many interior-reversed decoys. `chromatogram extract --level 2 --gate`, 4 workers:

  | Channels | Sweep | Trace | Default gate (50 peaks / 6 ions / 3 scans) |
  |---|---|---|---|
  | full b/y ladder, fragment charge ≤ 2 | 237 s | 83.1M rows, 3.2 GB | 231 targets, 34 decoys |
  | top 10 predicted (`--fragment-abundance library --top-fragments 10`), charge ≤ 3 | 60 s | 17.7M rows, 676 MB | 96 targets, 0 decoys |

  Every query the top-10 gate passed, the ladder's passed too. The top-10 output is the same bytes from
  one process as from 4 workers, and the same tables, value for value, as a sweep held wholly in
  memory (whose file differs in its row groups alone).

  Both levels (`--level both`, top 10) took 62 s, adding 9.45M MS1 matches to the 17.7M of MS2, with
  the same verdicts (the gate reads MS2); each target's MS1 sat at its own charge, one M+0 channel per
  target at its precursor m/z.

  The gate grid (`bench-sweep.py gate`, targets / decoys passing, of 1,756 each; 1,589 of each are
  eligible in some MS2 scan):

  | Gate (top peaks / ions / scans) | Full ladder | Top 10 |
  |---|---|---|
  | 50 / 6 / 3 (the default) | 231 / 34 | 96 / 0 |
  | 50 / 8 / 2 | 161 / 2 | 47 / 0 |
  | 50 / 6 / 2 | 306 / 71 | 118 / 0 |
  | 50 / 5 / 3 | 374 / 171 | 137 / 0 |
  | 50 / 5 / 2 | 533 / 354 | 161 / 0 |
  | 50 / 4 / 3 | 804 / 685 | 177 / 10 |
  | 50 / 4 / 2 | 1,057 / 974 | 230 / 25 |
  | 150 / 6 / 3 | 1,471 / 1,424 | 201 / 9 |
  | 150 / 5 / 3 | 1,574 / 1,568 | 318 / 76 |
  | 150 / 6 / 1 | 1,589 / 1,587 | 404 / 140 |
  | 20 / 6 / 2 | 88 / 1 | 47 / 0 |
  | 20 / 5 / 3 | 101 / 4 | 51 / 0 |
  | 20 / 4 / 3 | 171 / 34 | 89 / 1 |
  | 20 / 3 / 3 | 458 / 304 | 140 / 6 |

  With no peak cap every target and decoy passes the ladder's gate, and nearly every one the top 10's.

  Ranking the ions by the prediction separates targets from decoys at least as well as the whole
  ladder, at a quarter of the sweep's time and a fifth of its trace. But six of ten is a stricter bar
  than six of a ladder: the default gate passes 96 targets under top 10 against 231. At two decoys or
  fewer, top 10 at 50 / 5 / 2 passes as many targets (161, none of the decoys) as the ladder does at
  its best (50 / 8 / 2, 161 and 2). Targets less decoys, the estimate of the true passes, is best
  under top 10 at 150 / 6 / 1 (264, against the ladder's best 235 at 50 / 6 / 1–2). The defaults are
  unchanged: this is one run of a two-protein sample, the gate is a pre-filter whose job (P7) is to be
  settled on a complex sample, and the fragment charge ceilings differ between the two columns. Under
  top 10, five ions is the place to start.

## Precursor-major (2026-11-09, `feat/precursor-major-sweep`)

Why: at proteome scale with no retention-time window, three quarters of a GPF run's cells hold a
peak within 0.35 Th, so the scan-major sweep's map wrote every one of them (6.5 billion matches for one
GPF fraction against the human proteome: about 104 GB and an estimated 2.8 h), and its workers held
10.5–11 GB. A DIA query is only matched in scans of the windows holding its precursor, so read by
window the trace crosses nothing, and a unit of queries can be read, matched and reduced in one task,
writing the reducers' rows only (EncyclopeDIA's design). A scratch prototype of that (dense traces by
`bincount`, a stand-in scorer) ran the GPF fraction in 7.3 min and 16 MB.

**The layout** (`massspec.precursor_major`; named to pair with scan-major, PI). The same points again,
ordered by the precursor m/z each can be evidence for: `precursor_major/ms<L>/part-RRRRR.parquet` (L ≥ 2:
`lower`, `scan`, `mz`, `intensity`, sorted by the scan's isolation lower bound, so a scan's points stay
together in stored order) and `ms1/part-RRRRR.parquet` (`mz`, `intensity`, `scan`, `point`, sorted by
m/z), one file per peak part, so no DIA/DDA scheme is decided; `scans.parquet` (every scan's point
count); a stage key over the peak parts' names, sizes and footers and the scan metadata's. A copy, not
an index into `peaks/`: Parquet reads pages, and every row group of `peaks/` holds every window.
`convert` writes it from its range workers (`--no-precursor-major` to skip); `centroid`, the spectra-file
import and `massspec precursor-major BUNDLE` build it, byte-identical.

**The sweep** (`sweep(..., layout='precursor_major')`, `sweep.precursor_major`). A unit is a run of
queries in (precursor m/z, query) order: whole window segments while its cells stay within
`unit_cells` (or the plan's over 16, so a small plan is still spread) and its windows within 4M points,
a segment over the budget cut by cells. It reads every scan any of its queries is eligible in (their
windows, inclusive; unbounded scans; every scan for a query without a precursor m/z) and at MS1 each
isotope channel's m/z ± its tolerance and the m/z axis's largest correction, so where units are cut
never changes what a query is matched against. Its blocks span at most half its queries' typical RT
window. Inside, batches in m/z order (`batch_cells`) run the scan-major sweep's own `_map_block`,
`_batch` and `_reduce` on a plan taken for the batch (`QueryPlan.take`, numbering queries as in the
whole plan), one part per unit and output. With every reducer gated the batch decides the gate first
and matches only what it passed (`gate_first`, P7's intent). Refused: claims (a point is claimed across
units), the gate's MS1 isotope evidence (a unit reads a slice of each MS1 scan), a table of points.

**Held to scan-major.** For every query the same rows, bit for bit: 41 synthetic cases (DIA, staggered,
DDA; both levels; every rule; RT windows, ppm and Da, the gate, ranges, a 0.3 Th m/z correction,
windows overlapping by 1 Th, a precursor on a bound, scans without bounds, queries without a precursor
m/z, tiny units and batches, two workers) and the XIC parity oracle (72 more). On real data:
the GPF fraction with the chromatogram library's 7,540 targets and decoys at ±120 s, both levels,
0.35 Da: the same 36,052,672 zero-filled trace rows and the same summaries (precursor-major 21.6 s and
23.0 s against 13.8–24.7 s and 24.2 s scan-major on 4 workers; it took 65.7 s before units were spread
and blocks bounded in time); the wide-window run with the library's 152,895 targets and decoys, the
same way: the same summaries (418M MS2 matches; precursor-major 103 s against 172 s) and the same
33,716,272 trace rows for 10,000 of them; the full human space on a 2-minute slice of the GPF run (526k
queries, 105M matches): the same summaries.

**Measured** (this machine, 12 logical cores, WSL2; scratch inputs):

| | before | now |
|---|---|---|
| GPF 900–1000 × full human space (526,013 queries, top 10, 0.35 Th, no RT window, MS2, summary) | scan-major estimate ≈ 2.8 h, ≈ 104 GB of matches | 21.7 min on 4 workers, 7.6 GB peak, 27 MB written; 6,490,106,805 matches (the cells holding a peak, as counted independently); 0.61 µs per cell per worker |
| One unit of that (a 2-minute slice, 105M matches, one process) | 108 s | 42 s: the five-key match sort (57 %) became one combined-key sort (`order_by`), and the point-driven nearest no longer sorts its candidates |
| Scan-major map on the same slice, 4 workers | 10.5–11 GB per worker | 2.3 GB per worker (flushed per window group, runs capped at `run_rows`) |
| `convert`, GPF / wide-window, 8 workers | 23–25 s / 49–75 s | 33–36 s / 87–110 s with the layout (the same size as `peaks/`); peak memory about 11 → 17 GB on the wide run |
| The layout built afterwards, 4 workers | `mz_index`: 2 / 12 min | 11 s / about 30 s |
| Compile of the GPF prediction (526k targets, MS2 top 10) | 76 s (chunked by hand; the compile itself aborted on unpredicted precursors) | 72 s, 3.9 GB from an unsorted file; 56 s, 2.9 GB from one sorted by precursor |
| Prediction of the GPF library (JChronologer, 4 threads) | 300 s, 7.9 GB | 332 s, 4.8 GB; the same predictions bit for bit, written sorted by precursor |

The prototype's 7.3 min had a lighter workload (no RT test, the sum rule, no explicit match rows); the
remaining factor of three is the long-format match rows and their sort, which a dense kernel and view
for the scorer would avoid (deferred to the scorer, which decides their shape; PI). Not run: the
wide-window run against the full space unscheduled (about 177B cells, estimated 45–60 min on 4–8
workers) and the full space predicted (9.56M precursors, about 100 min at 4 threads), both beyond this
machine's run budget; nothing measured suggests either would not fit in memory.

## Still to do

- P7: the two-pass gate is built inside a precursor-major unit (`gate_first`, above); the scan-major
  sweep still writes every match. The bucket index layout, if a profile shows the per-group index
  build mattering.
- Precursor-major: claims across units, units cut by retention time as well (for calibrated windows),
  and the dense kernel and `QueryBatch.dense` view, all with the scorer.
- The scorer (elution peaks, co-elution evidence, library similarity, a verdict rule) as a reducer of
  several outputs over `QueryBatch.cells`, FDR, and output as an experiment directory the browser
  reads: the next plans.
- Targets from FASTA files are built (2026-11-05): `massspec search-space`
  ([search-space.md](search-space.md)) writes a library `chromatogram extract --targets` reads, with
  decoys flagged `is_decoy` through the queries and the query summary (schemas v3) instead of the
  bench's `target_id` offset. One MS1 query per precursor at its own charge is the default since
  2026-11-07 (`--charge-range` asks for a range).
- Profile-points rule, and with it a kernel choice that sees the points within tolerance of a channel
  (above) and an evidence gate for scans without peaks; isolation gating by the whole isotope envelope;
  mixture fits on a scan's competing queries.
