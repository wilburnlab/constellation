# Averaging a direct-infusion run (`massspec average`)

Status: PR #114 (Jeanie Edgington, 2026-11-19), reviewed and refined on the same branch on 2026-11-20: five commits, P1–P5. Code in `constellation/massspec/average.py`; the kind-of-scan key in `massspec.acquisition_structure.scan_kinds`; the bundle rules in `massspec.bundle`.

## What it does

A direct-infusion run repeats the same scans for its whole acquisition, so every repeat of a kind of scan is a replicate of it. `massspec average RUN.profile` (or a directory of them) writes `<run>.average`, one averaged spectrum per kind of scan:
- each profile point is placed on its kind's ion-trap m/z grid by index (nothing interpolated or re-binned);
- the output keeps each point's SD and replicate count (`average_points/`) and a summary per kind (`average_summary.parquet`).

`massspec centroid` makes a peak list from it. The command is ported from the lab's infusion-averaging script.

## Decisions (PI, 2026-11-20)

1. **The output stays a bundle; it says what its spectra are.** A run's bundles and an average are both spectra on an index, but a run's number its scans in time and an average numbers kinds of scan.
   - The manifest says so: `spectrum_index`, absent for a run's bundles, `kind_of_scan` for an average and the peak lists made from it.
   - `related_bundles` ties only bundles numbered alike, and `require_scans` refuses an average where scans in time are needed: registration, a sweep, a precursor-major layout, a spectra file.
   - The run's `source_sha256` stays, as provenance.
   - Moving the hash alone would not have been enough: `derived_from.source` also ties bundles into a family.
2. **What a kind of scan is belongs to `acquisition_structure`.** The key is:
   - the filter string (Thermo's own name for a kind of scan: polarity, scan rate, `cv=`, activation, energy, precursor, range);
   - level, analyzer, activation, energy, FAIMS voltage, precursor, isolation window and scan range, which also cover a bundle with no filter strings.

   The PR's key averaged two FAIMS voltages together and treated two scan rates as one kind.
3. **Per-kind scan metadata is true of the average:**
   - the defining columns;
   - the mean time and TIC, the median injection time and the mean spectrum's base peak;
   - other columns only where every replicate agrees;
   - null for one scan's readings and for scan numbers.
4. **A bundle made from another replaces only what it made** (`prepare_derived_bundle` / `finish_derived_bundle`, used by `average` and `centroid`):
   - it is written into a hidden sibling directory and renamed into place when whole;
   - only a finished bundle made from the same source is replaced, and only with `--force`;
   - nothing else is ever removed.

   `-o` names the directory `<run>.average` goes into, in both modes.
5. **Delivered as commits on the author's branch**, `centroid`'s guard included.

## Measured

Synthetic bundles of Stellar-sized MS2 scans (19,500 points), 58.5M points, this workstation:

| | 10 kinds × 300 | 300 kinds × 10 |
|---|---|---|
| PR as submitted (dense `np.bincount` per block) | 3.9 s | 8.2 s |
| Refined (`np.add.at` in place) | 3.7–4.2 s | 5.4 s |
| Arrow (Acero) group-by with `hash_stddev`, one thread | 11.9 s | 231 s |
| Arrow group-by, variance from sums (count, Σx, Σx²), one thread, reduction alone | 3.8 s | 6.9 s |

The first three rows are the whole average (the pass that finds each kind's extent, the reduction and the writing); the last is the reduction alone, with the grid given.

- **The reduction.** The refined version gives the PR's means, SDs and m/z to the last bits, and the same bytes on a rerun.
- **Why the first Arrow plan was slow** (corrected 2026-11-21; this said "the grid index is already a perfect hash", which the decomposition below does not support). The cause is Arrow's grouped standard deviation (`hash_stddev`, pyarrow 22.0.0), not the grouping. On the 300-kind bundle, one thread, adding one stage at a time:

  | Stage | Time |
  |---|---|
  | read the points | 2.5 s |
  | + join each point to its scan's kind and grid | 1.9 s |
  | + group by (kind, grid index, off-grid flag), count | 6.6 s |
  | group by one precomputed key, count and mean | 5.6 s |
  | + `hash_stddev` | 273.6 s (25.9 s with threads) |

  With 10 kinds `hash_stddev` adds 3.6 s; with 300 kinds and the same points, 268 s.
  - Its cost follows the number of groups, not the rows. Each batch of about 32,000 points touches about as many of the 5.85M groups, so the kernel evidently does work for every group on every batch: the pattern of the first version's grid-sized `np.bincount` per block, in smaller batches. A larger scan batch size did not change it.
  - This mechanism is inferred from the measurements, not read from Arrow's source.
- **Why numpy is kept.** With the variance taken from sums, an Arrow group-by is as fast as numpy. numpy is kept because it places each point by its address, with no join or hashing, in about the same time, deterministically on one thread.
- **RF30, as a stress test.** The profile bundle of a 100-min Stellar DIA run (4.59 billion points, read from `/mnt/e`) averaged in 493 s at 1.55 GB peak RSS:
  - 101 kinds (1 MS1, 100 windows);
  - 1,962,000 points;
  - 0 points off the grid;
  - centroided and summarised without complaint.

## Known limits

- **Kinds are only as distinct as the metadata says.** Two scan rates with the same filter string and no other difference, e.g. an imported spectra file with no filter strings, are one kind: the coarser grid takes one rate's points and the rest fall off it. That is now a warning at the command, not only a count in the manifest.
- **The SD comes from sums of squares.** It is exact to float64 for any realistic CV (it matched a two-pass computation to 4e-13 here), not for a CV below about 1e-6.
- **A directory is averaged one run at a time.**
- **The browser opens an average as an acquisition of its own:** a chromatogram of one point per kind, which is harmless.
- **Still to do:** re-run the 261118 RNA infusion set (69 runs, on OSC) with the refined code.
