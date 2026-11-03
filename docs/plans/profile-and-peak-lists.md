# Profile bundles, peak lists, an in-house centroider and the m/z axis

**Status (2026-10-31).** Steps 0-5 done. Step 6 (the island centroider) has not started and waits on the PI's sign-off of its spec below. Step 7 follows it.

## Why

`massspec convert` used to make profile and centroid an either/or choice, so a centroider could not be looked at against the profile it came from, or compared with another at scale. Thermo's `ToCentroid` gives M0 and M+1 one shared centroid in 21 % of 2+ and 54 % of 3+ precursor-scans (RF30), and the whole m/z axis on that run sits 0.10-0.19 Th low.

## The model

"Peaks" means centroids, always. The profile bundle is the run's data, kept so it can be reprocessed. A peak list is a complete ordinary bundle derived from it, and a run can have any number of them.

```
DIR/run.profile/          kind thermo_profile: manifest, scan and acquisition metadata, profile/part-*.parquet
DIR/run/                  the peak list convert makes, by --centroid-algo (thermo today)
DIR/run.apex-1a2b3c4d/    massspec centroid DIR/run.profile --centroid-algo apex
```

Separate bundles rather than layers inside one, because every consumer already treats a bundle as one list of points: the spectra-file writer, the peak index, XIC extraction and the per-scan columns stay correct with no change. Bundles of a run are tied by `source_sha256`. Only the current format (manifest v3) opens: an older bundle is refused and reconverted, never read as if it were current (PI, 2026-10-31; no outside users). That also protects the blank-trailer fix, since a v2 bundle may hold `iit = 0`.

| The raw file holds | Default | `--no-profile` | `--centroid-algo none` |
|---|---|---|---|
| Centroid scans only | `run/`, the instrument's centroids | no effect | no effect |
| Profile scans | `run/` **and** `run.profile/` | `run/` only | `run.profile/` only |

Both on profile data would write nothing, so that is refused. `--centroid-algo` defaults to `thermo`; changing that is one constant and the PI's decision.

## Measured on real data (RF30, 67 kTh/s, 2026-10-30/31)

**Conversion.** Peak bundle 3.40 GiB, profile bundle 11.66 GiB. Row for row against the old bundles: 280.5 M peak rows and 4.59 billion profile rows, with 9 and 2 rows flagged. Re-reading those scans showed all but one match; the exception was one stored value off by exactly 21.375, in a scan that reads back correctly fresh. A flipped bit on this machine's non-ECC RAM, not the reader.

**Blank trailers.** 31,300 of 236,340 scans have null `iit`, none has 0 left. They all have microscan count 0, and no other scan does.

**Telling a run has profile scans.** `IsCentroidScan` per scan costs 10-18 us, stops at the first profile scan, and is the flag the reader branches on, so the converter uses that rather than parsing `GetFilters`.

**The m/z axis.** RMS against the ID axis (`axis.tsv`, medians of identified fragments) over the 100 Th bins:

| Input | Scans | Time | MS2, 200-1200 m/z | MS1, 400-1100 m/z |
|---|---|---:|---:|---:|
| Profile bundle | 18,569 MS2, 200 MS1 | 37 s | 0.0094 Th (worst 0.018) | 0.0061 Th (worst 0.012) |
| Thermo peak list, against Thermo's axis | 20,226 MS2, 200 MS1 | 3 s | 0.0164 Th (worst 0.033) | 0.0238 Th, few knots |

**Weighting matters far more than picking.** On the same 6,517 MS2 scans, against the same axis:

| | Intensity | sqrt intensity | Unit |
|---|---:|---:|---:|
| Raw profile | 0.009 | 0.021 | 0.057 |
| Its local maxima | 0.011 | 0.018 | 0.059 |

The bench record's 0.015-0.06 Th MS2 bias came from a unit-weighted fold of filtered maxima, so it looks mostly like a weighting effect, though its filter differs.

**The reference.** Same measure, MS2, on the profile:

| Candidate | RMS, 200-1200 | Worst |
|---|---:|---:|
| Tryptic digest, y 1 and b 0.5 (chosen) | 0.0094 | 0.018 |
| RF30's own identified fragments | 0.0083 | 0.015 |
| Predicted library | 0.0123 | 0.023 |

None removes the residual at 300-500 m/z (about -0.015), so the reference is not its cause. The identified version is circular for RF30, and the small gain does not justify needing a search.

**A centroider's bias, with no identifications.** `massspec summarize` on RF30's bundles: Thermo's list reads 0.03-0.06 Th below the profile's curve across MS2; `apex` is within 0.013 Th below 1300 m/z. Thermo's MS1 curve has too little signal above 860 m/z to measure.

**Identified fragments.** The 1,200 most intense precursors of the lab's search of RF30, each quantified ion looked for in the MS2 scan nearest its apex, taking the strongest peak within 0.35 Th, so the errors are clipped at that limit:

| List | Median error, before and after | Within 0.05 Th | Within 0.1 Th | Robust SD, pooled | Robust SD, typical 100 Th bin |
|---|---:|---:|---:|---:|---:|
| Thermo | -0.146 to +0.020 Th | 5.5 % to 53 % | 18 % to 80 % | 0.079 to 0.063 | about 0.06, barely changed |
| `apex` | -0.114 to +0.013 Th | 10.5 % to 60 % | 33 % to 83 % | 0.075 to 0.056 | about 0.05, barely changed |

A smooth curve removes bias and can do little for what is left inside a bin, so the pooled spread narrows while the spread within a bin barely moves. What remains after correction at 300-500 m/z is about +0.03 Th, the residual the reference comparison above could not remove.

**The first search test (Step 4).** RF30's Thermo list, EncyclopeDIA 6.9.23 with Percolator v3-05, all three tolerances set together, 1 % FDR:

| Tolerance | Uncorrected | Corrected |
|---:|---:|---:|
| 0.35 Da | 31,093 precursors, 4,905 proteins | 30,933 precursors, 5,002 proteins |
| 0.2 Da | 23,895 precursors, 4,306 proteins | **34,301 precursors, 5,209 proteins** |

- **Noise floor:** a repeat of the uncorrected 0.35 Da search found the same precursors with identical retention times and quantities.
- **Corrected 0.2 Da against uncorrected 0.35 Da:** 27,908 shared, 6,393 gained, 3,185 lost. That is 10 % more precursors and 6 % more proteins than the best uncorrected search.
- **Are the gains plausible?** Their observed retention time agrees with the library as well as those both searches found (median 31 s and 30 s off; over 120 s off 3.3 % and 3.1 %), and their median PEP (0.027) is lower than that of the precursors only the uncorrected search found (0.042). Retention time is one of Percolator's features, so this is a sanity check and not independent evidence. There is no entrapment test.
- **Quantities move, and which are better is untested.** Over precursors shared by the corrected and uncorrected 0.35 Da searches, the median change is 0.17 in log2 (p99 8.1, log Pearson r 0.82), and only 10.5 % are within 1 %. The repeat changed nothing, so this is the correction's doing. Replicate precision or a dilution series would settle it; neither was run.
- **Limits:** one run, whose axis is unusually far off (about -0.15 Th). The two GPF fractions measured read between -0.08 and +0.13 Th, so expect less there. The library is the lab's six-fraction chromatogram library (400-1000 m/z, fragment m/z theoretical to within 1 mTh on 99.9 % of peaks), because the one the RF series search used (31,715 precursors, Percolator 3.01) is on OSC, and RF30 runs to 1200 m/z.

**Default output unchanged.** GPF 400-500 through this branch and through the commit before it: identical, 153.6 M peaks. GPF 900-1000 against MSRawJava's file with every metadata key compared: identical, 109 M peaks, and identical again with a curve stored. Against MSRawJava's 400-500 file, `run.gradient_length_seconds` differs in its last digit (6423.349381450999 against 6423.349381451) on both this branch and the commit before it, so that was already there; the 153.6 M peaks are identical. A curve named at export moves MS2 m/z by up to 0.078 Th on 400-500, as measured. Neither GPF fraction wrote a profile bundle.

**Island speed.** Padded PyTorch LM against `scipy` TRF, one joint fit from the same seeds on 10 scans, one thread: the same answer (median position difference under 1e-5 Th, p99 at most 0.004 Th, amplitude p99 at most 1.9 %), at 1-2x the speed, so batching alone will not do. A parallel peel gave 92-98 % identical seeds and no speed-up as written. Tight clusters with neighbours' tails held fixed were rejected: not converged after 4 sweeps, p99 position error 0.011-0.032 Th, amplitudes of strong peaks up to 100 % off.

## What each step shipped

- **Step 0.** A blank trailer (microscan count and elapsed time both 0) gives null in every typed trailer column; `trailer_extras` keeps the file's strings; the spectra-file writer writes null as 0, so its output is unchanged. Conversion warns with the count and the manifest records it. `_thermo_centroids` documents Thermo's one-grid-step offset.
- **Step 1.** `massspec.bundle`: `read_bundle_manifest`, `OutdatedBundleError`, `point_paths`, `points_dataset`, `is_profile_bundle`, `peak_list_name`, `related_bundles`, `list_mz_axes`; `peak_paths` refuses a profile bundle. `profile_mode` is gone (`ACQUISITION_METADATA_TABLE` v3). `--profile` is a tombstone naming its replacement. One worker pass writes both bundles: `Scan.FromFile(...).SegmentedScan` matched the separate read on 300 of 300 scans and left the centroider's output unchanged on 50 of 50, at 0.69 ms a scan against 1.42 ms.
- **Step 2.** A run's bundles are one acquisition with `peak_lists`; attaching any of them finds the others beside it. `SpectrumQuery.peak_list` is looked up, never made into a path. Profile draws as a line, peak lists as sticks over it, with peak area on a right-hand axis at a ratio fixed per scan (`areaPerHeight`) and a toggle to anchor sticks to the profile. `SPECTRUM_SCHEMA` gained a last column, `profile`, for the same reason `decimated` is a column: `response_headers` never sees the binding.
- **Step 3.** `core.signal.phase.PhaseFold` (generic), `massspec.mz_axis` (estimate, store, apply), `massspec mz-axis`, `massspec summarize BUNDLE` (`massspec.bundle_summary`: what a bundle holds, the run's other bundles, each list's curve against the profile's; reads what is stored and measures nothing), `scans.spectrum(mz_axis=)`, `scans.xic(mz_axis=)`, a browser toggle. Departures from FineTune: a direct sum rather than a 2**18-point FFT (a test holds the shifts equal); a window widens to a standard error rather than a TIC threshold; MS1 on the 2+ lattice with its branch from the MS2 curve. The straight line through a curve is reported as a scale error in ppm, which is what a frequency offset of the fold would be, so no separate period scan is needed.
- **Step 4.** Any peak bundle can be searched by giving it. A derived list or a named curve gets its own spectra file under `spectra/<run>/<name>[+<axis>]/`, because finished searches link to theirs; default paths and keys are unchanged. `--mz-axis` on `search`, `quantify` and `encyclopedia export-spectra`. `summarize` compares `peaks` and `m/z axis`. The precursor route says which list a search read, and the browser annotates that one.
- **Step 5.** `massspec centroid`, `massspec.centroiding` (`ALGORITHMS`, `centroid_bundle`, `apex`). `--centroid-algo` on `convert` reaches the registry. A partial list (`--stride`, `--scans`) is marked in its manifest and refused by the spectra-file writer.

## Not done from the plan as written

- The reference is checked against one other proteome only: a de novo translated one (`/mnt/e/Davis/260802-2229/proteins.fasta`) built into the same table differs from the packaged human digest by 0.0028 Th RMS for fragments and 0.0026 Th for precursors, a steady tilt of 0.008 Th across the range rather than scatter. That is well under the 0.0094 Th error of the curve itself, so the digest stands for peptides generally, though a tilt that consistent means it is not strictly organism-independent.
- The reference comparison is on RF30 alone, because a GPF run has no ID axis to score against in `axis.tsv`.
- The Step 4 search test uses a stand-in library (see the search test above), and it tests the Thermo list only. `apex` has not been searched; Step 7 does that with the island list.
- `massspec centroid --centroid-algo apex` on all of RF30: 7:54 wall on 8 worker processes, 1.9 GB resident, 4.5 GiB; 460 M MS2 peaks against Thermo's 278 M.

## Open: where a curve lives, and how it reaches a run's other bundles

Noted by the PI, 2026-10-31. Not to be fixed yet; it is a question about the design of `mz_axes`.

**The limitation.** A curve belongs to one bundle (`<bundle>/mz_axes/<key>.parquet`), and each list reads only its own. So one correction cannot be applied to a profile and its peak lists together: `massspec mz-axis` has to be run on each, and they give similar but not identical curves. In the browser, "Corrected m/z" therefore moves each list by a different amount.

**Both behaviours are legitimate, because they answer different questions.**
- *Each list's own curve* puts every list on the true axis. A list's curve includes its centroider's bias, so the bias is removed, and comparing the two curves is how `massspec summarize` measures it with no identifications. RF30's MS2: Thermo 0.03-0.06 Th below the profile, `apex` within 0.013 Th.
- *One curve for all* moves the whole run together. A centroider's offset from the profile stays visible, as it does uncorrected, with the instrument's error gone.

Today only the first exists.

**The workaround, and what it does.** The curve's Parquet file can be copied into another bundle's `mz_axes/`, and that bundle then uses it. Three things to know:
- **It replaces that bundle's own curve.** A curve's key is a digest of the method, reference and settings, not of the bundle or the values, so every bundle measured with the defaults has the same file name (`9844a736` on all of RF30's lists and on the GPF fractions). The own curve is gone until `mz-axis` is run again, and `summarize` then reports no bias for that list, since it compares a curve with itself.
- **A search in an existing experiment does not see the replacement.** The spectra file and the search are keyed by the curve's key, so the same key finds the file already written with the old values. Tested with a curve replaced under its key: the same unit, reading a file shifted by 0.1 Th where the curve now said 0.3. A new experiment directory, or removing that curve's `spectra/<run>/<name>+<key>/` and the search under `units/`, gets the new values. `scans.spectrum` and `scans.xic` read the curve each time and are not affected, nor is the browser once reloaded.
- **Nothing records that it was copied.** A curve's metadata does not name the bundle it was measured on.

**What a fix has to settle.**
- Where a curve lives: with the bundle it was measured on (as now), or with the run, which the bundles already share through `source_sha256`.
- How one is chosen: by naming another bundle's curve wherever `--mz-axis` is taken, and in the browser by choosing which list's curve applies, versus copying or linking.
- What a key means. Whatever else is decided, a key should identify the values, or the bundle and the values, so that a search cannot reuse a spectra file written from a different curve, and each curve should record what it was measured on.

## Island centroider: spec for review

**No island code or test is written until the PI signs this section off.** The design is `pick_island` from §6.6 of `results/e3_peak_picking_benchmark.md`, changed only where marked.

**Scope.** Ion-trap profile scans on a uniform grid (100 % of sampled steps were 1/15 Th).

**Steps for one scan.** `y` is ions per point, `intensity x IIT / 1000`.
1. **Baseline:** rolling 20th percentile over 5 Th, zeros included, fixed rather than fitted.
2. **Seeds by peeling:** the highest remaining residual point is the candidate centre (log-parabola, within +-1 tick, only a starting value), and its height over the peak shape's height gives the amplitude. It is kept when the matched-filter amplitude is at least 0.5 ions and 3 SE, with noise from baseline plus what has been peeled so far. Its shape is subtracted out to its reach; a failed candidate blocks +-1 tick. *Changed:* run in parallel, processing together all local maxima not covered by a higher candidate's reach. Must be shown to match the sequential seeds on the benchmark.
3. **Islands:** seeds whose reaches overlap. Reach is where the shape falls below 0.01 ions per point, at least 4 s.
4. **Joint fit of each island:** least squares on `sqrt(y + 0.1)`, baseline fixed, positions within +-0.5 FWHM of the seed, amplitudes non-negative. *Changed:* projected Levenberg-Marquardt in PyTorch instead of `scipy` TRF; measured to reach the same solution.
5. **Prune and merge until stable:** drop components under 0.5 ions or under 3 SE0 (the noise of everything else). Merge components closer than 0.2 FWHM.
6. **Re-seed from the island's residual,** up to 2 cycles. *New:* a re-seeded component stays only if dBIC < -10, in the `n ln(SSE/n) + k ln(n)` form of the old split pass.
7. **Report** m/z; `intensity = ions / (IIT/1000)`, Thermo's unit; `peak_mz_se` (`sqrt(PHI/4 diag((J'J)^-1))`) and `peak_snr` (amplitude / SE0).

**Constants, unchanged:** PHI 0.71, variance floor EPS 0.02, `FIT_C` 0.1.

**Peak shape, for now:** Student-t as benchmarked. MS1 s 0.1428, nu 3.02; MS2 s and nu interpolated over the 13 m/z knots. *Changed:* evaluated per component instead of once at the island's centre.

**Decisions for the PI**

| # | Question | Recommendation |
|---|---|---|
| A | Is the re-seed test right? It takes its scale from the residual rather than the known variance PHI/4, which would keep accepting splits at strong peaks, where the misfit is shape error. | Yes, evaluated within +-2 FWHM of the new component. If weak minors near strong peaks are not restored, a split trial for components of at least 10 ions comes back. |
| B | What to report at 1-2 ions? | Keep 3 SE0. `--min-snr 2` is just another peak list, and the search test decides. |
| C | Symmetric or split-t? | Symmetric first, to compare with the benchmark. Then use the peak list's curve against the profile's and the MS1 satellites to decide. |
| D | Peak shape at other scan rates? | A fixed table keyed by grid step, refusing other grids: 67 kTh/s only at first. Global sigma/nu regression follows; the risk is that a wider shape and fewer components trade off in MS1, so it is validated against `kernel.tsv`. |
| E | The 13 % of RF30 scans with no injection time? | Estimate from each scan's own TIC, `IIT = 1000 N / TIC`, capped at the level's longest injection time, with `N` the run's own median of `IIT x TIC / 1000` at that level (28,400 at MS1, 9,690 at MS2), so nothing has to be read from the method. Measured on RF30 by hiding the time on scans that have one: within 25 % on 86 % of MS1 and 84 % of MS2 scans, against 60 % and 47 % for the same window 180 s away, which is what the middle of its 349 s blank blocks would have to use. TIC is present on all 31,300 blank scans. One run only. |
| F | A truncated Jacobian for speed? | Yes, if it matches the dense fit on the benchmark. |
| G | Where do `peak_mz_se` and `peak_snr` go? `MS_PEAK_TABLE` has neither; its nullable `peak_noise` and `peak_baseline` are the instrument's. | Two new nullable columns, written only by a fitting centroider. The spectra file cannot carry them, so for now only the browser and a weighted fold of the peak list read them. |

**Speed route** (not measured beyond the above). Keep model and residual with full tails; truncate Jacobian columns to about +-1.5 FWHM for the normal equations, which become banded and can be solved by a batched block-tridiagonal Cholesky; converged islands leave the batch; islands are bucketed by size. Replace the all-pairs dominance check in the peel with a neighbour-limited one. One torch thread per worker. **Gate: at most 1 CPU-s per scan**, against 5.3 and 9.6 today, about 65 CPU-hours a run.

**Code.** `core/stats/peaks.py` `StudentTPeak`; `core/signal/peak_fit.py`, a generic batched fitter; `massspec/centroiding/island.py`.

**Acceptance.** The tables of the 150-scan benchmark within noise of `run_261030-1317`; the 7 % splits gone without losing weak minors; synthetic CI tests built ion by ion from §3's model.

## Step 7

RF30 with `{thermo, apex, island} x {no shift, shift} x {0.35, 0.2 Da}`, judged against a repeat search with `compare-results`. Ledger #15's promotion criterion, which also settles decision B.

## Not in this work

An XIC summing mode for profile points; correcting drift across retention time or by ion count; per-run estimation of the single-ion gain; Orbitrap or TOF centroiders; the browser's remaining steps H1-H3, whose mirror plot will share `CellData` and the y-domain with this overlay.
