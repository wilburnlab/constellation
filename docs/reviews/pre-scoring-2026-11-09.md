**Review of the search path before scoring, 2026-11-09**

Reviewed `main` at `730d719` (PR #109 merged). **Fixed on `fix/pre-scoring` (2026-11-10), one commit per phase P1–P7: the Status column says which; what changed and what was measured is in [../plans/sweep.md](../plans/sweep.md), "Pre-scoring fixes and the dense view".** The scope is everything a scorer will read:
- conversion and the bundle, the precursor-major layout and the m/z axis;
- the search space, decoys, content ids and library prediction;
- targets → queries → channels, with library abundances;
- the sweep, scan-major and precursor-major, and the reducer interface;
- the existing scoring functions;
- the downstream plumbing a native search would write into.

Five read-only reviews ran in parallel, one per layer, and a sixth surveyed the downstream plumbing. Each finding below was reproduced on synthetic data by a script, or traced in the code where marked *plausible*. The headline ones were re-read against the source. No production code was changed. The scripts sit in the session scratchpad and are not committed.

**Verdict.** The match-generating machinery is sound. The precursor-major path gives the same cells as scan-major in every configuration tried (33, including staggered and nested windows, every rule, charge ranges, m/z correction, gates, workers). No query is ever split across reducer calls, and the default decoy population is balanced against its targets before and after prediction. No defect found corrupts a DIA search on the default path, on a non-FAIMS run.

The defects that matter sit at the edges:
- inputs a scorer is likely to use next: PSM-derived targets, FAIMS runs, a reused work directory, a resumed search space;
- what an *unmatched* cell claims;
- how big a reducer call can get.

Scoring itself is almost entirely missing:
- **Spectrum-vector similarities exist but nothing calls them.** No time-domain operation exists: smoothing, background, peak picking, boundaries.
- **No statistics exist:** q-values, PEP, rescoring, protein inference.
- **A native search has nowhere to land:** the experiment directory, report and quantify are wired to the jar.

## 1. Defects, ranked

Severity is for a scorer and FDR built on this path. Paths are relative to `constellation/massspec/` unless they start with `constellation/`.

| # | Severity | Defect | Where | Status |
|---|---|---|---|---|
| A1 | high (FAIMS runs) | FAIMS compensation voltages share one window, so the gate fails every peptide of a multi-CV run and traces zig-zag | `sweep/schedule.py:194-199`, `precursor_major.py:137` | confirmed; refused (P3) |
| C1 | high if used for FDR | `targets_from_search` drops `is_decoy`, `decoy_set`, `precursor_id` and the run | `quant/targets.py:142-163` | confirmed; fixed (P6) |
| B1 | medium | `search-space --resume` keys on an install-time `__version__`, so an output from older code passes as current, and its prediction with it | `cli/search_space.py:205-209` | confirmed; fixed (P5) |
| B2 | medium | Peptide-level decoys have no proteins; `subset_proteins` drops decoys whose target was filtered away | `library/search_space.py:770-806`, `library/library.py:247-255` | confirmed; fixed (P5) |
| C2 | medium | One target with no fragment candidates aborts the whole compile, with an error blaming the library | `sweep/queries.py:614-618`, `library/fragment_abundance.py:214, 328-355` | confirmed; fixed (P6) |
| E1 | medium | Reusing a sweep `work_dir` makes an earlier `SweepResult` read the later sweep's rows (both layouts) | `sweep/precursor_major.py:495-501`, `sweep/engine.py:153-167, 1099-1103, 1146-1150` | confirmed; fixed (P1) |
| E2 | medium | Cells outside a scan's acquired m/z range are zero-filled as observed zeros | `sweep/reduce.py:47-62, 392-418` | confirmed; fixed (P3) |
| E3 | medium (OOM risk) | The default in-memory scan-major sweep holds and reduces too much at once. It tests `memory_rows` only after a whole block has been mapped (942 matches held at a limit of 100), and then reduces the whole plan in one call, counting neither gate nor zero-fill rows | `sweep/engine.py:680-689, 704-715, 754-759` | confirmed; fixed (P1) |
| D2 | low-medium | On disk, `point_claims` is out of scan order once a flush lands mid-block, and differs from the in-memory table; run names past 9,999 sort wrongly | `sweep/engine.py:850-851, 887-909, 1015-1026` | confirmed; fixed (P1) |
| A2 | medium | Building the layout holds a whole peak part in memory (~100 B/point); single-part writers make that the whole run | `precursor_major.py:114-155`; `readers/thermo/_convert.py:153, 801`; `io/encyclopedia/_dia_import.py:275` | confirmed; fixed (P7) |
| A5 | low-medium | An MS1 m/z curve with an undetermined branch (`ambiguous_levels`) is applied without a word | `mz_axis.py:132-144, 330-340` | confirmed; refused at that level (P7) |
| B3 | low-medium | A fixed rule silently disables a variable rule on the same slot (TMT + protein N-term acetyl, TMT@K + Acetyl@K) | `library/search_space.py:336-356`, `constellation/core/sequence/protein.py` | confirmed; fixed (P5) |
| B4 | low-medium | `reverse-c-fixed` moves residue 0 but not its N-terminal cyclization; Chronologer then refuses 437 decoys and no targets | `library/decoys.py:100-103`, `library/search_space.py:825-827` | confirmed; fixed (P5) |
| A3 | low (I/O) | One MS2 scan without bounds makes every unit read every window below it (9.8× rows on a test) | `sweep/precursor_major.py:262-275` | confirmed; fixed (P2) |
| A4 | low-medium | MS3 takes the MS2→MS3 isolation as its window, and MSX keeps only reaction 0 | `readers/thermo/_scans.py:450, 465, 471` | plausible; MS3 fixed, MSX refused (P7) |
| A6 | low | `scan_range_shards` raises when the last peak part is empty | `bundle.py:445-446` | confirmed; fixed (P7) |
| B5 | low | A `:max=` limit is pooled across rules for the same modification | `library/search_space.py:492-493` | confirmed; fixed (P5) |
| B6 | low | `reverse-protein:sets=2` duplicates set 0; `--count` books its decoys as targets | `library/search_space.py:538-548, 700-707` | confirmed; fixed (P5) |
| B7 | low | `<D>` and `<2H>` give different content ids for one molecule | `library/ids.py:189-199` | confirmed; fixed (P5); with it, ProForma read case-insensitively, as its spec says (residues, CV prefixes, names, labels) |
| C3 | low | `targets_from_precursor_quant` makes duplicate `target_id`s across runs and drops the integration window | `quant/targets.py:166-193` | confirmed; fixed (P6) |
| C4 | low | Two library rows naming one ion: the last overwrites, both count in the denominator | `library/fragment_abundance.py:289` | mechanism confirmed; fixed in `LibraryFragments` (P6); the MSP half did not reproduce |
| E4 | low (I/O) | A window is re-read once per cell-cut unit, not "at most twice" as documented | `sweep/precursor_major.py:18-19, 190-194` | confirmed; fixed (P2) |
| E5 | low | A unit of queries without a precursor m/z reads every MS2 point (only via `derive_precursor_mz=False`) | `sweep/precursor_major.py:198, 251-254` | confirmed; fixed with E4 (P2) |
| E6 | low (speed) | RT windows under ~1 s make 1 s blocks; precursor-major then runs 8× slower than scan-major | `sweep/precursor_major.py:385-392` | confirmed; fixed (P2) |
| S1 | low | `l1_normalize` lets a masked NaN poison the row and ignores the module's own masking rule | `constellation/core/stats/losses.py:77-93` | confirmed; fixed (P7) |

Checked and found correct, so not re-reviewed:
- the two kernels, bit for bit on 500 adversarial cases × 3 rules (duplicates, midpoints, exact bounds ± 1 ULP, float32 grids, zero and huge tolerances, empty scans);
- `order_by`, which falls back to `lexsort` past 2⁶²;
- inclusive float64 isolation gating in map and reduce;
- gate runs across shards;
- the layout's contents and order against `peaks/`, and its row-group reads;
- the m/z correction's sign, per-level curve, flat extrapolation and MS1 read window;
- masses across labels, modifications and termini, where b + y equals the precursor to 2e-13 on 12 sequences;
- chunked compile against whole compile;
- NaN and 0 abundances through `take` / `write` / `read`;
- the decoy population's balance by charge, m/z and length, before and after JChronologer;
- b/y complementarity of predicted fragments.

### Notes on the ones to fix first

**A1 — FAIMS.** `faims_cv` is read from the filter string into scan metadata, and nothing downstream reads it. In a method that alternates −45 and −65 V, a peptide transmitted at one CV shows evidence in every other scan of its window, so its longest run is 1. A synthetic run passed 6 of 6 peptides with one CV and 0 of 6 with two, in both layouts. Fix: make the CV part of the window key (ledger window index and the layout's sort key). Until then, refuse bundles with more than one CV.

**C1 — PSM targets.**
- **Decoys become targets.** ScribeTwo writes its decoys into `PSM_TABLE`, so re-extracting them for a scorer makes every decoy a target.
- **Library abundances are lost.** Without `precursor_id`, `LibraryFragments` gives NaN everywhere, and with `--top-fragments` the queries get no fragment channels (14 of 14 in a test).
- **Runs are mixed.** PSMs of every run become targets for the one swept, and `assigned_scans_only` pins them to another run's scan numbers.

Fix: carry the three columns, and take an `acquisition_id` as `targets_from_experiment` does.

**B1 — stale resume.** `__version__` is `0.0.1a2.dev123+gb2add4344`, from 2026-05-28, before `search_space.py` existed. A search space built before the 2026-11-05 and 2026-11-09 fixes therefore passes as current, and so does the prediction keyed on it. Fix: a `SEARCH_SPACE_LAYOUT` constant, bumped when output changes (as `PREDICTION_LAYOUT` is), plus the SHA-256 of `unimod.json`. Every other stage keyed on `__version__` has the same exposure; grep for it.

**B2 — decoys without proteins.**
- **No decoy edges.** Under the default method, 0 of 7,356 decoy peptides have a `PROTEIN_PEPTIDE_EDGE` row. A decoy reaches a protein only through `decoy_of`, and the PI has ruled that column provenance only.
- **The one route breaks under a filter.** After `--min-charge-probability 0.4`, 86 decoy peptides had lost their target peptide, and `subset_proteins(every protein)` then dropped those 86 decoys and no targets.

Protein-level FDR needs decoy proteins. Fix: emit `DECOY_<accession>` proteins with edges that mirror the target's, built from the decoy population and not from pairs, and have `subset_proteins` follow those edges.

**E2 — "measured nothing" vs "never measured".** Eligibility ignores each scan's `scan_window_lower/upper`, which the ledger does not carry. With `charge_range=(1,4)` and an MS1 range of 350–1250, 31% of MS1 cells lay outside the range, and 16% of MS2 cells fell below 200 Th. Each is zero-filled exactly like an empty cell. A censored likelihood (Counter's design) or a "matched / expected" feature reads them as evidence of absence. Fix: carry the scan window into `Ledger`, and give cells an `acquired` flag or leave them out. Empty scans have the mirror problem: a scan with no points is not in the cell grid at all.

**E3 — the in-memory path.** This is what `sweep()` and `extract_xic()` do from a notebook (one worker, no `work_dir`), and it fails in two ways.
- **The memory check comes too late.** The 2026-11-09 per-window-group flush went into `_map_stream` only, so here the check runs after a whole block. A GPF fraction against the whole space with no RT window makes about 61 matches per point, so the first 1M-point block yields about 61M matches before an 8M check: 12–30 GB at the plan's measured 200–500 B per held match, on a 31 GB VM.
- **The reduce takes everything at once.** It reduces the whole plan in one call and counts neither gate nor zero-fill rows. With `memory_rows`, `batch_cells` and `reduce_rows` all at 50,000, the paths differ:

| Path | Reducer calls | Largest call |
|---|---|---|
| Scan-major, in memory | 1 | 1,962,000 cells |
| Scan-major, on disk | 42 | 48,000 cells |
| Precursor-major | 42 | 49,320 cells |

A dense scorer on a proteome plan would get billions of cells in one call. Fix:
- drive the in-memory loop part by part (`_map_block_parts`), handing the rest to `_map_stream` when it spills;
- spill on `_reduce_weights`, not on matches alone;
- bucket `_reduce_in_memory` as the disk path does.

**E1 — work directories.** A calibrant pass and a target pass sharing `work_dir` is exactly what a two-pass scorer will do. Give each sweep its own `reduce-<id>/`, or stamp and check a sweep id in every part.

## 2. Gaps the scorer will hit in the existing layers

None of these is a defect, but each changes what a scorer can compute.

- **Channels that overlap within tolerance carry no flag.** On 3,000 tryptic peptides (b/y, fragment charge ≤ 2), 17% of queries have two channels overlapping at 20 ppm and 62% at 0.35 Th; with H₂O/NH₃ losses, 60% and 99%. Some coincide exactly, e.g. b(2k)²⁺ = b(k)¹⁺ for "AA…", and some y−H₂O equal a b. The gate counts `min(keys, peaks)`, so it is safe, but a scorer summing cells counts one point as two ions. EncyclopeDIA makes its ions unique within `-ftol`.
- **Channel kinds not generated:** the unfragmented precursor in MS2 (and its losses), immonium ions, internal ions, fragment isotopes.
- **Channels know nothing about the scan or the precursor.** 9% of fragment channels fall below 200 Th. In 65% of queries some fall within ±12.5 Th of their own precursor, where co-isolated precursors sit; EncyclopeDIA's XCorr removes precursor ±10.
- **2⁺ precursors get 2⁺ fragments by default.** `fragment_charge_below_precursor=1` reproduces EncyclopeDIA and is not the default, which dilutes `Uniform` shares with mostly empty channels.
- **No ion-count conversion is possible without care.** Injection time is null for 13% of scans of one Stellar run (NaN in the ledger). No per-scan detection floor (lowest intensity) is stored, and that is what a censored likelihood needs.
- **DDA context is not in the ledger:** precursor m/z, charge, monoisotopic m/z. A scorer must join `scan_metadata` itself.
- **Missing scans are silent.** Scans that fail to read are dropped with a warning, and a partial `centroid --scans` list is not refused, so scan numbers can have gaps that look like silence.
- **I/L-equivalent targets are separate peptides** with separate protein edges. Their decoys are twins now, but a scorer sees two targets fitting one signal, and protein grouping must build I/L classes itself.
- **Any refusal drops the precursor.** Chronologer refuses Deamidated N/Q, HexNAc@N and Oxidation@W, and Sculptor refuses TMT. Refusals are symmetric and reported, but a "keep, with this property unset" mode would keep those precursors searchable.
- **`charge_probability` is P(z) over charges 1–6**, not renormalised over the charges searched. Filtering is per precursor, so after `--min-charge-probability` the target and decoy counts drift slightly (10,327 against 10,266 at 0.05).
- **Decoy-named FASTA entries become targets** (`DECOY_P1`, `rev_P2`), without a warning.
- **Entrapment sets live in the same tables** as `decoy_set > 0`. Every FDR step must choose which sets compete; nothing marks it.
- **Isolation offset sign is untested.** No fixture or reference run has a non-zero MS2 isolation offset. Check one Orbitrap DDA run that uses one before trusting DDA windows.
- **A curve's key does not pin its values.** It hashes the method and settings only (`mz_axis.py:381-386`).
- **`nearest` ignores intensity.** At 0.35 Th, a noise centroid nearer the channel beats the real fragment, which is visible only under `all`, and that multiplies the rows. Ion-trap scoring will want a "most intense within tolerance" rule.
- **A `sum` match's m/z is not a calibration error.** It is an intensity-weighted mean clipped by the window: biased toward the channel on profile points, and between the two when two centroids share a window. Fit mass calibration on `nearest` errors.
- **The gate's evidence peaks are the whole scan's top 50.** In wide windows at proteome scale those belong to a few abundant precursors, and the defaults were measured on a two-protein sample.
- **Gate runs interleave overlapping windows.** A precursor near a window edge, poorly transmitted in one window, cannot make a run of 3.
- **Isolation is tested on the monoisotopic m/z only.** A precursor whose M+0 sits just outside a window while M+1 and M+2 are inside is never matched there.
- **One window group is unbounded in the point-driven kernel.** `kernel_elements` bounds only the channel-driven kernel. An MS1 group against a whole space with no RT window cannot run scan-major.
- **`n_matched_ms*` counts match rows.** Under `all` those are (cell, point) pairs, not the cells the schema comment says. The spill also goes to system temp with no space check: about 100 GB at GPF scale, left behind if the process is killed.

## 3. The reducer contract as it stands

This is what a scorer written as a reducer receives today, measured on both layouts.

- **A query arrives once, whole.** One call holds a contiguous run of queries, with every match at every level over the whole run (or `scan_range`). No query was ever split across calls, blocks or units.
  - Precursor-major calls run in precursor m/z order, at most `batch_cells` (4M) eligible cells each.
  - Scan-major on disk cuts at `reduce_rows` of weighted rows.
  - Scan-major in memory sends the whole plan (E3).
- **MS1 and MS2 come in the same call**, as `cells(1)` and `cells(2)`. Nothing links an MS2 scan to its MS1 cycle; scans are ordered by (RT, scan).
- **Query numbers are local in precursor-major.** There, `batch.queries` and `batch.plan.targets` are a taken plan's, numbered 0…n−1. Anything keyed by query number must go through `plan.global_query`, or it silently reads the wrong rows. `batch.settings.mz_axis` is the resolved key in precursor-major but can still be `latest` in scan-major.
- **What a cell row holds.** One row per match (several under `all`), plus a `matched=False` row per eligible empty cell: intensity 0, m/z NaN, `point_index` null. The error is observed − theoretical, with the observed m/z axis-corrected. Abundance NaN means not modelled; 0 means predicted absent. RT, TIC and injection time are the cell's own scan's. Isolation bounds are reachable only via `cells.scans`, not `Cells.table()`.
- **Gated reducers see `batch.passing()` only.** With every reducer gated, a precursor-major sweep never matches the failing queries.
- **A query's rows may depend only on that query** (`Reducer` docstring). That rules out interference and competition features inside the sweep, and claims are refused on the precursor-major path.
- **Not available today:**
  - a dense scans × channels tensor;
  - a scan's unmatched points or local point density;
  - the scan's m/z range;
  - neighbouring queries;
  - anything across batches, such as a calibration (that needs its own pass).
- **Reducers are pickled to spawned workers per unit.** A scorer holding a model or a library should hold a path and cache the loaded object per process.

## 4. Scoring: what exists

| Piece | Where | State |
|---|---|---|
| Similarities over aligned channel vectors: cosine / normalized dot, Pearson, spectral angle, spectral-entropy similarity (unweighted), KL, G² multinomial deviance, `compare_spectra` | `constellation/core/stats/losses.py`, `spectra/similarity.py` | See note 1 below. |
| Dirichlet-multinomial fragment model with an intensity → ion gain | `spectra/noise.py` | One peptide's stack of scans, one fit per peptide; no mask, so a NaN (unmodelled) abundance propagates. |
| Peak shapes: Gaussian, EMG (exact mode), HyperEMG, each with `integrate` / `bounds` | `constellation/core/stats/peaks.py` | Log-space EMG checked. `Parametric.fit` fits one model per call; DE vmaps a population of one problem, not many problems. |
| Counter: MS1 isotope × charge likelihood | `counter/` | See note 2 below. |
| Evidence gate | `sweep/evidence.py` | A pre-filter, not a score. |
| RT alignment from pairs | `rt/` | Ready for a native pass 1: `library_pairs` takes any table with rt, qvalue, is_decoy and ids; `align` gives `to_observed` and a band. |
| Predicted fragments, RT, CCS, charge probability | `library/prediction/` | On targets and decoys alike. |

1. **The similarities.** They are correct, scale-free and batched, and their masking matches the sweep's convention: NaN or negative is dropped, 0 is kept. Their 189 tests pass. Nothing on the search path calls them; `spectra.consensus` is the only caller.
2. **Counter.** It models a HyperEMG elution flux, charge fractions, isotope fractions and a gain α(z)·IIT, with a censored Poisson term for unobserved cells and a Student-t on m/z error, and panels add a background term. That makes it the physically grounded scorer in embryo, but it is MS1 only and fits per target (DE + L-BFGS or VI). It needs a `GlobalCalibration` from identified calibrants, reads `XIC_TRACE_TABLE` rather than cells, and is flagged for revision.

One property of the deviance matters for scoring. G² grows linearly with N when the reference is even slightly wrong, and a predicted spectrum always is: ×10 the counts gives ×10 the deviance. As a score it would rank intense true matches below faint ones. The overdispersed Dirichlet-multinomial (`FragmentationNoiseModel`'s α₀) is the likelihood that survives model error; G²/N (= 2·KL) is the scale-free summary.

## 5. Scoring: what is missing

In rough dependency order.

1. **A dense view of a query's evidence.** `QueryBatch.dense(level)` is deferred in `sweep.md`. It would give a torch tensor `[queries, scans, channels]` with masks for not eligible and not acquired (E2), the abundance `[queries, channels]` and the scan axis (RT, TIC, IIT, cycle). Each MS2 scan also needs the MS1 cycle it belongs to, for precursor–fragment correlation. The cell view is long-format numpy, and every feature below is a few lines on the dense form.
2. **Trace operations in `core.signal`**, batched over `[queries, channels, scans]`: Savitzky–Golay smoothing, background (moving median), local maxima with blocking, boundary walk-out. The module's docstring lists them as "to come"; none exist.
3. **Candidate peaks per query.** Build a consensus trace (EncyclopeDIA uses the ~3rd-highest normalised ion per scan), take K candidates, then refine boundaries and choose ions as `TransitionRefiner` does: correlation with the median, keeping 3–5 ions. The output is a candidates table, written alongside per-ion quantities by a multi-output reducer.
4. **Features per candidate.** Most are one-liners once the dense view exists:
   - library similarity at the apex and over the peak, with functions that already exist;
   - Scribe, ln(1/SSE);
   - co-elution: each ion's r with the consensus, and the count with r > 0.75 or 0.9;
   - MS1 isotope dot and MS1–MS2 correlation;
   - abundance-weighted signed mass error for MS1 and MS2;
   - shape: correlation to a Gaussian or EMG, width against expected;
   - matched against expected channels, longest ladder, fraction of predicted intensity matched;
   - after calibration, the RT residual in band units.

   Keep metadata (RT, m/z, charge, length) out of the learned score, or know you have put it in (algorithms doc, observation 8).
5. **Evidence without decoys.** Purified-protein searches have tens of queries, too few to estimate an FDR from. Three routes fit invariant 1:
   - a per-query e-value from that query's own score distribution over the run (EncyclopeDIA's `EValueCalculator`, already specified);
   - a chance-match null per channel from the scan's local point density, giving a binomial tail on k co-eluting matches; this needs per-scan point density by m/z, and the ledger carries only `n_points`;
   - a Counter-style likelihood ratio against background, extended to MS2, with censoring on the measured-and-empty cells only (E2).
6. **Ion counts.** The multinomial, Dirichlet-multinomial and censored likelihoods need counts, not intensities. Counter has an α(z) for MS1; nothing exists for MS2, and the detection floor is not stored.
7. **Batched fits.** A peak model per candidate (~500k queries × ~16 candidates) needs independent problems on a leading axis: vmapped L-BFGS per problem, or moment estimators. `Parametric.fit` fits one at a time.
8. **Statistics, all of it.**
   - Target–decoy q-values over populations, best candidate per query, (D+1)/T, choosing which `decoy_set` competes.
   - PEP by isotonic regression.
   - Semi-supervised rescoring (Percolator / mokapot / JavaPot's linear SVM) in torch, with folds grouped by peptide.
   - Peptide- and protein-level FDR, with protein grouping and picked-protein FDR, which need B2 fixed.

   `search/encyclopedia/percolator.py` only checks whether the jar's Percolator can start; nothing in Constellation computes a q-value.
9. **Competition across queries.** Interference, shared fragments, chimeric deconvolution and DIA group suppression (ScribeTwo) all need either:
   - a stage after the sweep over the candidates of one window and RT; or
   - a window-level reducer on the precursor-major path, whose units already hold whole window segments.

   The reducer contract forbids both inside a batch today.

## 6. Plumbing to a pipeline

1. **Calibration loop (two passes).** The PI's rule (2026-11-07): calibrants are their own entity, kept filterable before statistics.
   - **Pass 1:** calibrant queries → sweep + scorer → q-values → `rt.library_pairs` → `rt.align`.
   - **Then:**
     - a **calibrated-target builder**, mapping `rt_library` through `RtAlignment` and its band to `rt_start` / `rt_end`; it does not exist, and windows around model-scale times are refused until it does;
     - a **mass-error-over-RT fit** writing `MASS_CALIBRATION` (none native);
     - **applying a ppm(RT) correction** while matching; the sweep has only the per-bundle m/z-axis curve;
     - **tolerances re-estimated** from the calibrants.
   - **Precursor-major units cut by RT**, so calibrated windows save reading and not just matching.
2. **Result schemas.**
   - **A typed candidates table:** features with a recorded feature-set version, in place of an untyped `percolator_feature`.
   - **`pass`, `role` and `decoy_set` on the score tables.** `PRECURSOR_SCORE_TABLE` is keyed (modified_sequence, charge, acquisition, engine, is_decoy), so a calibrant pass and a target pass of one engine collide.
   - **A rule for `PRECURSOR_QUANT`:** it holds one peak per precursor, so only the chosen candidate goes there.
   - **`PSM_TABLE` does not fit DIA:** it is per spectrum, with MaxQuant columns.
3. **A native engine in the experiment directory.**
   - **The readers already need nothing jar-specific.** `results.identifications`, `rt_mapping`, `targets_from_experiment`, `summarize`'s generic view and the browser read only the tables, a `SearchManifest` with `inputs.source = {kind: 'bundle', …}`, and `_SUCCESS`.
   - **The writer is jar-bound.**
     - `workflows/encyclopedia.py` holds the engine-neutral parts privately: acquisition registration, `_write_tables`, the locks and the manifest updates.
     - `content_keyed_tables` writes the whole library per unit, which is prohibitive for a 9.56M-precursor space; write the detected subset, or a reference to the library.
   - **Places hard-wired to `engine == "encyclopedia"` or the jar:**
     - report and quantify (`workflows/encyclopedia.py:1615, 1786`; the report is the jar's `-libexport`);
     - `results.searched_library_spectrum`, which opens the library through sqlite3;
     - tolerances read from `-ftol` / `-ptol`;
     - the FDR line read from `-percolatorThreshold` (`experiment.py:243, 339`).
4. **A verb and an orchestrator.** `massspec search` always runs the jar, and `--search-mode` picks between the jar's two programs. What is missing is a native search verb, chaining search space → prediction → plan → pass 1 → calibration → pass 2 → statistics → quantities, with stage keys between passes. `quantify` gives the pattern: `--resume`, `--detach`, `--jobs`, locks. `chromatogram extract` takes a bare `--acquisition-id` that is not tied to a registered acquisition.
5. **Quantities.** Integrate the chosen candidate into `PRECURSOR_QUANT` / `FRAGMENT_QUANT` / `PRECURSOR_CHROMATOGRAM`, then roll up to peptides and proteins. Today the jar's projection is the only producer.
6. **Several runs.** Chromatogram libraries from GPF, alignment across runs (`rt.align` already takes run-to-run pairs), match-between-runs, experiment-level FDR and a native report.
7. **DDA.** A peptide-centric sweep over DDA works but is sparse. A spectrum-centric search (fragment index, ScribeTwo / Sage style) does not exist; open searches would need it.

## 7. Suggested order

1. **Fix what would corrupt a scorer's inputs or its FDR:**
   - C1 (PSM targets), C2 (compile abort), B1 (stale resume; audit every `__version__` key);
   - A1 (refuse multi-CV now);
   - E2 (carry the scan window; an `acquired` flag on cells);
   - E3 (bucket the in-memory reduce), E1 (work directories);
   - A5 (ambiguous branch), A6;
   - then B2 (decoy proteins) before any protein FDR.
2. **Give the scorer its input.** The dense view, with acquired and eligible masks and the MS1 cycle link. Add the scan's m/z range and detection floor to the ledger.
3. **Add batched trace operations and candidate peaks** to `core.signal`.
4. **Write scorer v1 as a multi-output reducer**: candidates with features, a fixed score and a per-query e-value. Check it two ways:
   - on the purified BSA / lysin run, where it must stand without decoys;
   - on a GPF fraction against EncyclopeDIA's result for the same bundle (Jaccard of precursors at 1%, apex RT agreement). The algorithms doc's plan of rebuilding EncyclopeDIA's 35-column feature rows for its own chosen peaks is the sharpest test of the features.
5. **Statistics:** TDC q-values and PEP with `decoy_set`, then semi-supervised rescoring.
6. **Two-pass calibration**, using the pieces in §6.1.
7. **A native engine writer and verb**, then quantities, protein inference and the multi-run work.

## 8. Review of the fixes (Codex, 2026-11-10)

An independent review of `fix/pre-scoring` found six defects the fixes had made. Each is fixed with a test that fails on the code before.

| # | Severity | What | Where | Status |
|---|---|---|---|---|
| R1 | P1 | Claims were sorted as one shard to put them in scan order, and a sweep spilled to disk is one shard, so its claims undid the bound that made it spill (1,600 held where main held 160) | `sweep/engine.py` `_claims_part` | fixed: claims are written for whole blocks only, so runs follow in scan order and are made into the table one at a time, as on main (835 claims, at most 64 held) |
| R2 | P1 | A group of units read MS1 for the whole group, and an MS1-only plan is one segment, so every unit read all the units' m/z slices (40 units: 800 points where each needed 20) | `sweep/precursor_major.py` `_group_task` | fixed: fragment levels are read once per group, MS1 per unit |
| R3 | P2 | A limit counted only the first rule that reached a slot, so N-term and Protein N-term rules gave forms that depended on their order, and one order broke a limit | `core/sequence/protein.py` `_plan_modforms` | fixed: a placement counts toward every rule that reaches it |
| R4 | P2 | A fixed cyclization removed residue 0's fixed modification from every form, also where a variable N-terminal modification replaced the cyclization (`[UNIMOD:1]-C` without its carbamidomethyl) | `core/sequence/protein.py` | fixed: residue 0 is the cyclization's only in the forms that keep it, and its variable modifications are offered in the forms that free it |
| R5 | P2 | Scan-major corrected the m/z of every level it read, the plan's or not, so a curve undetermined at MS1 refused a sweep of MS2 alone | `sweep/engine.py` `_make_block` | fixed: only the plan's levels are corrected |
| R6 | P2 | Two decoy methods under one set index gave two decoy proteins one accession | `library/search_space.py` | refused: a set is one method's decoys, and two under index 0 would also double the decoys competing with the targets; the CLI numbers its sets apart |
