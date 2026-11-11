**Review of the search path before scoring, second pass, 2026-11-10**

Reviewed `main` at `b051506` (PR #110 merged), the same scope as the first pass ([pre-scoring-2026-11-09.md](pre-scoring-2026-11-09.md)) plus what PR #110 added:
- conversion and the bundle, the precursor-major layout, the m/z axis, and the ledger's new scan context (acquired range, parent scan);
- the search space, decoys and decoy proteins, content ids, modification planning, ProForma read case-insensitively;
- targets → queries → channels, with library abundances;
- the sweep (scan-major in memory and on disk, precursor-major), the reduce, claims, and the reducer interface;
- the dense view (`QueryBatch.dense`), checked adversarially and then used by a throwaway scorer on real data;
- the existing scoring functions, the downstream plumbing, and the scoring decisions that are now due.

Six read-only reviews ran in parallel, one per item above. Each finding was reproduced by a script, on synthetic data or the real GPF, wide-window and purified bundles, or traced in the code where marked *plausible*. The headline findings were re-read against the source. No production code was changed. The scripts sit in the session scratchpad (`/tmp/claude-1000/-home-dbwilburn-projects-constellation/806b57b2-e0ed-4820-b3ed-8694972c7a4d/scratchpad/review/`) and are not committed. The full test suite at `b051506` passes (5,444 passed, 14 skipped, 1 xfailed), and `ruff check .` is clean.

**Verdict.** The first pass's fixes hold, also on inputs their own tests do not cover. What a scorer reads is correct on the default path, a DIA run converted from `.raw`:
- **Parity.** Every sweep path gives identical outputs: 124 adversarial cases on six paths, 1,560 comparisons, 0 differences. On the real GPF fraction with 2,500 predicted precursors (51M acquired cells), scan-major on disk, precursor-major and a spilled in-memory sweep give identical outputs.
- **The dense view equals the cell view** on 161,456 synthetic cells (40 sweeps) and 8.2M real cells.
- **Modification planning** matches a brute-force enumerator on 8,960 cases (0 mismatches).
- **No content id changed** with the ProForma case fix (checked on 140k real strings and a whole search space built at both commits).
- **The decoy population is balanced** before and after prediction and charge filtering. The decoy proteins mirror the targets for every method.

What remains is at the edges again: inputs a scorer will reach soon (ScribeTwo PSMs, imported `.dia` files, search spaces built before PR #110) and conventions that make a forgotten line expensive. In the scoring code the first pass did not test, two functions misbehave on the inputs a scorer will feed them (§4). A toy scorer on the dense view found real signal in the GPF fraction, but only once retention times were calibrated (§5). That, and the decisions in §6, set the order in §8.

## 1. Defects, ranked

Severity is for a scorer and FDR built on this path. Paths are relative to `constellation/massspec/` unless they start with `constellation/`. Numbering continues the first pass's letters (A acquisition, B search space, C targets, E sweep, S scoring), with V for the dense view.

| # | Severity | Defect | Where | Status |
|---|---|---|---|---|
| C5 | high if PSM targets are scored | ScribeTwo decoy PSMs reach the queries with no `precursor_id`. They get no library abundances, and under top-N no fragment channels, while their targets get N | `io/encyclopedia/_scribe2.py:154-158, 173`; `quant/targets.py:186-206` | confirmed |
| A7 | medium | A `.dia` import records no MS2 acquired range, and a Thermo filter with several ranges parses to none. Such scans count as having acquired every m/z, so E2 is unfixed on those paths | `io/encyclopedia/_dia_import.py:210-211`; `readers/thermo/_filter.py:47`; `sweep/reduce.py:139-144`, `sweep/dense.py:224-231` | confirmed (real GPF `.dia`) |
| V1 | medium | The dense view has intensity 0 on cells never acquired, while the similarities treat 0 as observed and take no mask. Pass `d.intensity` straight in and the E2 bias is back | `sweep/dense.py:20-24, 200-231`; `constellation/core/stats/losses.py:36-38, 301, 334, 359` | confirmed (real GPF) |
| S2 | medium | Every similarity drops negative values as the −1 sentinel, so a correlation of signed traces is wrong. Signed traces include background-subtracted, centred, log and derivative ones | `constellation/core/stats/losses.py:36` (`_valid`), `:334` | confirmed |
| S3 | medium (high if reused) | Counter's censored term clamps P(count < floor) at 1e-30. Past about 70 predicted ions, the penalty for signal predicted in an empty cell stops at −69.08, with zero gradient | `counter/channels.py:184` | confirmed |
| V2 | medium (cost) | `np.unique` over the matched cell keys, only to look for duplicates, is 74% of the dense build. On real data the dense build costs more than the toy scorer's scoring | `sweep/dense.py:216` | confirmed (real GPF) |
| S4 | medium | Injection times the bundle did not record reach `Dense.iit` as NaN, and any count term then turns NaN | ledger → `sweep/dense.py`; `counter/iit.py:44` | confirmed (RF30: 13.2% of scans) |
| S5 | medium | No detection floor is stored, and what one would mean depends on the centroider. Counter assumes a constant 1 ion | `sweep/reduce.py` (Ledger); `counter/panel.py:51` | confirmed (GPF, RF30) |
| S7 | medium (plumbing) | `PRECURSOR_SCORE_TABLE` has no `pass`, `role` or `decoy_set`, so a calibrant pass and a target pass of one engine collide; `results` silently keeps the first row | `search/schemas.py:83-95`; `search/search.py:134-147`; `results.py:261` | plausible (traced) |
| B8 | low-medium | A search space does not record its layout, so one built before PR #110 (no decoy proteins, B2's asymmetric subset) passes `predict-library` and `subset_proteins` without a word | `library/search_space.py:111-128, 985-993`; `cli/_predict_jchronologer.py:126-135` | confirmed |
| C6 | low-medium | Top-N is chosen at compile, blind to the MS2 acquired range: 42% of GPF queries have at least one top-10 channel that is never acquired | `peptide/channels.py:582-583`; `sweep/queries.py:614-624` | confirmed (real GPF, purified) |
| C7 | low-medium | A target source of one run is accepted against any bundle and stamped acquisition 0; `--assigned-scans-only` then pins another run's scan numbers | `cli/chromatogram.py:422`; `quant/targets.py:142-167` | confirmed |
| C8 | low-medium | The MS1 envelope counts the heavy atoms of isotopic-substitution modifications (SILAC, TMT) as natural atoms: M+0 share 7–9% low, M+2 11–16% high; m/z exact | `peptide/envelope.py:107-108, 161-163` | confirmed |
| E7 | low-medium | An in-memory sweep holds every reducer output whole. `memory_rows` bounds the map only, so a per-cell output outgrows it unspilled (1.11 GB held at 5% of `memory_rows`) | `sweep/engine.py:831-858, 952-988` | confirmed (real GPF) |
| E8 | low-medium (claims) | On the map side a whole block's claims are held, not `run_rows` (14.9× in a test), and every later window group writes a run of its own | `sweep/engine.py:1118-1174` | confirmed |
| V3 | low-medium | A scan with no points is on no query's axis (cells and dense alike). The trace shows a gap, not a zero, and an MS2 slot whose parent is empty gets −1. This is the "mirror problem" noted under E2, still open | `sweep/schedule.py:193`; `sweep/precursor_major.py:270` | confirmed (synthetic; real runs have none) |
| S6 | low-medium | `EMGPeak.integrate` stops at apex + 4σ + 3τ (−1.7% of the area at τ = 5σ, −3.7% at 20σ) and cuts gradients through its limits | `constellation/core/stats/peaks.py:344-361, 472-490` | confirmed |
| A8 | low | The layout writer splits a scan whose points span 3+ build batches when the held limit trips in the middle one; MS1 `point` restarts. Unreachable on centroids (needs > 2²⁰ points in one scan) | `precursor_major.py:140-152, 273` | confirmed |
| A9 | low | Regression from A5's fix: `massspec summarize` raises on a bundle whose curve is undetermined at MS1 | `bundle_summary.py:66` | confirmed |
| A10 | low | `parent_scans` takes any recorded master scan > 0 unchecked (itself, a later scan, one not in the run). A CV recorded on MS2 but not MS1 passes the multi-CV refusal and leaves every MS2 scan without a parent | `acquisition_structure.py:126-131, 144-147` | confirmed |
| A11 | low | An MS3 after supplemental activation (EThcD) records the MS2's supplemental activation, not its own | `readers/thermo/_scans.py:447`; `readers/thermo/_filter.py` | confirmed |
| A12 | low | `scans.scan_table(levels=[2])` filters MS1 out before deriving parents, so every DIA MS2 scan gets a null parent (no caller passes `levels` today) | `scans.py:222, 227` | confirmed |
| A13 | low | The multi-CV and MSX refusals do not reach the browser (`scans.py`) or the `.dia` export | `scans.py:236-248`; `io/encyclopedia/_dia_write.py:286` | browser confirmed; export plausible |
| A14 | low | `.dia` 0.3.0 / 0.4.0 files fail import with a raw `sqlite3.OperationalError`, not a version refusal | `io/encyclopedia/_dia.py` | confirmed |
| B9 | low | The search-space resume key does not cover `proteases.json` or `elements.json` | `library/search_space.py:114-137` | confirmed |
| B10 | low | B4's remainder: `reverse-c-fixed` still moves residue 0 from under residue-specific N-terminal modifications other than 26/27/28, among them UNIMOD:385, the EncyclopeDIA reader's pyro-carbamidomethyl (524 of 32,683 decoys) | `library/decoys.py:245, 251` | confirmed |
| B11 | low | `PEPTIDE_MODIFICATION_TABLE.rule` depends on rule order when two rules reach one placement (R3's provenance remainder) | `constellation/core/sequence/protein.py:727-728, 987` | confirmed |
| B12 | low | A ProForma group carrying its modification twice keeps one in the mass; case folding now merges `#G1` into `#g1` (1174.42 → 1094.45 Da) | `constellation/core/sequence/proforma.py:781` | confirmed |
| B13 | low | A name under another CV's prefix (`M:`, `R:`, `X:`, `G:`) is looked up in UNIMOD; `[M:phospho]` is now UNIMOD:21 | `library/ids.py:132-135` | confirmed |
| B14 | low | A labile `{glycan:Hex}` keeps its lower-case CV prefix where `[glycan:…]` is normalised: two keys for one form | `constellation/core/sequence/proforma.py:425` | confirmed |
| C9 | low | A partly linked PSM table (ids plus null ids from another raw file) passes as one run | `quant/targets.py:149-159` | confirmed |
| C10 | low | A library row with a null intensity becomes abundance 0 (predicted absent), not NaN | `library/fragment_abundance.py:224, 288` | confirmed |
| C11 | low | The CLI manifest records only the last compile chunk's abundance report | `cli/chromatogram.py:450`; `library/fragment_abundance.py:215-216` | confirmed |
| E9 | low | The spill's free-space estimate runs low when the spill comes in an early block (0.16× of rows in a test; about 42 against 105 GB projected at GPF scale); on-disk sweeps are not checked | `sweep/engine.py:343-348, 846-897` | confirmed |
| E10 | low | Scan-major never resolves `mz_axis='latest'`; each worker resolves it again, and a curve saved mid-sweep is named on a result corrected with the older one | `sweep/engine.py:762-767, 832, 1088, 1371` | confirmed |
| E11 | low | `point_claims.query` holds a taken plan's local numbers where the summary holds global ones | `sweep/claims.py:167` | confirmed |
| E12 | low | A killed sweep leaves its spawned workers alive and its directories behind | `sweep/engine.py:913-933, 1380-1414`; `sweep/precursor_major.py:597-642` | confirmed |
| V4 | low | `dense(queries=)` is not checked: unsorted input raises an obscure error, a duplicate comes back empty | `sweep/dense.py:154-161` | confirmed |
| V5 | low | `dense(dtype=)` is not checked: float16 turns every 1e10 intensity into `inf` | `sweep/dense.py:199` | confirmed |
| V6 | low | A reducer that builds dense views without declaring `zero_fills` gets buckets sized by matches alone (5× larger in a test) | `sweep/reduce.py:373-379`; `sweep/engine.py:1297-1316` | confirmed |
| V7 | low (cost) | Every `dense()` call re-sorts the ledger and re-filters the batch's matches; one query at a time cost 7.9× one build | `sweep/dense.py:196-205`; `sweep/reduce.py:115-118` | confirmed |
| V8 | low | `parent_slots` gives −1 for four reasons a scorer cannot tell apart, and only the left MS1 bracket | `sweep/dense.py:111-137` | confirmed |
| S8 | low | `FragmentationNoiseModel.log_prob` has no mask: NaN abundance gives NaN, a −1 sentinel a finite wrong value | `spectra/noise.py:76-90` | confirmed |
| S9 | low | In float32 the similarities are not scale-free at extremes (cosine 0 at 1e-24 and ≥ 2e19) | `constellation/core/stats/losses.py:63` | confirmed |
| S10 | low | The dense view does not carry `point_index`, so one point matched by two channels, or shared between queries, is invisible without going back to `batch.matches` | `sweep/dense.py:62-104` | traced |
| S11 | low | RT pairs and `RT_ALIGNMENT` do not record the library's clock; JChronologer times enter as "seconds" | `rt/pairs.py:32-41, 95-132`; `rt/__init__.py:158, 285` | traced |

### Notes on the ones to fix first

**C5: PSM decoys without precursor ids.** The ScribeTwo reader derives content ids for targets only (`precursor_id` is written with `mask=is_decoy`), as `PSM_TABLE`'s comment says. C1 carried the column through, but a decoy still has nothing to look up. Measured against the real GPF library (which holds the decoys):
- 2,000 target PSMs average 10.00 fragment channels under top-10; 2,000 decoy PSMs get 0.
- Without top-N, 100% of decoy channel abundances are NaN, against 0% of targets'.
- Nothing warns; the compile report counts `no_positive_fragment: 2000`.

Target–decoy competition on such queries passes nearly every target. Fix: derive the missing id (`content_ids(canonical_precursor_key(modseq, z))`) in the reader or in `targets_from_search`, which restores 2,000 of 2,000. Also refuse a compile whose `no_positive_fragment` is lopsided between targets and decoys.

**A7: imported runs claim every m/z.** `_dia_import` writes `scan_window_*` for MS1 only, and a NaN bound fails both comparisons, so every MS2 cell counts as acquired. The same GPF run, 1,000 precursors:

| Same run | MS2 cells | Matched | Share matched |
|---|---|---|---|
| Converted from `.raw` | 3,444,688 | 2,663,373 | 77.3% |
| Imported from its reference `.dia` | 4,597,282 | 2,663,432 | 57.9% |

Any matched/expected feature or censored likelihood then takes different values for one run depending on how it arrived. Fix: write each level's range into the `.dia` the writer emits and read it back. For a foreign `.dia`, parse the method text it carries ("Scan Range (m/z) = 200-1500"), or mark the range unknown so a scorer can choose. Parse multi-range filters.

**V1 and S2: two conventions for "missing".** The dense view marks a never-acquired cell by `acquired = False` and puts 0 in `intensity`. The losses module drops only NaN and negative values, and treats 0 as an observed zero. Neither is wrong alone, but one forgotten line reintroduces E2. On the real GPF run with library top-10, 3.7% of eligible MS2 cells are unacquired, and 40.6% of MS2 slots get a different library cosine with and without masking (95th percentile +0.047, largest +0.31). The same rule makes the similarities wrong on signed traces (S2). On a trace with 15 of 41 points negative, Pearson gives r = 0.339 where numpy gives 0.795, and anti-correlated tails give +1.000 where numpy gives −0.790. Fix both in one place: a `mask=` argument on every similarity (the sentinel rule kept as the default for spectrum vectors) and `Dense.observed()` returning NaN where not acquired.

**S3: the censored term saturates.** `log(gammaincc(floor, λ).clamp(min=1e-30))` is −69.08 for any λ past about 70 (floor 1). The exact value at λ = 10⁴ is −10,000, and the gradient there is 0. Real MS1 cells run to hundreds or thousands of ions (nb42c: a floor of about 200), so Counter today barely penalises predicted signal in an acquired, empty cell. Any MS2 reuse inherits it. Fix: compute it in log space (−λ + logsumexp over j < floor of j·log λ − lgamma(j+1)), in `core.stats` beside `Poisson`, which should gain a `log_cdf`.

**V2: the dense build's cost.** On one real precursor-major batch (23 queries, 376,510 MS2 cells, 283,169 matched) the build took 0.220 s, 456 ms of it in `np.unique`; `cells(2)` took 0.126 s. The matches arrive sorted, so `(np.diff(cell) > 0).all()` suffices, with `np.unique` kept as a fallback. Every tensor stays identical and the build drops to 0.051 s. Projected to a GPF fraction against the whole human space (about 11.2B padded cells), the check alone is about 65 CPU-minutes against the sweep's 87. The plan's "6× faster than the cell view" was measured on a synthetic run with few matches; real ion-trap data at 0.35 Da matches 81% of acquired cells.

**B8: old search spaces pass silently.** `SEARCH_SPACE_LAYOUT` exists only inside the `_SUCCESS` digest, not in the library's metadata. `search-space --resume` rebuilds an old directory, but `predict-library --from-library OLD` takes it as it is. On 60 human proteins built at `730d719`, the prediction ran on all 43,520 precursors with no warning. It had 0 decoy proteins, and a 10% target filter then cost `subset_proteins` 124 of 8,204 decoys. The human space and GPF prediction in the earlier session's scratch are such inputs. Fix: write the layout into `x.constellation.search_space`, and have readers warn on or refuse a missing or older one.

**C6: top-N blind to the acquired range.** With library abundances, b/y, fragment charge ≤ min(3, z) and top 10:

| Run | Queries with ≥ 1 never-acquired top-10 channel | 2+ / 3+ / 4+ | Predicted abundance lost (median) | If chosen among acquirable ions: < 10 usable |
|---|---|---|---|---|
| GPF 900–1000 (HCD; 50k sample) | 42.1% | 51.7% / 32.3% / 29.4% | 7.7% | 0.2% |
| Purified BSA + lysin (CID; all 3,512) | 65.3% | 66.0% / 59.4% / 71.5% | — | 2.3% |

On the GPF run these are mostly y1⁺ below 200 Th and 1⁺ y ions above 1,500 from 2+ precursors. "Top 10" is effectively top 9 on half the 2+ precursors. Targets and decoys lose alike on GPF (42.2% vs 42.0%) but not on the purified run (69.0% vs 61.6%). Fix: give `compile_queries` the MS2 range per isolation window and choose top-N inside it; at least count never-acquirable channels in the plan and summary.

## 2. The first pass's findings

Every fix holds, including on inputs its own test does not cover. Each holds in full except for the remainders below:

| First pass | Remainder found now |
|---|---|
| A1 (FAIMS) | A CV recorded on MS2 but not MS1 passes the refusal and loses every parent (A10); the browser is not refused (A13). The message says "Split it by CV first", but no command does. |
| A2 (bounded layout build) | A scan spanning 3+ build batches is split (A8), unreachable on centroids. The wide-window layout rebuilt in pieces (46 files a level instead of 16) gives all 294,142,728 rows identical and identical sweeps. |
| A4 (MS3, MSX) | MS3 activation after supplemental activation (A11); MSX not refused in the browser or the `.dia` export (A13). |
| A5 (undetermined branch) | `massspec summarize` regression (A9). |
| B1 (stale resume) | `proteases.json`, `elements.json` not keyed (B9); a search space does not record its layout for readers (B8). No stage key in the package keys on `__version__` any more. Five `--resume` paths still trust a bare `_SUCCESS` for any inputs: the EncyclopeDIA `predict-library` backend, `library-export`, `collision-filter`, `classify-novel-peptides`, and the frozen transcriptome pipeline. |
| B4 (`reverse-c-fixed`) | Other residue-specific N-terminal modifications (B10). |
| B7 and the ProForma case change | B12–B14. |
| C1 (PSM targets) | ScribeTwo decoys lack `precursor_id` (C5); a one-run source is not checked against the bundle (C7); partly linked tables (C9). |
| C4 (an ion named twice) | A null-intensity row scores 0 (C10). |
| E2 (unacquired cells) | `.dia` imports and multi-range filters (A7); empty scans (V3). |
| E3 (bounded in-memory sweep) | Outputs held whole (E7). |
| R1 (claims) | The map side (E8). |
| R3 (limits across rules) | Provenance rows still depend on rule order (B11). |

## 3. Checked and found correct

- **Sweep parity.** 124 cases on two adversarial runs (DIA with staggered, overlapping and nested windows, a precursor exactly on a bound; DDA; scan-number gaps, empty scans, unbounded scans, a NaN RT, acquired ranges partial, null and one-sided). Each case ran in memory, with a forced spill, on disk with 1 and 4 workers, and precursor-major with 1 and 4 workers, across levels, all three rules, RT windows, `scan_range`, `rt_range`, claims, small buckets, queries without precursor m/z, `charge_range`, gates, `min_isotopes` and an m/z curve. That gives 1,560 comparisons of about 8.0M rows a path, with identical ledgers: **0 differences.** Every query arrived in exactly one reducer call on every path. No multi-query call exceeded `reduce_rows` or `batch_cells`.
- **Real GPF parity.** 2,500 predicted precursors (half decoys), MS1 + MS2 top 10, 0.35 Da (ITMS at both levels), no RT window: 50,953,262 acquired cells, 81% matched.

| Path | Sweep wall | Largest process | Outputs |
|---|---|---|---|
| Scan-major on disk, 4 workers | 51.1 s | 1.80 GB | reference |
| Precursor-major, 4 workers | 38.4 s | 1.23 GB | identical |
| In memory, one process (spilled at 8,389,007 rows) | 74.4 s | 2.54 GB | identical |

- **Dense view.** It equals `cells()` field by field (values, masks, channel identity, abundance, rt, TIC, IIT, parent row, slot order, clean padding):
  - on 10 synthetic scenarios × 4 paths: 161,456 cells;
  - on 400 real GPF queries with library top-10: 49 batches, 8,178,452 cells.

  A gated view equals the ungated view of the passing queries. `dense_chunks` gives every query exactly once at any budget. Float32 values stay within 6e-8 relative at 1e10–1e11, and every index is int64.
- **Abundances and identifiers.** 0 and NaN survive compile → write → read → `cells` → `dense` on 6 paths with 0 mismatches. `target_id`, `is_decoy`, `decoy_set` and `precursor_id` through a batch's local plan equal the whole plan's on every path. `Uniform` shares sum to 1 (within 6e-8) over 160 settings groups. M+0 equals the library's `precursor_mz` within 3e-10 ppm on 53,512 precursors, including global labels.
- **Search space.**
  - A brute-force enumerator written from the documented rules gives 0 form, count and origin mismatches on 8,960 cases (46,998 forms), and the modification table matches the forms on about 324k peptides.
  - `format(parse(s)) == s` and `content_id(s) == peptide_id` hold for 71,938 strings. The same space built at `730d719` and now is identical row for row. 140k strings from `ss_full`, the purified space and two EncyclopeDIA results keep their keys.
  - Decoy proteins mirror their targets over 7 decoy configurations (21,874–65,613 decoys each). Shared peptides' decoys sit on every decoy protein, and I/L twins have twin decoys.
  - Balance is exact by charge, m/z, length, modification count and missed cleavages, through JChronologer's symmetric refusals. After `--min-charge-probability 0.2` no axis is significant (largest χ² 4.13 on 14 degrees of freedom).
- **Acquisition.**
  - The layout in pieces is valid on 6 synthetic configurations.
  - Parents: 100% of MS2 scans on the GPF, wide-window and purified bundles have one. On the DDA run the recorded master scan equals the derived parent for all 179,532 MS2 scans.
  - Every Thermo bundle's stored ranges equal the method text, and no scan's range excludes its own window.
  - The three Thermo bundles have no null injection times, no empty scans and no scan gaps.

## 4. Gaps a scorer will hit (new, or changed since the first pass)

**Acquisition**
- **The ion-trap CID low-mass cutoff.** On the purified CID run nothing is measured below about 0.28 × precursor m/z, but the ledger calls 200–1500 acquired. 1.0% of fragment channels of real identifications fall in that band and are zero-filled as measured.
- **Unread scans look like empty scans.** In a ranged sweep, unread scans have `n_points` 0, the same as empty ones.
- **A dropped MS1** makes its cycle's MS2 take the previous cycle's MS1 as parent, a cycle late.
- **The acquired-range test** compares theoretical m/z with the raw scan window. It ignores the m/z-axis shift and the tolerance, so channels within about 0.15 Th of a bound (RF30's shift) can fall on the wrong side (*plausible*).
- **Unchanged:** isolation-offset sign (0 on every scan of every bundle here), injection-time nulls, the curve key not pinning values.

**Channels and targets**
- **Overlaps and generation.** Overlapping channels are still unflagged. No new channel kind is generated, and `fragment_charge_below_precursor` is still not a CLI flag.
- **MS1 envelope precision.** The envelope's cut-off share is not booked (median 13–19%, up to 58% at three isotopes). The binned ¹³C spacing puts M+1 to M+3 3–6 ppm off the exact isotopologue m/z: immaterial at 0.35 Da, but a large part of a 10 ppm Orbitrap tolerance.
- **Identifiers past the summary.** `query_summary` has no `precursor_id` or `peptide_id`. `xic_trace` has only `target_id`, and its uniqueness is never checked.
- **Target sources.** `targets_from_search` takes every scored PSM (ScribeTwo's best and its others) with no q-value filter. `targets_from_experiment` drops `is_decoy` / `decoy_set`, which is harmless while the jar writes target quantities only.

**Search space and proteins**
- **Protein-level FDR** has decoy proteins now, but no `decoy_set` column on `PROTEIN_TABLE` (the set is in the accession only), no grouping, no decoy protein lengths, and no native writer of `PROTEIN_SCORE_TABLE`.
  - The mirror drifts after per-precursor filters: at charge probability 0.2, 36 of 60 proteins differ from their decoy by up to 7 peptides.
  - `subset_precursors` keeps empty proteins.
- **Decoy-named FASTA entries** are now refused on a clash with a decoy made here, and warned about for common prefixes. Mascot `###REV###` and OpenMS `_DECOY` suffixes are not warned about.
- **JChronologer refusals** include UNIMOD:385 at the N-terminus, the EncyclopeDIA reader's spelling of pyro-carbamidomethyl. With Deamidated N/Q variable, refusals dropped 64% of one space.

**Sweep**
- **Counts.**
  - A zero-filling reducer receives about half of `reduce_rows` in cells: matched cells are weighted twice, and unacquired cells are weighted though dropped. Dense padding is not counted.
  - `n_eligible_ms*` counts (scan, query) pairs, not acquired cells.
- **Memory.** Precursor-major has no memory knob: one window segment's points are unbounded by any setting (1.2 GB a worker measured).
- **Disk.**
  - On-disk sweeps have no free-space check.
  - The CLI's `.sweep/` is removed only after success.

**Dense view (from the toy scorer)**
- **Per-query metadata.** `target_id`, `is_decoy`, `decoy_set` and `rt_library` are not on `Dense`; they come through `plan.targets.take(local)`.
- **Channel identity.** `loss` indexes `loss_ids`, which `Dense` does not carry, and there is no evidence key.
- **No charge grouping.** Under `charge_range`, the MS1 channels of several charges share one axis with no group index.
- **Mixed dtypes.** Values are float32, while rt, TIC, IIT and m/z are float64; MPS has no float64.
- **No axis length.** Per-query axis lengths are not a field. A window clamped at the padded S−1 made one feature depend on which queries shared the batch.

## 5. The reducer contract and the dense view as they stand

- **A query arrives once, whole, on every path.** Scan-major in memory is now cut into the same buckets as on disk. Precursor-major calls hold at most `batch_cells` eligible cells.
- **MS1 and MS2 come in the same call**, and `batch.scans.parent` / `SWEEP_SCAN_TABLE.parent_scan` link each scan to its parent, identically on all paths.
- **Query numbers are local in precursor-major.**
  - Outputs must go through `plan.global_query`; `point_claims` does not (E11).
  - `batch.settings.mz_axis` is resolved on precursor-major and still `'latest'` on scan-major (E10).
- **A cell row.** An unmatched cell outside its scan's acquired range is no longer a row; a matched one is. A scan with no recorded range counts as acquired (A7).
- **Gated reducers see `batch.passing()` only.** With every reducer gated, precursor-major decides the gate first.
- **A query's rows may depend only on that query.** Claims are still refused on precursor-major.
- **Outputs.** They are held whole in memory (E7). On disk there is one part per bucket (scan-major) or per unit (precursor-major).
- **Failures.** A failing call is reported after the stage's other tasks; precursor-major runs up to 2 × workers more first.
- **Now available:** the dense view, the scan's acquired range and its parent.
- **Still missing:** a scan's unmatched points and local point density, neighbouring queries, the next MS1 slot and interpolation weights, a detection floor, `point_index` on the dense view, and anything across batches.

What the dense view delivers against the first pass's request (§5.1 there):

| Asked for | Delivered |
|---|---|
| `[queries, scans, channels]` torch tensors | yes: intensity, signed m/z error, `matched`, `acquired` |
| Masks for not eligible / not acquired | yes: `slot_valid` × `channel_valid`, and `acquired` (intensity 0 there, not NaN: V1) |
| Abundance `[queries, channels]` | yes, float32: NaN not modelled, 0 predicted absent |
| Scan axis with RT, TIC, IIT | yes, per slot, float64 (IIT NaN where unrecorded: S4) |
| The MS1 scan each MS2 belongs to | `parent_slots`: left bracket only; no cycle index, weights or interpolation (V8) |
| Scan m/z range, detection floor | range in the ledger, used for `acquired`, not on `Dense`; no floor (S5) |

## 6. Scoring: what exists

| Piece | Where | State | Changed since 2026-11-09 |
|---|---|---|---|
| Dense view | `sweep/dense.py`, `QueryBatch.dense` | Correct (§3). Costly to build (V2). Missing `point_index`, floor, next-MS1 bracket | new |
| Similarities: cosine / normalized dot, Pearson, spectral angle, spectral entropy, KL, G², `compare_spectra` | `constellation/core/stats/losses.py`, `spectra/similarity.py` | Correct on spectra. The sentinel rule breaks signed traces (S2); no `mask=` (V1); float32 extremes (S9). Still no caller on the search path | `l1_normalize` masking (S1) |
| Dirichlet-multinomial fragment model | `spectra/noise.py` | One peptide's scans per fit; no mask (S8) | — |
| Peak shapes: Gaussian, EMG (exact mode), HyperEMG | `constellation/core/stats/peaks.py` | Values and gradients finite for τ/σ from 1e-4 to 1e4. Parameters are 0-d, one peak a module. `integrate` truncates the tail (S6) | — |
| `Parametric.fit`, `core.optim` (LBFGS, Adam, DE) | `constellation/core/stats/parametric.py`, `constellation/core/optim/` | One problem per fit. On 512 Gaussian peaks: one at a time 6.2 s; one L-BFGS over the stacked problems 216 ms; a 40-line batched Levenberg–Marquardt 35 ms, all three equally accurate (apex error median 0.031 s). Nothing in the package fits many independent problems | — |
| Trace operations | `constellation/core/signal/` | Still none: no smoothing, background, peak picking or boundaries | — |
| Counter (MS1 likelihood) | `counter/` | The channel terms broadcast to `[Q,S,C]`; `iit.accumulated_count` and the EMG log-pdfs are reusable. `Progenitor`, `Panel` and `estimate_n` are per target and read `XIC_TRACE_TABLE`. The censored term saturates (S3) | per-cell floor (2026-11-09) |
| Evidence gate | `sweep/evidence.py` | A pre-filter. Decided first on precursor-major. Defaults measured on BSA/lysin only | — |
| RT alignment | `rt/` | Ready for a native pass 1. No calibrated-target builder; the library's clock is not recorded (S11) | — |
| Statistics | — | None: no q-value, PEP, isotonic regression or rescoring code | — |

### A toy scorer on the dense view

To see what a scorer meets, a throwaway reducer was written on the dense view (scratch only). It computes a smoothed library cosine per MS2 slot, a 7-slot co-elution correlation, an apex, the MS1 isotope dot at the apex's parent, and a naive combined score.
- **Targets:** 1,200 precursors EncyclopeDIA found at 1% in this fraction, plus 300 random library targets it did not find.
- **Decoys:** 1,500 library decoys stratified by charge and 1-Th m/z bin. This is sampling, not `decoy_of` pairing.
- **Sweep:** library top-10, precursor-major, 4 workers, 0.35 Da.

| | No RT window | Crude calibrated window (cubic fit on EncyclopeDIA's IDs, ±234.5 s) |
|---|---|---|
| Padded cells / padding | 63.8M / 0.02% | 5.9M / 18.2% |
| Dense build / toy scoring (ns per cell) | 1,262 / 555 | 646 / 505 |
| Wall, largest process | 62 s, 1.28 GB | 22 s, 0.92 GB |
| Found targets' apex within 30 s of EncyclopeDIA's (median \|Δrt\|) | 91.2% (5.6 s) | 93.4% (5.1 s) |
| AUC, found targets vs decoys: smoothed cosine / co-elution / combined | 0.877 / 0.779 / 0.688 | 0.915 / 0.927 / 0.839 |
| AUC of MS1 features | 0.47–0.54 | 0.47–0.54 |
| Crude target–decoy count at 1% / 5% | 0 / 91 | 202 / 414 |

**Caveat.** This is not an FDR. Most targets are EncyclopeDIA's own identifications, the window was fitted on them, and targets and decoys were not matched on RT. Three things still stand out:
- **Calibrated RT is the largest lever.** It took the 1% count from 0 to 202.
- **MS1 carries no signal at 0.35 Da on this ion-trap MS1.** A scan holds about 1,139 centroids over 350–1250 Th, a median spacing of 0.75 Th, so a ±0.35 Da channel finds some point about 93% of the time by chance.
- **A per-query z-score of the apex was useless** (0 targets at 5%). An e-value needs a real tail model, not a median and MAD.

On DDA (computed from the lab DDA run's metadata, not swept), MS2 padding is 44% without a window and 79% with one. The MS1 axis is 8,977 scans a query (80.8M MS1 cells against 4.3M eligible MS2 cells), so on DDA MS1 sizes memory.

Friction met (beyond V1–V8 and §4's dense items):
- **Trace operations had to be written.** About 100 lines: masked smoothing, a windowed leave-one-out Pearson via `unfold`, per-row interpolation via `searchsorted`, and a masked Pearson. The losses reduce over channels only.
- **The match rule is a sweep-time choice.** EncyclopeDIA sums the peaks within tolerance, which means sweeping with `sum`; the scorer cannot choose afterwards.
- **A reducer cannot see a scan's other points.** So `xCorrLib`, `xCorrModel` and the immonium part of `lnSp` (3 of EncyclopeDIA's 18 scores) are out of reach, as is a chance-match null from local density. Immonium channels are not generated.

## 7. Decisions for the scoring session

Each is a question the PI's answer changes the code for, with the evidence and a recommendation. "Where" names the modules, keeping to the import DAG: generic trace operations, fits and statistics in `core`, mass-spec specifics in `massspec`.

**D1. What is scorer v1's score?**
- **Options:**
  - (A) a count likelihood: Counter extended to MS2, elution shape × predicted fragment shares × gain, censored on acquired, empty cells;
  - (B) EncyclopeDIA-style per-candidate features, then semi-supervised rescoring;
  - (C) both, with likelihood terms as features.
- **Evidence:** invariant 1 favours A, but A's inputs are missing or unreliable today. There is no floor (S5), injection times are NaN on 13% of RF30 scans (S4), the censored term saturates (S3), the gain is "irreducible" (`counter-real-data-deconstruction.md` §7), and HyperEMG fits hit the τ_R bound on 26 of 29 peptides (`peak-boundary-refinement-research.md`). EncyclopeDIA reaches 31,833 precursors on the wide-window run with 18 scores plus 3 calibration features, and its 35-column feature file is a seam to test against.
- **Recommendation: C.** v1 computes, per candidate on the dense view:
  - EncyclopeDIA's scores: primary (Scribe × co-elution), correlation to a Gaussian and to the precursor, ion counts, mass errors, isotope dot;
  - two likelihood terms that need no gain: a Dirichlet-multinomial fragment-share log-likelihood ratio at the apex, and an elution-shape log-likelihood ratio on the consensus trace;
  - a fixed score with no learning.

  The censored count likelihood on MS2 is v2, after D2.
- **Where:** `massspec.search.score` (new) for the features; the likelihood terms in `core.stats`.

**D2. Intensities or ion counts?**
- **What is there now:**
  - On instrument-centroided ion-trap data (GPF), the floor is constant in intensity × injection time (min(I)·IIT median 359 at MS2, 420 at MS1; corr(log min I, log IIT) = −0.83). But values above it are not quantized, so the gain stays unidentified.
  - On Thermo-centroided profile data (RF30), the floor is the centroider's 1.0 a.u., an artefact.
  - Orbitrap centroids carry `peak_noise` per peak.
- **Options:**
  - (A) intensities only, with scale-free features;
  - (B) counts from a per-run gain prior plus the per-scan floor;
  - (C) counts with the gain fitted on calibrants by shot noise.
- **Recommendation:** A for v1. Now: carry a per-scan `floor` (lowest stored intensity) and the scan's `centroider` into the ledger and `Dense`, and impute missing injection times from neighbours of the same level and window, with a flag. Then v2 can work in I·IIT units with an overdispersed count model whose signal-per-ion is fitted on pass-1 calibrants, refused on Thermo-centroided profile data until there is a profile rule.

**D3. How are candidate peaks found and shaped?**
- **Options:**
  - (A) K candidates from a consensus trace, with ±2w blocking (about 16 per entry), then `TransitionRefiner`-style boundaries, as EncyclopeDIA does;
  - (B) an elution model fitted per query over the whole axis (Counter's DE + L-BFGS per target);
  - (C) A, then a shape fitted to each surviving candidate, all at once.
- **Cost:** at GPF scale (526k queries × 16 candidates = 8.4M fits), one at a time is about 28 h, and the batched Levenberg–Marquardt about 10 min single-threaded.
- **Recommendation: C**, batched over `[Q, C, S]` with `slot_valid` masks:
  - trace operations in `core.signal`: `smooth` (Savitzky–Golay normalised by valid weights), `background` (moving median), `peaks` (max-pool non-maximum suppression, then top-k), `boundaries` (walk-out), `interpolate` (per-row, masked, along RT);
  - fits in a new `core.optim.batched_least_squares` over functional peak forms taking `[B, P]` parameters.

  Trace operations run on slot index, as EncyclopeDIA does. Shape fits run on time: cycles are irregular (RF30 2.46 s median, up to 10.3 s).

**D4. Where does competition between queries live?** (interference, shared fragments, chimeric spectra)
- **Options:**
  - (A) inside a reducer: forbidden today, since a query's rows depend only on that query;
  - (B) a window-level reducer contract on precursor-major (10⁴–10⁵ queries a segment at proteome scale);
  - (C) a stage after the sweep, over the candidates table grouped by (acquisition, window, RT).
- **Recommendation: C.** Each chosen candidate carries its apex-region point keys (needs S10's `point_index`) and per-ion intensities. The stage computes the shared-point fraction, the best competing score within ±w in its window, and ScribeTwo's sibling and other-charge features; NNLS deconvolution is v2. The reducer contract stays, and the stage serves both layouts.

**D5. What is the evidence when decoys are too few to count?** (purified proteins, tens of queries)
- **Options:**
  - (A) a per-query e-value from its own per-slot score distribution over the run (EncyclopeDIA's `EValueCalculator`, a tail fit, not the median/MAD z-score the toy showed useless);
  - (B) a binomial chance-match null from local point density, which needs a per-scan point histogram by m/z made in the map;
  - (C) a likelihood ratio against background (after D2);
  - (D) a large decoy population: `shuffle-interior:sets=100` gives about 10⁴ decoys for 100 targets, an empirical null to p ≈ 1e-4. It is still a population, so it fits the PI's rule.
- **Recommendation:** A + D for v1, and B as a feature once the map can count points. Report a p-value per target, and claim no FDR below a minimum number of targets.

**D6. How is FDR estimated, and is a protein-level pairing allowed?**
- **Constraint:** EncyclopeDIA competes each target with its own decoy before Percolator, which the PI has ruled out. JavaPot competes on a spectrum key, which a peptide-centric DIA search does not have.
- **Options:**
  - (A) separate target and decoy populations, q = π₀ · (D + 1) / T with π₀ = 1 (conservative, no estimation);
  - (B) the same with π₀ estimated by mix-max (Keich et al. 2015; Percolator's `-y`);
  - (C) competition on a constructed key (window × RT), which is non-standard.
- **Recommendation: A for v1, B behind an option once checked against entrapment.** Further rules:
  - `decoy_set == 0` competes; sets ≥ 1 are entrapment, scored and left out of q-values, with FDP by the combined estimator N_E(1 + 1/r)/(N_T + N_E).
  - Precursor level takes the best candidate per query; peptide level the best precursor per modified peptide, with every charge kept in the quantities.
  - PEP by isotonic regression (PAVA).
  - A gated-out query gets a candidates row with a null score, so both population sizes are known.
- **For the PI:** picked-protein FDR keeps the better of each target protein X and its decoy `DECOY_X` before counting. That is a pairing, made possible only by the accession naming, which was built from `decoy_of`. Either allow it at protein level and make it explicit (a column naming the target accession on decoy proteins), or use plain competition on protein scores, which is known to inflate at proteome scale. Protein grouping (parsimony from `PROTEIN_PEPTIDE_EDGE`, I/L classes) is needed either way.
- **Where:** `core.stats.fdr` (q-values, π₀, PAVA); `massspec.search.fdr` (which sets compete, levels, grouping).

**D7. Rescoring: in-house, Percolator, or none for v1?**
- **Options:**
  - (A) none for v1;
  - (B) a torch port of JavaPot (about 7,700 lines, deterministic, checkable value for value);
  - (C) Percolator: 3.01 cannot start on this WSL kernel, and no other build is installed.
- **Recommendation:** A, then B, with folds grouped by I/L-normalised stripped peptide, raw RT / m/z / mass / charge kept out of the features, and JavaPot's start direction and fallback kept. Check the port on EncyclopeDIA's own `features.txt` against Percolator's 31,833, which isolates the rescorer from our features.
- **Where:** the trainer in `core.stats`; the feature rows in `massspec.search.rescore`.

**D8. How do calibration and the two passes work?**
- **Why it is urgent:** the toy scorer shows retention time is the largest lever. EncyclopeDIA puts it in as a feature after one unwindowed pass; a windowed second pass puts it in as a window.
- **Pass 1** (`role='calibrant'`, `pass=1`): the v1 scorer on a 5–10% random subset of targets, with decoys in proportion, no RT window and wider tolerances. It emits, per precursor:
  - the chosen candidate's apex RT and boundaries;
  - q-value, `is_decoy`, `decoy_set` and ids;
  - abundance-weighted signed MS1 and MS2 ppm errors with ion counts.
- **Fits:**
  - RT with `rt.align`;
  - ppm(RT) with `core.stats.density.grid_kde_2d` and `monotone_path` (moves both ways), written to `MASS_CALIBRATION`.

  Fit on each precursor's identification row only, never every match of a passing peptide: ScribeTwo does the latter, and its 7.3-min RT window became 68.1 min. Use MAD-scaled bands.
- **Back into the sweep:**
  - `quant.targets.calibrate_targets(targets, alignment, width)` fills `rt_start` / `rt_end`;
  - a per-scan ppm shift is applied where `mz_axis` is (`sweep/engine.py:579`);
  - precursor-major units are cut by RT.
- **Pass 2:** the whole space, calibrated. Calibrants are kept filterable before statistics (PI, 2026-11-07).

**D9. What does the output look like?**
- **Candidates:** a typed, wide `CANDIDATE_TABLE` in `search/schemas.py`:
  - keys: acquisition, precursor, peptide, `is_decoy`, `decoy_set`, `role`, `pass`, rank, `chosen`;
  - apex, boundaries and apex scan;
  - one column per feature;
  - `x.massspec.feature_set` (name and version) in its metadata.
- **Score and quantity tables:**
  - `PRECURSOR_SCORE_TABLE` gains `pass`, `role` and `decoy_set` (S7).
  - `PRECURSOR_QUANT` holds the final pass's chosen candidate.
  - `FRAGMENT_QUANT` gains nullable ion identity; today it has m/z only.
  - `PRECURSOR_CHROMATOGRAM` holds the consensus over the boundaries.
- **The experiment writer:** move registration, `_write_tables`, the locks and the manifest updates out of `workflows/encyclopedia.py` into a generic module. A native unit writes `SearchManifest(engine='constellation', parameters={tolerances, fdr})`. Its library rows are only the precursors it scored, plus the search space's path and SHA-256 under `inputs`.

**D10. How is v1 judged?**
- **Purified BSA/lysin:** the known peptides pass on D5's p-values, and the shuffled decoys' pass rate is reported.
- **A GPF fraction against the jar's result for the same file:** Jaccard of precursors at 1%, apex |Δrt| at most one cycle, Spearman of quantities. A repeat search moves about 10 precursors across the line, which is the noise floor.
- **The features:** ours computed at EncyclopeDIA's own chosen peaks on the wide-window run, compared column by column with its 76,399 × 35 rows. This is the sharpest test of the features.
- **Entrapment:** one equal-size shuffled set, FDP ≤ 1% at q = 0.01.

**D11. Which channels does the scorer see, and by which rule?** These are `ChannelSettings` choices, not sweep changes:
- `fragment_charge_below_precursor=1` (EncyclopeDIA's choice) as the default;
- top-N chosen among acquirable ions (C6);
- channels below 200 Th and within ±10 Th of the precursor left out (EncyclopeDIA's XCorr removes precursor ±10);
- M−1 added at MS1 for the isotope dot;
- a per-channel overlap group, so the dense view can de-duplicate a point matched by two channels;
- the match rule (`sum` to reproduce EncyclopeDIA, `nearest` for mass-error features), chosen per search, since a scorer cannot change it after the sweep.

**D12. What may a reducer see?** The scorer runs as a gated multi-output reducer (`zero_fills=True`) on `layout='precursor_major'`, with `dense_chunks` under a cell budget. Two of EncyclopeDIA's scores (XCorr) and any chance-match null need a scan's other points, which no reducer can see. Either:
- the map hands a reducer the apex scan's points on request; or
- XCorr is left out of v1 and a per-scan point histogram is made in the map for the null (D5 B).

Recommend the second for v1. The gate's defaults were measured on BSA/lysin only (2.5% of reversed queries pass); check them on a GPF fraction before relying on `gate_first` at proteome scale.

**D13. What is MS1 evidence worth on ion-trap MS1?** At 0.35 Da it is chance (§6). Options:
- (A) leave MS1 out of v1's score on ion-trap MS1;
- (B) tighten the MS1 tolerance once the m/z axis is corrected (0.0094 Th RMS on RF30's profile; the corrected 0.2 Da search found 34,301 precursors against 31,093 at 0.35 Da);
- (C) score MS1 against a density-aware chance null.

Recommend B, with C as the feature form. Decide it before the features are fixed, since it changes which MS1 features exist.

## 8. Suggested order

1. **Small fixes that change a scorer's inputs:**
   - C5 (PSM decoy ids), A7 (ranges on imports and multi-range filters), B8 (layout in the metadata);
   - the masking pair V1 / S2 (`mask=` on the similarities, `Dense.observed()`);
   - S3 (log-space censored term in `core.stats`), V2 (the uniqueness check), S4 / S5 (IIT imputation, `floor` and `centroider` in the ledger and dense view), S10 (`point_index` on the dense view);
   - C6 (top-N among acquirable ions), D11's channel defaults, S7 (score-table columns).

   The rest of §1 can follow at leisure; none corrupts a DIA search on the default path.
2. **Batched trace operations and fits** in `core.signal` and `core.optim` (D3), including per-row masked interpolation along RT (V8).
3. **Scorer v1** as a multi-output reducer on precursor-major: candidates, EncyclopeDIA's features, the two likelihood terms, an e-value, and quantity rows for the chosen candidate (D1, D5, D9). Check the features against the jar's 35-column rows at its own peaks (D10).
4. **Minimal statistics** in `core.stats.fdr` / `massspec.search.fdr`: q-values with π₀ = 1, PAVA PEP, sets and entrapment (D6). Pass 1 needs q-values to choose calibrants, so this comes before calibration.
5. **Two passes** (D8): calibrants, `rt.align`, ppm(RT), `calibrate_targets`, units cut by RT. On the toy scorer this is what turns signal into identifications.
6. **A native engine writer and verb**, one pass then two, read by `summarize` and the browser (D9). Validate on BSA/lysin and on a GPF fraction against the jar (D10).
7. **Rescoring:** the JavaPot port, checked on the jar's own feature file (D7).
8. **v2:** competition after the sweep (D4), the count likelihood once floor and gain are calibrated (D2), protein inference once the PI rules on the pairing (D6), several runs.
