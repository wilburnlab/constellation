# nb42c: Peak Boundary Refinement & Interference Detection

## Context

From nb42b, the corrected EM pipeline produces stable alpha (z=2 ≈ 20-25) and per-peptide ν_I, but **τ_R hits the 60s bound for 26/29 peptides** — the HyperEMG right tail extends far past the real chromatographic peak because the global InterferenceMixture is mis-specified (per-peptide signal logL varies by ~7 nat-log units). A region-growing prototype (per-peptide threshold) showed promise (63.8% acceptance vs 92% global) but has a bridging problem where stray scans reset the stop counter.

Two new ideas to explore:
1. **LogL spatial patterns** — per-ion logL within the peak is remarkably flat ("square pulse"), confirming N(t) calibration. Characterize this pattern to derive per-peptide boundaries.
2. **Zero-observation censored likelihood** — scans where no ions are detected constrain peak tails. Currently ignored entirely. Absence of signal means intensity < detection floor (~4k counts ≈ ~200 z=2 ions), which penalizes the model for predicting significant N(t) where nothing was observed.

## Notebook: `notebooks/42c_peak_boundary_refinement.ipynb`

Output dir: `results/calibration/nb42c/`

---

## Section 0: Setup (Cells 1-4)

**Cell 1** — Imports, paths, constants. Same as nb42b plus `scipy.stats.t`, `scipy.ndimage.label`. Constants: `WIDE_RT_WINDOW=180`, `SEED_RADIUS=5`, `FIT_RT_WINDOW=60`.

**Cell 2** — Load data: spectra, PSMs, search results, scanmeta. Extract MS1 chromatograms (±60s, drop_unmatched=True). Run Bayes3 scoring. Build CalibrationData. Same as nb42b Cell 3.

**Cell 3** — Reconstruct nb42b global model from `42b_summary.parquet` and `42b_settings.parquet`. Avoids re-running the 15-min EM.

**Cell 4** — Wide extraction (±180s, drop_unmatched=True) + per-ion logL computation for all 29 peptides. Store in `pep_viz_data[seq]` dict with `wide_obs`, `per_ion_logL`, `complete_mask`.

---

## Section 1: LogL Landscape Characterization (Cells 5-9)

Goal: Validate the "square pulse" hypothesis — logL is flat within the peak, drops sharply at boundaries.

**Cell 5** — Per-peptide plateau statistics. For each peptide:
- Seed: scans within ±5s of Bayes3 apex → `seed_median`, `seed_mad`
- Valley: scans outside HyperEMG 99% window → `valley_median`, `valley_mad`
- `transition_sharpness = (seed_median - valley_median) / seed_mad`
- → `plateau_df` (29 rows) → `42c_plateau_stats.parquet`

**Decision point:** If sharpness consistently >5, a simple threshold boundary works.

**Cell 6** — Figure: LogL vs RT gallery (6×5 grid, 29 peptides). Points colored by charge, horizontal lines at seed_median (green) and threshold (red), seed region shaded, HyperEMG bounds in grey. → `logL_landscape_gallery.png`

**Cell 7** — Plateau flatness test. Within seed: Spearman correlation of logL with |RT−apex| and with log(N_pred). If |ρ| is small, the plateau is noise not signal → boundary is classification, not regression.

**Cell 8** — Cross-peptide comparison: seed_median vs log10(N_total), vs ν_I, transition_sharpness vs τ_R. → `plateau_cross_peptide.png`

---

## Section 2: Intensity Floor Discovery (Cell 9)

**Cell 9** — From `df_wide` (all matched ions): histogram of all observed intensities (log10), minimum per isotope, 1st percentile. Convert to ion count: `N_floor = I_floor * median_iit / alpha(z=2)`. → `I_FLOOR` scalar, `intensity_floor_histogram.png`

**Decision point:** Sharp floor (~4k) → clean censoring boundary. Fuzzy → need probabilistic floor (logistic transition).

---

## Section 3: Zero-Observation Analysis (Cells 10-12)

**Cell 10** — Re-extract with `drop_unmatched=False` (±60s). Compare row counts with matched-only extraction. Report occupancy fraction.

**Cell 11** — Per-peptide zero spatial analysis at dominant charge: fraction of scans with signal in 5s RT bins. This is the empirical peak shape through the detection floor lens. → `zero_observation_spatial.png` (6×5 grid, detection fraction vs RT with HyperEMG overlay)

**Cell 12** — Overlay zeros on the logL landscape gallery: same as Cell 6 but with red × markers at zero-observation RTs. → `logL_with_zeros.png`

**Decision point:** If zeros are sparse (<10 per peptide in ±60s), censored likelihood has minimal effect → focus on region-growing. If hundreds per peptide, censored likelihood is the dominant correction.

---

## Section 4: Censored Likelihood Prototype (Cells 13-15)

The peak fit operates in ion-count space: `N_obs = I_sum * iit / alpha(z)`. For zero-observation scans, we know `N_obs < N_floor = I_floor * iit / alpha(z)`.

**Cell 13** — Define `compute_censored_logL()`. For zero scans within ±60s of apex:
```
N_pred = HyperEMG(t)  # predicted ion count at scan t
N_floor_scan = I_FLOOR * iit_scan / alpha(z)
# If N_pred >> N_floor, the model predicts ions where none were seen → penalty
# logL_cens = log CDF_t(z_floor; nu_I) where z_floor = (N_floor - N_pred) / scale
```
Test on one peptide: plot censored logL vs RT.

**Cell 14** — Single-peptide refit prototype. Pick one with τ_R=60s. Modified loss:
```
total_loss = data_loss + CENS_WEIGHT * (-censored_logL_sum)
```
Sweep `CENS_WEIGHT` ∈ [0.01, 0.1, 0.5, 1.0, 5.0]. → `censored_prototype_single.png` (chromatogram, τ_R vs weight, loss components)

**Decision point:** If τ_R drops smoothly from 60s to ~2-10s, approach is robust. If it jumps/oscillates, needs softer integration.

**Cell 15** — Apply best weight to all 29 peptides. Record new peak params. → `censored_tau_r_comparison.png` (paired before/after), `42c_censored_sweep.parquet`

---

## Section 5: Region-Growing Refinement (Cells 16-19)

Fix the bridging problem with three variants:

**Cell 16** — Define three variants of `grow_peak_per_charge`:
- **A: Contiguous component** — after walk, `scipy.ndimage.label` on accepted mask, keep only component containing apex. Guarantees single connected block.
- **B: Maximum RT gap** — stop extension if gap to previous accepted scan > `max_gap_s` (sweep 10, 15, 20s).
- **C: Lookahead** — require ≥2 of next 3 scans pass threshold before extending.

**Cell 17** — Run all variants + original on 29 peptides, all charges. Record acceptance rate, boundary RTs, n_components. → `variants_df`, `42c_region_variants.parquet`

**Cell 18** — Re-fit peaks with each variant's weights. Compare τ_R distributions. → `region_growing_variants_tau_r.png` (boxplot per variant)

**Decision point:** Variant producing τ_R in 2-10s range without over-shrinking acceptance wins. Contiguous component (A) is expected winner.

**Cell 19** — Chromatogram gallery with best variant: refit N(t) + accept/reject coloring + nb42b overlay in grey. → `best_variant_chromatograms.png`

---

## Section 6: Combined Approach (Cells 20-23)

**Cell 20** — Combined refit: best region-growing weights on observed scans + censored logL on zero scans. All 29 peptides.

**Cell 21** — Diagnostic comparison (3×3 panel): τ_R/τ_L/σ distributions across methods, scatter comparisons, acceptance rates, NLL improvement, N_total stability, FWHM comparison. → `combined_diagnostics.png`

**Cell 22** — Final chromatogram gallery with combined fit, logL coloring, zero markers, nb42b overlay. → `combined_chromatograms.png`

**Cell 23** — Save all results:
- `42c_plateau_stats.parquet` — logL landscape statistics
- `42c_region_variants.parquet` — region-growing comparison
- `42c_censored_sweep.parquet` — censored weight sensitivity
- `42c_combined_summary.parquet` — final peak params (main output for downstream)
- `42c_settings.parquet` — hyperparameter record

---

## Key Files

| File | Role |
|------|------|
| `scripts/calibration_model.py` | GlobalParamsModel, fit_all_peaks, e_step — modify loss for censored term |
| `notebooks/42b_corrected_em.ipynb` | Source for region-growing prototype, visualization code |
| `cartographer/data/chromatogram.py` | `extract_ms1_chromatogram` with `drop_unmatched` param |
| `cartographer/peak_shape.py` | HyperEMGPeak (6 params) |
| `results/calibration/nb42b/42b_summary.parquet` | Starting point: nb42b converged params |

## Verification

1. **τ_R sanity:** Combined fit should produce τ_R ∈ [2, 15]s for most peptides (not 60s bound)
2. **N_total stability:** Should not change dramatically from nb42b (peak area is well-constrained by observed data)
3. **Visual check:** Chromatogram gallery — N(t) should return to baseline within ~30s of apex, not extend for minutes
4. **Alpha stability:** Global alpha should not shift when re-running with new weights (it's driven by the bulk of the data, not the tails)
