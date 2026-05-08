# nb43 — Profile vs Centroid MS1 Investigation

## Context

We've been characterizing MS1 interference in ProteomeTools PROCAL peptides through
the nb41–42 series (EM pipeline, per-ion logL scoring, peak boundary detection). The
nb42b corrected EM pipeline reveals interference through shape parameter distortion
(tau_r bound-hitting, bimodal eta, extreme sigma) but the peak boundary problem
(nb42c) remains unresolved.

The user reprocessed the pool 1 HCD28_DDA_Orbi raw file to produce profile-mode data
and made initial observations (in an OSC notebook) that profile peaks look Gaussian,
widths may vary with m/z, and centroid center/height perfectly predict profile
mean/amplitude. This notebook systematically investigates whether profile peak shapes
carry information about interference that centroids miss.

**Central hypothesis:** If two ions overlap in m/z, the profile peak should show
broadening, asymmetry, or bimodality that a single centroid value cannot capture.

**Known limitation:** TFGTETFDTFK and YGFSSEDIFTK are exact isobars (same elemental
composition) — profile cannot distinguish them. This case serves as a negative control.

---

## Step 0: Pull existing OSC notebook + verify profile data

1. `rsync` the existing notebook from OSC:
   ```
   scp osc-cardinal:/users/PAS1309/damienbwilburn/WilburnLab/DBW/MS_Quant/260417\ Centroid\ vs\ Profile\ comparison.ipynb notebooks/
   ```
2. Extract useful code (pyteomics loading, Gaussian fitting, file paths)
3. Verify profile mzpeak file size and row count on OSC:
   ```
   ssh osc-cardinal "ls -lh /users/PAS1309/damienbwilburn/WilburnLab/Cartographer_Data/Unmodified/ProteomeTools/Zolg2017/01625b_GA1-TUM_first_pool_1_01_01-DDA-1h-R2.profile.mzpeak"
   ssh osc-cardinal "python3 -c \"import pyarrow.parquet as pq; f=pq.ParquetFile('...profile.mzpeak'); print(f.metadata)\""
   ```
4. Check if scanmeta exists for the profile file (needed for IIT)
5. Create `notebooks/43_profile_vs_centroid.ipynb`

**Decision point:** If the profile mzpeak is enormous (>10 GB / billions of rows),
we may need to work with the profile mzML directly via pyteomics streaming instead
of loading the full mzpeak table. Plan both paths.

---

## Step 1: Setup + load both datasets

- `%matplotlib inline` first line
- Load centroid data via `read_table()` (the pool 1 file we always use)
- Load profile data — two strategies depending on size:
  - **If mzpeak is manageable (<5 GB):** `read_table()` with the profile mzpeak path
  - **If too large:** use `pyteomics.mzml.MzML()` on the profile mzML, streaming scan-by-scan
- Load PSMs via `read_psm_tsv()` for PROCAL RT anchors
- Load nb42b summary (`results/calibration/nb42b/42b_summary.parquet`) for interference parameters
- Define target peptides:
  - **Interference cases:** TFGTETFDTFK (isobaric negative control), TFAHTESHISK (nearby RT contaminant)
  - **"Clean" cases:** HDTVFGSYLYK, HEHISSDYAGK (user-selected, though nb42b shows both have elevated tau_r)
- Compute theoretical m/z for each target at z=2, z=3 (M+0, M+1, M+2)

**Key files:**
- Centroid mzpeak: `data/Zolg2017/HCD28_DDA_Orbi/` (local) or ESS path
- Profile mzpeak: `/users/PAS1309/damienbwilburn/WilburnLab/Cartographer_Data/.../01625b_...profile.mzpeak`
- PSM TSV: `data/Zolg2017/HCD28_DDA_Orbi/01625b_...-DDA-1h-R2.tsv`

---

## Step 2: Profile data structure exploration

**Questions answered:** Is the profile data truly profile? What does it look like?

1. For one MS1 scan near a PROCAL apex, compare:
   - Number of data points (centroid vs profile — expect 10-100x more for profile)
   - m/z spacing in profile (uniform grid? if so, what spacing?)
   - Baseline between peaks (noise floor vs exact zeros?)
2. Plot a ~2 Da window around a PROCAL M+0 at z=2 from both profile and centroid
3. Check for FT sidelobes (sinc ringing) — true Orbitrap FID data shows these;
   reconstructed profile would be clean Gaussian

**Output:** 1-2 figures, text summary of data structure

---

## Step 3: Single-peak Gaussian fitting

**Questions answered:** Are profile peaks Gaussian? Does centroid perfectly predict profile?

1. Helper function `fit_profile_gaussian(mz, intensity, center, window_mda=50)`:
   - Model: `I(m/z) = A * exp(-(m/z - mu)^2 / (2*sigma^2)) + baseline`
   - Fit via `scipy.optimize.curve_fit` with reasonable bounds
   - Returns: mu, sigma, amplitude, baseline, FWHM, R-squared, residuals
2. Fit M+0 profile peak for each target peptide at z=2, at the PSM apex scan
3. Compare:
   - `profile_mu` vs `centroid_mz` (should be identical if reconstructed)
   - `profile_amplitude` vs `centroid_intensity` (should be identical if reconstructed)
   - Residual patterns — systematic asymmetry or just noise?
4. Also fit a Voigt profile (Gaussian + Lorentzian convolution) to test if there's
   a non-Gaussian component

**Output:** 4-panel figure per peptide (data+fit, residuals, mu comparison, amplitude comparison)

---

## Step 4: Width vs m/z relationship (all PROCAL, N=29-40)

**Questions answered:** Does FWHM follow Orbitrap physics? Is width independent information?

1. For all PROCAL peptides with PSM hits, fit Gaussian to M+0 at z=2 at apex scan
2. Plot FWHM vs m/z
3. Fit theoretical Orbitrap model: `FWHM = mz / R` where R is resolution at that m/z.
   For Orbitrap, R scales as `R(mz) = R_ref * sqrt(mz_ref / mz)`, so
   `FWHM = mz^1.5 / (R_ref * sqrt(mz_ref))`
4. Also fit simple linear: `FWHM = a * mz + b`
5. Compare extracted R_ref to nominal 60K (at m/z=200)
6. Identify outliers — do they correlate with nb42b interference parameters?

**Output:** FWHM vs m/z scatter + fit, residuals colored by nu_I or tau_r

---

## Step 5: Isotopic envelope in profile space

**Questions answered:** Do M+0/M+1/M+2 each show clean profile peaks? Any "extra" peaks?

1. For 2-3 PROCAL peptides at z=2, plot the full isotopic region (~1.5 Da window)
   from profile data
2. Mark theoretical m/z positions (with d_mz corrections: d_mz_1 ≈ 9.89e-4 Da)
3. Independently fit Gaussians to M+0, M+1, M+2
4. Compare: widths (should be equal at same m/z), amplitude ratios vs
   `expected_isotope_envelope()`, fitted centers vs theoretical
5. Look for unexpected peaks between isotopes (would indicate unknown interferers
   at specific m/z offsets)

**Output:** Full isotopic envelope plots with fits, summary table of per-isotope params

---

## Step 6: Centroid-profile correspondence (systematic)

**Questions answered:** Is there a 1:1 mapping? What info does profile add?

1. Take 3-5 MS1 scans at different RTs
2. For all centroid peaks in a ~50 Da window, find corresponding profile peaks
3. Fit Gaussian to each profile peak
4. Scatter plots: centroid_mz vs profile_mu, centroid_int vs profile_amplitude
5. FWHM vs m/z for all matched peaks (extends Section 4 beyond PROCAL)
6. Histogram of (centroid_mz - profile_mu) — should be centered at zero
7. Any profile peaks without centroid counterparts? Any centroid peaks without
   profile peaks?

**Output:** Correspondence scatter plots, FWHM relationship, mismatch inventory

---

## Step 7: Interference detection from profile peak shape (CENTRAL SECTION)

**Questions answered:** Can profile shapes detect interference?

### 7a. TFGTETFDTFK (isobaric negative control)
- Extract profile peaks at 5-10 scans spanning RT 39-42 min
- Fit Gaussians — sigma should NOT change (same composition = same peak shape)
- Plot sigma and R-squared vs RT
- Explicit demonstration that isobaric interference is invisible in profile space

### 7b. TFAHTESHISK (nearby RT contaminant, ~0.6 min separation)
- The interferer may have a different m/z — check whether two profile peaks appear
- Extract profile peaks at RTs spanning 12.0-14.0 min
- Fit single Gaussian vs double Gaussian at each RT; BIC comparison
- Look for broadening, asymmetry, or secondary peaks near the contaminant RT
- Helper: `fit_double_gaussian(mz, intensity)` for bimodal testing

### 7c. HDTVFGSYLYK and HEHISSDYAGK
- Same protocol as 7b
- If the interferer is at a different m/z, profile might resolve it
- Track sigma and R-squared vs RT across the elution window

### 7d. Clean control comparison
- Pick 1-2 PROCAL peptides with lowest tau_r from nb42b (e.g., TFGTETFDTFK
  actually has lowest tau_r=47.7 but is the isobar case; next lowest are
  HLTGLTFDTYK at 50.8, VSSIFFDTFDK at 50.1)
- Same protocol — expect constant sigma, high R-squared, no secondary peaks

### 7e. Summary table
Per peptide: fitted sigma at apex, R-squared, single vs double Gaussian BIC,
any detected secondary peaks, nb42b tau_r/nu_I for comparison

**Output:** Multi-panel figures per peptide, summary comparison table

---

## Step 8: Profile peak evolution along chromatogram

**Questions answered:** Does profile shape change where interference is present?

1. For 2 interference + 2 clean peptides, extract profile peaks from every MS1 scan
   in ±60s around the apex
2. Time series plots: sigma vs RT, R-squared vs RT, amplitude vs RT (= chromatogram),
   fitted mu vs RT (= m/z drift)
3. Compare amplitude-vs-RT to centroid intensity-vs-RT (should overlay)
4. Look for sigma changes correlating with interference onset
5. If nb42b per-ion logL data is available, overlay as color

**Efficiency:** Iterate profile data once, extract all target m/z windows per scan.
If using mzpeak table: filter by RT range first, then by m/z windows. If using
pyteomics streaming: single pass with windowed extraction.

**Output:** Time-series panels per peptide

---

## Step 9: Assessment — real vs reconstructed profile

**Questions answered:** Is profile data worth investing in?

Synthesize evidence from Sections 2-6:
1. Does centroid perfectly predict profile? (mu, amplitude correspondence)
2. Is FWHM at the physically expected value for 60K Orbitrap?
3. Are there FT sidelobes or is it clean Gaussian? (sidelobes = real transient data)
4. Is baseline noise or exact zeros? (zeros = reconstructed)
5. Is m/z grid uniform? (uniform = reconstructed grid)
6. Final verdict: real FID-derived profile vs Thermo-reconstructed profile

---

## Step 10: Conclusions

1. Which interference types are detectable from profile? (different-mass: maybe;
   isobaric: no; different-charge same-m/z: depends on width difference)
2. Does profile width add discriminative information beyond centroid?
3. Should cartographer add profile support? (probably not unless Section 7 shows clear wins)
4. Implications for nb42 series peak boundary work

---

## Critical files to modify/create

| File | Action |
|------|--------|
| `notebooks/43_profile_vs_centroid.ipynb` | Create — main investigation notebook |

## Critical files to read

| File | Purpose |
|------|---------|
| OSC: `~/WilburnLab/DBW/MS_Quant/260417 Centroid vs Profile comparison.ipynb` | Pull first — extract working code/paths |
| `results/calibration/nb42b/42b_summary.parquet` | nb42b interference parameters for all 29 PROCAL |
| `data/Zolg2017/HCD28_DDA_Orbi/01625b_...-DDA-1h-R2.tsv` | PSM RT anchors |
| OSC: `.../01625b_...profile.mzpeak` | Profile data (main input) |
| `cartographer/data/spectra.py` | Reference for `read_table()` / mzpeak loading |

## Verification

1. Run notebook on OSC via Jupyter tunnel (sinteractive, 8 cores, 32GB)
2. Check that Gaussian fits converge for all target peptides (R-squared > 0.99 for clean peaks)
3. FWHM vs m/z should show a clear power-law relationship
4. Section 7 produces a definitive answer on whether profile adds interference discrimination
5. Figures saved to `results/figures/nb43/`
