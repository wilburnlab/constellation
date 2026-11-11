# Peak detection along a continuous axis

Status: survey and plan, 2026-11-11. Round 1 is implemented in `constellation/core/signal/` (P7–P9 of the KL-scorer branch). Every reference in §6 was checked in November 2026 against Crossref or a publisher or indexing record; entries without a DOI say so.

## 1. Why

A peak is a localized region where a trace rises above what its surroundings predict, so peak detection is localized outlier detection along a continuous axis (retention time, chemical shift, migration time, m/z; PI, 2026-11-10). It has four steps:

1. Estimate the trace without peaks (the baseline).
2. Estimate how much the trace varies without peaks (the noise).
3. Find connected regions that depart from the baseline by more than the noise explains.
4. Describe each region by its location, extent, size and shape.

The methods in §2 differ mainly in which of these steps they make explicit and what prior they place on peak shape. Several consumers need this, and they stress different parts of the problem:

| Consumer | Points per peak | Baseline | Overlap | Scale |
|---|---|---|---|---|
| DIA fragment-ion XICs (first) | 5–50, sampling irregular in time | near zero, sparse or zero-inflated | interference on single fragments; co-elution across fragments of one precursor | many traces per query, millions of queries |
| NMR spectra | 5–50 per line, 10⁴–10⁵ points per spectrum | roll, phase errors (dispersive, negative lobes) | multiplets, crowding | moderate |
| Capillary electrophoresis | few to tens | drift | ladders of narrow, closely spaced peaks | moderate |
| HPLC UV/Vis | tens to hundreds | drifting with the gradient | tailing, shoulders without a local maximum | moderate |
| Profile mass spectra | 5–20 along m/z | low | isotope envelopes; width varies with m/z | thousands of peaks per spectrum |

So there is no picker per instrument. Constellation provides one layer of interchangeable stages (smooth, baseline, noise, detect, bound, refine) on batched tensors `[..., N]` with coordinates `x` and a validity mask. A method is swapped by replacing one stage, and each stage records an identity, so every result states how it was produced.

## 2. The families

"Batching" is how well the family maps to tensor operations on `[..., N]` with masks: full, loop (vectorized inside a short outer loop), or sequential.

| Family | Idea | Outputs | Inputs and priors | Batching | Suited to | References |
|---|---|---|---|---|---|---|
| 1. Local maxima plus criteria | A sample larger than its neighbours, filtered by height, threshold, minimum distance (greedy by height), prominence, width at a relative height and plateau size (`scipy.signal.find_peaks`, `peak_prominences`, `peak_widths`). Prominence is the 0-dimensional persistence of the superlevel-set filtration, except at borders (§5). | apex, prominence and its bases, width, plateau edges | none intrinsic; thresholds are absolute unless divided by noise; distance and width in samples | full: max-pool non-maximum suppression; range-minimum tables for prominence and width; distance in rounds | CE, profile-MS centroiding (with apex interpolation), smoothed XICs. Misses shoulders. | Virtanen 2020; Edelsbrunner 2002; Huber 2021; Kirmse 2017 |
| 2. Derivative-based | Smooth and differentiate. The apex is a +/− zero crossing of y′ or a minimum of y″; y″ minima also reveal shoulders. Start and end are set by slope thresholds. Waters ApexTrack detects apexes by curvature and places baselines with Liftoff % / Touchdown %; the Agilent integrator compares slope and curvature filters with a slope sensitivity; MS-DIAL uses first and second derivatives. | apex, inflection points, start and end, shoulder flags | uniform spacing (Savitzky–Golay coefficients); window near the peak width; slope or curvature threshold | full: convolution and elementwise logic; walk-outs as masked argmax | HPLC UV (the industry standard), CE. Noisy below about 10 points per peak. | Savitzky 1964; Dyson 1998; Waters theory guide; Agilent help; Tsugawa 2015 |
| 3. Matched filter | Correlate the trace with the expected shape. Zero-mean kernels (negative second derivative of a Gaussian, Ricker) cancel constant and linear baselines. XCMS matchedFilter, CRAWDAD's Gaussian-derivative filters, OpenMS `PeakPickerChromatogram` (`crawdad`). | apex, filter response as score, support-based bounds | width prior; noise of the response | full: convolution; a width bank is extra channels | DIA XICs (width nearly constant within a run), NMR (Lorentzian template), CE | Turin 1960; Danielsson 2002; Smith 2006; Finney 2008; Finney 2012 |
| 4. CWT ridge lines | Continuous wavelet transform over a range of scales; local maxima linked across scales into ridges; long ridges with high SNR kept. The best scale estimates width; noise comes from the smallest scales. MassSpecWavelet, XCMS centWave, MZmine ADAP, OpenMS PeakPickerCWT, `scipy.signal.find_peaks_cwt`. | apex, scale, ridge length, SNR | scale range; uniform spacing | transform full (convolution bank); ridge linking loops over scales | profile MS, LC-MS metabolomics, CE (Ridger), NMR (WaVPeak). Coarse at 5–10 points per peak; parameter-sensitive. | Du 2006; Tautenhahn 2008; Lange 2006; Wee 2008; Myers 2017; Liu 2012; Yang 2009; Libiseller 2015 |
| 5. Baseline then threshold | Estimate the baseline (morphological opening, rolling quantile, SNIP, Whittaker, AsLS, airPLS, arPLS) and a robust noise level (MAD); mark runs above kσ; the apex is each run's maximum. The "smoothed z-score" is a causal streaming variant. | runs (start, end), apex, height above baseline, SNR, the baseline | baseline stiffness or element width; k | opening and quantile full; SNIP loops over iterations; AsLS needs banded solves | HPLC drift, NMR roll, CE drift. Sparse XICs need a noise floor (MAD can be 0). | Perez-Pueyo 2010; Ryan 1988; Morháč 2009; Eilers 2003; Eilers & Boelens 2005; Zhang 2010; Baek 2015; Donoho 1994; van Brakel 2014 |
| 6. Statistical significance | Give each local maximum of the smoothed trace a p-value under a noise model, then control FWER or FDR across candidates. MACS uses a local Poisson rate λ_local = max(λ_BG, λ_1k, λ_5k, λ_10k). DIA controls error at the peak-group level with decoys. | statistic, p and q per candidate; no bounds | noise model; smoothing kernel; decoys | full: elementwise; Benjamini–Hochberg as a sort | all domains; false-peak control in DIA | Schwartzman 2011; Cheng & Schwartzman 2017; Benjamini 1995; Abbas 2013; Zhang 2008; Reiter 2011; Rosenberger 2017 |
| 7. Model-based deconvolution | Fit a baseline plus a sum of shapes (Gaussian, EMG, Lorentzian, Voigt) by nonlinear least squares, or NNLS / L1 over a dictionary of shapes. Mnova GSD reduces a spectrum to line parameters; CRAFT is Bayesian, in the time domain. | overlapping components, residual, covariance | component count and starting values; shape family; irregular x is natural | loop: batched Levenberg–Marquardt with fixed K_max and iterations; NNLS by projected gradient | HPLC shoulders and tailing, NMR, centroid refinement. A 4-parameter EMG is underdetermined on fewer than 10 points. | Grushka 1972; Lan 2001; Kalambet 2011; Lawson 1995; Tibshirani 1996; Slawski 2012; Cobas 2011; Krishnamurthy 2013 |
| 8. Bayesian | Posterior over the number and parameters of peaks: reversible-jump MCMC, variable selection with SAMC, BATMAN for NMR metabolites; Bayesian Blocks (optimal piecewise-constant segmentation); Gaussian-process regression. | posterior samples or summaries, per-location probability, blocks | likelihood, priors | MCMC sequential; Bayesian Blocks O(N²) DP; GP a Cholesky per trace | NMR quantification. Too costly for millions of DIA queries. | Green 1995; Cheng 2014; Hao 2012; Scargle 2013; Rasmussen 2006; Spyrou 2009 |
| 9. Change-point and segmentation | Split the trace into regimes by penalized cost (PELT, binary segmentation), an online run-length posterior (BOCPD) or an HMM; peaks are elevated segments. | segments and change points; bell shapes become staircases | cost, penalty, minimum segment length | PELT and BOCPD sequential; DP and HMM loop over N, vectorized over batch and states | elution windows, baseline steps | Killick 2012; Scott 1974; Adams 2007; Truong 2020; Rabiner 1989; Särkkä 2021 |
| 10. Multi-trace co-elution | Detect on several aligned traces jointly through a consensus (sum, median, k-th largest, best-correlated fragment) or per-trace picks merged into groups, scored by co-elution and shape similarity. OpenMS MRMTransitionGroupPicker, mProphet, Skyline, DIA-NN, EncyclopeDIA, DIA-Umpire, Avant-garde. | group apex and bounds, per-trace measures, interference flags, group scores | aligned traces, library intensities, a retention-time prior | full: reductions along the trace axis, batched correlations; greedy picking in masked rounds | DIA, PRM, SRM; analogues in NMR multiplets, CE dye channels, diode arrays | Röst 2014; Röst 2016; Reiter 2011; MacLean 2010; Demichev 2020; Searle 2018; Tsou 2015; Vaca Jacome 2020 |
| 11. Learned detectors | CNN classifiers on candidate regions (peakonly, NeatMS, Kantz et al.); PeakBot classifies each local maximum and regresses its centre and box; networks for peak evaluation and NMR picking (DEEP Picker, NMRNet). | probability per candidate, boxes, per-point masks | fixed-length windows; labelled data; shifts across instruments | full | metabolomics curation; 2-D NMR | Melnikov 2020; Gloaguen 2022; Bueschl 2022; Kantz 2019; Risum 2019; Li 2021; Klukowski 2018 |

How the family-10 tools work. OpenMS MRMTransitionGroupPicker smooths each chromatogram, picks peaks with PeakPickerHiRes, takes the largest peak across chromatograms and integrates every transition within its boundaries, and continues in order of intensity until `stop_after_feature` or `stop_after_intensity_ratio` (options `use_consensus`, `boundary_selection_method` largest or widest, `recalculate_peaks` using the median boundary for outlying transitions). mProphet scores co-elution and shape and estimates error with decoys. DIA-NN takes the fragment best correlated with the others as the reference elution profile and computes 73 peak scores. Avant-garde refines boundaries and transitions using all runs.

## 3. Boundaries and quality measures

**Boundary rules.**
- *Prominence bases (valley):* the minima between the peak and higher ground on each side. Good for separating neighbours; on a slope or for an isolated peak they can lie far out, so poor integration limits.
- *Valley between neighbours:* drop-line or tangent skim, as in chromatography data systems (Dyson 1998).
- *Fraction of apex:* walk out until the signal falls below f × height (or a fraction of prominence, as `peak_widths`) and interpolate the crossing.
- *Slope or curvature:* ApexTrack's Liftoff % / Touchdown %; the Agilent integrator's slope accumulators.
- *Return to noise:* extend until S/N falls below a threshold (OpenMS `signal_to_noise`).
- *Inflection points:* from the second derivative; ±σ for a Gaussian.
- *Fitted-shape quantiles:* central quantiles of a fitted Gaussian or EMG; stable with few points.
- *Consensus:* median boundaries across transitions (OpenMS `recalculate_peaks`) or across runs (Avant-garde, Skyline imputation).

**Quality measures.**

| Measure | Definition | Source and notes |
|---|---|---|
| SNR | (apex − baseline) / σ, σ from 1.4826·MAD, MAD of first differences / √2, the SD outside the peak, the smallest CWT scales, or a local intensity histogram (OpenMS) | the definitions are not comparable; record which was used |
| Prominence | height above the higher of the two bases | Virtanen 2020; convention in §5 |
| FWHM and width ratios | width at half height, at other fractions, against the expected width | standard |
| Asymmetry and tailing | back over front half-width at a fixed fraction of height (10 % asymmetry, 5 % tailing factor in chromatography) | standard |
| Gaussian similarity | correlation with a fitted or idealized bell | Zhang 2014; Kumler 2023 |
| Sharpness | summed relative differences between neighbouring points either side of the apex | Zhang 2014 |
| Zig-zag index | squared slope changes between neighbours, normalized by an intensity scale | Zhang 2014 (in MetaClean) |
| Jaggedness | share of points where intensity changes direction, ignoring changes below a flatness factor (MetaClean default 0.05) | Toghi Eshghi 2018; Chetnik 2020 |
| Modality, symmetry, FWHM consistency, co-elution and shape similarity | per-peak and per-group metrics for targeted proteomics | Toghi Eshghi 2018 |
| Cross-trace scores | cross-correlation maximum and lag, shape correlation, agreement with library intensities | Reiter 2011; Demichev 2020 |

On combining them: Zhang & Zhao (2014) compared six metrics (sharpness, Gaussian similarity, SNR, peak significance level, TPASR, zig-zag index); sharpness, significance level and zig-zag did best, and combinations beat any single one. Kumler et al. (2023) fitted a logistic regression on two (an SNR and bell-curve similarity) and cut false positives from 70–80 % to 1–5 %. MS-FLO (DeFelice 2017) removes duplicate, isotope and adduct features from a feature list; it is not a shape metric.

## 4. The contract in Constellation

**Modules** (`constellation/core/signal/`):

| Module | Contents |
|---|---|
| `smooth.py` | `moving_average`; `gaussian`; `savitzky_golay`, an exact masked local-polynomial regression |
| `baseline.py` | `rolling_quantile` (moving median), `opening` (morphological), `snip` |
| `noise.py` | `diff_mad`, `residual_mad`, `rolling_mad` |
| `consensus.py` | `collapse` of K traces by sum, mean, median, quantile or `kth_largest` (k traces must rise together, as in EncyclopeDIA) |
| `_rows.py` | `compact` (padding to each row's end); `RangeTable`, a sparse table of range min/max with binary-lifting searches for the nearest sample past a level, O(B·N·log N), used by prominence, widths and hysteresis |
| `interpolate.py` | `bracket`, `interp_rows` |
| `peaks/` | the stage classes and `PeakPicker` |
| `peak_fit.py` | `fit_peaks`, a peak form fitted to every peak's extent at once (`core.optim.batched_least_squares`, `core.stats.peaks.PEAK_FORMS`) |

**Stages.** Frozen dataclasses with a kind, a name and a version, and an `identity()`, in per-kind registries `SMOOTHERS`, `BASELINES`, `NOISE`, `DETECTORS`, `BOUNDARIES`, `REFINERS`; added with `register`, rebuilt with `make_stage` or `stage_from_identity`.
- *Detectors:* `LocalMaxima` (scipy `find_peaks` criteria plus `min_snr`; score = prominence); `MatchedFilter` (a bank of zero-mean Gaussian or Ricker kernels over widths; local maxima of the response over its MAD at or above a threshold); `RobustZ` (rolling-median baseline and MAD noise; runs above `k_low` reaching `k`).
- *Boundaries:* `Valley` (prominence bases), `DetectorBounds`, `Fraction`, `WalkOut` (EncyclopeDIA's getIndexRange rule).
- *Refiners:* `NoRefine`; `Parabolic` (log option: three-point Gaussian interpolation); `ShapeFit` (a batched Levenberg–Marquardt fit of `gaussian` or `emg`).

**The picker.** `PeakPicker(detect, smooth, baseline, noise, boundaries, refine, max_peaks, recenter)` takes `y [..., N]`, `x` and `valid`, and an optional `boundary_trace` (detect on one trace, bound and measure on another, the apex re-centred on it within `recenter` samples). It returns `Peaks [..., M]`, best score first, with `apex`, `apex_x`, `boundary_apex`, `left` / `right` and their x, the prominence bases, `height`, `prominence`, `snr`, `score`, `width` (FWHM in x), `area`, `asymmetry`, `jaggedness`, `extras` (method-specific) and `components` (fitted shapes). `identity()`, `key()` and `from_identity()` record and restore provenance.

**How each family maps onto the contract.**

| Family | Stage kind(s) | Fields it determines (the rest from the shared measures) | What does not fit, or is needed |
|---|---|---|---|
| 1. Local maxima | `DETECTORS` (`LocalMaxima`, implemented) | apex, prominence, bases, width, score | fits |
| 2. Derivative | `DETECTORS` plus new `BOUNDARIES` (inflection, slope) | apex at a zero crossing, left and right; shoulder flags in `extras` | derivative output from the smoother (`savitzky_golay(derivative=)` exists) |
| 3. Matched filter | `DETECTORS` (`MatchedFilter`, implemented) | apex; score from the normalized response; best width in `extras` | fits |
| 4. CWT ridges | `DETECTORS` | apex, score (ridge SNR), width from scale; scale and ridge length in `extras` | fits once ridges are reduced to apexes |
| 5. Baseline then threshold | `BASELINES`, `NOISE`, `DETECTORS` (`RobustZ`, implemented) | apex, run edges through `DetectorBounds`, snr | fits |
| 6. Significance | a new kind (scorer or filter) after detection | score; p and q in `extras` | detects nothing, gives no bounds; needs a stage that can drop candidates |
| 7. Model-based | `REFINERS` (`ShapeFit`, implemented) | `components`, apex_x | overlapping components need bounds that need not be disjoint; per-component covariance |
| 8. Bayesian | `DETECTORS` plus `REFINERS` | posterior summaries in the usual fields | samples and an uncertain count do not fit `[..., M]` |
| 9. Change-point | `DETECTORS` that emit segments | edges through `DetectorBounds`; apex = segment maximum | the segmentation is dense `[..., N]`; needs a decoder from segments to peaks |
| 10. Multi-trace | `consensus.collapse` before detection; `boundary_trace` for bounds | group-level fields | no per-trace areas, apex shifts or interference flags `[..., K, M]` yet; needs a per-trace pass within fixed bounds |
| 11. Learned | `DETECTORS`, with `DetectorBounds` for boxes | score = probability | per-point masks need a decoder or go in `extras` |

Three domains need more than these fields: NMR multiplets are a hierarchy (lines grouped with coupling constants); dispersive or negative lines need sign handling; profile MS can give thousands of peaks a spectrum, for which a padded `[..., M]` is wasteful and a ragged form may be needed.

**Batching.** Round 1 uses only operations that vectorize over `[..., N]` with masks: max-pool suppression; the distance rule as greedy-by-height in parallel rounds; `RangeTable` queries; convolution for smoothing and matched filtering; pooling and unfold for opening and rolling quantiles; a fixed loop for SNIP; `compact` to keep valid entries contiguous; batched Levenberg–Marquardt with per-row convergence. Of what is deferred, CWT ridge linking, NNLS, Bayesian Blocks and HMM programs, short-trace GPs and AsLS-type baselines batch inside a short loop; RJMCMC and SAMC chains, PELT pruning, BOCPD, the smoothed z-score with influence below 1, and event-driven integrators are inherently sequential.

## 5. Round 1, and what is deferred

**What round 1 implements, and why.** Families 1, 3 and 5 as detectors, and parts of 2 (Savitzky–Golay, with derivatives), 7 (`ShapeFit`) and 10 (consensus traces, a separate boundary trace). Because:
- all of it batches (pooling, convolution, sparse-table queries, elementwise logic, batched solves), which millions of DIA queries require;
- it needs no training data, and its parameters stay interpretable (prominence, width, k in units of σ);
- at 5–50 points per peak, simple detectors plus refinement are more robust than derivative or scale-space methods;
- `LocalMaxima` has a reference to be held to: `scipy.signal.find_peaks`.

`savitzky_golay` equals `scipy.signal.savgol_filter` to 1e-10 inside a full row; at edges and gaps it fits the samples that are present, by design, rather than scipy's edge modes.

The first consumer is the DIA scorer v1 (`massspec.search.score`): per MS2 scan, KL(observed fragment shares ‖ library shares); peaks detected in that trace; bounded and measured on an intensity consensus of the fragment traces.

**Parity with scipy.** `LocalMaxima` matches `find_peaks` on continuous data in every criterion: height, threshold, distance, prominence with `wlen`, width at `rel_height`, plateau size. When exactly equal heights fall within `distance`, scipy's choice depends on numpy's sort, which is not stable past 16 candidates; Constellation keeps the later one. Exact ties are rare on continuous data but occur on integer-valued or saturated traces. The prominence convention (the search stops at a higher peak or at the border or `wlen`; the higher of the two side minima is the base) equals topological persistence only with higher ground on both sides: near a border it can be smaller, and the global maximum gets a finite prominence where its persistence is infinite.

**Deferred.**
- *CWT ridges.* The transform is a convolution bank; linking ridges across scales is a loop with matching at each step. `MatchedFilter`'s width bank covers multi-scale detection without linking. Revisit for profile MS and CE.
- *AsLS, airPLS, arPLS.* A pentadiagonal solve per trace and reweighting iteration; torch has no batched banded solver, and dense batched Cholesky is O(N³), fine only for short traces. A custom kernel or a CPU fallback (`scipy.linalg.solve_banded`). HPLC and NMR need these.
- *Bayesian methods.* Sequential and costly per trace; posterior outputs do not fit `Peaks`.
- *Change-point methods.* Segments rather than peaks; PELT pruning and BOCPD are sequential.
- *Learned detectors.* Need labelled data per domain and may not generalize across instruments; they fit the contract as detectors later.

Not yet scheduled, rather than deferred: derivative detection and boundaries (family 2); a significance or FDR stage (family 6, a new stage kind); per-trace measures for multi-trace groups.

## 6. References

- Abbas A, Kong X-B, Liu Z, Jing B-Y, Gao X. 2013. Automatic peak selection by a Benjamini–Hochberg-based algorithm. *PLoS ONE* 8:e53112. doi:10.1371/journal.pone.0053112
- Adams RP, MacKay DJC. 2007. Bayesian online changepoint detection. arXiv:0710.3742
- Agilent Technologies. OpenLab CDS Data Analysis help: "The peak recognition algorithm." No DOI. https://openlab.help.agilent.com/2.8/en/mergedProjects/DataAnalysis/11016492171-1-54043206835275531-7.htm
- Baek S-J, Park A, Ahn Y-J, Choo J. 2015. Baseline correction using asymmetrically reweighted penalized least squares smoothing. *Analyst* 140:250–257. doi:10.1039/C4AN01061B
- Benjamini Y, Hochberg Y. 1995. Controlling the false discovery rate. *J R Stat Soc B* 57:289–300. doi:10.1111/j.2517-6161.1995.tb02031.x
- Bueschl C, et al. 2022. PeakBot: machine-learning-based chromatographic peak picking. *Bioinformatics* 38:3422–3428. doi:10.1093/bioinformatics/btac344
- Cheng D, Schwartzman A. 2017. Multiple testing of local maxima for detection of peaks in random fields. *Ann Stat* 45:529–556. doi:10.1214/16-AOS1458
- Cheng Y, Gao X, Liang F. 2014 (online 2013). Bayesian peak picking for NMR spectra. *Genomics Proteomics Bioinformatics* 12:39–47. doi:10.1016/j.gpb.2013.07.003
- Chetnik K, Petrick L, Pandey G. 2020. MetaClean: a machine learning-based classifier for reduced false positive peak detection in untargeted LC–MS metabolomics data. *Metabolomics* 16. doi:10.1007/s11306-020-01738-3
- Cobas C, Seoane F, Domínguez S, Sýkora S, Davies AN. 2011. A new approach to improving automated analysis of proton NMR spectra through Global Spectral Deconvolution (GSD). *Spectroscopy Europe* 23(1):26–30. No DOI; Mestrelab lists the year as 2010.
- Danielsson R, Bylund D, Markides KE. 2002. Matched filtering with background suppression for improved quality of base peak chromatograms and mass spectra in liquid chromatography–mass spectrometry. *Anal Chim Acta* 454:167–184. doi:10.1016/S0003-2670(01)01574-4
- DeFelice BC, et al. 2017. Mass Spectral Feature List Optimizer (MS-FLO). *Anal Chem* 89:3250–3255. doi:10.1021/acs.analchem.6b04372
- Demichev V, Messner CB, Vernardis SI, Lilley KS, Ralser M. 2020 (online 2019). DIA-NN: neural networks and interference correction enable deep proteome coverage in high throughput. *Nat Methods* 17:41–44. doi:10.1038/s41592-019-0638-x
- Donoho DL, Johnstone IM. 1994. Ideal spatial adaptation by wavelet shrinkage. *Biometrika* 81:425–455. doi:10.1093/biomet/81.3.425
- Du P, Kibbe WA, Lin SM. 2006. Improved peak detection in mass spectrum by incorporating continuous wavelet transform-based pattern matching. *Bioinformatics* 22:2059–2065. doi:10.1093/bioinformatics/btl355
- Dyson N. 1998. *Chromatographic Integration Methods*, 2nd ed. Royal Society of Chemistry. ISBN 978-0-85404-510-5
- Edelsbrunner H, Letscher D, Zomorodian A. 2002. Topological persistence and simplification. *Discrete Comput Geom* 28:511–533. doi:10.1007/s00454-002-2885-2
- Eilers PHC. 2003. A perfect smoother. *Anal Chem* 75:3631–3636. doi:10.1021/ac034173t
- Eilers PHC, Boelens HFM. 2005. Baseline correction with asymmetric least squares smoothing. Unpublished report, Leiden University Medical Centre. No DOI.
- Finney GL, et al. 2008. Label-free comparative analysis of proteomics mixtures using chromatographic alignment of high-resolution μLC-MS data. *Anal Chem* 80:961–971. doi:10.1021/ac701649e
- Finney GL. 2012. PhD dissertation, University of Washington (CRAWDAD; Gaussian second- and first-derivative peak filters). No DOI. https://digital.lib.washington.edu/researchworks/bitstreams/8cc3fe53-5b43-4552-85e7-a8f10183fdc7/download
- Gloaguen Y, Kirwan JA, Beule D. 2022. Deep learning-assisted peak curation for large-scale LC-MS metabolomics. *Anal Chem* 94:4930–4937. doi:10.1021/acs.analchem.1c02220
- Green PJ. 1995. Reversible jump Markov chain Monte Carlo computation and Bayesian model determination. *Biometrika* 82:711–732. doi:10.1093/biomet/82.4.711
- Grushka E. 1972. Characterization of exponentially modified Gaussian peaks in chromatography. *Anal Chem* 44:1733–1738. doi:10.1021/ac60319a011
- Hao J, Astle W, De Iorio M, Ebbels TMD. 2012. BATMAN—an R package for the automated quantification of metabolites from NMR spectra using a Bayesian model. *Bioinformatics* 28:2088–2090. doi:10.1093/bioinformatics/bts308
- Huber S. 2021. Persistent homology in data science. Springer. doi:10.1007/978-3-658-32182-6_13
- Kalambet Y, et al. 2011. Reconstruction of chromatographic peaks using the exponentially modified Gaussian function. *J Chemometrics* 25:352–356. doi:10.1002/cem.1343
- Kantz ED, Tiwari S, Watrous JD, Cheng S, Jain M. 2019. Deep neural networks for classification of LC-MS spectral peaks. *Anal Chem* 91:12407–12413. doi:10.1021/acs.analchem.9b02983
- Killick R, Fearnhead P, Eckley IA. 2012. Optimal detection of changepoints with a linear computational cost. *J Am Stat Assoc* 107:1590–1598. doi:10.1080/01621459.2012.737745
- Kirmse A, de Ferranti J. 2017. Calculating the prominence and isolation of every mountain in the world. *Prog Phys Geogr* 41:788–802. doi:10.1177/0309133317738163
- Klukowski P, et al. 2018. NMRNet: a deep learning approach to automated peak picking of protein NMR spectra. *Bioinformatics* 34:2590–2597. doi:10.1093/bioinformatics/bty134
- Krishnamurthy K. 2013. CRAFT (complete reduction to amplitude frequency table). *Magn Reson Chem* 51:821–829. doi:10.1002/mrc.4022
- Kumler W, Hazelton BJ, Ingalls AE. 2023. Picky with peakpicking: assessing chromatographic peak quality with simple metrics in metabolomics. *BMC Bioinformatics* 24:404. doi:10.1186/s12859-023-05533-4
- Lan K, Jorgenson JW. 2001. A hybrid of exponential and Gaussian functions as a simple model of asymmetric chromatographic peaks. *J Chromatogr A* 915:1–13. doi:10.1016/S0021-9673(01)00594-5
- Lange E, Gröpl C, Reinert K, Kohlbacher O, Hildebrandt A. 2006. High-accuracy peak picking of proteomics data using wavelet techniques. *Pac Symp Biocomput* 11:243–254. doi:10.1142/9789812701626_0023
- Lawson CL, Hanson RJ. 1995 (SIAM reprint of the 1974 edition). *Solving Least Squares Problems*. doi:10.1137/1.9781611971217
- Li D-W, et al. 2021. DEEP picker is a deep neural network for accurate deconvolution of complex two-dimensional NMR spectra. *Nat Commun* 12. doi:10.1038/s41467-021-25496-5
- Libiseller G, et al. 2015. IPO: a tool for automated optimization of XCMS parameters. *BMC Bioinformatics* 16. doi:10.1186/s12859-015-0562-8
- Liu Z, Abbas A, Jing B-Y, Gao X. 2012. WaVPeak: picking NMR peaks through wavelet-based smoothing and volume-based filtering. *Bioinformatics* 28:914–920. doi:10.1093/bioinformatics/bts078
- MacLean B, et al. 2010. Skyline: an open source document editor for creating and analyzing targeted proteomics experiments. *Bioinformatics* 26:966–968. doi:10.1093/bioinformatics/btq054
- Melnikov AD, Tsentalovich YP, Yanshole VV. 2020. Deep learning for the precise peak detection in high-resolution LC–MS data. *Anal Chem* 92:588–592. doi:10.1021/acs.analchem.9b04811
- Morháč M. 2009. An algorithm for determination of peak regions and baseline elimination in spectroscopic data. *Nucl Instrum Methods A* 600:478–487. doi:10.1016/j.nima.2008.11.132
- Myers OD, Sumner SJ, Li S, Barnes S, Du X. 2017. One step forward for reducing false positive and false negative compound identifications from mass spectrometry metabolomics data. *Anal Chem* 89:8696–8703. doi:10.1021/acs.analchem.7b00947
- OpenMS. TOPP MRMTransitionGroupPicker documentation. No DOI. https://abibuilder.cs.uni-tuebingen.de/archive/openms/Documentation/release/latest/html/TOPP_MRMTransitionGroupPicker.html
- Perez-Pueyo R, Soneira MJ, Ruiz-Moreno S. 2010. Morphology-based automated baseline removal for Raman spectra of artistic pigments. *Appl Spectrosc* 64:595–600. doi:10.1366/000370210791414281
- Rabiner LR. 1989. A tutorial on hidden Markov models and selected applications in speech recognition. *Proc IEEE* 77:257–286. doi:10.1109/5.18626
- Rasmussen CE, Williams CKI. 2006. *Gaussian Processes for Machine Learning*. MIT Press. ISBN 0-262-18253-X
- Reiter L, et al. 2011. mProphet: automated data processing and statistical validation for large-scale SRM experiments. *Nat Methods* 8:430–435. doi:10.1038/nmeth.1584
- Risum AB, Bro R. 2019. Using deep learning to evaluate peaks in chromatographic data. *Talanta* 204:255–260. doi:10.1016/j.talanta.2019.05.053
- Rosenberger G, et al. 2017. Statistical control of peptide and protein error rates in large-scale targeted data-independent acquisition analyses. *Nat Methods* 14:921–927. doi:10.1038/nmeth.4398
- Röst HL, et al. 2014. OpenSWATH enables automated, targeted analysis of data-independent acquisition MS data. *Nat Biotechnol* 32:219–223. doi:10.1038/nbt.2841
- Röst HL, et al. 2016. OpenMS: a flexible open-source software platform for mass spectrometry data analysis. *Nat Methods* 13:741–748. doi:10.1038/nmeth.3959
- Ryan CG, Clayton E, Griffin WL, Sie SH, Cousens DR. 1988. SNIP, a statistics-sensitive background treatment for the quantitative analysis of PIXE spectra in geoscience applications. *Nucl Instrum Methods B* 34:396–402. doi:10.1016/0168-583X(88)90063-8
- Särkkä S, García-Fernández ÁF. 2021. Temporal parallelization of Bayesian smoothers. *IEEE Trans Autom Control* 66:299–306. doi:10.1109/TAC.2020.2976316
- Savitzky A, Golay MJE. 1964. Smoothing and differentiation of data by simplified least squares procedures. *Anal Chem* 36:1627–1639. doi:10.1021/ac60214a047
- Scargle JD, Norris JP, Jackson B, Chiang J. 2013. Studies in astronomical time series analysis. VI. Bayesian block representations. *Astrophys J* 764:167. doi:10.1088/0004-637X/764/2/167
- Schwartzman A, Gavrilov Y, Adler RJ. 2011. Multiple testing of local maxima for detection of peaks in 1D. *Ann Stat* 39:3290–3319. doi:10.1214/11-AOS943
- Scott AJ, Knott M. 1974. A cluster analysis method for grouping means in the analysis of variance. *Biometrics* 30:507. doi:10.2307/2529204
- Searle BC, et al. 2018. Chromatogram libraries improve peptide detection and quantification by data independent acquisition mass spectrometry. *Nat Commun* 9. doi:10.1038/s41467-018-07454-w
- Slawski M, et al. 2012. Isotope pattern deconvolution for peptide mass spectrometry by non-negative least squares/least absolute deviation template matching. *BMC Bioinformatics* 13. doi:10.1186/1471-2105-13-291
- Smith CA, Want EJ, O'Maille G, Abagyan R, Siuzdak G. 2006. XCMS: processing mass spectrometry data for metabolite profiling using nonlinear peak alignment, matching, and identification. *Anal Chem* 78:779–787. doi:10.1021/ac051437y
- Spyrou C, Stark R, Lynch AG, Tavaré S. 2009. BayesPeak: Bayesian analysis of ChIP-seq data. *BMC Bioinformatics* 10. doi:10.1186/1471-2105-10-299
- Tautenhahn R, Böttcher C, Neumann S. 2008. Highly sensitive feature detection for high resolution LC/MS. *BMC Bioinformatics* 9. doi:10.1186/1471-2105-9-504
- Tibshirani R. 1996. Regression shrinkage and selection via the lasso. *J R Stat Soc B* 58:267–288. doi:10.1111/j.2517-6161.1996.tb02080.x
- Toghi Eshghi S, Auger P, Mathews WR. 2018. Quality assessment and interference detection in targeted mass spectrometry data using machine learning. *Clin Proteomics* 15:33. doi:10.1186/s12014-018-9209-x
- Truong C, Oudre L, Vayatis N. 2020. Selective review of offline change point detection methods. *Signal Process* 167:107299. doi:10.1016/j.sigpro.2019.107299
- Tsou C-C, et al. 2015. DIA-Umpire: comprehensive computational framework for data-independent acquisition proteomics. *Nat Methods* 12:258–264. doi:10.1038/nmeth.3255
- Tsugawa H, et al. 2015. MS-DIAL: data-independent MS/MS deconvolution for comprehensive metabolome analysis. *Nat Methods* 12:523–526. doi:10.1038/nmeth.3393
- Turin GL. 1960. An introduction to matched filters. *IRE Trans Inf Theory* 6:311–329. doi:10.1109/TIT.1960.1057571
- Vaca Jacome AS, et al. 2020. Avant-garde: an automated data-driven DIA data curation tool. *Nat Methods* 17:1237–1244. doi:10.1038/s41592-020-00986-4
- van Brakel JPG. 2014. Robust peak detection algorithm (using z-scores). Stack Overflow answer 22640362 to question 22583391. No DOI. https://stackoverflow.com/questions/22583391
- Virtanen P, et al. 2020. SciPy 1.0: fundamental algorithms for scientific computing in Python. *Nat Methods* 17:261–272. doi:10.1038/s41592-019-0686-2
- Waters Corporation. Empower 3 Software Data Acquisition and Processing Theory Guide (715005481), ApexTrack chapter. No DOI. https://support.waters.com/@api/deki/files/12972/715005481.pdf
- Wee A, Grayden DB, Zhu Y, Petkovic-Duran K, Smith D. 2008. A continuous wavelet transform algorithm for peak detection. *Electrophoresis* 29:4215–4225. doi:10.1002/elps.200800096
- Yang C, He Z, Yu W. 2009. Comparison of public peak detection algorithms for MALDI mass spectrometry data analysis. *BMC Bioinformatics* 10:4. doi:10.1186/1471-2105-10-4
- Zhang W, Zhao PX. 2014. Quality evaluation of extracted ion chromatograms and chromatographic peaks in liquid chromatography/mass spectrometry-based metabolomics data. *BMC Bioinformatics* 15(Suppl 11):S5. doi:10.1186/1471-2105-15-S11-S5
- Zhang Y, et al. 2008. Model-based analysis of ChIP-Seq (MACS). *Genome Biol* 9:R137. doi:10.1186/gb-2008-9-9-r137
- Zhang Z-M, Chen S, Liang Y-Z. 2010. Baseline correction using adaptive iteratively reweighted penalized least squares. *Analyst* 135:1138. doi:10.1039/b922045c
