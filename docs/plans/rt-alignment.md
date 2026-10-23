# Retention-time alignment

## Status

- **Plan approved** by the PI on 2026-10-22. The approved original is
  `~/.claude/plans/a-critical-feature-currently-gleaming-pixel.md`. Work is on
  `feat/rt-alignment`.
- **Decisions (PI, 2026-10-22):**
  - **Scope:** the algorithm, a library API, this review and verification. No CLI, no
    writes into experiment directories, no browser changes yet.
  - **Default method:** EncyclopeDIA's, replayed exactly, until a benchmark shows a better
    one; flipping the default is a later change. **Flipped the same day, after the
    benchmark:** the default is now `global` refined by `modal` (PI, 2026-10-22, accepting
    its 12–19 s per fit; see "Decision" under "Benchmark").
  - **Refinement:** both refinement objectives are built and the benchmark chooses.
  - **Extrapolation:** soft linear tails by default, clamp available.
  - **Naming:** no identifier says "ridge" (it reads as ridge regression). EncyclopeDIA's
    "ridge walk" is `greedy_walk` / `method="greedy"`, and the dynamic-programming path is
    `method="global"`.
- **PR A (survey, parity, foundations):** implemented.
  - The jar's fit is replayed bit for bit: all four lab reference sidecars (205,000 rows,
    every warp, delta and 95 % flag) and ten synthetic cases the jar itself evaluated.
  - Files: `core/stats/density.py`, `core/stats/monotone.py` (base), `core/signal/modal_path.py`
    (`greedy_walk`), `core/signal/alignment.py` (`method="greedy"`), `massspec/rt/encyclopedia.py`,
    `scripts/survey-encyclopedia-rt-kde.py`.
- **PR B (global path and API):** implemented.
  - The algorithm and API: `monotone_path`, `method="global"`, `massspec/rt/{__init__, pairs}.py`.
- **PR C (refinement and benchmark):** implemented.
  - Code: the fittable `RationalQuadraticSpline`, both refinements, and
    `scripts/bench-rt-alignment.py`.
  - The results and the decision are under "Benchmark".
- **Default flipped (2026-10-22):** `align` and `fit_monotone_alignment` default to
  `method="global"` with `refine="auto"`, which is `modal` for `global` and no refinement
  for `greedy` and `encyclopedia`.

## Methods and defaults

**What a call does today without options:**

| Call | Method | Curve | Refinement | Beyond the fitted range |
|---|---|---|---|---|
| `massspec.rt.align(pairs)` (also `align_to_library`, `align_runs`) | `global` | rational-quadratic spline | `modal` | `soft` tails |
| `core.signal.alignment.fit_monotone_alignment(x, y)` | `global` | rational-quadratic spline | `modal` | `soft` tails |

`refine` defaults to `"auto"`: `modal` for `global` (unless `curve="linear"`), nothing for
`greedy` and `encyclopedia`. `refine=None` gives the unrefined global path.

**To reproduce EncyclopeDIA:** `method="encyclopedia"` is the jar's curve *inside* the range
it fitted. Beyond that range it is not the jar's unless `extrapolation="clamp"` is also given,
because soft tails replace the jar's clamp. With clamp it reproduces the jar end to end, for
more than 20 points (below 21 the jar's least-squares line extrapolates unclamped).

`soft` continues the curve's end slope, shrunk toward its whole-range slope where few points
support that end, for up to a quarter of the range, and holds constant beyond.

**The methods, in two stages and an optional third:**

| | Density | Path through it | Curve | Refinement |
|---|---|---|---|---|
| `encyclopedia` | the jar's: 3000² grid over each axis's min–max, cosine kernel, jar bandwidth, float32, point by point | greedy walk | the jar's piecewise-linear knots | not allowed |
| `greedy` | the same density, built by FFT in float64 (options: padding, trimming, robust bandwidth) | greedy walk | piecewise-linear (staircase, C0) | — |
| `global` | standardized axes rotated 45°, Gaussian kernel, narrowed across the band in a second pass | Viterbi (best path overall), slope prior | rational-quadratic spline (smooth, strictly increasing, invertible) | — |
| `greedy` + `refine="modal"` | as `greedy` | as `greedy` | spline, refitted | the spline fitted to the raw pairs |
| `global` + `refine="modal"` | as `global` | as `global` | spline, refitted | the spline fitted to the raw pairs |

- **`encyclopedia` vs `greedy`:** the same algorithm. `greedy` builds the grid by FFT in
  float64 instead of stamping point by point in float32, so it is not bit-identical (on the
  reference fits its curve is within 0.05 min of the jar's over the inner 90 % of library time). It is faster at large sizes and
  takes weights and options.
- **`greedy` vs `global`:** two things differ.
  - The path search: a greedy walk commits one step at a time from the densest cell, while
    Viterbi scores every monotone path and keeps the best one.
  - The density it searches: the global method's density is symmetric in the two axes and
    narrow across the band.
- **`+modal`:** the modal refinement replaces the curve with a spline fitted to the pairs
  themselves.
  - The fit: an annealed robust kernel on each pair's distance from the curve, which needs
    no grid, so the curve follows curvature that grid smoothing flattens.
  - The base method is only its starting point. `greedy+modal` and `global+modal` came out
    almost the same in the benchmark.
  - It costs 5–19 s per fit against 1–3 s for the others.
- **`refine="density"`:** also built. It refines the spline against the grid's density
  instead of the pairs, and the benchmark found it no better than `global` alone.

## The problem

Two runs, or a library and a run, measure the same peptides' retention
times on different axes. A monotone map between them is the basis of
several things:

- **RT windows:** where to look for a library peptide in a run.
- **Cross-run work:** carrying an identification across runs.
- **Display:** putting runs on one axis.
- **Scoring:** a peptide's distance from where it should elute is a scoring feature (the
  jar's `deltaRT`).

The pairs come from identifications, so some are wrong: about 1 % at 1 % FDR,
and many more for a predicted library or a loose threshold. A least-squares
fit is pulled by them. EncyclopeDIA's method follows where the pairs are
densest instead (Searle et al. 2018, Nat Commun 9:5128; Wilburn et al.
2023, bioRxiv 10.1101/2023.05.30.542978), and that robustness is what this
work keeps while removing the method's rough edges.

## Review of the existing implementations

### Cartographer `cartographer/rt.py` (176 lines; from Chronologer's `kde_alignment.py`)

It has no callers and no tests beyond the import check. The three implementations differ in their details:

| | EncyclopeDIA 6.9.23 (Java) | Skyline `KdeAligner.cs` | Cartographer / Chronologer (Python) |
|---|---|---|---|
| Grid | 3000 (paper: 1000) | 1000 | 3000 |
| Kernel σ cap | 75 cells (res/40) | res/40 | none |
| Walk tie rule | diagonal if the diagonal is densest or the side neighbours tie | same as Java | `np.argmax` over (0,1),(1,0),(1,1): ties go to the y-only step |
| Several y per x | corner cut at each turn | midpoint | only x-advancing steps recorded |
| Beyond the data | clamp | clamp | flat pad + `np.clip` |

**Defects of the Python port, each removed here:**

- **Bandwidth.** Each axis spans the data's minimum to maximum, so one outlier stretches the
  grid. The bandwidth's standard deviation also includes outliers.
- **Edges.** Stamps are cut at the grid edges, so density near the ends of the gradient is
  biased low. A stamp wider than the grid breaks the edge-trim code.
- **Ties.** A tie goes to the y-only step. In an empty stretch the walk therefore climbs
  vertically to the edge, and the tail is padded flat.
- **Ends.** Where density falls toward the data's ends, y-only steps beat x-advancing ones,
  which bends both ends toward vertical.
- **Locality.** The walk is greedy: it can follow a secondary dense band and cannot recover.
- **Output.**
  - The result is a C0 staircase at the grid's resolution, about 2.4 s on a 120-min
    gradient.
  - It is monotone but not strictly, so it cannot be inverted.
  - It is evaluated one value at a time in Python, and inputs beyond the fitted range are
    clamped.

### EncyclopeDIA 6.9.23, from its bytecode (`javap -c`)

`Encyclopedia.getRescoringModel` → `ScoredPSMFilter` →
`RetentionTimeFilter.getFilter(points, 0.95f)`.

**Inputs.** The points are x = library RT / 60 and y = observed RT / 60, in minutes. They are
the PSMs passing Percolator at 1 %, including the few decoys that share a target's sequence.
The jar writes them, in the order it fitted them, to `<run>.encyclopedia.txt.{first,final}.rt_fit.txt`.

**Choice of model.**
- More than 20 points: `TwoDimensionalKDE(points, 3000).trace()`.
- 2–20 points: a float32 least-squares line, which is not clamped.
- Fewer: the identity.

The arithmetic, as `massspec/rt/encyclopedia.py` replays it (float32 unless noted):

1. **Axis ranges and cells.** Each axis's range is the float32 minimum to maximum. A value
   maps to `Math.round((2999/range)·(v − start))` (half up), clamped to the grid; a zero
   range maps every value to cell 1500.
2. **Bandwidth.** `h = Math.pow(n, (double)(-1f/6)) · (sd_i + sd_j)/2` (double), where
   `sd_i` and `sd_j` are the standard deviations of the cell indices computed with float32
   running sums (`torch.sum` accumulates in double and gets a different answer). Then
   `σ = min(75, h / 2.3548)`. In every reference search seen, σ was 42–56 cells and the cap
   was never reached.
3. **Stamps.** Each point adds a stamp of radius `Math.round(2f·σ)` to a float32 grid, in
   point order, cut at the edges. A stamp entry is `(float)(A·cos(π r/4σ))`, with:
   - `A = π/8σ`;
   - r the Euclidean distance in cells;
   - the cosine a parabolic approximation (`B = 4/π`, `C = −4/π²`, `P = 0.225`);
   - zero beyond 2σ, and entries ≤ 0 skipped.
4. **Walk.** It starts at the first strict maximum in x-major order and walks down, then up,
   appending to one list.
   - **Each step:** append the current cell; stop at any edge.
   - **Choice:** step diagonally if the diagonal neighbour is the densest or the two side
     neighbours tie (so empty stretches are crossed diagonally); otherwise step toward the
     denser side.
   - **Corner cut:** before an x-step, drop the last point if the one before it has the same
     mapped x; likewise for y. That cuts each corner, while straight runs keep their points.
5. **Knots.** Cells map back as `(range/2999)·i + start`, clamped, and are sorted by (x, y).
   `LinearInterpolatedFunction` evaluates them:
   - the query is cast to float32 and located with `Arrays.binarySearch`;
   - interpolation is done in double, then cast to float32;
   - equal knot x gives the mean y;
   - beyond the knots the end values hold. The inverse is the mirror image.
6. **Residual band.** `delta = (float)y − f(x)`. The 95 % band is the sorted finite deltas
   at `floor((n−1)·t)` and `ceil((n−1)·(1−t))`, with `t = (1 − (double)0.95f)/2`.

**Where its defect shows in the lab's data.**
- In the gas-phase-fraction first pass, the up-walk reached the top edge with x left
  over: 7 points at library RT 26.13–27.41 all get the maximum observed RT, 105.644.
- The down-walk stopped at 10.58 min, although observed RTs start at 0.98 min.

**Code that exists in the jar but is not on this path:**
- a Gaussian-plus-uniform mixture of residuals (`RetentionTimeFilter.generateMixtureModel`,
  used only by the median and Theil–Sen filters);
- an RT-dependent residual width (`RTFitMixtureModel`, no callers);
- `MonotonicCubicSplineInterpolatedFunction` (Percolator's D2PEP only).

`-libexport -a` aligns each run to a seed run with the same KDE at resolution 10000; the
reference data has no such report.

## Strategies for a monotone curve through modal density

A survey, with what was built marked. The robust core is that a fit
follows the *mode* of the paired times. Everything below is a way to do
that, smoothly, invertibly and with sensible ends. Least squares, LOESS and
isotonic regression on the pairs were reviewed and rejected as the alignment
itself, because they follow the mean.

| Approach | Monotone | Robust | Smooth | Notes | Here |
|---|---|---|---|---|---|
| Greedy walk on a KDE (EncyclopeDIA) | non-strict | high | C0 staircase | local; stops at the first edge | `greedy` |
| Global monotone path, dynamic programming / Viterbi (Felzenszwalb & Huttenlocher 2012; Calib-RT, Zhang et al. 2024, robust to 75 % FDR where LOESS and RANSAC failed) | strict, with bounded moves | high | C0 path, smoothed by the spline | exact optimum; forward–backward gives a per-column posterior (its mean is an option, not the default: see Benchmark) | `global` |
| DTW lattice (3 moves) | non-strict | high | C0 | step-count bias; only slopes 0, 1, ∞ | (subsumed by `global`) |
| Monotone cubic Hermite (Fritsch & Carlson 1980; Steffen 1990) | yes | n/a (interpolant) | C1 | end cubic can turn over; no closed inverse | `steffen_slopes` only |
| Rational-quadratic spline (Gregory & Delbourgo 1982; Durkan et al. 2019) | strict | n/a / as fitted | C1 | closed-form inverse; natural linear tails | `RationalQuadraticSpline` |
| I-splines, SCAM cumulative-exp (Ramsay 1988; Pya & Wood 2015) | strict | as fitted | C2 | linear in coefficients | not built |
| Modal regression / Welsch loss, annealed (Yao & Li 2014; graduated non-convexity) | per curve family | high, redescending | per curve | needs a good start; the gridless limit of the KDE | `refine="modal"` |
| Density-surface fit along the curve | per curve family | high | per curve | the continuous form of the DP's objective | `refine="density"` |
| Gaussian + uniform mixture (EncyclopeDIA's paper; MLESAC) | per curve | high | per curve | gives per-point probabilities | deferred (residual model) |
| Isotonic / PAVA, L1 isotonic, COBS | non-strict | low / moderate | steps | mean or median | rejected |
| LOESS / LOWESS (TRIC, XCMS, alphaDIA) | no | low unless robust iterations | C1 | mean-seeking | rejected |
| Spectronaut segmented Theil–Sen (Bruderer et al. 2016) | not guaranteed | moderate | C0 | linear extrapolation | `theil_sen` for < 21 points |

**Boundary handling.**

| Option | Here |
|---|---|
| Pad the grid by about 3σ so stamps are not cut | `pad_sigmas`; the global path always pads |
| Renormalize by the kernel mass inside the data's support, `D / (1_box ⊛ K)` (Jones 1993), with fractional edge cells | `support_mask` |
| Extrapolate beyond the data: `clamp` (EncyclopeDIA, Skyline, Chronologer) | `Extrapolation` |
| Extrapolate beyond the data: `linear` with the end slope (Spectronaut, Calib-RT, OpenMS) | `Extrapolation` |
| Extrapolate beyond the data: `soft`, a slope shrunk in log space toward the curve's whole-range secant by the support near the end, for a bounded margin (a quarter of the range by default) | `Extrapolation` |
| Uncertainty that grows with distance from the data | deferred |

## Design

**Placement.** The generic algorithms live in `core`, so other calibrations (electrophoresis
ladders, HPLC) can use them. The jar's exact arithmetic lives in `massspec`.

- **`core.stats.density`** — gridded 2-D kernel densities:
  - linear or nearest deposition and FFT convolution;
  - Gaussian or EncyclopeDIA's cosine kernel;
  - `support_mask` renormalization;
  - bicubic sampling, differentiable;
  - `weighted_quantile`.
- **`core.stats.monotone`** — monotone curves:
  - the `MonotoneCurve` protocol and `Knots`;
  - `PiecewiseLinear` (ties averaged, so a staircase inverts), `LinearMap`, `steffen_slopes`;
  - `RationalQuadraticSpline`: a `Parametric`, vmap-safe, fitted with `core.optim`;
  - `Extrapolation` / `TailedCurve`.
- **`core.signal.modal_path`** — paths through a density grid:
  - `greedy_walk`, the jar's rules, with exact or tolerant ties and corner cuts by index or
    by mapped value;
  - `monotone_path`: Viterbi plus forward–backward over columns, vectorized per column.
- **`core.signal.alignment`** — the generic entry point `fit_monotone_alignment(x, y,
  weights, method, curve, refine, extrapolation)`, returning a `MonotoneAlignment`
  (`forward`, `inverse`, `residuals`, `band`, `within_band`, `is_extrapolated`). Fewer than
  21 points give a Theil–Sen line.
  - **`greedy`:**
    - EncyclopeDIA's algorithm on the FFT grid;
    - optional trimming, padding, a robust bandwidth and renormalization (renormalization hurts a greedy walk and is off by default: see Benchmark);
    - the walked cells as `PiecewiseLinear`, or as a spline.
  - **`global`:**
    - **Standardize and rotate.** Each axis is standardized by its median and 2.5–97.5 %
      span. The plane is rotated 45° (u along the band, v across it); swapping x and y sends
      v to −v, so the fit is the same either way round.
    - **Grid and score.** Gaussian KDE with σ from Scott's rule, read as a FWHM as the jar
      does. Columns are σ/4 apart; rows are 1/K of a column (K = 8).
      - Score: `log(D + 0.05·D̄)`; the floor is what a column of only wrong pairs scores.
      - Move prior: −λt², where t is the move per column and 0 is the span ratio.
      - Both terms are scaled per σ of path, so the posterior does not depend on the grid.
      - Moves are bounded by (K−1)/K, so every path is strictly monotone: local slopes lie
        in 1/15–15 times the span ratio.
    - **Path.** The default is the Viterbi path, the mode. The posterior mean is an option: it is smoother and sub-cell, but where the band bends away from the prior slope it sits between the two (see Benchmark).
    - **Second pass.** The kernel across the band is narrowed to the residual spread (MAD),
      with finer rows. The jar's single bandwidth scales with the RT range, not with the
      band's width.
    - **Domain and curve.** The domain is the x-range of the pairs within 3 spreads of the
      path. The path, thinned to a knot every σ/2 with Steffen slopes, becomes a
      `RationalQuadraticSpline`.
  - **Refinement:** a `RationalQuadraticSpline` (one bin per two σ of curve, 8–64), started
    from either curve and fitted by L-BFGS with an annealing schedule.
    - **`density`:** maximize log-density along the curve, sampled bicubically.
      - Annealing: kernels 4, 2 and 1× σ.
      - Sampling: points uniform in x and in y.
      - Prior: the global path's slope prior applies too.
    - **`modal`:** maximize the mean Gaussian kernel of each pair's first-order orthogonal
      distance (in standardized units) from the curve. The kernel starts at 3× the residual
      spread and shrinks by 1.5 per stage to the spread.
    - **Length reward.** Two choices were needed, found in the benchmark, against the curve
      growing hooks at its ends. Within an annealing stage, the arc measure (density) and the
      orthogonal scaling (modal) are fixed from the curve at the start of the stage, as in
      IRLS. Differentiated, both pay a curve to turn vertical at its ends, where a vertical
      stretch passes through the whole band. The roughness penalty holds neighbouring bins'
      log-secants together, and each knot derivative to the secants beside it; a penalty on
      the knot derivatives alone does not smooth a rational-quadratic spline.
- **`massspec.rt.encyclopedia`** — `fit_encyclopedia(x, y, backend="exact" | "fft")`
  returns an `EncyclopediaFit` with:
  - its `JarLinearFunction` or `JarLinearRegression` curve;
  - its float32 band;
  - its σ, stamp radius and apex.

  `exact` stamps point by point, 1.6–3 s on a reference search. `fft` is near-parity:
  97.8–99.8 % of rows identical and at most 0.035 min apart, in about 0.9 s.
- **`massspec.rt`** — `align(pairs, method="encyclopedia" | "greedy" | "global", refine,
  extrapolation)` returns an `RtAlignment` in seconds:
  - methods: `to_observed`, `to_reference`, `residuals`, `within_band`, `is_extrapolated`;
  - pair builders (`massspec.rt.pairs`):
    - `library_pairs`: a library's `rt_predicted` against a run's identifications;
    - `run_pairs`: two runs' identifications joined on the peptide;
    - `engine_pairs`: the engine's own points from `rt_alignment`, in order, with decoys;
  - `align_to_library` and `align_runs` read the pairs from an experiment;
  - `to_rt_alignment_table` gives `RT_ALIGNMENT` rows with `engine="constellation"`
    (library → run only).

  The EncyclopeDIA replay fits in minutes, as the jar does.

## Verification

- **Parity with the jar, on real data:** `tests/test_massspec_rt_parity.py`, opt-in with
  `CONSTELLATION_RT_FIT_REFERENCE=/mnt/e/261010_u2os`.
  - Each reference sidecar is refitted from its own rows in order.
  - Every `warpToactual`, `delta` and `within95p` must be identical: 76,459 + 67,595 +
    31,833 + 29,103 rows, all identical.
- **Parity with the jar, in CI:** `scripts/survey-encyclopedia-rt-kde.py` compiles a small
  harness with `javac --release 21`.
  - **The run:** the jar's bundled Temurin 21 (the JVM a search uses) runs
    `RetentionTimeFilter.getFilter` on ten synthetic point sets, in its own session with a
    private temp directory. Only synthetic files are given to Java.
  - **The cases:**
    - 1, 2, 20 and 21 points;
    - a zero-width axis;
    - values on half cells;
    - a lattice of exact ties;
    - plateaus and jumps;
    - a walk that ends on the top edge;
    - 30 % outliers.
  - **The record:** `tests/data/encyclopedia_rt_kde/6.9.23/*.json` (192 KB) holds the warp,
    delta, 95 % flag and inverse per point, and the warp at 201 query times past both ends.
    `tests/test_massspec_rt.py` holds the replay to them exactly.
- **Unit tests** (`tests/test_core_stats_{density,monotone}.py`,
  `tests/test_core_signal_{modal_path,alignment}.py`, `tests/test_massspec_rt.py`):
  - **Density:**
    - deposition conserves mass;
    - FFT equals direct convolution;
    - nearest deposition equals a stamp per point;
    - Gaussian KDE equals a direct kernel sum;
    - renormalization is flat to the edge.
  - **Curves:**
    - the spline passes through its knots and slopes, inverts, and matches autograd and
      finite differences;
    - the spline is vmap-compatible (`tests/test_core_optim_de.py`);
    - a Welsch fit ignores 30 % outliers;
    - each tail policy behaves as specified.
  - **Paths:**
    - every walk branch is checked on a hand-built grid;
    - Viterbi and the posterior equal brute force;
    - mirrored rows give a mirrored path.
  - **Alignment:**
    - each method recovers curved warps through 20 % outliers;
    - the global fit is the same either way round and ignores a dense off-curve cluster.
  - **Pairs:** filters, joins and order.
- **Benchmark:** `scripts/bench-rt-alignment.py` (results below).

## Benchmark

`python3 scripts/bench-rt-alignment.py --out-dir DIR --reference-dir /mnt/e/261010_u2os --seeds 3`,
run 2026-10-22 on the working tree over `242c4e6`, torch 2.12.0 (CPU, 6 threads), peak RSS 1.3 GB.

**Methods.**
- `encyclopedia`: the replay.
- `greedy`: the generic form.
- `greedy+`: with robust bandwidth, 0.1 % trim, 3σ padding, renormalization and linear deposition.
- `global`: the Viterbi path.
- `global-mean`: the posterior mean.
- Refinements: `global+density`, `global+modal`, `greedy+modal`.

**Synthetic setup.** Each scenario varies one factor from a baseline: a curved warp
`2 + 100((x+5)/105)^0.75` over 0–100 min, 0.3 min Gaussian noise, 10 % uniformly wrong pairs,
and 10,000 pairs. Scores are against the true warp over the central 98 % of the correct
pairs' x, as median over 3 seeds.
- **Other warps:** linear; kinked; saturating (slope 0.2× at the ends); predicted (shaped
  like the lab's predicted-library fit, 1.9 min noise, 2 % wrong).
- **Noise:** Student-t ν = 3, or heteroscedastic.
- **Wrong pairs:** 0 / 25 / 50 %; a dense off-curve cluster (5 %); a parallel false band
  (8 %, +6 min).
- **Gap:** no correct pairs in x = 40–55.
- **Sizes:** n = 50 … 100,000.

**Mean absolute error (min):**

| scenario | encyclopedia | global | global-mean | global+density | global+modal | greedy+modal |
|---|---|---|---|---|---|---|
| baseline | 0.034 | 0.031 | 0.026 | 0.029 | **0.016** | **0.015** |
| linear | 0.018 | 0.020 | 0.010 | **0.008** | 0.017 | 0.017 |
| kinked | 0.086 | 0.083 | 0.120 | 0.069 | **0.019** | **0.018** |
| saturating | 0.111 | 0.272 | 0.875 | 0.345 | 0.031 | **0.016** |
| predicted | 0.139 | 0.214 | 0.485 | 0.221 | **0.079** | **0.078** |
| Student-t noise | 0.034 | 0.030 | 0.025 | 0.024 | **0.018** | **0.018** |
| heteroscedastic | 0.036 | 0.035 | 0.027 | 0.029 | **0.018** | **0.018** |
| 50 % wrong | 0.057 | 0.040 | 0.034 | 0.030 | **0.020** | **0.020** |
| dense cluster | 0.034 | 0.031 | 0.024 | 0.025 | **0.015** | **0.016** |
| parallel false band | 0.094 | 0.031 | 0.026 | 0.027 | **0.015** | **0.016** |
| gap | 0.470 | 0.074 | 0.061 | 0.059 | **0.018** | **0.017** |
| n = 50 | 0.410 | 0.251 | **0.206** | 0.396 | 0.292 | 0.268 |
| n = 200 | 0.176 | 0.149 | **0.098** | 0.298 | 0.122 | 0.125 |
| n = 1,000 | 0.066 | 0.070 | 0.055 | 0.077 | 0.056 | **0.055** |
| n = 100,000 | 0.019 | 0.021 | 0.013 | 0.010 | **0.005** | **0.006** |

**Summary over the 17 scenarios:**

| method | MAE ≤ the replay's | worst MAE ratio to the replay | geometric-mean ratio | runs with max error > 2 min | time, 10k pairs | time, 100k pairs |
|---|---|---|---|---|---|---|
| encyclopedia | — | 1 | 1 | 5 / 49 | 0.92 s | 3.7 s |
| greedy | 71 % | 1.00 | 1.00 | 5 / 49 | 0.80 s | 0.86 s |
| greedy+ | 0 % | 4.08 | 1.53 | 8 / 49 | 1.7 s | 1.5 s |
| global | 71 % | 2.45 | 0.84 | 1 / 49 | 0.40 s | 0.67 s |
| global-mean | 82 % | 7.87 | 0.77 | 7 / 49 | 0.37 s | 0.70 s |
| global+density | 76 % | 3.10 | 0.75 | 10 / 49 | 2.8 s | 2.8 s |
| global+modal | **100 %** | **0.93** | 0.39 | 1 / 49 | 4.7 s | 7.5 s |
| greedy+modal | **100 %** | **0.93** | **0.37** | **0 / 49** | 5.1 s | 6.5 s |

**Real EncyclopeDIA fits.** Each method refits the jar's own points:
- *inlier MAD:* the robust spread of the residuals of the pairs the jar called inliers;
- *vs jar:* the largest difference from the jar's warp over the inner 90 %;
- *held out:* fitted on the inner 80 % of library time, the median error on held-out inliers
  at the ends, under soft / clamp tails, and the share within 1 min (soft).

| file | method | inlier MAD | vs jar | held out, soft | held out, clamp | within 1 min | time |
|---|---|---|---|---|---|---|---|
| wide-window, final | encyclopedia | 0.718 | 0 | 0.687 | 4.86 | 75 % | 1.6 s |
| | global | 0.509 | 1.04 | 0.300 | 5.16 | 98 % | 0.24 s |
| | global+modal | **0.463** | 1.10 | **0.147** | 4.98 | 97 % | 12.9 s |
| wide-window, first | encyclopedia | 0.739 | 0 | 1.261 | 4.55 | 35 % | 1.5 s |
| | global | 0.539 | 0.92 | 0.248 | 5.15 | 96 % | 0.51 s |
| | global+modal | **0.480** | 1.07 | **0.143** | 5.00 | 96 % | 11.9 s |
| GPF build, final | encyclopedia | 1.797 | 0 | 3.020 | 3.91 | 11 % | 2.6 s |
| | global | 1.802 | 0.32 | 1.276 | 5.21 | 42 % | 0.66 s |
| | global+modal | **1.790** | 0.21 | **1.275** | 5.28 | 42 % | 19.0 s |
| GPF build, first | encyclopedia | 1.903 | 0 | 3.262 | 3.93 | 9 % | 2.2 s |
| | global | 1.906 | 0.35 | 1.368 | 5.17 | 40 % | 0.64 s |
| | global+modal | **1.897** | 0.20 | **1.328** | 5.33 | 40 % | 17.6 s |

The GPF build is a predicted library against the run, and its residuals are dominated by
prediction error (about 1.8 min) whatever the curve.

**Extrapolation.** Fit on the 10th–90th percentile of x; mean error on the held-out correct
pairs.

| | curved | saturating | predicted |
|---|---|---|---|
| encyclopedia: clamp / linear / soft | 5.61 / 0.84 / 0.92 | 1.64 / 3.92 / 3.98 | 4.12 / 3.92 / 3.84 |
| global: clamp / linear / soft | 6.11 / 0.83 / 0.87 | 1.81 / 3.17 / 3.24 | 4.73 / 2.30 / 2.31 |
| global+modal: clamp / linear / soft | 6.08 / **0.41** / 0.52 | 2.40 / **1.95** / 2.02 | 4.71 / **1.48** / **1.48** |

### What the benchmark found and changed

- **Viterbi is the global default, not the posterior mean.**
  - The posterior mean is the Bayes estimate under squared loss. Where the band bends away
    from the prior slope (saturating ends, the predicted-library shape), the posterior
    spreads over "follow the band" and "continue straight", and the mean sits between them.
    That is mean-seeking, against the method's point.
  - Viterbi is the mode. It cut those errors 0.88 → 0.27 and 0.48 → 0.21 min and failures
    7 → 1, at a small cost on smooth warps (baseline 0.026 → 0.031).
  - Other settings changed little. The prior's strength λ ∈ {0.1, 0.3, 1} and finer slope
    steps barely mattered; the second pass is essential (without it, baseline error rose
    10×).
- **Soft tails reach a quarter of the range, not a tenth.**
  - With 10 %, the tail stopped before most held-out points did. At ≥ 25 %, soft and linear
    were within about 5 % of each other on all seven hold-outs.
  - Soft keeps a bound on how far a curve is carried. Clamping is the worst policy wherever
    the curve keeps rising (median held-out error 4–5 min on every real file).
- **Both refinements rewarded length until fixed.**
  - Differentiated, the density objective's arc measure and the modal objective's
    orthogonal scaling pay a curve to turn vertical at its ends: a vertical stretch passes
    through the whole band. The first runs showed hooks of +18 min.
  - Both are now fixed per annealing stage.
  - A roughness penalty on the knot derivatives alone did not smooth the spline. One on the
    log-secants and on each derivative against them does.
- **Renormalization hurts a greedy walk** (saturating 0.11 → 0.32 min): it amplifies the
  box's sparse corners, which hold only wrong pairs, so it stays off for `greedy`. That is
  why `greedy+` lost everywhere. Trimming and padding were neutral.

### Decision: `global` + `modal` is the default

The proposed rule for replacing `encyclopedia` as the default:
1. In-range error at or below the replay's in ≥ 90 % of scenarios, and never above 1.25× it.
2. No more failures than the replay.
3. A lower inlier spread on all four reference fits.
4. At most twice its runtime at 76,000 pairs.

**`global+modal` meets the first three:**
- it is never worse than the replay, and 2.6× more accurate on average;
- it has fewer failures;
- its inlier spread is lower on all four real fits, 35 % on the wide-window run;
- on the real hold-outs it predicts ends 4–9× better.

**It fails the fourth:** 12–19 s against the replay's 1.5–2.6 s on the real fits.
`global` alone is the fastest method (0.2–0.7 s) and the best on gaps and false parallel
bands, but it is behind the replay on saturating and predicted shapes.

**Options put to the PI:**
- (a) flip to `global` + `refine="modal"` accepting about 15 s per fit (a search takes
  minutes);
- (b) first make the modal refinement cheaper (subsample the pairs, fewer stages), then
  flip;
- (c) keep `encyclopedia` and offer the others.

**The PI chose (a) on 2026-10-22.** The runtime criterion is waived; making the refinement
cheaper stays open (see Deferred).

## Deferred

- **CLI and experiments:** a `massspec align-rt` CLI and writes into experiments,
  including a table for run-to-run fits.
- **Browser:** `results.rt_mapping` and the browser's predicted RT moved to these curves;
  alignment QC panels.
- **Residual model:** a residual model with per-point probabilities and an RT-dependent
  width (the jar's mixture and `RTFitMixtureModel`); uncertainty growing in the tails.
- **Alignment to a seed run:** cross-run alignment to a seed run at resolution 10000, as
  `-libexport -a` does.
- **Cost of the default:** a cheaper modal refinement (subsampling the pairs, fewer
  annealing stages), now that it is the default.
