# Counter → Cartographer: Architecture & Gap Audit

**Scope.** A design-and-audit document, not an implementation plan. Produces (a) the architectural flow of the scoring/quantification engine as PyTorch-style modules, (b) a Counter v2 → Cartographer gap map, (c) the additive-progenitor extension to the joint likelihood, and (d) the data-slicing pattern that makes all of the above efficient.

Writing this doc IS the task. Code comes after the user reviews and approves.

---

## Context

The last session of notebook work (nb42a–c) hit a wall on peak-boundary refinement. The nb42c state-of-play doc (`docs/nb42c_state_of_play.md`) catalogs both implementation mistakes (wrong DE, improper censored scaling) and conceptual mistakes (binary interference mixture, fudge-factor cens_weight). Rather than another round of iteration on a mis-specified loss, we're stepping back to redesign.

Three strategic shifts inform this redesign:

1. **Interference as additive progenitors, not binary mixture.** Instead of "signal vs noise", observations are explained by a sum over co-contributing species (same-peptide isotopologues, different peptides, unrelated background). The binary mixture of Counter v2 §8 is a special case.
2. **Seeded and global as a spectrum, not a dichotomy.** Even in PROCAL calibration, interferers are unpredictable: crude synthetic peptides carry solid-phase synthesis artifacts, degradation products, and modification adducts that are not in any library. Both seeded and global modes therefore need the ability to *discover* progenitor masses in-panel; they differ only in whether one mass is locked to a PSM target. The architecture should not commit to a fixed progenitor set at either end.
3. **Overlapping, center-weighted panels.** Current code filters by m/z first (ppm window, mass-sorted). For correlated-peak detection and efficient batching, 2D panels of (narrow m/z × RT chunk) are the natural unit — a mini in-silico DIA window. Hard non-overlapping panels leak signal across boundaries given the ppm-scale m/z errors; instead, panels overlap and each panel's "active" parameters are those near its center, enforced either via gated `requires_grad` or via proximity-scaled priors.

The near-term deliverable is unchanged: a tool that produces N_total estimates with credible intervals from joint MS1+MS2 likelihood on ProteomeTools PROCAL. MS2 is expected to slot in with modest effort once the architecture supports arbitrary `(mass, z, k)` channels with learnable fractions — fragment ions are just another `p_k`-like array at a different resolution scale.

---

## Design principles (inherited from Counter v2 §0, with additions)

1. **Physics first, statistics second, engineering last.** Unchanged.
2. **Identification and quantification are the same problem.** Unchanged.
3. **Priors are named and defended.** Unchanged.
4. **Every component has a falsifiable prediction.** Unchanged.
5. **One scoring function; orchestration decides what it's applied to.** (New.) The seeded and global modes differ only in whether a target mass is locked (and, softly, the progenitor-discovery prior). The scorer itself is agnostic.
6. **Panel-local additivity, run-level composition.** (New.) Within a 2D (m/z × RT) panel, observations are modeled as an additive sum over the progenitors active in that panel. Panels overlap in m/z; each panel's parameters are center-weighted so a progenitor is "owned" by the panel whose center it's nearest to, with soft falloff.
7. **Discover, don't assume, the progenitor set.** (New.) Even in seeded mode, the set of masses contributing to a panel is partially unknown (synthesis artifacts, modifications, degradation products). The architecture must support introducing new `Mass` instances into a panel during fitting, not just fitting a fixed list.

---

## Architecture: modules and their responsibilities

Sketch in PyTorch-style `nn.Module` terms. The names are provisional.

### `GlobalCalibration` (per file)
State: `α₀, α₁, d_mz_1, d_mz_2, ρ_R, ν_mz, α_mz`.
Role: converts between ion counts and intensities, applies isotope-spacing corrections, parameterizes the m/z precision as a function of ion count. Fit once per file on PROCAL; frozen during downstream panel fits.

### `Mass` (the core progenitor object)
State (learned, per-Mass):
- `m` — monoisotopic mass (typically known for seeded peptides; candidate for free parameter in global mode)
- `p_z = {f_z}` — charge-state fractions (simplex)
- `p_k = {p_k}` — isotope fractions per charge (simplex, could share across charges or vary)
- `peak` — HyperEMG(1,1) parameters: `N_total, μ, σ, τ_R, τ_L, η`
- `ν_I` — per-peptide intensity d.o.f.
- `c_mz` — per-peptide m/z precision constant

For fragment ions (MS2), `p_k` becomes a longer array indexed over fragment channels, with a second `ν_I_MS2` and resolution-scaled α. Same module, longer observation axis.

Method: `predict(t, m/z grid) → {I_pred, σ_I, m/z_pred, σ_mz}` per (z, k) channel. Integrates HyperEMG to get `N(t)`, partitions by `p_z · p_k`, applies `α(z)` and IIT scaling, returns predicted observables.

### `Panel` (the batchable unit)
A 2D slice of the run: narrow m/z range × RT chunk, with a defined *center* in (m/z, RT). Contains:
- Observed ions inside its (possibly overlapping) bounds — MS1 peaks from the spectra table, plus optionally MS2 fragments within this precursor window.
- A list of `Mass` instances whose predicted signal overlaps the panel. Each `Mass` has a "home panel" (the one whose center it's closest to) but may contribute to neighbors.
- A center-weighting function `w_center(m_q − m_panel_center, t_q − t_panel_center)` that attenuates how strongly panel observations inform each `Mass`'s parameter updates. In the home panel, `w ≈ 1`; in neighbor panels, `w` decays. This can be realized either (a) hard: non-home `Mass` instances contribute to the forward pass but have `requires_grad = False` in the non-home panel's loss, or (b) soft: gradients from non-home panels are weighted by `w_center`.
- Shared `GlobalCalibration` reference.

Method: `forward() → logL`. Computes, for each observed ion, the additive-progenitor prediction from all `Mass` instances overlapping the panel, evaluates the joint m/z + intensity likelihood, sums. Zero-ion regions inside the panel contribute via the censored term (§Zeros). Also supports adding newly-discovered `Mass` candidates during iteration (see *Progenitor discovery* below).

### `PanelSet` / orchestrator
Decides how panels tile the run (with overlap), which `Mass` instances start in each, and when/how to introduce new ones. Seeded mode: panels centered on seed peptides' priors; a target `Mass` per seed, plus library neighbors as warm-start candidates. Global mode: panels tile (m/z, RT) uniformly; initial `Mass` candidates come from a feature-detection pass. In both modes, the orchestrator can add new `Mass` instances mid-fit from unexplained signal (residual peaks exceeding a detection threshold).

### Progenitor discovery (bridging seeded and global)
A pass that inspects the residual `(I_obs − I_pred)` inside a panel and proposes new `Mass` candidates for any coherent unexplained signal — the same detection logic that drives global feature finding, applied inside each panel on the residual. This is the mechanism by which PROCAL calibration can handle synthesis-artifact interferers: the seeded peptide gets its target `Mass`, and any additional progenitors with coherent RT/isotope/charge structure in the residual are discovered and added, without requiring a library lookup.

### Scoring function (the unit that serves both modes)
`score(panel, mass_query) → {N_total posterior, marginal logL, Λ vs null}`. Internally: fit the panel with `mass_query` included vs. excluded, compute the likelihood ratio. Same function whether `mass_query` came from a PSM seed or a global peak detection.

---

## The additive-progenitor observation model

This is the math gap in Counter v2. Two equivalent views; both should be articulated so the implementation choice is informed.

### View A: observation-as-graph

Every observed ion (scan t, centroid m_obs, intensity I_obs) is a node. Every progenitor `q` with non-negligible predicted contribution at (t, m_obs, z) is a node. Edges carry the predicted contribution `I_pred(t; q, z, k)` and predicted m/z `m_pred(q, z, k)`. Observation-attribution is soft: the observed ion's likelihood under the full set of edges is a mixture weighted by predicted intensities.

Posterior attribution weight of observation i to progenitor q (edge belief):
```
γ_{i,q} = I_pred(t_i; q) / Σ_q' I_pred(t_i; q')
```
evaluated with the edge's m/z likelihood as a prior modulation. This is a concrete replacement for Counter v2 §8's two-component mixture: instead of signal-vs-noise for each observation, we have a full soft assignment across all candidate progenitors.

### View B: convolution-as-prediction

The predicted observable at location (t, m) is a convolution of progenitor contributions:
```
I_pred(t, m) = Σ_q Σ_{z,k} I_pred(t; q, z, k) · φ(m − m_pred(q, z, k); R, N_{q,z,k})
```
where `φ(·; R, N)` is the instrument response kernel in m/z. The shape is *not* a simple Gaussian or Lorentzian with width ∝ `c_mz · N^{-α_mz}`; functionally it's closer to a **smoothed square wave**: approximately flat within a resolution-determined width (set by FT transient length, `Δm/z ≈ m / R`), then smooth shoulders reflecting the per-ion centroiding precision (the `N_k`-dependent Student-t we already have). The flat core captures the observation that any ion within the resolution-limited window contributes essentially equally to the centroid estimate; the shoulders capture sub-resolution precision gained from ion statistics. Shoulder shape is an instrument-level prior (per analyzer type, possibly per resolution setting).

A single observed centroid then has expected m/z equal to the intensity-weighted mean over progenitors whose kernels overlap the observation — the partially-resolved multiplet case. Expected intensity is the integral of the kernel sum over the observation's m/z window.

This view is the continuous generalization. It naturally handles:
- Partial ¹⁵N/¹³C fine structure (progenitors are different N-substitution states of one peptide; their m_pred differ by fractions of a Da).
- Interference (progenitors are different peptides sharing a m/z window).
- Low-resolution instruments (wider flat core; same math).

### Equivalence and implementation choice

View A is discrete-observation; View B is continuous-prediction. They agree when the m/z kernels are well-separated relative to observation bin width. They diverge when kernels overlap: View A treats the overlap via soft attribution (edge weights), View B treats it via averaged centroids.

**Practical recommendation:** implement View A for the MS1 Orbitrap case (well-resolved isotopes; observations are centroided peaks that map cleanly to single progenitors at high resolution). Use View B as the conceptual frame when generalizing to ion-trap MS2, low-resolution MS1, or fine-structure modeling — places where the resolved-single-progenitor assumption breaks. Both views should live behind the same `Panel.forward()` interface.

### Zero observations

A zero-ion region inside a panel is positive information: the model predicts `I_pred_total(t, m)` from the progenitor sum, and the probability that no ion was observed in that region given the prediction is a proper log-probability (Poisson/binomial tail, or Student-t CDF depending on which likelihood approximation we're using at that N). The cens_weight fudge factor of nb42c disappears once this term is written as an actual log-probability on the same scale as the data term.

The key design point: the zero-term is computed over *panel-local m/z grid cells with no observation*, not over "scans with no matched ion for peptide q" as in nb42c. The panel-local formulation automatically gets the right spatial scope; the peptide-centric one required us to decide what counted as "the peptide's region" and that was the whole trouble.

---

## Data-slicing pattern: panels as batchable in-silico DIA windows

Current Cartographer flow: `extract_ms1_chromatogram` returns a long table of (scan, peptide, z, k) matches within ±ppm of each target. Mass-first; one row per matched ion per peptide. Good for seeded extraction, wasteful for overlapping peptides.

Proposed panel pattern:

1. Define panels by (m/z center, RT center, m/z half-width, RT half-width). Width tuned so that a typical panel contains O(1–10) progenitors and O(100–10K) observed ions. For a 10 ppm-wide panel at m/z 700, that's ~0.007 Th; at DDA scan cadence over a 2 min RT window, O(thousands) of spectra rows.
2. **Panels overlap** in both m/z and RT. An observed ion contributes to every panel whose bounds it falls within, but the *ownership* of a given `Mass`-parameter update belongs to the panel whose center that mass is closest to (the home panel). Non-home panels can still include the `Mass` in their forward prediction (for correctness of the additive-progenitor sum over observations in the overlap region), but the `Mass`'s gradient signal from that panel is attenuated by the center-weighting function. This avoids the double-counting that would occur with naive overlap while preventing the boundary-leak failures of hard non-overlapping panels.
3. Panels are batchable across the run. When overlap is handled via a home-panel convention, different home panels can be updated in parallel as long as their shared `Mass` instances coordinate (which they do implicitly via the home-panel rule).
4. Seeded use case: one panel per seed peptide, centered on its m/z and prior RT. Initial progenitor list: the target `Mass`, plus any library neighbors within the panel bounds. Progenitor discovery runs inside the panel to add synthesis-artifact / modification / degradation masses.
5. Global use case: tile the (m/z, RT) plane uniformly with overlapping panels; seed each panel's initial progenitor list from a feature-detection pass; then run the same fit-plus-discover loop.

**Design implication for Cartographer:** add a panel-tiling utility alongside the existing search-results-driven extraction. The existing extraction becomes a special case ("one panel per PSM seed").

---

## Counter v2 → Cartographer gap audit

Per section of `Counter_technical_reference_v2.md`, classify: (E)xists cleanly; (P)artial / needs fixing; (M)issing.

| Counter v2 section | Status | Notes |
|---|---|---|
| §1 Source & charge/isotope partitioning | E | Cartographer has `expected_isotope_envelope`; fitted `p_k` live in nb results |
| §2 Ion accumulation, Poisson, IIT | E | `read_scan_metadata` delivers IIT; math applied in calibration_model |
| §3.1 Orbitrap detection, α(z) | E | α(z) in calibration_model with /iit variance (nb42a fix in place) |
| §3.2 Ion trap | M | Placeholder only in v2; cartographer has no IT-specific α model |
| §3.3 TOF | M | Placeholder only |
| §3.4 Astral | M | Placeholder only |
| §4 Intensity likelihood (Student-t) | E | Per-peptide ν_I, IIT in variance |
| §5 m/z likelihood (Student-t, $N_k$-conditioned) | E | In calibration_model; d_mz corrections present |
| §6 Joint (conditional-independent) likelihood | E | Used in nb41b/c per-ion logL |
| §7 HyperEMG peak model | E | `HyperEMGPeak`, `fit_all_peaks`, cartographer DE |
| §8 Interference mixture | **P** | Global 2-component mixture known mis-specified. **Replace with additive-progenitor model (this doc).** |
| §9.1 Per-file vs global params | E | Respected in stepwise EM |
| §9.2 PROCAL calibration | E | `nb41e` path |
| §9.3 α–β degeneracy | E | β removed (nb42a); α₀ small-but-nonzero open question |
| §9.4 Stepwise EM | E | Exists; currently PSM-seeded |
| §9.5 Laplace credible intervals on N_total | M | Called out as nb42e; not implemented |
| §10 Marginal likelihood ratio Λ for identification | M | Conceptualized; no implementation. Required for the scoring function to serve identification. |
| §11 Hsu recovery special case | — | Conceptual; no code needed |
| §12.1 α₀ = 0 formal test | M | Open; candidate for a small notebook |
| §12.2 ν_I biophysical | — | Open science question, not engineering |
| §12.5 MS2 integration | **M** | The biggest concrete missing piece for the MVP. User's framing: add fragment channel as another `p_k`-like array, resolution-scaled α, learnable fractions. Fits cleanly in the `Mass` module. |
| §12.6 Bayesian peak model (GP) | M | Future work |
| §12.7 Non-Orbitrap analyzers | M | Future work |

**Additional gaps not covered in Counter v2:**

| Capability | Status | Notes |
|---|---|---|
| Panel-based data slicing | M | New utility needed; replaces mass-first extraction for the deconvolution use case |
| `Panel.forward()` with additive-progenitor logL | M | Core new module |
| Scoring function (seeded vs global agnostic) | M | Wraps panel fits; returns N_total posterior + Λ |
| Zero-region censored term as log-prob | P | Present in nb42c but not on a proper likelihood scale |

---

## What the architecture does NOT commit to yet

These are open design questions deliberately kept open until the audit is agreed:

- **Global feature detection.** Whether a standalone Dinosaur/Biosaur-style MS1 feature detector is built, or whether progenitor discovery from residuals inside seeded panels is sufficient to unify the two modes. Recommendation below.
- **Fragment ion integration timeline.** Whether MS2 enters the `Mass` module immediately or as a second pass after MS1-only quantification is proven.
- **Attribution algorithm.** View A (graph / soft edge weights) vs. View B (convolutional with square-wave-like kernel) for the initial implementation. Recommendation above (View A for MS1 Orbitrap MVP, View B as conceptual generalization that becomes load-bearing when fine structure or low-resolution data are modeled).
- **`p_z` as learned vs fixed-per-peptide.** Counter v2 treats `f_z` as constant over elution; whether we fit it or take it from a prior is open.
- **Center-weighting realization.** Hard (`requires_grad`-gated on non-home panels) vs soft (gradient multiplied by a smooth `w_center(Δm/z, Δt)`). Hard is simpler; soft is more forgiving near panel boundaries.
- **Panel width, overlap fraction, and tiling stride.** Tuning parameters; depend on observed interference density and m/z error distribution.
- **Progenitor discovery threshold.** The detection criterion for introducing new `Mass` candidates from residuals. Too loose → spurious masses; too tight → synthesis-artifact interferers missed. Needs empirical calibration on PROCAL residuals.

---

## Recommended MVP scope (for discussion, not committed)

**Near-term (weeks):**
1. Write the additive-progenitor joint NLL on paper (extends Counter v2 §6, §8). Includes the zero-region censored term on the same log-probability scale, and the center-weighted gradient convention for overlapping panels. Resolves the cens_weight-style issues of nb42c.
2. Implement `Mass`, `Panel`, `Panel.forward()`, and the overlap / home-panel logic as nn.Modules. Implement progenitor discovery from residuals.
3. Re-run PROCAL calibration inside the new architecture. One panel per PROCAL peptide; target `Mass` locked to PSM; discovered progenitors populate with synthesis-artifact / modification interferers. Target: alpha, d_mz, ν_I match nb42b within uncertainty; peak shapes converge without τ_R-at-bound artifacts; the number and identity of discovered interferer masses is itself a diagnostic output.
4. Add MS2 fragment channel to `Mass`. Re-run; produce combined MS1+MS2 N_total with Laplace credible intervals.

**Deferred (months):**
5. Global feature detection → panels → library-matching. This is the manuscript-scale vision. MVP should be designed to accept global feature input as a drop-in replacement for seeded progenitor lists, but global detection is not built in the MVP.

---

## Verification

This doc itself doesn't need end-to-end verification — it's a design document. The verification is user review: does this capture the architectural moving parts accurately, does the additive-progenitor reframing resolve the nb42c conceptual problems, is the panel-based data slicing the right abstraction, and does the MVP scope align with the near-term deadline pressure.

If this doc is approved, the next concrete artifact is the additive-progenitor joint NLL written out on paper (or in Markdown), which would then become the basis for the `Panel.forward()` implementation.
