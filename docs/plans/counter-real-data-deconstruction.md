# Counter — real-data deconstruction (Stage-0 design note)

**Status:** design note, not a spec. Captures a converged design dialogue about how
the Counter module (as of the `feat/counter-samegrid-discovery` branch, post-PR-78)
must be deconstructed and rewired to estimate ion counts on real acquisitions given
a peptide list. No code has been written against it yet. Treat every parameter value
and prior below as a *starting point the data should reshape*, not a fixed constant —
this is an ongoing research project, not a port of a finished method.

Companion docs: [`cartographer-counter-port.md`](cartographer-counter-port.md) (port
plan / gap audit), [`profile-vs-centroid-research.md`](profile-vs-centroid-research.md),
[`peak-boundary-refinement-research.md`](peak-boundary-refinement-research.md).
Provenance for the m/z model + units: the `counter-mz-model-provenance` memory
(trust nb41/nb42b; the draft technical reference is non-authoritative).

---

## 1. Scope and what was decided *against*

The goal is to adapt the **existing XIC-centric, peptide-centric** Counter pipeline
into a working real-data form — not to replace it. Until the data are searched, an
acquisition is just a large set of `(rt, m/z, intensity)`; a peptide list (from a
search) is the entry condition.

**Decided against:** an ID-free Stage-0 instrument calibrator that fits global
parameters from M+0/M+1/M+2 isotope trios in raw MS1 *before* any IDs. The conclusion
was that priors must be imposed *somewhere* regardless (within-scan isotope spacing
and RT-linkage covariance are each just different priors), and the trio route does not
clearly beat finding abundant, high-confidence calibrant peptides — whose very presence
*is* a prior. The calibrant-peptide / prior-bearing route is the one pursued here.

**Governing philosophy:** parameters enter as **soft priors, not hard constraints**.
The RT seed is already an informative prior on the elution-peak center, not a hard
extraction window (priors.py). Global instrument parameters get generous round-0 priors
and the data shape the posterior.

## 2. Reframing interference: the near-isobaric blend is the pervasive case

The dominant real-data problem is **not** two chromatographically/spectrally resolved
centroids competing for one channel (that is an assignment problem). It is the
**near-isobaric blend**: a *single* observed centroid that contains contributions from
two or more species whose m/z differ by less than the centroiding resolution. At a
10–20 ppm match window with true errors of ~1–2 ppm on high-res Orbitrap data, a *lot*
of statistically inconceivable candidates pass the window. The community keeps these
rather than discarding them, because an interfered (blended) ion is still informative —
better to model its shared contribution than to drop it.

The right model for this is the one Counter already has the bones of: an **additive**
latent count across candidates plus **soft, intensity-weighted attribution** of each
observed peak (the EM responsibility `γ_q = I_pred_q / Σ I_pred`). A single blended
centroid is fractionally attributed to every candidate that could produce it; the m/z
likelihood is the responsibility-weighted mixture marginal (`logsumexp_q log γ_q + log
t_q`). This is the principled limit when candidate kernels coincide, and it is exactly
what `attribution.py` computes.

### The one structural gap: per-progenitor m/z centers

`panel_log_prob` (panel.py:109) currently computes the m/z center **once** from
`obs.channel_mz` and shares it across all progenitors — i.e. every candidate is scored
as if it sits at the *same* m/z. That is the PR-78 **same-grid** model (discover.py
clones the target's `channel_mz`), correct only for *exactly* co-isobaric interferers.

The near-isobaric blend requires **per-progenitor centers**: candidate `q`'s Student-t
centered at *its own* theoretical m/z relative to the observed peak, so the blend is
resolved by the candidates' distinct mass defects. The same-grid case is the `Δ=0`
special case.

Crucially, for multi-progenitor panels these centers are **not a static input** — they
are re-resolved on **each γ (E-step) update**, alongside the responsibilities, as the
partial attribution is recomputed each cycle. So the right shape is a **general scorer
over arbitrary `(observed, expected)` arrays** (observed m/z error + intensity vs an
expected center/scale/fraction set), with two thin wrappers:

- the **static peptide-default** case (expected = the theoretical isotope envelope), and
- the **per-cycle re-attribution** case (expected re-derived each γ update for the
  candidate set currently in the panel).

This keeps the core m/z + intensity math generic (consistent with the project's
generic-abstractions invariant) and isolates the changing-context logic to the wrapper.

Each candidate's center is **`global_offset + candidate_theoretical_Δ`**: the global m/z
offset (D5) and the per-isotope `d_mz_k` stay on `GlobalCalibration` as the *single shared
calibration anchor*, and only the candidate's theoretical position varies. Nothing
per-candidate drifts the offset — so admitting more candidates into a blend never
multiplies the free calibration degrees of freedom.

## 3. Architectural spine: a precursor-level candidate index

Your MSFragger/Sage analogy lands at the precursor level. Half of it already exists.

- **Observed side (exists):** `quant/peak_index.py` re-lays `MS_PEAK_TABLE` into a
  mass-sorted, isolation-window-partitioned dataset where a query m/z resolves to a
  contiguous `(rt, scan, intensity)` slice — "scale with query count rather than
  n_scans." Its docstring already anticipates a spectrum-centric search.
- **Theoretical side (missing — the new piece):** index the peptide list's theoretical
  precursor-isotope m/z (peptide × charge × isotope) into m/z bins → *"if a given m/z is
  observed, which candidate progenitors could it be?"* This is **data-independent given
  the peptide list** (independent of the raw peaks).

Composed, the two are a merge-join in m/z space — the open-search architecture at the
precursor level. Two payoffs:

1. **Speed:** extraction/matching scales with query count, not scans, and the candidate
   set per observed peak is a bin lookup.
2. **Channel-overlap components fall out for free:** two progenitors collide iff their
   indexed bins overlap within tolerance. The connected components of the collision
   graph are the natural *fit units* (see §4, L8).

**Precedent + contrast:** `search/collision.py` already does graph → connected-components
clustering (group by window, pair by RT, count shared ions, take components) — but it
*drops* collision losers (keeps the best-scoring peptide per cluster). Counter wants the
**same structural pattern with inverse semantics: keep every co-isobaric candidate and
soft-attribute, never drop.**

The risk of admitting many weak candidates is small (soft γ down-weights them), and it
shrinks further after round-0 calibration tightens the effective tolerance.

## 4. Decisions log

| # | Decision |
|---|---|
| D1 | **L8 = soft-γ attribution spanning panels**, not hard exclusive assignment. "Never assigned to two progenitors" means *never double-**counted***; a blended ion legitimately maps to one-or-more progenitors with fractional γ. |
| D2 | **Parallelize by channel-overlap connected components**, not per-target. Singletons (the vast majority) stay embarrassingly parallel; only overlapping clusters fit jointly as one panel. (collision.py pattern, inverse semantics.) Optimal parallelization of large/dense components is an open sub-problem. |
| D3 | **Unify onto a prior-bearing panel path.** Retire `estimate_panel`'s MAP-only restriction; give the panel likelihood + VB a prior hook so L6 biasing and credible intervals exist on the path that actually handles interference. |
| D4 | **α(z): linear model with soft priors.** `α₁ ~ N(12, 2)` (Hsu et al.'s +1-ion value; charge states are near-perfect linear scalers of it), `α₀ ~ N(0, 1)` (the *soft* form of linear-origin — α∝z physics, but let the data nudge the intercept). |
| D5 | **Global m/z offset `~ N(0, 7)` ppm**, shared by all ions in all charge states (the single strongest round-0 constraint) and the **single calibration anchor** — per-candidate centers are `global_offset + candidate_theoretical_Δ` (D6), never a per-candidate free offset. Generous priors on the other globals likewise. |
| D6 | **Per-progenitor m/z centers, re-resolved on each γ update** for multi-progenitor panels (§2). The core scorer generalizes to arbitrary `(observed, expected)` arrays; thin wrappers cover the static peptide-default case and the per-cycle re-attribution context. Generalizes the same-grid model to near-isobaric blends. |
| D7 | **Persist an ion ↔ progenitor ↔ γ map** as tracked metadata, recorded **per iteration** so attribution migration across rounds is observable. Foundation for "what's left after targets are searched." |
| D8 | **`calibration_to_table` round-trips all four shape-prior slots** (currently drops `log_tau_l` + `logit_eta`). |
| D9 | **Benchmarking knob:** an optional multi-progenitor-panel mode that **excludes blended cells** — any `(S,C)` no progenitor owns at `γ ≳ 0.99` — from the loss while still *inferring* their contribution. Lets us A/B whether letting interfered ions contribute to the fit helps or hurts vs exclude-but-still-infer. Reuses the L4 cell-level exclude-state (§6 step 4). |

## 5. The bootstrap ordering (round 0 → convergence)

The key subtlety: the shared-`N(t)`/`μ` joint pull across isotopes is a *steady-state*
property of the likelihood (one `n_flux` scored against every channel via `f_z·p_k`,
model.py:223-230). It only exists **after** a HyperEMG fit. Round 0 has no `N(t)` — each
ion is whatever was independently extracted from the XIC, and the per-ion m/z and
intensity likelihoods are **independent across channels**. That independence is what
makes **concurrence** the bootstrap signal: M+0/M+1/M+2 of one charge, each divided by
its expected fraction `p_k`, should give *consistent* per-channel N; agreement is
evidence, a one-off match is not.

The current seed (`seed_peak_from_observation`, orchestrate.py:72-87) pools channels by
summing raw intensity and dividing by summed expected weight — the *naive* version, with
no consistency check and no robustness to a spurious bright channel. The fix is to gate
on per-ion logL first:

0. **Prior-seed** the global model (D4/D5; instrument-prior α(z), `N(0,7)` offset).
1. **Score each independently-extracted ion's per-ion logL** (m/z-error fit + intensity-
   implied-N consistency) — the L3 per-cell decomposition, used *before* the joint fit.
2. **Separate the concurrent, well-fit ions from the rest.** This is also the answer to
   "N-weighted μ blends interferers": the interferer ion, scored under the target's model,
   scores *low* (wrong m/z error / wrong isotope ratio) and is held out before it can pull μ.
3. **Form the first `N(t)` / `μ`** from the survivors, **N-weighted** (not raw-intensity-
   weighted — intensity-weighting biases μ toward short-injection-time apex scans).
4. **Fit the joint HyperEMG** (DE broad + polish); now the shared-`N(t)` likelihood ties
   μ across the surviving channels.
5. **Iterate (L7):** select high-scoring progenitors → re-fit globals → re-derive XIC
   under the refined m/z model → re-fit → repeat to convergence.

**Round-0 separation is genuinely tricky and should not be over-engineered up front.**
The mature mechanism is likely some form of **outlier / mixture modeling** (a well-fit
component vs an interfered/spurious component) that will need tuning — *not* a hard logL
threshold. But round 0 does not need precision: empirically the well-fitting ions form a
**strongly biased, separable distribution**, so even the soft round-0 priors (D4/D5) pick
them out well enough to *jump-start* the optimization. The selection criterion is
**fit-quality-per-unit-evidence** (high `N/N_total` share **and** good m/z fit), never raw
logL (abundance-confounded — bright ions always win).

Round-0 calibration is a chicken-and-egg (you need calibration to separate, you separate
to calibrate); it resolves because the *global* m/z offset is shared across all ions and
charges, so the most obvious high-SNR, many-concurrent-ion peptides dominate even a crude
first separation and pin the first global calibration, which then tightens. That
tightening — and the maturing of the separation toward a tuned mixture model — *is* the
L7 loop's job.

## 6. Build order (cheap → invasive)

**Status:** steps 0–4 **landed** (branch `feat/counter-realdata-estimation`). Two
implementation refinements vs the original sketch, both in the raw-preservation spirit:
(1) per-ion N is a `CounterObservation.recovered_count(calibration)` **method**, not a stored
field — the observation stays raw; the count is a derived projection. (2) Loss exclusion is a
scorer/`Panel` argument (`exclude_mask` + `gamma_loss_threshold`), not a third state baked into
`obs.mask` — excluded cells leave both the observed and censored sets (so they are *inferred,
not scored*), which avoids the "excluded leaks into the censored tail" N-bias by construction.
Steps 5–7 remain. `panel_log_prob` is byte-identical when no exclusion is set (the full counter
suite — incl. the DE/vmap fit path — stays green).

0. ✓ **Fixed the stale doc:** the counter CLI + batch driver ship (`massspec counter
   {calibrate, estimate}`, cli.py); `massspec/CLAUDE.md` updated.
1. ✓ **Per-ion N at the observe layer:** `CounterObservation.recovered_count(calibration)`
   = `I·τ/α` on observed cells (0 on non-detections), via the now-live `iit.accumulated_count`.
   The `sum(N_guess)` / N-weighted-μ array L6 needs. No new physics.
2. ✓ **Per-cell decomposition (L3):** `panel_cell_log_prob` → `PanelCellTerms` (the `(S,C)`
   terms + `(Q,S,C)` `component_mz`/`γ`/`n_count`/`i_pred` + the effective masks) with no
   trailing `.sum()`; `panel_log_prob` is the masked sum (byte-identical default). The m/z
   term stays the **mixture-marginal** (the per-progenitor `component_mz` is also exposed for
   callers that want the bare component). **Per-progenitor centers** (D6) land here as a
   general `(observed, expected)` scorer: `mz_center_ppm` gains `reference_mz`, each candidate
   scored at `global_offset + (channel_mz − reference_mz) + d_mz_k/z`; multi-progenitor panels
   re-resolve centers + responsibilities together each call (E-step). `PanelCellTerms` is a
   NamedTuple, so the decomposition flows through the DE/vmap path unchanged.
3. ✓ **Persist γ (D7):** a stable `(scan, mz_observed)` physical-peak key threaded from the
   input trace through `observe.py` onto `CounterObservation` (`scan` axis + `(S,C) source_mz`);
   `emit.panel_attribution_table` re-derives γ from a fitted panel and emits the sparse
   `(scan, mz_observed, progenitor_index, is_target, responsibility, n_obs_count, iteration)`
   rows (`COUNTER_PEAK_ATTRIBUTION_TABLE`); `CounterResult` gains an optional `peak_attribution`
   slot. Observed cells **no progenitor owns** (≈0 predicted flux — the high-value
   unexplained/interferer peak) surface as `progenitor_index = -1` **residual rows**
   (responsibility = the unmodeled share), so the anti-join never loses a raw peak. The key is
   `(scan, mz_observed)` — NOT a per-`(target,scan,ion)` trace-row index — precisely so the SAME
   physical peak extracted under different targets reconciles: "what's left" = anti-join the full
   peak set on the owned (`progenitor_index ≥ 0`) `(scan, mz_observed)` keys, and a peak two
   panels both claim shares one key. Component co-fits stamp each member's real `target_id` via
   `panel_attribution_table(progenitor_target_ids=…)`. *Follow-up:* the CLI `estimate --emit-
   attribution` path is wired; still to do — back-match discovered interferers to peptide identity
   (today `progenitor_index` is the panel-local key for an anonymous same-grid clone).
4. ✓ **Cell-level loss exclusion (L4 filter + D9 benchmark knob):** `panel_log_prob` /
   `Panel` accept `exclude_mask` (per-`(S,C)`) + `gamma_loss_threshold`. Excluded cells leave
   **both** the observed and censored sets — *inferred, not scored* — so they cannot leak into
   the censored tail and bias N down. Two uses on one mechanism: (a) the **bootstrap
   separation** (§5) holds out spurious/poorly-fit ions (selection wiring is step 5); (b) the
   **benchmark mode** (D9) drops well-modeled-but-*blended* cells (`max_q γ_q < threshold`) to
   A/B contribute-vs-exclude. The γ-threshold touches only observed cells (non-detections keep
   their censored contribution).
5. **L6 priors on the panel path (D3):** extend `make_log_prior` with `N_total →
   log(sum(N_guess))` and an N-weighted-μ center; give the panel likelihood/VB the prior
   hook. Switch the seed centroid weight from `i_tot` to step-1's per-ion count.
6. **Precursor-candidate index + channel-overlap components (§3, D2):** build the
   theoretical m/z → candidate index; cluster targets into connected components; fit each
   component as one shared panel (`Panel.add_progenitor` already supports multi-progenitor).
7. **L7 outer driver + cross-panel reconciliation (D1):** a new top-level driver holding the
   raw peak source that selects high-scoring progenitors, re-fits globals via
   `StagedCalibration`, re-derives/re-extracts XIC under the refined m/z, and loops to a
   Δ-global-param convergence with an iteration cap. Most invasive (breaks the per-target
   parallel worker model — re-broadcast a fresh frozen calibration each round). Depends on 1–6.

Plus **D8** (`calibration_to_table` shape-prior round-trip) wherever calibration persistence
is touched.

## 7. Dragons (the ones that survive the design)

- **α(z) absolute scale is irreducible.** Gain is only weakly identified from shot-noise
  (excluded from `StagedCalibration`'s default globals), so on real data it is essentially
  the prior (D4). Per-ion `N = I·τ/α` is a trustworthy *relative* weight for L4/L6 ranking;
  treated as an *absolute* count it is only as good as the gain prior. Cross-run transfer
  can break silently if AGC/source tuning changed.
- **Round-0 calibration chicken-and-egg** (§5) — mitigated, not eliminated, by the shared
  global offset + loose-then-tight filtering.
- **Dense-grid one-peak-per-cell limit.** The `(S,C)` observation holds one value per
  `(scan, channel)`; near-isobaric candidates are expressed via *separate progenitors with
  per-progenitor centers* (D6), not multiple peaks in one cell. The `(scan, mz_observed)`
  attribution side-table (step 3) is what preserves the raw contenders for the index.
- **Per-ion logL is abundance-confounded** — selection must normalize (N-share + m/z fit),
  never raw magnitude.
- **L7 convergence discipline.** The loop re-fits global params on fixed observations
  (no re-extraction — see §9), so the observed-ion set is *stable* across rounds; but a
  refined offset still moves N, so converge on Δ-global-param with a hard iteration cap, never
  a naive fixed-point on N.
- **Component fit cost/memory** for large dense overlap clusters is the open cost of D2 —
  a mandatory size cap → singleton fallback keeps the pool from stalling on one mega-panel.

## 8. Open questions

- Optimal parallelization / memory strategy for large or dense overlap components (D2).
- How far to lift `max_candidates` (currently 1; gated on parsimony/VB-on-panel machinery)
  once per-progenitor centers + the index make multi-candidate blends routine.
- Profile vs centroid for the m/z mixture (multi-point kernel) — out of scope here; see
  [`profile-vs-centroid-research.md`](profile-vs-centroid-research.md).
- Whether the per-iteration γ map (D7) should also drive a convergence diagnostic
  (attribution stability across rounds) in addition to Δ-global-param.

## 9. Steps 5–7 — detailed sub-PR plan (decisions locked 2026-06-18)

Decisions: **step 5** ships posterior-MAP-with-prior (VB-on-panel deferred to a later opt-in
`inference='vb'`); **step 7** round-0 uses explicit `--calibrants` first (auto-selection a
separate PR); and — the load-bearing simplification, per the PI — **step 7 does NOT
re-extract.** The candidate↔peak *linkage* is fixed by the index binning; a refined global m/z
offset enters at *scoring* time (`mz_center_ppm` already consumes `cal.mz_offset_ppm`), not
extraction. So the loop re-fits parameters on the fixed, wide-extracted observations and
`MS_PEAK_TABLE` never enters the loop contract. **Constraint:** the round-0 extraction window
+ the index bins (PR-D) must carry enough m/z margin (`all_in_window`, tolerance ≫ true error
+ plausible offset) that an offset refinement never moves a peak across a candidate boundary.

### Ship order (10 sub-PRs, 3 phases)

**Status:** PR-A + PR-B **shipped** (PR #80, "step 5 foundations"); PR-C + PR-D **shipped**
(PR #81 — calibration 4-slot round-trip + theoretical candidate index); PR-E **shipped**
(channel-overlap connected components, PR #82); PR-G **shipped** (component co-fit +
μ-anchoring, PR #83); PR-H **shipped** (CLI component worker-model + size cap; attribution
emission + fitted-param return deferred to PR-H2). A key finding (2026-06-18)
re-scoped step 5: a **post-DE gradient polish to apply the prior to the panel POINT estimate
is unsafe** on the same-grid-blend surface — Adam drifts off DE's basin along the flat
target↔interferer N-split direction (a 2× over-count in testing), unbounded L-BFGS escapes to a
NaN η/τ region, and bounded L-BFGS collapses the target to the N=1 bound. So **the "MAP-with-prior
via polish" path (decision-1 approach A) does not work**; the prior-bearing panel POINT estimate
moves to the **VB-on-panel** PR (it optimizes the ELBO globally with the η-taming priors already in
place — the right tool, and the reason the single-progenitor path is VB). PR-B's `N_total` term is
therefore wired into the **single-progenitor `estimate_n` VB path** (opt-in `n_total_prior=False`,
the safe home: no blend to contaminate the seed center, no post-hoc polish). PR-F below is
superseded by VB-on-panel.

**Phase 1 — foundations (independent, low-risk, parallelizable):**
- **PR-A** — `seed_peak_from_observation` weights the μ-centroid by `recovered_count` (not raw
  intensity); the N-weighted-μ fix in isolation. Improves every caller (`estimate_n` +
  `StagedCalibration`). Tests: constant-IIT no-regression + varying-IIT divergence.
- **PR-B** — `make_log_prior` gains an `N_total → log(sum(N_guess))` Gaussian + an `n_total_key`
  arg (parameterized like `mu_key`, so a panel caller passes `progenitors.{i}.peak.log_N_total`).
  Pure additive; returns `None` when inactive. The bare-key-vs-panel-namespace mismatch is the
  #1 silent-no-op dragon — test the namespaced key resolves.
- ✓ **PR-C** — **D8**: `calibration_to_table` round-trips all four shape-prior slots (`log_tau_l`
  + `logit_eta` were dropped); schema v1→v2 with a `.get` back-compat read (v1 parquet still loads).
- ✓ **PR-D** — `counter/candidates.py` `TheoreticalCandidateIndex`: peptide list → mass-sorted
  `(target_id, charge, isotope, m/z)` via `peptide_envelope` (same `mode='binned'`/`n_isotopes`
  the fit uses — verified index m/z == `Progenitor.for_peptide` channel grid). `candidates(mz,
  tolerance_ppm)` binary-search query; tolerance ≫ true-error + offset → linkage stable under
  calibration refinement. Precursor-scoped but not precursor-*only* (MS2 analogue later). The
  channel-overlap connected components (PR-E) build on it next.

**Phase 2 — step 5 deliverable + step 6 components:**
- ✓ **PR-E** — `channel_overlap_components(index, collide_ppm)`: union-find over collision edges
  from a mass-sorted sweep → a partition of targets (singletons stay size-1 frozensets; sorted by
  min target_id for determinism). `search/collision.py`'s union-find core was **copied** (not
  imported — avoids `counter`→`search` coupling) with **keep-and-attribute** semantics. `collide_ppm`
  static; the calibration-aware loose-then-tight widening is a future step-7 opt-in.
- **PR-F** — ~~posterior-MAP gradient polish on the panel path~~ **SUPERSEDED** (see Status above):
  a post-DE polish is unsafe on the same-grid-blend surface (Adam drifts, unbounded L-BFGS NaNs,
  bounded L-BFGS collapses). The panel prior-bearing POINT estimate moves to VB-on-panel. The
  `laplace_cov` already accepts a `log_prior`; wiring it stays a clean follow-up once a *consistent*
  posterior mode exists (i.e. under VB). Step-5 foundations (PR-A + PR-B) shipped instead.
- ✓ **PR-G** — `counter/component.py` `estimate_component(members, obs, rt_priors_ms=…)`: co-fits a
  component as one `Panel`, reporting each member's N + Laplace CI. Members keep their own
  `channel_mz`; #79's per-progenitor centers score each at its mass defect on `members[0]`'s grid —
  **zero new likelihood code**. No union grid (off-reference channels under-scored — v1 limit). A
  1-member component matches `estimate_panel` (no discovery loop). **Added:** same-grid co-eluting
  members are *exchangeable* (one would absorb the whole blend), so each member's `μ` is anchored to
  a `± mu_window_ms` **search bound** around its PSM RT (not the unsafe prior-polish); `fit_panel`
  gained a `bounds` override for it. Verified: a 2e5/1e5 same-grid blend recovers both (≈2.5%/5%).
- ✓ **PR-H (worker-model + size cap)** — CLI `_cmd_counter_estimate` builds **co-fit units** in the
  parent (`_counter_cofit_units`: candidate index → m/z-overlap components → RT co-elution refinement
  → size cap) and `pool.map`s `_counter_worker_unit` over **units**: a singleton routes to today's
  per-target discovery worker (unchanged); a multi-member unit is one joint `estimate_component`
  panel. New args `--collide-ppm`, `--rt-overlap-s` (default `--rt-window`), `--max-component-size`
  (oversized → singleton fallback). **RT-refinement is an *efficiency* split, not correctness** — the
  N(t)-weighted soft γ + μ-anchoring separate RT-different members within one window fine; we just
  don't waste a panel on non-overlapping RT. The component obs is the reference member's
  pre-extracted `all_in_window` trace over its `rt_window` (no re-extraction; edge members partial —
  a v1 limit). **Codex review fixes (correctness):** (i) `restrict_to_reference_star` re-cliques each
  transitive m/z unit into reference *stars* — a member only co-fits when it DIRECTLY overlaps the
  reference (else its peaks were never extracted onto the reference grid → silent ~0 N); transitive
  members split off. (ii) `--rt-overlap-s` is clamped to `--rt-window` (a unit can't span beyond the
  obs window) + a defensive μ-bound guard in `estimate_component` (skip narrowing for a prior outside
  the obs window — no `lo>hi` inversion crash). (iii) an empty reference grid falls back to per-member
  singleton fits, not a blanket `no_signal`.
- ✓ **PR-H2 (attribution emission):** `counter estimate --emit-attribution` writes the sparse
  ion→progenitor soft-attribution map (`COUNTER_PEAK_ATTRIBUTION_TABLE`, the "what's left" /
  interference bookkeeping) alongside the count table. The worker contract is now uniform —
  `(records, attribution)` per unit; the singleton path has its panel in hand, the component path
  gets it via `estimate_component(return_panel=True)`; the parent concatenates + saves it in the
  `CounterResult` bundle. Opt-in (default off → byte-identical count output). The fitted-`parameters_dict`
  return for step 7's `StagedCalibration` rebuild stays deferred to step 7 (its needs are clearer there).
- ✓ **Hardening (follow-up 1 of 3):** `collide_ppm` is now auto-clamped to the trace's **recorded XIC
  extraction tolerance** — `chromatogram extract` stamps `x.massspec.extraction_tolerance` into the
  trace schema metadata; `counter estimate` reads it (overridable via `--extraction-tolerance-ppm`)
  and clamps + warns, since a member beyond the extraction tolerance has no signal on the reference
  grid (silent under-scoring otherwise). **Remaining hardening:** (2) best-signal reference vs
  min-`target_id` — minor (the empty-reference singleton-fallback + star-restriction cover the
  correctness case; a faint-but-present reference's theoretical grid is still valid); (3) the **union
  grid** — the substantial fidelity fix that *recovers* the (transitive / off-reference) blends the
  reference-star restriction conservatively drops, by merging members' own traces onto a superset
  grid with shared co-isobaric cells (an architectural Panel change — the next major effort, not a
  quick hardening).

**Phase 3 — step 7 (no re-extraction):**
- **PR-I** — `counter/run.py` + `massspec counter run`, calibrant-anchored: loop over the explicit
  calibrant set = {`StagedCalibration` re-fit globals on the calibrant panels} ↔ {re-broadcast a
  **fresh frozen** calibration parquet} ↔ {re-fit calibrant panels} until Δ-global-param < tol (+
  hard `max_rounds`). Then a single final parallel all-targets estimate (PR-H) under the converged
  calibration. **No trace re-extraction.** Re-broadcast keeps the live mutated calibration off the
  process boundary (#79). Convergence on Δ-global-param, never a fixed-point on N.
- **PR-J** — `counter/select.py` auto-selection of high-scoring progenitors: rank by N-share
  (`N_target/N_panel`) **and** a normalized m/z-fit score (median `|mz_error|/σ` from
  `panel_cell_log_prob`), the abundance-de-confounded criterion — **never raw logL**. Selection
  scalars extracted under `@torch.no_grad` (never in the DE objective). Cross-panel γ reconciliation
  = Arrow `group_by` on the attribution `(scan, mz_observed)` key (stable across targets, unlike a
  trace-row index). Default when `--calibrants` omitted. Riskiest /
  untuned — **hand-labeled clean/blend/interfered fixtures first** (per
  [[feedback-synthetic-test-fixtures]]); the test must include a bright *interfered* peptide that
  raw-logL would wrongly select, asserting it is rejected.

### Cross-cutting
- **Fixture strategy** (per the synthetic-fixture practice): hand-curate clean/blend/interfered
  cases *before* the PR-J selector, not after.
- **Identity back-match:** a component interferer's panel-local `progenitor_index` → real peptide
  identity is a step-6/7 follow-up (the attribution table should carry enough to resolve it later).
