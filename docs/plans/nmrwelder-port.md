# nmrwelder → constellation: Port Plan

**Scope.** Greenfield rewrite of `nmrwelder`'s 1D NMR processing pipeline into `constellation.nmr` (NMR-specific code) and `constellation.core.signal` (generic 1D-trace primitives). nmrwelder is reference reading material, not source. The standalone nmrwelder repo is not maintained going forward.

This plan is also a teaching artifact — the work doubles as practice in Python package construction (Damien is aware of and supports this framing). Sections that introduce a packaging idiom call it out and explain *why* it is idiomatic here, alongside the alternatives considered.

---

## Context

`nmrwelder` was scaffolded as a from-scratch PyTorch port of nmrglue, predating constellation's design. The 1D Bruker pipeline is **complete and tested** there: Bruker FID reader, apodization, zero-fill, FFT with GRPDLY correction, phase correction (with auto-ph0 from FID first point), polynomial/spline baseline correction, ppm referencing — all torch-tensor end-to-end, 99 tests passing across 7 test files.

Two design tensions surfaced once constellation existed:

1. **Architectural mismatch.** nmrwelder is a free-standing package: native Python dataclasses for FID containers, no Arrow schemas, no `RawReader` ABC, internally-numpy baseline / spline / reference, Python 3.13 (constellation pins `<3.13`).
2. **Generality leakage.** Baseline correction, smoothing, peak picking, polynomial calibration are *not* NMR-specific — they are generic 1D-trace primitives. Constellation's CLAUDE.md already reserves `core.signal` for exactly these operations. nmrwelder's `processing/baseline.py` belongs in `core.signal`, not `nmr`. Other modalities (HPLC-DAD chromatography, capillary electrophoresis) need the same primitives.

This plan also resolves a known regression in nmrwelder: the spline baseline overshoots in the NH region of protein spectra because it only has outer anchors. The fix (ArPLS or Whittaker smoother) lands directly in `core.signal` during the port — never in nmrwelder.

**Strategic decisions already made (as of 2026-05-08):**

- **1D-first.** Integrate the working 1D pipeline into constellation now. The nmrwelder TODO 2D HSQC machinery (LP, Hilbert, hypercomplex transpose, 2D FFT, NMRPipe writer, peak volume integration) gets built **inside constellation**, not finished in nmrwelder first. 2D code benefits from constellation's invariants from day one.
- **ArPLS/Whittaker baseline fix lands in `core.signal` after the move.** No further work in nmrwelder.
- **Branch workflow.** `todd471` continues through this port (the work is already in flight on it) and merges to main when the port completes, then retires. Sync regularly with `git fetch origin && git rebase origin/main` to keep the eventual merge small. Push at meaningful milestones (build-sequence step boundaries), not after every commit. **After the port:** default to short-lived feature branches off main (`feat/nmr-2d-hsqc`, `fix/<thing>`, etc.) with self-review-via-PR — catches hygiene mistakes that Damien's in-person content review doesn't. Avoid both long-lived parallel branches *and* direct commits to main. NMR development is single-author; other lab members may *use* the module and read explainers but won't write NMR code.

---

## Design principles for this port

1. **Constellation invariants are non-negotiable.** PyArrow tables in memory, Parquet on disk, no pandas inside the package, PyTorch for all numerical work, PascalCase classes, snake_case functions, `-1.0` "not observed" sentinel, schema metadata via PSI-MS CV accessions where they exist or `x.nmr.<key>` fallback. (Constellation's `core.io` already self-registers Arrow schemas; readers subclass `core.io.readers.RawReader`.)
2. **Generality split is decided per-function, not per-file.** A module name is not protective — `processing/baseline.py` lives in `core.signal` not `nmr.processing` because the *math* is not NMR-specific. A module that mixes generic and modality-specific operations gets split.
   - **The criterion is structural context, not line count.** A function gets a place in `core` if it carries non-trivial mathematical structure that another modality will plausibly use — *or* if it's already a named primitive elsewhere in `core` that would otherwise be duplicated. Polynomial phase correction qualifies (the `exp(iθ)` rotation along a polynomial profile from a pivot point is a real, reusable pattern; complex-spectrum modalities like FT-MS, beamforming, and optics all need it). A Gaussian apodization window qualifies (the Gaussian itself is already `core.stats.NormalDistribution`). A trivial wrapper around `torch.fft.fft` does *not* qualify — there's no additional structure. Line count alone is misleading.
   - **NMR-fluent wrappers compose the core primitives, not the other way around.** The wrapper holds the domain conventions (degrees vs radians, Hz vs time-domain σ, ppm vs Hz, line-broadening parametrization); the core primitive holds the math. Each wrapper is ~5–15 lines that translate domain parameters into the primitive's native arguments and call it.
3. **The plan is the teaching artifact.** Each architectural decision below names the packaging idiom it leans on (ABC, registry, schema, dataclass, etc.) and gives a one-line *why* — what the idiom buys versus an alternative implementation.
4. **No half-ports.** A port lands when (a) the rewrite is in `constellation/`, (b) the rewrite has tests, (c) the rewrite uses `core.io` schemas at the I/O boundary, (d) the corresponding nmrwelder code is no longer the canonical implementation. Partial states ("rewritten but uses native dicts") are not acceptable intermediate landing points; they undo invariant 3 (PyArrow as canonical handoff).
5. **Explainers are reviewed during port, not mechanically moved.** nmrwelder's explainers were Sonnet-assisted and have known failure modes (overconfident phrasing, missing edge cases, glossed math). Each one gets read and edited as it ports.

---

## Architecture: where each piece lands

### The generality split

| nmrwelder module | Lands in | Rationale |
|---|---|---|
| `io/bruker.py` | `constellation/nmr/io/bruker.py` | Bruker-specific binary FID + acqus parser. Becomes a `core.io.readers.RawReader` subclass. |
| `io/schemas.py` | folded into `constellation/nmr/io/__init__.py` (or removed) | Existing schemas re-emit `core.io` `Trace1D` with `x.nmr.*` metadata. nmrwelder's bespoke dataclass is replaced by Arrow. |
| `processing/apodization.py` | **Split four ways — see *Apodization split* below.** | Distribution shapes (Gaussian envelope, Lorentzian-derived exponential decay) consume `core.stats`; DSP windows (sine bell, future Hann/Hamming/etc.) live in new `core.signal.windows`; the multiply-trace-by-window operation is `core.signal.apodize`; NMR-conventional wrappers (lb/gb in Hz, dwell-time conversion) stay in `nmr`. |
| `processing/zerofill.py` | `constellation/nmr/processing/zerofill.py` | Zero-padding before FFT. Generic in principle, but driven entirely by NMR/spectroscopy use. Promotion to `core.signal` deferred until a second modality wants it. |
| `processing/fourier.py` | `constellation/nmr/processing/fourier.py` | FFT itself is a trivial wrapper around `torch.fft.fft` and would not earn `core.signal` status on its own. The module is GRPDLY-aware (Bruker digital filter group-delay correction) — that *is* NMR-specific structural context, so the whole module stays in `nmr`. |
| `processing/phase.py` | **Split:** `core.signal.phase.apply_polynomial_phase()` + thin `constellation/nmr/processing/phase.py` wrapper. | The `exp(iθ)` rotation along a polynomial profile pivoted at an index is real reusable structural context — same pattern serves FT-MS calibration, beamforming, radar pulse compression, optics wavefront sensing — none of which exist in constellation today, but the math is genuine. Core primitive supports arbitrary polynomial order; NMR wrapper exposes the conventional `phase_correct(ph0, ph1, unit='deg')` signature. `absorption()` / `magnitude()` helpers are dropped — `.real` and `.abs()` directly is fine. |
| `processing/baseline.py` | **`constellation/core/signal/baseline.py`** | Polynomial / spline baseline. Generic 1D-trace operation. ArPLS / Whittaker fix lands here at port time, not in nmrwelder. |
| `processing/reference.py` | **Split:** `find_reference()` → `core.signal.calibration.find_peak_in_window()`; `to_ppm()` → `core.signal.calibration.linear_axis_calibration()`; `SOLVENT_PPM` dict + the table-aware wrapper (renamed to `set_ppm_scale`) stay in `constellation/nmr/processing/reference.py`. | The two math primitives are generic: windowed argmax and affine x-axis transform. Same primitives serve HPLC retention-time alignment against internal standards, MS lock-mass calibration, and CE size-ladder calibration. The NMR layer keeps the standards database and the Hz-↔-ppm conventions. |
| `examples/pipeline_1d_*.py` | `constellation/examples/nmr/` (or removed) | Port to import from `constellation.nmr`; keep one as a smoke-test example. The diagnostic / temperature-overlay scripts can be archived or rewritten at user discretion. |
| `context/explainers/*.md` + `*.py` | **`docs/explainers/nmr/`** with figure scripts alongside; PNGs committed | Explainers are self-contained reading material. PNGs are referenced inline in `.md` and need to ship with the docs. Reviewed (not mechanically moved) during port. |
| `context/sessions/*.md` | NOT migrated — local-only sessions in constellation start fresh | Session notes are personal learning records; nmrwelder's history stays in nmrwelder. |
| `nmrglue/` (vendored reference) | Stays in nmrwelder; already in its `.gitignore` | External reference reading material; not in scope for the rewrite. |

### Apodization split

The four-tier breakdown of nmrwelder's `processing/apodization.py`:

| Lives in | Provides | Notes |
|---|---|---|
| `core.stats` *(already shipped)* | `NormalDistribution` (Gaussian), `StudentT(df=1)` (= Lorentzian, the FT-dual of exponential time-domain decay), `GaussianPeak`, plus future named `Lorentzian` alias (see open questions) | These are physically grounded distribution shapes. The Gaussian envelope and the FT-of-exponential-decay are the same parametric forms that show up in noise modeling, peak fitting, and other modalities. |
| `core.signal.windows` *(new submodule)* | `sine_bell(n, off, end, power)` ported from nmrwelder, plus thin wrappers over `torch.signal.windows.*` for `hann`, `hamming`, `blackman`, `tukey`, `kaiser` (only ship what nmrwelder uses today + obvious DSP standards; don't over-build). | DSP windows that are not probability distributions — they don't integrate to anything special, parametrized by windowing geometry rather than statistical moments. |
| `core.signal.apodize(trace, window)` *(new, top-level export)* | The `trace * window.to(trace.dtype)` operation as the canonical name across modalities. | One-liner that earns its place because the name standardizes the call site across `nmr`, future `chromatography`, and anywhere else apodization-style multiplication makes sense. |
| `nmr.processing.apodization` *(thin wrappers)* | `em(fid, lb, dwell)`, `gaussian(fid, lb, gb, dwell)`, `sine_bell(fid, off, end, power)` | Holds NMR domain conventions: lb / gb parametrized in Hz, time-domain σ derived from sweep width + dwell time, the FID + dwell signature. Each wrapper is ~10 lines: build the window from `core.stats` and/or `core.signal.windows`, call `core.signal.apodize`. |

The pedagogical payoff: the *same* `NormalDistribution` serves three roles — apodization window (time domain), peak-fit shape (frequency domain, via `GaussianPeak`), noise model (via `.log_prob`). Per architecture invariant 5 ("generic, nested abstractions exposing statistical similarities"), this unification is the design's whole point.

### `core.signal` build-out (new module)

`core.signal` is a scaffold-only slot in the import DAG today. This port creates it. Initial scope:

```
constellation/core/signal/
├── __init__.py        # top-level export: apodize(trace, window)
├── baseline.py        # polynomial, spline, ArPLS, Whittaker
├── calibration.py     # find_peak_in_window, linear_axis_calibration
├── windows.py         # sine_bell, hann, hamming, blackman, tukey, kaiser
├── phase.py           # apply_polynomial_phase
└── (smoothing.py, peak_picking.py — deferred until 2D port wants them)
```

`core.signal` may import `core.stats` (per the import DAG: signal sits between stats and structure) — relevant for smoothing / peak_picking when they land, not for the windows / phase / baseline / calibration primitives shipped in this port, which are leaf-style on `torch`. No imports from `chem`, `sequence`, `io`. Schema-stamped outputs (e.g. corrected `Trace1D`) go through `core.io` at the call site, not inside the algorithm. Domain modules (`nmr`, future `chromatography`) freely import from both `core.signal` and `core.stats`.

### Bruker reader: registering with `core.io`

This is the biggest packaging idiom in the port. nmrwelder's Bruker reader is a free function `read_bruker(path)` returning a dataclass. Constellation's idiom is the **`RawReader` ABC + suffix/modality registry** in `core.io.readers`.

```python
class BrukerReader(RawReader):
    suffixes = (".d/",)        # or whatever Bruker dir convention is
    modality = "nmr"
    def read(self, path: Path) -> Bundle: ...
```

**Why this is idiomatic:** the registry decouples *which file format is provided* from *how it is loaded*. A caller who has a path can ask `core.io.readers.find(path)` and get the right reader without importing `nmr` directly — important because `core.io` is a leaf and must not know about `nmr`. Self-registration on import means adding a new instrument format is a single new file, no central "register me" list to maintain.

**Alternative considered:** keep the bare function and add a thin shim. Rejected — it leaves the `core.io.readers.find()` mechanism inconsistent (some formats registered, some not), and it forces every downstream caller to know whether the source is Bruker before picking a reader.

---

## Build sequence

Each step is a self-contained change that lands `main`-ready (tests pass, invariants hold) before the next begins.

1. **Scaffold local session-notes infrastructure.**
   - Create `context/sessions/` and `context/explainers/_drafts/` at constellation root.
   - Add both to `.gitignore` (mirrors nmrwelder's pattern).
   - Add a session-note seed file `context/sessions/2026-MM-DD_session-01_port-kickoff.md` capturing this plan's resolution and the open question states.
   - **Idiom note:** `.gitignore`'d top-level dirs let personal records coexist with the package without polluting the published distribution. Standard alternative is a separate sibling repo; rejected because the notes need to live adjacent to the code being written.

2. **Build `core.signal/` skeleton.**
   - Create `constellation/core/signal/__init__.py` exporting `apodize` at the package root, plus modules `baseline.py`, `calibration.py`, `windows.py`, `phase.py`.
   - `baseline.py` — port nmrwelder's polynomial + spline implementations, **plus** ArPLS and Whittaker implementations that fix the NH-region overshoot.
   - `calibration.py` — `find_peak_in_window(trace, axis, expected_position, search_half_width)` and `linear_axis_calibration(axis, observed_position, target_position, scale)`. Lifted directly from nmrwelder's `find_reference` and `to_ppm` with the NMR-specific names generalized.
   - `windows.py` — `sine_bell(n, off, end, power)` extracted from nmrwelder's FID-multiplier signature into a bare window function (the multiplier becomes `core.signal.apodize`). Plus thin wrappers over `torch.signal.windows.hann/hamming/blackman/tukey/kaiser` for completeness. Don't over-build.
   - `phase.py` — `apply_polynomial_phase(spectrum, coefficients, pivot)`. Polynomial order is set by `len(coefficients)`; NMR's 0+1 case is `coefficients=[ph0_rad, ph1_rad]`. Coefficients in radians (unit conversion at the wrapper boundary).
   - `apodize(trace, window)` lives in `__init__.py` as the one-line multiply primitive.
   - Tests: `tests/test_signal_baseline.py` (four methods + ArPLS regression on protein-spectrum fixture), `tests/test_signal_calibration.py`, `tests/test_signal_windows.py` (window shapes + apodize semantics), `tests/test_signal_phase.py` (polynomial phase at orders 0, 1, 2 — order 0 is identity at pivot, order 1 matches NMR's hardcoded version, order 2 verifies the formula generalizes).
   - **Idiom note:** building this once now (vs putting it in `nmr` and "extracting later") matters because extraction-later requires re-doing the test surface. Test code reflects API choices, so testing at the wrong layer means the tests don't carry over.

3. **Port Bruker reader as `BrukerReader(RawReader)`.**
   - Create `constellation/nmr/io/bruker.py` with the class subclass + suffix/modality registration.
   - Output: a `core.io.Bundle` carrying a `Trace1D` (FID) plus a parameter table (Arrow), with `x.nmr.*` metadata keys for: spectrometer frequency, sweep width, GRPDLY, dimensionality, nucleus.
   - Tests: roundtrip a real Bruker dataset through the new reader; assert metadata fidelity vs nmrwelder's reader output.
   - **Idiom note:** `Bundle` is constellation's "multi-table return value" pattern — a reader returns a `Bundle` of named Arrow tables rather than a tuple of DataFrames or a custom dataclass. Caller does `bundle["fid"]`, `bundle["params"]`. Idiomatic because every reader returns the same shape, downstream code does not need format-specific unpacking.

4. **Port the NMR-specific processing wrappers.**
   - One commit per module: `apodization` → `zerofill` → `fourier` → `phase`. Each operates on `Trace1D` in / `Trace1D` out, torch-native end-to-end.
   - **Apodization wrappers** consume `core.stats` and `core.signal` per the four-tier split: `em` builds an exponential decay window directly (or via the to-be-added `Lorentzian` alias) and calls `core.signal.apodize`; `gaussian` builds the combined Lorentzian + Gaussian window using `core.stats.NormalDistribution` for the Gaussian factor and calls `core.signal.apodize`; `sine_bell` calls `core.signal.windows.sine_bell` and `core.signal.apodize`. The wrappers concentrate Hz-↔-time-domain unit conversion.
   - **Phase wrapper** is a ~10-line shim around `core.signal.phase.apply_polynomial_phase`: convert ph0 / ph1 to radians per `unit` arg, pack into a coefficient list, pass through. `absorption()` and `magnitude()` are dropped (callers use `.real` / `.abs()`).
   - **Reference wrapper** (`set_ppm_scale`) composes `core.signal.calibration.find_peak_in_window` + `linear_axis_calibration` + the NMR `SOLVENT_PPM` lookup; stamps `x.nmr.ref_hz`, `x.nmr.ref_ppm`, `x.nmr.reference_compound` into the Arrow metadata.
   - Re-run the existing nmrwelder test fixtures against the new implementations as regression checks (port the test data, not the test code; rewrite tests against the new API).

5. **Port reference / ppm calibration with the `core.signal.calibration` split.**
   - `core.signal.calibration.linear_offset_from_peak()` is the generic primitive.
   - `nmr.processing.reference.set_ppm_scale()` is the NMR wrapper that calls it, knows about TMS / DSS / water as standards.
   - Tests in both modules.

6. **Port the 1D pipeline example.**
   - `constellation/examples/nmr/pipeline_1d.py` — end-to-end: read Bruker → apodize → zerofill → FFT → phase → baseline (via `core.signal`) → reference → emit ppm-scaled `Trace1D`.
   - This is also a smoke test that the import DAG is intact (it pulls from both `core` and `nmr`).

7. **Explainer review + port.**
   - For each of nmrwelder's 9 explainers + figure scripts: read for accuracy (flag overconfident phrasing, missing edge cases, glossed math), edit, port figure script imports to `constellation`, regenerate PNGs, land in `docs/explainers/nmr/<topic>.{md,py,png}`.
   - **Idiom note:** PNGs are committed because explainers reference them inline (`![](apodization_em_effect.png)`); a reader cloning the docs subdirectory needs the images present. Alternatives — generate on doc build, or host externally — add infrastructure cost not justified at this scale.

8. **Update root and module CLAUDE.mds.**
   - `constellation.nmr` row in the module index moves from "scaffold" to "partial (1D pipeline shipped)".
   - `core.signal` row updates similarly.
   - Per-module `constellation/nmr/CLAUDE.md` created (mirrors `core/CLAUDE.md` and `massspec/CLAUDE.md` patterns), capturing 1D status, 2D TODOs, and reference back to nmrwelder for historical implementation notes.

---

## Test strategy

- **Unit tests per module** in `tests/test_<module>.py`, mirroring nmrwelder's test layout.
- **Regression fixtures.** Real Bruker dataset from nmrwelder's test fixtures (Lysin 24°C, `todd-20260219-2-8R/1`) imported into `tests/data/nmr/` (Git LFS or tracked binary depending on size — decide at port time). Pipeline output compared element-wise against nmrwelder's known-good output within numerical tolerance.
- **Integration test.** `examples/nmr/pipeline_1d.py` invoked under pytest as an end-to-end smoke test.
- **Import DAG.** `tests/test_imports.py` extended to cover `core.signal` and `nmr.processing.*`.

---

## Open questions (deferred / not committed by this plan)

- **Bruker reader Arrow metadata schema.** Concrete `x.nmr.*` key list: spectrometer frequency, sweep width, GRPDLY, nucleus, dimensionality are obvious. Pulse program, acquisition mode (DQD vs digital), receiver gain — useful but not covered yet. Decide at port time alongside the first reader test.
- **Test-fixture binary handling.** Git LFS, vendored, or generated? The Bruker fixture is small enough (~1–5 MB) to commit directly without LFS, but lab-wide practice going forward should be deliberate. Recommended default: vendor for this port; revisit before any multi-dataset or 2D fixture work. **Damien's call** on long-term policy.
- **`Lorentzian` as a `core.stats` named alias for `StudentT(df=1)`.** Mathematically equivalent; spectroscopists (NMR, MS, IR) will look for `Lorentzian` and not find it without the alias. Not a hard blocker — the apodization wrappers can use `StudentT(df=1)` or implement `exp(-π·lb·t)` directly — but the domain-fluent name matters for readability and pedagogy. **Damien's call** whether `core.stats` should grow domain-fluent aliases or stay statistically pure.
- **`Distribution.evaluate(normalized=False)` API.** Apodization windows want the unnormalized PDF shape (`exp(-x²/(2σ²))`, not `1/(σ√(2π)) · exp(...)`), since the window is being multiplied into a signal where overall scaling doesn't matter. `core.stats.Distribution` currently exposes `.log_prob` and `.cdf`; whether unnormalized evaluation is supported is not yet checked. If only `.log_prob` exists, the apodization wrapper computes the unnormalized form directly rather than going through the `Parametric` API. Decide at apodization-port time after reading `core.stats.parametric`.
- **`zerofill` promotion to `core.signal`.** Punt — promote when a second modality (FTIR? FT-MS?) wants it.
- **2D pipeline plan.** Out of scope for this plan; will get its own `nmrwelder-2d-port.md` once 1D lands. Notable: 2D HSQC is the user's first NMR analysis target after the 1D pipeline is in constellation, so this is not "nice to have" — it is the next plan.

---

## Verification (definition of done for this plan)

- 1D pipeline runs end-to-end inside `constellation` against the Lysin 24°C dataset; output matches nmrwelder's known-good output within tolerance.
- All seven nmrwelder processing modules have a constellation equivalent. Disposition:
  - **Whole in `nmr`** (3): `zerofill`, `fourier` (GRPDLY-aware), Bruker I/O.
  - **Whole in `core.signal`** (1): `baseline` + new ArPLS / Whittaker.
  - **Split between `core` and `nmr`** (3): `apodization` (four-tier — `core.stats` + `core.signal.windows` + `core.signal.apodize` + NMR wrappers), `phase` (`core.signal.phase.apply_polynomial_phase` + NMR wrapper), `reference` (two `core.signal.calibration` primitives + NMR `set_ppm_scale` wrapper + `SOLVENT_PPM` dict).
- `core.signal` exists with submodules `baseline`, `calibration`, `windows`, `phase`, and a top-level `apodize` export.
- ArPLS / Whittaker baseline fixes the NH-region overshoot regression on a protein-spectrum test fixture.
- `BrukerReader` is registered with `core.io.readers` and returns a `Bundle` of Arrow tables.
- `apply_polynomial_phase` verified at orders 0, 1, 2 — order 1 matches nmrwelder's `phase_correct` numerically on the existing test fixtures.
- All 9 nmrwelder explainers reviewed, edited, and ported to `docs/explainers/nmr/` with PNGs.
- `tests/test_imports.py` passes; full pytest suite green; ruff clean.
- `constellation/nmr/CLAUDE.md` exists and accurately describes shipped 1D state + 2D TODO list.
- Root CLAUDE.md module-status table updated (`nmr` partial, `core.signal` partial with shipped scope).

---

## Out of scope (this plan)

- 2D HSQC pipeline (LP, Hilbert, hypercomplex transpose, 2D FFT, NMRPipe writer, peak volume integration). Separate plan.
- DEEP picker (Li et al. 2021) integration as a 2D peak-picking backend. Separate plan once 2D lands.
- Solvent suppression post-processing (`sol_boxcar`, `sol_gaussian`). Deferred — current Lysin data uses `zgesgp` water suppression in the pulse sequence, no post-processing needed.
- High-level `pipeline.py` / `summarize.py` convenience wrappers. Deferred — example script is sufficient until external users appear.
- nmrwelder repository decommissioning (archive flag, README pointer to constellation). Decision deferred until the 1D port is complete and validated.
