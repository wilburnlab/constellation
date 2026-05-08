# Plan: Integrate Agilent HPLC-DAD and Fragment/ProteoAnalyzer into Constellation

## Context

Two reverse-engineering efforts in adjacent repos have produced working readers for instruments that are heavily used in the lab and that complement the omics-scale modalities Constellation is being built for:

- **`~/projects/hplc_analysis/dx_reader.py`** — Agilent OpenLab `.dx` (HPLC-DAD). OPC zip with a 4 KB-header `.UV` for the 2D DAD matrix and 6 KB-header `.CH`/`.IT` for chromatograms and irregular telemetry. The `.CH` decode is bit-exact against the OpenLab CSV export; the `.UV` matrix correlates ≥0.99 with each `.CH` channel at the predicted wavelength column.
- **`~/projects/analyzer_analysis/read_raw.py`** — Agilent Fragment Analyzer / ProteoAnalyzer `.raw` (capillary electrophoresis). 1992 B big-endian header, 12-uint16 capillary-position table at 0x3E8, then 1-Hz frames of 751 × uint16 BE pixels. Per-capillary trace = 5-pixel sum centred on `cap_pos[k] + 4`. Validated against the 16-peak DNF-473-55 ladder.

Both are used as companion instruments to omics workflows (FA pre/post Nanopore library prep; HPLC-DAD for purity/quantitation upstream of MS or sequencing). They are "low-res" relative to MS or POD5 but produce clean, well-structured data, and Constellation's design principles (Arrow in / Parquet on disk / Torch for math; physically grounded statistical models; generic abstractions across modalities) apply directly. Bringing them in now — *before* the Cartographer port begins — has a useful side effect: the foundations they need (`core.io.readers`, `core.io.schemas`, `core.io.bundle`, `core.stats` peak shapes, `core.optim` DE) are the *exact same* foundations the MS port needs. So this round bootstraps the rest of the platform.

This is a **clean rewrite, not a vendoring**: the two prototype scripts are reference implementations to be re-expressed in Constellation idiom (Arrow tables, `RawReader` subclass, `Bundle` composition, schema metadata, lazy decoders). No code is copied verbatim.

Coding work will land across several PRs — this plan defines the scope and ordering, not a single-session sprint.

---

## Architectural decisions (confirmed in clarification round)

1. **Two sibling domain modules**: `constellation.chromatography` (HPLC-DAD; extensible to LC-MS later) and `constellation.electrophoresis` (FA, PA; extensible to other CE platforms). Mirrors the existing `massspec`/`sequencing`/`codon`/`structure`/`nmr` cadence. Shared math goes into core, never cross-imported between domains.

2. **`core.signal` as a new core submodule** (deliberated below).

3. **ProteoAnalyzer is pinned for now.** The user has a 12-capillary PA, so the same `(N_CAPILLARIES, FRAME_BYTES)` constants will apply when we get there. The FA reader will expose those constants as a `RunGeometry` dataclass so PA support is later set-the-numbers, not a rewrite. PA test fixtures will be added when the user provides a sample run.

4. **DAD 2D matrix on disk**: one Arrow row per spectrum, with `spectrum: fixed_size_list<float64>[N_wavelengths]`. Wavelength axis lives in schema metadata. Tensor reconstruction routes through Arrow's zero-copy buffer view → `torch.frombuffer` (no numpy intermediate held as a working type). Avoids the per-row schema explosion of the wide layout and the 226× row blow-up of the tall layout.

5. **Torch-first numerics; numpy only at I/O boundaries.** Per Principle 3 (PyTorch for numerical work) and the user's instruction to minimise wasted np ↔ torch round-trips: `core.signal`, `core.stats`, and `core.optim` operate on `torch.Tensor` end-to-end. The only places numpy is permitted:
   - **Big-endian byte decode in FA `.raw`** — torch has no native big-endian dtype; `np.frombuffer(buf, ">u2")` is the pragmatic decoder, immediately handed off via `torch.from_numpy(arr.copy())` (one boundary-crossing per file).
   - **Arrow ↔ torch bridge** — pyarrow doesn't expose torch directly; `pa.Array.to_numpy(zero_copy_only=True)` + `torch.from_numpy(...)` is the documented zero-copy route. Both legs share memory; nothing is recomputed.
   - Little-endian dtypes (everything in HPLC `.dx`: `<f8` for `.CH`/`.IT`/`.UV` payloads, `<u4` for the per-block `time_ms`) decode straight through `torch.frombuffer` without numpy at all.

   Module entry points in `core.signal` accept `torch.Tensor | np.ndarray` and coerce once via `torch.as_tensor(...)`; everything downstream is torch. Return type is always `torch.Tensor`. Type annotations make this explicit. No numpy variables held across function boundaries inside core.

### Deliberation: `core.signal` vs. fold into `core.stats`

The user's instinct — that signal processing is "more art than science, and while there often is a formal Bayesian interpretation to the operations, the priors are imprecisely specified" — is the right framing, and it actually argues *against* folding into `core.stats`, not for it.

**The split that honors the philosophy:**
- `core.stats` is for **parametric forms with well-defined likelihoods** that can be fit via NLL minimisation through `core.optim` (the existing scaffold's stated purpose). Gaussian / EMG / HyperEMG / Voigt peak shapes; Student-t error model; KLD and spectral-angle losses. Every member has a `.log_prob()` and a `.fit(data)`; the priors, when they exist, are explicit.
- `core.signal` is for **transforms of observed traces** where the operation is heuristic-with-a-Bayesian-shadow-but-imprecise-priors: AsLS / arPLS / SNIP baselines, Savitzky–Golay smoothing, prominence-based peak picking, CWT detection, polynomial calibration curves, SNR estimators. None of these have a `.log_prob()` we'd actually trust as a likelihood; pretending they do by hosting them under `Parametric` would be architectural dishonesty. They are reused identically by mass spec chromatograms, HPLC, CE, and (eventually) NMR FT spectra — Principle 4 territory.

**Why not fold in:** if `core.stats` becomes "anything with numbers", the `Parametric` ABC stops being load-bearing and the namespace loses its discipline. The boundary cost (one extra core module to document) is small compared to the conceptual cost.

**The honest dependency:** `core.signal.peaks.fit_peaks(trace, found_peaks)` will call into `core.stats.distributions.EMG` to fit a parametric shape to a region heuristically located by `find_peaks`. So `core.signal` imports `core.stats`, sitting downstream of it in the DAG (same level as `nn` and `graph`).

**Updated import DAG (one line added):**
```
chem ─► sequence ─► structure ─► {stats, graph, nn, signal} ─► optim
  │         │            │              │   │                    ▲
  └─────────┴────────────┴──────────────┴───┴─► io ◄─────────────┘
```

`signal` imports `stats` (for parametric peak fits) but nothing else in core; everything else is a leaf-style dependency on numpy/torch. CLAUDE.md's Module index gets one new row.

---

## Build sequence

The first six items are **modality-agnostic foundations**. They unblock the MS port and any other future port; the chromatography/electrophoresis readers in items 7–8 are the first consumers but not the only ones.

### 1. `core.io.readers` — `RawReader` ABC + `READER_REGISTRY`

`constellation/core/io/readers.py`:
- `class RawReader(ABC)`: `suffixes: tuple[str, ...]`, `modality: str | None`, `read(path | Bundle) -> ReadResult`. The `ReadResult` is an attribute bag of `pa.Table` references (`primary`, `companions: dict[str, pa.Table]`) plus run-level metadata `dict[str, Any]` to be embedded as schema metadata.
- `READER_REGISTRY`: suffix-keyed dict-of-list (multiple readers can claim the same suffix; `modality` hint disambiguates). `register(cls)` decorator; `find(path, modality=None)` resolver that raises `ReaderNotFoundError` with a clear message listing registered options.
- Domain modules call `register()` at import time on their reader subclass.

### 2. `core.io.schemas` — canonical Arrow schemas

**Where these live and why.** The user's instinct ("Trace1D could definitely apply across chromatograms, electropherograms, pod5 traces, etc.") is exactly the principle here: these are *generic data shapes*, not modality semantics. They go in `core.io.schemas` because the alternative — a parallel `chromatography.schemas.Chromatogram1D`, `electrophoresis.schemas.Electropherogram1D`, `sequencing.schemas.Pod5Trace1D` — would all be `(time_s: float64, intensity: float64)` with cosmetic field-name differences, and any cross-modality consumer (a peak-fitter, a baseline-corrector, a CNN that ingests 1D traces regardless of source) would spend its life translating between them.

**The discipline that keeps it honest.** A schema only earns a slot in `core.io.schemas` if its column layout is genuinely universal across ≥2 modalities. Modality-specific information rides as **namespaced schema metadata** (`x.<domain>.<key>`), not as a column. So `Trace1D` is `(time_s, intensity)` everywhere; an HPLC consumer reads `x.hplc.wavelength_nm`, a CE consumer reads `x.ce.capillary_index`, a POD5 consumer reads `x.pod5.read_id` / `x.pod5.channel_id`. Producers stamp the metadata; downstream math doesn't care which modality the bytes came from.

**Limits of the abstraction.** `SpectraMatrix2D` as specced (regularly-gridded second axis stored as a fixed-size list) is correct for DAD wavelength axes and 2D NMR but **not** for MS scans, where m/z is per-scan-irregular. When the MS port lands, that case will need a separate `Spectra2DIrregular` (likely tall: `scan_idx, mz, intensity` + a `scan_metadata` companion). That's fine — it lives in the same `core.io.schemas` namespace and `SchemaRegistry`, as a sibling shape, not a forced unification.

**The schemas:**
- `Trace1D`: `time_s: float64`, `intensity: float64`. Schema metadata (universal): `channel_name`, `device`, `units`, `slope`, `sampling_rate_hz`, `instrument_id`, `run_id`, `run_datetime`. Schema metadata (domain extras): `x.<domain>.<key>` — e.g. `x.ce.capillary_index`, `x.hplc.wavelength_nm`, `x.pod5.read_id`, `x.ms.precursor_mz`. Stacked tables across channels/capillaries/reads use a `channel_id: string` column (still part of the generic schema) so `groupby('channel_id')` is the universal slice operator.
- `SpectraMatrix2D`: `time_ms: int64`, `spectrum: fixed_size_list<float64>[n_axis]`. Schema metadata: `axis_name` (e.g. `"wavelength_nm"`, `"chemical_shift_ppm"`), `axis_values` (JSON-encoded list), `units`, `scale_factor`, plus standard run metadata. Generic on the inner axis name so the same schema works for DAD (wavelength) and 2D NMR (chemical shift).
- `PeakTable`: `peak_id: int32`, `channel: string`, `retention_time_s: float64`, `area: float64`, `height: float64`, `fwhm_s: float64`, `asymmetry: float64`, `snr: float64`, `baseline: float64`, `calibrated_value: float64` (nullable; size in bp, amount in pmol, m/z, etc.), `calibrated_unit: string`. Schema metadata: `detection_method`, `baseline_method`, `peak_shape`. Domain extras: `x.ms.mz`, `x.ms.charge`, `x.ce.ladder_name`.
- `cast_to_schema(table, target_schema)` for forward-compat reads (mirroring Cartographer's `_cast_to_schema`).
- `SchemaRegistry` keyed by name, lazily populated. Domain modules don't add their own schemas to this registry unless the shape is genuinely universal — they consume the canonical ones and stamp `x.<domain>.*` metadata.

**Helpers also colocated here:**
- `spectra_to_tensor(table) -> torch.Tensor` (zero-copy via Arrow buffer view).
- `trace_to_tensor(table, channel=None) -> tuple[torch.Tensor, torch.Tensor]` (returns `(time, intensity)` tensors, optionally filtered to one channel).
- `pack_metadata(d: dict) -> dict[bytes, bytes]` / `unpack_metadata(meta) -> dict` for the JSON-encoded extras convention.

### 3. `core.io.bundle` — `Bundle` for multi-file containers

`constellation/core/io/bundle.py`:
- `class Bundle`: abstract, `members() -> list[str]`, `open(name) -> bytes`, `path: str`.
- `class OpcBundle(Bundle)`: zip-backed; for HPLC `.dx`. Reads `[Content_Types].xml` to map extension → content-type and `_rels/.rels` for the manifest pointer. Caches members in memory by default; supports `lazy=True` for streaming.
- `class DirBundle(Bundle)`: directory of files, primary file plus companions in the same directory. For FA/PA `.raw` runs (the `.raw` plus `*.txt`, `*.ANAI`, `*.current`, `Timing.txt`, `ExpTime.txt`, `method.mthd`).
- `Bundle.from_path(p)` dispatches on whether `p` is a zip, a directory, or a single file (single file becomes a degenerate `DirBundle` with no companions).

### 4. `core.stats.distributions` — `Parametric` ABC + peak shapes

`constellation/core/stats/distributions.py`:
- `class Parametric(nn.Module, ABC)`: `forward(x) -> y`, `log_prob(x) -> float`, `.fit(x, y, optimizer="de")` driving through `core.optim`.
- Densities: `Normal`, `StudentT`, `GeneralizedNormal` (port the Cartographer error model later when the MS port lands).
- Peak shapes: `Gaussian`, `EMG` (exponentially-modified Gaussian — chromatographic workhorse), `HyperEMG` (sum of EMGs, recoverable from `cartographer/peak_shape.py@0cee299`), `Voigt` (deferred unless an obvious need surfaces). All as `Parametric` subclasses with `(amplitude, center, sigma, ...)` learnable parameters.
- **Calibration models** as `Parametric` subclasses (these satisfy `CalibrationCurve` from `core.signal.calibration` automatically — `apply` is `forward`, `inverse` is solved analytically per family, `.fit()` flows through `core.optim`):
  - `Sigmoidal` — `y = A + (B − A) / (1 + exp(−(x − x0) / k))`. Captures the plateau-rise-plateau shape of CE mobility-vs-size.
  - `GeneralizedLogistic` — Sigmoidal with an asymmetry exponent; better for ladders where the small-fragment plateau is sharper than the large-fragment one.
  - `Hill` — `y = (B · x^n) / (K^n + x^n) + A`. Sigmoidal in log-x; common biology default.
  - `LogLinear` — `log(size) = a + b · log(t)`. The simple physics baseline; useful for sanity-checks and Ogston-regime small-fragment calibration.
  - `MonotonicMLP`-backed (when `core.nn.MonotonicMLP` lands) — neural calibration with monotonicity enforced architecturally; deferred to a follow-up but architectural slot reserved.
  - **Out of scope for first cut, on the roadmap**: Slater–Noolandi (free-draining) and reptation-with-stretching DNA-mobility models. These have ~3 physically interpretable parameters and are the gold-standard physics-grounded calibration for gel CE; they live in `electrophoresis.physics` (a new domain submodule) when developed because their parameters are CE-specific and don't generalise. They satisfy the same `CalibrationCurve` protocol.
- `losses.py`: `kld`, `spectral_angle`, `spectral_entropy_loss` (deferred until first MS consumer).

### 5. `core.optim.de` — `DifferentialEvolution`

`constellation/core/optim/de.py`:
- Port from `cartographer/optimize.py@0cee299` (recoverable via `git -C ~/projects/cartographer show 0cee299:cartographer/optimize.py`).
- Sobol initialisation, rand/1, best/1, current-to-best/1 mutation, optional LBFGS polish folded in.
- Operates on `nn.Module` parameters so `Parametric.fit()` can call it uniformly.
- `OPTIMIZER_REGISTRY` in `core/optim/registry.py` exposing `"de"` plus the `torch.optim.*` family under one interface.

### 6. `core.signal` — heuristic 1D-trace operations

New core submodule. Updates the Module index in CLAUDE.md.

`constellation/core/signal/__init__.py`:
- Re-exports the four submodules below.

All operations are torch-native (Principle 3 + the no-wasted-roundtrip rule from §Architecture). Tensors are 1D for single traces and 2D `(N, T)` for batched application (e.g. baseline correction across all DAD wavelengths or all CE capillaries simultaneously). No held numpy variables.

`constellation/core/signal/baseline.py`:
- `asls(y: torch.Tensor, lam=1e7, p=0.001) -> torch.Tensor` — Eilers asymmetric least squares; uses `torch.linalg.cholesky_solve` on the penalty matrix `D.T @ D`.
- `arpls(y: torch.Tensor, lam=1e5, ratio=1e-6) -> torch.Tensor` — Baek 2015 robust variant.
- `snip(y: torch.Tensor, n_iter=40) -> torch.Tensor` — Statistics-sensitive Non-linear Iterative Peak-clipping; pure tensor ops (max-pool style envelope iteration), GPU-friendly.
- All operate batched on the leading dim, so DAD's `(W, T)` matrix can be baseline-corrected across all 226 wavelengths in one call.

`constellation/core/signal/smoothing.py`:
- `savitzky_golay(y: torch.Tensor, window, order) -> torch.Tensor` — local polynomial fit via precomputed Savitzky–Golay kernel and `torch.nn.functional.conv1d`.
- `gaussian_smooth(y, sigma)` — kernel + `conv1d`.
- `moving_average(y, n)` — `conv1d` with uniform kernel.

`constellation/core/signal/peaks.py`:
- `find_peaks(y: torch.Tensor, prominence, height=None, distance=None) -> torch.Tensor` — torch-native prominence-based peak picker. Local-maxima via `y[1:-1] > y[:-2]` + `y[1:-1] > y[2:]`, prominence by left/right minimum walks expressed as cumulative ops. Returns indices as `int64` tensor.
- `cwt_peaks(y, widths)` — multi-scale wavelet peak detection (Ricker-wavelet `conv1d` ridge tracking).
- `peak_snr(y, peak_idx, baseline)` — local SNR via tensor slicing.
- `fit_peaks(y, peak_idx, shape="emg")` — calls `core.stats.distributions.EMG.fit()` per peak window (this is the documented dependency on stats). Returns fitted-param tensor stack `(n_peaks, n_params)`.

`constellation/core/signal/calibration.py`:
- `class CalibrationCurve(Protocol)`: `apply(x: torch.Tensor) -> torch.Tensor`, `inverse(y: torch.Tensor) -> torch.Tensor`, `residuals_at(x_known, y_known) -> torch.Tensor`, `r_squared: float`, `model_name: str`. Anything that satisfies this protocol can be returned by a fitter and consumed by `apply_calibration(traces, curve)`.
- Non-parametric / heuristic fitters (this module's home turf — ProSize-compatibility-first, not method-first):
  - `polyfit_calibration(x, y, order=5, log_x=False, log_y=False) -> PolynomialCurve` — torch-native via `torch.linalg.lstsq` on a Vandermonde matrix. **Kept primarily for ProSize-compatible re-analysis**, not as a recommended default. Returns a `PolynomialCurve` dataclass with `coefficients`, `residuals`, `apply` / `inverse`.
  - `monotonic_spline_calibration(x, y, n_knots=None, monotone="decreasing") -> SplineCurve` — natural cubic spline with monotonicity enforced (Fritsch–Carlson PCHIP-style or constrained least squares). Reasonable non-parametric default with fewer artefacts than a high-order polynomial outside the ladder span.
- The genuinely **parametric** calibration families live in `core.stats.distributions` (next section), because they have well-defined functional forms with interpretable parameters and are fit via `Parametric.fit()` through the optimizer registry. `core.signal.calibration` re-exports them under the `CalibrationCurve` protocol for convenience, but does not own them.

### 7. `constellation.chromatography` — HPLC-DAD domain

```
constellation/chromatography/
├── __init__.py                 (module docstring; re-exports `read_dx`)
├── readers/
│   ├── __init__.py             (registers AgilentDxReader at import time)
│   └── agilent_dx.py           (AgilentDxReader subclass + decoders)
├── peaks.py                    (chromatogram → PeakTable workflow)
└── tests/                      (alongside repo-level tests/)
```

`agilent_dx.py`:
- `class AgilentDxReader(RawReader)`: `suffixes = (".dx",)`, `modality = "hplc-dad"`, `read(path) -> ReadResult` returning:
  - `primary`: `Trace1D` table for the *user-selected default* chromatogram channel (typically DAD1A) — selected by reading the manifest and picking the first `Signal179` whose `Description` includes `"Sig="`.
  - `companions`:
    - `chromatograms`: stacked `Trace1D` table with `channel_id` column for all `Signal179` channels.
    - `instrument_traces`: stacked `Trace1D` table for all `InstrumentTrace179` (PMP1A pump pressure, AFC temps, etc.). Schema flag `irregular: true` because timestamps aren't on a regular grid.
    - `dad_spectra`: `SpectraMatrix2D` table when a `Spectra131` is present in the manifest. Skipped when "Spectrum: All" was off in the method.
  - Run-level metadata embedded as schema metadata on every table: `sample_name`, `operator`, `run_datetime`, `vial`, `injection_volume_uL`, `acquisition_method`.
- Decoders mirror `dx_reader.py`'s logic but wrap into Arrow with **direct `torch.frombuffer` decode** (the `.dx` payloads are little-endian `<f8` and `<u4`, both natively supported — no numpy intermediate):
  - `_decode_signal179(buf, meta)` → `torch.frombuffer(buf[0x1800:], dtype=torch.float64)` × `slope` → `Trace1D` chunk via `pa.array(...)` from the tensor's underlying buffer.
  - `_decode_it179(buf, meta)` → same; the paired `(time_ms, value)` interleave is a strided view.
  - `_decode_spectra131(buf, meta)` → strided `torch.frombuffer` reads at `0x1000 + i * SPEC_STRIDE + 22` into a single `(N_RECORDS, SPEC_N_VALS)` tensor; the per-block `time_ms` (uint32 LE at offset 4) decoded as a separate `torch.frombuffer` view. Wavelength axis stored as schema metadata; the per-block 14-byte mystery header is preserved as a single bytes blob in metadata (`x.dx.block_header_bytes`) for forward-compat once it gets cracked.
- Constants `SPEC_STRIDE`, `SPEC_N_VALS`, `WL_START`, `WL_STEP`, `SPEC_SCALE` are class attributes of `AgilentDxReader` so subclasses or hot-fixes can override per run if/when the per-block header is decoded.

`peaks.py`:
- `detect_peaks(trace_table, baseline="arpls", shape="emg") -> PeakTable`
- Pipeline: `core.signal.baseline.arpls` → `core.signal.peaks.find_peaks` → `core.signal.peaks.fit_peaks` (which delegates to `core.stats.distributions.EMG.fit`) → `PeakTable` Arrow table.

### 8. `constellation.electrophoresis` — FA / PA domain

```
constellation/electrophoresis/
├── __init__.py
├── readers/
│   ├── __init__.py             (registers AgilentFaReader)
│   └── agilent_fa.py           (AgilentFaReader; FA + PA share this class)
├── calibration.py              (ladder fit; bp lookup tables)
└── peaks.py                    (electropherogram → PeakTable workflow)
```

`agilent_fa.py`:
- `@dataclass(frozen=True) class RunGeometry`: `header_size, n_capillaries, frame_bytes, frame_pixels, cap_table_offset, cap_pixel_offset, window_half`. Two presets: `FA_12CAP_4_0_X` (current FA firmware 4.0.x with 12 caps, 1502 B/frame) and `PA_12CAP` (provisional, same numbers — confirm against user-supplied sample). Auto-detection helper `_detect_geometry(header_bytes, file_size)` that probes the cap-position table length (read uint16s at 0x3E8 until a zero or until consistency breaks) and divides through to find the matching `frame_bytes`.
- Byte decode: the `.raw` is **big-endian uint16**, which torch does not natively support. The numpy boundary is `arr = np.frombuffer(data, dtype=">u2", offset=HEADER_SIZE).reshape(n_frames, frame_pixels)`, immediately handed off via `frames = torch.from_numpy(arr.copy())` (the `.copy()` is required because the byte-swapped numpy buffer is a non-contiguous view in native order). Per-capillary trace extraction (`frames[:, lo:hi].sum(axis=1)`) runs in torch.
- `class AgilentFaReader(RawReader)`: `suffixes = (".raw",)`, `modality = "ce"`, `read(bundle) -> ReadResult`.
  - `primary`: stacked `Trace1D` with `channel_id = capillary_index` for all capillaries. Time in seconds from frame index (1 Hz nominal; corrected against `Timing.txt` if present in the bundle).
  - `companions`:
    - `current_log`: 3-column `Trace1D` from `*.current` TSV (Current/Voltage/Pressure), all 1 Hz, useful as QC overlay.
    - `method`: small `pa.Table` with one row of method.mthd INI fields.
    - `analysis_settings`: `pa.Table` of the `.ANAI` per-capillary settings + size-calibration block (ladder list, polynomial order).
    - `sample_map`: `capillary_index → well → sample_id` from the run-summary `.txt`.
  - Run metadata: `instrument_id`, `cartridge_id`, `firmware`, `run_date`, `run_time`, `cap_pixel_positions` (the table at 0x3E8 — kept for diagnostics).
- Skip `.PKS`, `.GANNT`, `.raw2D`, `CameraImage.bmp`, `ExpTime.txt`, `Timing.txt` for first cut. Document each as an open companion to round out later.

`calibration.py`:
- `@dataclass(frozen=True) class LadderSpec`: rich enough to drive both peak-detection and calibration choice, plus to stamp provenance:
  - `name: str` — registry key (e.g. `"DNF-473-55"`).
  - `platform: Literal["FA", "PA"]` — Fragment Analyzer or ProteoAnalyzer.
  - `vendor_part: str` — Agilent kit number for traceability.
  - `sizes: tuple[float, ...]` — fragment sizes in bp (FA) or kDa (PA).
  - `size_unit: Literal["bp", "kDa"]`.
  - `size_uncertainty: tuple[float, ...] | None` — vendor-quoted ±tolerance per fragment; used to weight the calibration fit so small-fragment markers (which the vendor specifies tightly) drive the fit more than the 6000 bp band (which is loose by ~50 bp).
  - `lower_marker: float` — alignment marker at the small end; FA uses 1 bp.
  - `upper_marker: float | None` — at the large end; some kits use it as an internal QC peak rather than for sizing.
  - `default_model: str` — recommended `Parametric` calibration family for this ladder (`"sigmoidal"`, `"hill"`, `"monotonic_spline"`, `"polyfit_loglog"`); set by the kit's design and our empirical experience, overridable per call.
  - `expected_peak_count: int` — sanity-check for peak detection.
- `LADDERS`: registry loaded from `constellation/data/calibration_ladders.json` at import. **Adding a ladder is a JSON edit + a test row, not a code change.** Initial registry seeded with what the lab actually uses; the user will provide the inventory before this lands. Likely entries:
  - FA: `DNF-473-55` (NGS, 1–6000 bp, 16 fragments) — the only one tested in `analyzer_analysis`.
  - FA / PA: additional kits to be specified by the user. Probable candidates from Agilent's catalogue: `DNF-377` (DNA HS NGS), `DNF-486` (DNA 50kb), `DNF-915` (Genomic DNA 165 kb), `P-241` (ProteoAnalyzer Standard Protein Ladder). The plan intentionally does not commit to the full list until verified.
- `calibrate_capillary(trace, ladder=LADDERS[name], model=None, prominence=None) -> CalibrationCurve`:
  1. Detect ladder peaks via `core.signal.peaks.find_peaks`; assert the count matches `ladder.expected_peak_count` (raise with diagnostic if not).
  2. Choose calibration family: `model` arg if given, else `ladder.default_model`. Resolve to a `Parametric` subclass from `core.stats.distributions` (sigmoidal/Hill/etc.) or a non-parametric fitter from `core.signal.calibration` (polyfit/spline).
  3. Fit using ladder uncertainties as weights when present.
  4. Return any object satisfying the `CalibrationCurve` protocol.
- Diagnostics: residuals tensor at each ladder peak, R², condition number, and a `goodness_of_fit_warning(curve, ladder)` helper that flags fits where any residual exceeds `2 × size_uncertainty[i]`.
- `apply_calibration(traces, curve) -> traces_with_size`: adds a `size_<unit>: float64` column (`size_bp` for FA, `size_kDa` for PA) to the `Trace1D` table; stamps `x.ce.calibration_model = curve.model_name`, `x.ce.calibration_params` (JSON-encoded), `x.ce.ladder_name`, and `x.ce.calibration_residual_max` in schema metadata for full provenance.

**Calibration approach is a known active development area, not a settled choice.** The plan deliberately keeps the API surface (`CalibrationCurve` protocol + `default_model` per ladder) flexible so swapping in a better physical model later is a one-line registry change, not a rewrite. The Slater–Noolandi / reptation models are flagged as roadmap items in `electrophoresis.physics`. Empirical comparison of polyfit / sigmoidal / spline / physics fits across the lab's actual ladder inventory should happen as a separate analysis once the basic readers + ladder registry land — that comparison is the data the model choice should be driven by.

### 9. CLI surface

Add to `constellation/cli/__main__.py`:
- `constellation hplc convert <run.dx> <out_dir>` — write `chromatograms.parquet`, `instrument_traces.parquet`, optional `dad_spectra.parquet`.
- `constellation hplc inspect <run.dx>` — print signal manifest + run metadata (no parquet output).
- `constellation hplc peaks <chromatograms.parquet> --channel DAD1A` — write `peaks.parquet`.
- `constellation ce convert <run.raw>` — write `traces.parquet`, `current_log.parquet`, `analysis.parquet`. Bundle resolution from sibling files automatic.
- `constellation ce calibrate <traces.parquet> --ladder DNF-473-55 --capillary 12` — fit ladder, materialise `size_bp` column.
- `constellation ce peaks <traces.parquet>` — peak detection + integration on each capillary.

Each subcommand follows the existing `_cmd_doctor` pattern and stays a thin wrapper around the domain API so the subcommands ARE the integration tests.

---

## Critical files to modify or create

### New core files
- `constellation/core/io/readers.py` — RawReader ABC + READER_REGISTRY
- `constellation/core/io/schemas.py` — Trace1D, SpectraMatrix2D, PeakTable
- `constellation/core/io/bundle.py` — Bundle / OpcBundle / DirBundle
- `constellation/core/stats/distributions.py` — Parametric ABC + Gaussian/EMG/HyperEMG
- `constellation/core/optim/de.py` — DifferentialEvolution port
- `constellation/core/optim/registry.py` — OPTIMIZER_REGISTRY
- `constellation/core/signal/__init__.py` — module entry
- `constellation/core/signal/baseline.py` — AsLS / arPLS / SNIP
- `constellation/core/signal/smoothing.py` — Savitzky–Golay etc.
- `constellation/core/signal/peaks.py` — find_peaks / CWT / fit_peaks (delegates to stats)
- `constellation/core/signal/calibration.py` — polyfit_calibration / CalibrationCurve

### New domain files
- `constellation/chromatography/__init__.py`
- `constellation/chromatography/readers/__init__.py`
- `constellation/chromatography/readers/agilent_dx.py`
- `constellation/chromatography/peaks.py`
- `constellation/electrophoresis/__init__.py`
- `constellation/electrophoresis/readers/__init__.py`
- `constellation/electrophoresis/readers/agilent_fa.py`
- `constellation/electrophoresis/calibration.py`
- `constellation/electrophoresis/peaks.py`

### New data files
- `constellation/data/calibration_ladders.json` — ship DNF-473-55 with full `LadderSpec` schema (sizes, uncertainties, default_model, vendor_part). Additional FA and PA ladders added per the user's inventory before this module lands; schema is set up to accept them as JSON edits with no code change.

### Edits
- `constellation/core/io/__init__.py` — re-export real symbols
- `constellation/core/stats/__init__.py` — re-export Parametric + peak shapes
- `constellation/core/optim/__init__.py` — re-export DE + registry
- `constellation/cli/__main__.py` — add `hplc` and `ce` subcommands; register both readers
- `CLAUDE.md` — update Module index (add `core.signal`, `chromatography`, `electrophoresis`); update import-DAG ASCII to include `signal`; document the `.dx` and `.raw` formats in a new "Companion-instrument data" section; remove the "scaffold" tag from `core.io`/`core.stats`/`core.optim` once shipped
- `pyproject.toml` — no new top-level deps required; numpy/pyarrow/torch/scipy already present. (No `rainbow-api` — clean rewrite.)

### Existing code to reference (do not vendor)
- `~/projects/hplc_analysis/dx_reader.py` (reference for `agilent_dx.py`)
- `~/projects/hplc_analysis/FORMAT_NOTES.md` (authoritative format spec)
- `~/projects/analyzer_analysis/read_raw.py` (reference for `agilent_fa.py`)
- `~/projects/analyzer_analysis/REPORT.md` (authoritative format spec)
- `~/projects/cartographer @ 0cee299` — `optimize.py` (DE port), `peak_shape.py` (HyperEMG), `error_model.py` (Student-t)

---

## Test strategy

Mirrors the `core.chem` / `core.sequence` gold-standard pattern (`tests/test_<module>_<topic>.py`, parametrize-heavy, no fixtures unless data-bearing).

### Unit tests
- `tests/test_core_io_readers.py` — registry registration, suffix dispatch, modality disambiguation, error messages.
- `tests/test_core_io_schemas.py` — schema construction, `cast_to_schema` round-trip, metadata preservation.
- `tests/test_core_io_bundle.py` — `OpcBundle` from a synthesised tiny zip; `DirBundle` from a tmp dir.
- `tests/test_core_stats_emg.py` — EMG forward shape, fit on synthetic data recovers parameters within tolerance.
- `tests/test_core_optim_de.py` — DE fits a known parametric form; checks Sobol init reproducibility with a seed.
- `tests/test_core_signal_baseline.py` — AsLS/arPLS/SNIP on a synthetic chromatogram (Gaussian peaks + linear drift) recover the drift to within tolerance.
- `tests/test_core_signal_peaks.py` — `find_peaks` recovers known peak indices on synthetic data; `fit_peaks` round-trips EMG params.
- `tests/test_core_signal_calibration.py` — `polyfit_calibration` recovers a known polynomial; log-log mode handled.

### Integration tests (data-bearing)
- `tests/data/hplc/Mario-2026.dx` — single bundled .dx fixture (or symlink into `~/projects/hplc_analysis/...`).
- `tests/data/ce/all_ladder.raw` — single bundled .raw fixture (companion files included as a directory).
- `tests/test_chromatography_reader.py`:
  - `read_dx` returns the expected channels (DAD1A, …, PMP1A, etc.).
  - `chromatograms.parquet` round-trips bit-exact for DAD1A against the bundled OpenLab CSV (the gold standard from FORMAT_NOTES.md).
  - DAD `SpectraMatrix2D` reconstructs DAD1A within 5% across the run (per FORMAT_NOTES validation).
- `tests/test_electrophoresis_reader.py`:
  - `read_raw` recovers 12 capillary traces of the expected length (5397 frames).
  - Capillary 12 (the ladder) yields exactly 16 detected peaks aligning with the DNF-473-55 declared sizes after calibration.
  - `current_log.parquet` parses the `.current` TSV correctly.

### Smoke
- Update `tests/test_imports.py` to import `constellation.core.signal`, `constellation.chromatography`, `constellation.electrophoresis`.

---

## End-to-end verification

Once shipped, the following commands should run cleanly on the existing reverse-engineering artefacts (zero code edits to either `~/projects/hplc_analysis` or `~/projects/analyzer_analysis`):

```bash
# HPLC with full DAD acquisition
constellation hplc inspect ~/projects/hplc_analysis/Mario-2026*.dx
constellation hplc convert ~/projects/hplc_analysis/Mario-2026*.dx /tmp/mario_2026/
ls /tmp/mario_2026/  # chromatograms.parquet  dad_spectra.parquet  instrument_traces.parquet
constellation hplc peaks /tmp/mario_2026/chromatograms.parquet --channel DAD1A

# HPLC, discrete-channel-only (no DAD spectrum)
constellation hplc convert ~/projects/hplc_analysis/Mario-2022*.dx /tmp/mario_2022/
# dad_spectra.parquet should be absent

# Fragment Analyzer
constellation ce convert "~/projects/analyzer_analysis/all ladder/2025 06 04 17H 25M.raw" /tmp/fa_run/
constellation ce calibrate /tmp/fa_run/traces.parquet --ladder DNF-473-55 --capillary 12
constellation ce peaks /tmp/fa_run/traces.parquet
```

Programmatic smoke test in a notebook:

```python
from constellation.chromatography import read_dx
from constellation.electrophoresis import read_raw
from constellation.core.io.schemas import spectra_to_tensor

# HPLC: round-trip and tensor reconstruction (zero-copy bridge through Arrow buffer)
inj = read_dx("Mario-2026.dx")
chroms = inj.companions["chromatograms"]                     # pa.Table, Trace1D schema
dad = inj.companions["dad_spectra"]                          # pa.Table, SpectraMatrix2D
matrix = spectra_to_tensor(dad)                              # torch.Tensor (T, W), zero-copy

# CE: read, calibrate, peak-detect (all torch, no numpy intermediate held)
run = read_raw("all ladder/...raw")
from constellation.electrophoresis.calibration import calibrate_capillary, apply_calibration, LADDERS
curve = calibrate_capillary(run.primary, ladder=LADDERS["DNF-473-55"], capillary_index=12)
sized = apply_calibration(run.primary, curve)                 # adds size_bp column
```

`spectra_to_tensor` lives in `core.io.schemas` and encapsulates the Arrow-buffer → torch view (`pa.Array.buffers()[1]` for the data buffer → `torch.frombuffer` of dtype float64 → reshape to `(T, W)`). One canonical helper, used by both notebook code and the CLI.

`pytest tests/test_chromatography_reader.py tests/test_electrophoresis_reader.py -v` should pass with the bundled fixtures. `pytest tests/` overall green. `ruff check .` clean.

---

## Out of scope / open items

- **ProteoAnalyzer fixture and verification.** User has a 12-capillary PA, expected to share constants with FA; will revisit when a sample `.raw` is in the test fixtures. The `RunGeometry` dataclass + auto-detect helper will accept a PA file gracefully or fail with a clear "couldn't auto-detect, please supply a RunGeometry" message.
- **`.UV` per-block 14-byte header.** Currently undecoded; treated as opaque metadata. Cracking it requires 2–3 short HPLC runs with deliberately different DAD spectrum ranges (FORMAT_NOTES.md item 1). Nothing depends on it for first-cut readers.
- **`.UVD` n_marker = 81.** Same status — ignored, preserved.
- **`.PKS`, `.GANNT`, `.raw2D` decoders.** Skipped intentionally; ProSize-derived artefacts that we'd be replacing anyway.
- **LC-MS hyphenation.** When the MS port lands, `constellation.massspec.readers` will register alongside `chromatography.readers`; for an LC-MS run the chromatography reader handles the LC time-axis and the MS reader handles m/z scans. The `Bundle` abstraction is designed to host both side-by-side.
- **GC and other chromatography flavours.** `chromatography` namespace is sized for them; not built now.
- **Compound identification, library matching, downstream stats.** All deferred.
- **Physically grounded CE calibration models** (Slater–Noolandi free-draining, reptation-with-stretching, Ogston-regime small-fragment limit). Active development area; landed in `electrophoresis.physics` once we've benchmarked polyfit / sigmoidal / spline / physics families across the lab's full ladder inventory. The `CalibrationCurve` protocol + per-ladder `default_model` field future-proofs the swap.
- **Full ladder inventory.** Initial registry ships with `DNF-473-55` only. The user will provide the lab's complete FA and PA kit list before the electrophoresis module lands; each kit needs sizes, vendor-quoted uncertainties, and an empirically-chosen default model.
- **`pyteomics`-style mzML support.** Lives in `massspec`, not here.
