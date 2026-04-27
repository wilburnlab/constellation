# CLAUDE.md — Constellation

## Project Purpose

Integrative bioinformatics platform: a shared core of physically grounded primitives (compositions, distributions, optimizers, graphs) with modality modules (mass spectrometry, sequencing, structure, codon optimization, NMR) layered on top. The goal is to connect experimental modalities through common abstractions — mathematical structures exposed once in `core/` and reused everywhere — rather than siloed per-modality code.

Constellation is absorbing four existing lab packages as a **clean rewrite** (no vendored source): **Cartographer** (MS proteomics, first port target), **NanoporeAnalysis** (POD5 → protein), **CoLLAGE** (codon optimization), **Contour** (PDB → MD prep). Chronologer (RT prediction) has already been absorbed into Cartographer.

Current state: **`core.chem`, `core.sequence`, `core.io`, `core.structure`, `core.stats`, and a minimal `core.graph.Network` shipped; everything else scaffold.** Functional code lives in `constellation.core.chem` (full periodic table, Composition arithmetic, binned + isotopologue-resolved isotope distributions, UNIMOD vocabulary with subsetting), `constellation.core.sequence` (canonical + IUPAC alphabets, generic ops, nucleic translation/ORF/rev-complement, protein cleavage / peptide composition), `constellation.core.io` (`RawReader` ABC + suffix/modality registry, `Bundle`/`OpcBundle`/`DirBundle` for multi-file containers, canonical Arrow schemas `Trace1D` / `SpectraMatrix2D` / `PeakTable` with namespaced metadata extras, torch-tensor bridges), `constellation.core.graph` (Arrow-backed `Network[NodeT, EdgeT]` with neighbor lookup / induced subgraphs / connected components), `constellation.core.structure` (`AtomTable` Arrow schema for per-atom identity, separate `(F, N, 3)` torch coord tensor on `Ensemble`, `Topology` with bond/angle/dihedral graphs and `infer_bonds` from covalent radii, composable `pa.compute.Expression` selection helpers, tensor-native geometry — translate / rotate / centroid / mass_centroid / radius_of_gyration / principal_axes; Kabsch / RMSD / superposition deferred until `core.optim` ships), `constellation.core.stats` (two-tier ABC: `Parametric(nn.Module)` with `.fit(data, *, optimizer, ...)` driver, `Distribution` requires `.log_prob` + `.cdf`, `PeakShape` requires `.integrate` + `.bounds`; 9 distributions, 2 peak shapes, 3 calibration models, 5 losses, CODATA units; optimizer-agnostic — `core.optim` ships LBFGS/DE next session), `constellation.thirdparty.registry` (tool discovery), and `constellation.cli.__main__` (the `constellation doctor` command). All other subpackages are scaffold-only. Core primitives and domain modules are staged for the 90-day roadmap (see `docs/roadmap.md` when written).

## Architecture invariants (load-bearing — design principles from the PI)

1. **Physically grounded statistical models** with explicit priors — preferred over heuristics.
2. **PyArrow tables in memory, Parquet on disk.** No pandas inside the package; pandas exits at the port boundary and stays at the edge.
3. **PyTorch for numerical work** wherever possible — multithreaded performance + optimizers + ML.
4. **Generic, nested abstractions** — trees, networks, composition tensors, distributions, optimizers expose statistical similarities across modalities that are otherwise treated as orthogonal.
5. **Unified treatment of optimizers and distributions.** Peak shapes and probability densities share a single `Parametric` ABC; `DifferentialEvolution` and `torch.optim.*` share one registry.
6. **Close-to-instrument data access.** Provide readers that sit directly above the vendor/instrument format; document the assumptions each processing layer adds.
7. **CLI-first** (single `constellation <subcommand>` binary with thin legacy shims for Cartographer muscle memory); GUI later.
8. **Linux primary, WSL second, macOS best-effort.**
9. **Dual audience** — importable library for Jupyter-notebook lab research + polished external-user tool.

### Code-level invariants

- **PascalCase for all new classes.** Legacy `snake_case` class names from Cartographer (`resnet_block`, `resnet_unit`) do not come across.
- **No pandas inside the package.** Arrow in, Arrow out.
- **Degenerate-alphabet compatibility rule.** `core.sequence.alphabets` supports IUPAC degenerate codes (N, R, Y, B, Z, X, U, O, ...). Canonical alphabets expose chemical compositions; degenerate alphabets do not. Functions gate via `requires_canonical` / `degenerate_ok`; composition/mass/cleavage reject degenerate tokens, while kmerize/reverse-complement/ORF tolerate them. `translate()` resolves degenerate codons by enumerating canonical expansions and collapsing the residue set — wobble-position synonymy stays unambiguous (`CTN → L`), two-residue ambiguities map to `B`/`Z`/`J`, anything else (including stop+coding mixes) collapses to `X`.
- **Pluggable codon tables.** `translate(seq, codon_table=STANDARD)` accepts any entry from `CODON_TABLES` (keyed by NCBI transl_table number). Shipped: 1 (Standard), 2 (Vertebrate Mito), 3 (Yeast Mito), 4 (Mold/Protozoan/Coelenterate Mito + Mycoplasma), 5 (Invertebrate Mito), 6 (Ciliate), 11 (Bacterial/Archaeal/Plastid). Stored in `constellation/data/codon_tables.json` as overrides on top of table 1.
- **Cleavage uses regex, not pyteomics.** `core.sequence.protein.cleave` is hand-rolled with the third-party `regex` module (overlap-aware). The 17 built-in protease patterns (Trypsin family, LysC/N, ArgC, Chymotrypsin, AspN, GluC, Pepsin, ProteinaseK, Thermolysin, No_enzyme) are lifted from ExPASy PeptideCutter and ship in `constellation/data/proteases.json`. Pyteomics still earns its place later in `massspec` for mzML I/O — but `core.sequence` does not import it.
- **Third-party tool env-var convention:** `$CONSTELLATION_<TOOL>_HOME` (uniform across all tools — no `_JAR` / `_DIR` / `_HOME` mixing).
- **Generic-vs-modality split for schemas.** A schema earns a slot in `core.io.schemas` only if its column layout is genuinely universal across ≥2 modalities (`Trace1D`, `SpectraMatrix2D`, `PeakTable` qualify). Modality-specific information rides as **namespaced schema metadata** (`x.<domain>.<key>`), never as columns. So `Trace1D` is `(time_s, intensity)` everywhere; HPLC reads `x.hplc.wavelength_nm`, CE reads `x.ce.capillary_index`, POD5 reads `x.pod5.read_id`. Producers stamp the metadata; downstream math doesn't care which modality the bytes came from. Round-trip metadata through `pack_metadata` / `unpack_metadata` from `core.io.schemas` to keep the JSON encoding consistent.
- **Torch-first numerics; numpy only at I/O boundaries.** Per Principle 3, `core.signal`, `core.stats`, and `core.optim` operate on `torch.Tensor` end-to-end. Numpy is permitted only at: (a) big-endian byte decode where torch lacks the dtype (FA `.raw` ``>u2``) — one boundary-crossing per file via `torch.from_numpy(arr.copy())`; (b) the Arrow ↔ torch bridge (`pa.Array.to_numpy(zero_copy_only=True)` + `torch.from_numpy(...)` is zero-copy in both legs). Little-endian dtypes decode straight through `torch.frombuffer` without numpy.
- **`core.stats` parameterization conventions.** Positive parameters live in log-space (`log_sigma`, `log_tau`, `log_N_max`) and bounded parameters in logit-space (`logit_eta`, mixture weights) so any unconstrained gradient-based optimizer can fit them cleanly. Default dtype is float64 — peak-shape and density numerics are precision-sensitive. `Parametric.parameters_dict()` decodes back to natural units for inspection.
- **`Parametric.fit()` requires a caller-provided optimizer.** `core.stats` ships zero optimizer implementations — `fit(data, *, optimizer, ...)` accepts any object exposing `step(closure) -> Tensor`. The real `LBFGSOptimizer` + `DifferentialEvolution` ship in `core.optim` and operate generically on `nn.Module`. Tests wrap `torch.optim.LBFGS` in a 10-line `LBFGSAdapter` shim in `tests/conftest.py`.
- **CDF gap (pinned).** `StudentT.cdf` and `Beta.cdf` raise `NotImplementedError` in v1 — torch lacks `betainc` as of 2.11, a scipy bridge would break autograd, and a custom autograd Function is heavier than the v1 use case justifies. Lift when torch ships `betainc` or when a concrete gradient-aware-CDF use case appears (Cartographer error-model port goes through `log_prob` end-to-end and is unaffected). All other CDFs are pure torch.
- **Cartographer scoring inheritance.** `core.stats.losses` mirrors Cartographer's `kld` / `spectral_angle` formulae but as **free functions** (Cartographer's `KLD` / `Spectral_Entropy_Loss` `nn.Module` wrappers used `snake_case` class names against the PascalCase invariant). `spectral_angle` clamps the dot product to `[-1, 1]` instead of `[-1+ε, 1+ε]` — the latter pulls identical-vector similarity below 1 by ~3e-4 in float32. `spectral_entropy_loss` completes the truncated Cartographer implementation (return statement was commented out upstream).

## Import order — project-wide DAG

Load-bearing invariant. Violating it creates circular imports or module-init ordering bugs. Enforced by convention, not linting.

### Inside `core/`

```
chem ─► sequence ─► {graph, stats, nn, signal} ─► optim ─► structure
                                                              │
                          io ◄─── (leaf, consumed by all) ────┘
```

Tertiary structure / dynamics are *consequences* of chemistry + sequence + (graph) topology + thermodynamics + optimization, not building blocks for them — so `core.structure` sits at the bottom of the math chain. Bond graphs route through `core.graph`; non-trivial structural manipulation (alignment, fitting, constrained geometry) wants the objective-function + optimizer machinery in `core.stats` + `core.optim`.

- `chem` — never imports from `sequence`, `structure`, or `io`.
- `sequence` — imports `chem`; never imports `structure`.
- `graph` — generic over node/edge types, independent of `chem` and `sequence`.
- `stats`, `nn` — downstream math; may import from `chem`, `sequence`. Do NOT import `structure`. `core.stats` ships zero optimizer implementations; `Parametric.fit(data, *, optimizer, ...)` requires the caller to pass any object exposing `step(closure)`.
- `signal` — heuristic 1D-trace operations (baseline, smoothing, peak picking, polynomial calibration). Imports `stats` for parametric peak-fitting helpers; otherwise leaf-style. Distinct from `stats` because the operations have imprecisely-specified priors and don't earn membership in the `Parametric` namespace.
- `optim` — operates generically on `nn.Module` (which `Parametric` subclasses are); ships `LBFGSOptimizer` + `DifferentialEvolution` + `Optimizer` Protocol. The DAG question (does `optim` import `stats` to type its Protocol's consumer, or vice versa?) is settled when `optim` lands; v1 `core.stats` prejudges nothing and uses duck-typed `optimizer: object`.
- `structure` — most-downstream core module. May import any of `chem`, `sequence`, `graph`, `stats`, `nn`, `optim` as features land. Currently imports just `chem`, `sequence`, `graph`, `io`; `stats` / `optim` enter when Kabsch / RMSD / refinement land in `core.structure.geometry`.
- `io` — leaf. Format/codec concerns only. Never imports other core. The `AtomTable` schema self-registers with the `core.io` schema registry from `core.structure.atoms`.

### Between layers

```
core  ─►  {massspec, sequencing, structure, codon, nmr,
           chromatography, electrophoresis}  ─►  cli
           (domain modules — never cross-import)
```

Domain modules import from `core` and from `thirdparty` adapters; they **never cross-import to another domain module**. If a workflow legitimately spans modalities (e.g. the transcriptomic → proteomic pipeline), it lives as a thin top-level script under `constellation/` that imports from the relevant domains — no dedicated `bridges/` or `pipelines/` folder until we have ≥2 such workflows to compare.

Adding a new module: confirm its place in this DAG before touching imports. Adding a new reader (Bruker `.d`, Sciex `.wiff`, DCD trajectory, ...) follows the reader-subclasses-core.io.readers rule without special handling.

## Module index

| Module | Role | Status |
|---|---|---|
| `constellation.core.chem` | Elements (full H–Og via `ElementTable` / `ELEMENTS`), Composition (int32 tensor wrapper, plain-Python — hashable, bool `==`), binned + isotopologue-resolved isotope APIs, UNIMOD ModVocab with first-class subsetting | **shipped** |
| `constellation.core.sequence` | Alphabet ABC + canonical/IUPAC instances, generic ops (kmerize, sliding_window, hamming, modified-sequence parse/format), nucleic (reverse_complement, 7 NCBI codon tables, degenerate-codon-aware translate, find_orfs / best_orf, gc_content), protein (17-protease registry, cleave with N→C ordering and missed-cleavage enumeration, peptide_composition / peptide_mass with UNIMOD lookup) | **shipped** |
| `constellation.core.structure` | `AtomTable` Arrow schema (per-atom identity, coords held separately as `(F, N, 3)` torch tensor), `CoordinateFrame` (units, PBC), `Topology` (atoms + bonds/angles/dihedrals; bond view via `core.graph.Network`), distance-based `infer_bonds` (covalent radii from `core.chem`), `Ensemble` (NMR / cryoEM / MD frames unified, `frame_metadata` Arrow table), composable `pa.compute.Expression` selection helpers, tensor-native `translate` / `rotate` / `centroid` / `mass_centroid` / `radius_of_gyration` / `principal_axes`. Kabsch / RMSD / superposition / alignment deferred until `core.optim` ships. | **shipped** |
| `constellation.core.io` | `RawReader` ABC + suffix/modality registry, `Bundle`/`OpcBundle`/`DirBundle` for multi-file containers, canonical Arrow schemas (`Trace1D`, `SpectraMatrix2D`, `PeakTable`) with namespaced `x.<domain>.<key>` metadata extras, torch-tensor bridges (`trace_to_tensor`, `spectra_to_tensor`, `tensor_to_spectra`), forward-compat `cast_to_schema` | **shipped** |
| `constellation.core.stats` | Two-tier ABC under `Parametric(nn.Module)`: `Distribution` (requires `.log_prob` + `.cdf`) and `PeakShape` (requires `.integrate` + `.bounds`). Densities: NormalDistribution, StudentT, GeneralizedNormal, Beta, Gamma, LogNormal, Poisson, Multinomial, Dirichlet. Peak shapes: GaussianPeak, EMGPeak (with `_erfcx` two-branch numerics + `emg_pdf`/`emg_log_pdf` free fns; HyperEMG/Warped/Spline deferred to post-optim). Calibration models (`Parametric` leaf): Sigmoidal/4PL, Hill, LogLinear. Losses: kld (sentinel-`-1` masked + pseudocount), spectral_angle (with `clamp(±1)` stability fix vs Cartographer), spectral_entropy_loss (completion of Cartographer's truncated form), l1_normalize, l2_normalize. units: re-exports `core.chem.elements` constants + CODATA 2018 (k_B/R/h/c/ε_0/e) + ppm/Da/kT helpers. | **shipped** |
| `constellation.core.signal` | Heuristic 1D-trace ops — baseline correction (AsLS/arPLS/SNIP), smoothing (Savitzky–Golay), prominence peak picking, polynomial / monotonic-spline calibration. Torch-native; imports `core.stats` for Parametric peak-fits | scaffold |
| `constellation.core.optim` | `DifferentialEvolution` (with LBFGS polish), optimizer registry | scaffold |
| `constellation.core.nn` | ResNet blocks, MonotonicMLP, transformer scaffolds (PascalCase) | scaffold |
| `constellation.core.graph` | `Network[NodeT, EdgeT]` (Arrow-backed nodes/edges, neighbor lookup, induced subgraphs, connected components) shipped; `Tree[T]` still scaffold (lands when phylogeny work needs it) | partial |
| `constellation.massspec` | MS domain — mzpeak I/O, peptides, chromatograms, scoring ("Counter"), Koina, EncyclopeDIA | scaffold (first port target) |
| `constellation.sequencing` | POD5 → protein pipeline; Phred codec lives here (not in `core.io`) | scaffold |
| `constellation.codon` | CoLLAGE-style codon optimization | scaffold |
| `constellation.structure` | Contour replacement; PDB/mmCIF + MD trajectories unified via `core.structure.Ensemble` | scaffold |
| `constellation.nmr` | Placeholder | scaffold |
| `constellation.chromatography` | HPLC-DAD (Agilent OpenLab `.dx` first; LC-MS hyphenation when MS lands). Reader subclass of `core.io.RawReader`; peak workflow via `core.signal` + `core.stats` EMG | scaffold |
| `constellation.electrophoresis` | Capillary electrophoresis — Fragment Analyzer + ProteoAnalyzer (`.raw`, same LabVIEW-derived big-endian format). Ladder calibration via `LadderSpec` registry; physically-grounded models (Slater–Noolandi, reptation) reserved for `electrophoresis.physics` | scaffold |
| `constellation.models` | NN architectures assembled from `core.nn` | scaffold |
| `constellation.cli` | `constellation <subcommand>` dispatcher; `doctor` wired, others stubbed | partial (`doctor` works) |
| `constellation.thirdparty` | Tool discovery (`registry.find`); EncyclopeDIA adapter registered | partial |
| `constellation.data` | Packaged data: `elements.json` (NIST AME2020, full periodic table + per-isotope exact masses), `unimod.json` (1560 entries from upstream UNIMOD XML), `codon_tables.json` (7 NCBI tables as overrides on table 1), `proteases.json` (17 enzymes from ExPASy PeptideCutter). Raw vendored sources under `data/_raw/` regenerable via `scripts/build-{elements,unimod,codon-tables,proteases}-json.py` | partial |

## Environment

Conda env `constellation`; Python 3.12 only (pinned `<3.13` by **koinapy 0.0.10 + PyTorch 2.5.x** — same ceiling Cartographer uses). `torch`, `koinapy`, and `pythonnet` are all pip-installed via the `pip:` block in `environment.yml` — PyPI's default `torch` wheel is CUDA-capable, which works on CPU-only hosts and lets GPU nodes (OSC Cardinal, etc.) use CUDA without changing env.yml.

**Additive CUDA overlay:** `environment-cuda.yml` exists as a stub; apply on top of the base env on GPU hosts when GPU-specific deps (dorado builds, CUDA mmseqs) get wired in.

**Optional tool pattern:**
- `scripts/install-<tool>.sh` — idempotent, hash-pinned, writes into `third_party/<tool>/<version>/` and repoints `third_party/<tool>/current`.
- `$CONSTELLATION_<TOOL>_HOME` — user override pointing at a shared/external install.
- Per-tool adapter in `constellation/thirdparty/<tool>.py` declares the registry contract (artifact path, version probe, install-script hint).
- `constellation doctor` prints a tool-status table.

```bash
conda env create -f environment.yml && conda activate constellation
constellation doctor
```

**Reproducibility status:** fresh-machine install pending end-to-end validation. Installs `-e .[dev]` via the pip block in `environment.yml`, so code changes are picked up without re-installing.

## Testing

**Tools:** pytest + ruff (lint/format). Config in `pyproject.toml`.

```bash
pytest                          # verbose by default (addopts)
pytest tests/test_imports.py    # smoke test: import every subpackage
ruff check .                    # lint
```

**Test layout:** `tests/test_<module>.py`. `test_imports.py` is import-only smoke — update it when adding subpackages.

## Conventions

- **Functions:** `snake_case`.
- **Classes:** `PascalCase` (always — no legacy carryover).
- **Modification identifiers:** UNIMOD format (`UNIMOD:N`) is canonical; mass notation (`+N.NNN`) is a secondary representation. The full UNIMOD XML ships built-in (`UNIMOD: ModVocab`); EncyclopeDIA-style `[+mass_delta]` translation goes through `UNIMOD.find_by_mass(mass, tolerance_da)` in a thirdparty adapter, not in `core.chem`.
- **Modified-sequence bracket syntax is alphabet-agnostic.** `core.sequence.ops.parse_modified_sequence("PEPC[UNIMOD:4]TIDE")` returns `("PEPCTIDE", {3: "UNIMOD:4"})` — index keys are 0-indexed positions in the stripped sequence. Mass-notation values stay as `float`; UNIMOD/alias strings stay as `str`. The legacy Cartographer `[nterm]` / `[cterm]` markers are NOT carried forward — N-terminal mods land at position 0; the distinction between "on residue 0" and "on the N-terminus itself" is intentionally collapsed.
- **Cleavage output is N→C ordered, deduplicated.** `cleave()` returns peptides sorted by `(start, end)`, with sequence-level deduplication keeping first occurrence (pyteomics returns a `set` — random order). `return_spans=True` yields `Peptide(sequence, start, end, n_missed)` records with parent-protein coordinates intact, no dedup. `cleave_sites()` always anchors on `[0, ..., len(seq)]`.
- **Peptide mass uses heavy-isotope correction.** `peptide_mass(seq, modifications=...)` adds `mod.delta_composition` to the peptide composition for all mods; for mods carrying `mass_override` (TMT10, SILAC ¹³C₆, ...) the difference `mod.mass_override - mod.delta_composition.mass` is added on top so the result matches the canonical UNIMOD value. Light-skeleton compositions remain in the returned `Composition`; the heavy-isotope correction lives in the mass calculation, not in the composition itself.
- **Vocabulary vs enablement.** `core.chem.modifications` defines the *universe* (built-in `UNIMOD`); each downstream tool declares its *enabled subset* via `UNIMOD.subset([...])` and ships that with its model checkpoint / pipeline config. The chemistry layer never knows what any specific tool accepts. `register_custom(...)` is the typed escape hatch for genuinely novel non-UNIMOD mods; the legacy `user_modifications.txt` text format from cartographer/Chronologer is **not** carried forward.
- **Composition is plain-Python, not a torch.Tensor subclass.** Hashable, equality-as-bool, only chemistry-meaningful ops exposed. Batched / GPU work uses raw `(B, N_elements)` tensors via `stack()` / `batched_mass()` free functions — `Composition` objects are not in the hot path.
- **Isotope data carries per-isotope exact masses (NIST AME2020), not just abundances.** `isotope_distribution` / `isotope_envelope` use the binned ¹³C-spacing approximation (cartographer-equivalent, fast); `isotopologue_distribution` / `isotope_envelope_exact` resolve ¹⁵N vs ¹³C as distinct peaks for high-res MS, ¹⁵N-labeled samples, and NMR-adjacent work.
- **Composition tensor dtype is int32.** Counts are always whole; `.mass`/`.average_mass` cast to float64 internally for the dot-product. Averaged "compositions" (mixtures) are a downstream concept and are NOT a `Composition`.
- **Tolerance:** default fragment tolerance 20 ppm; functions accept `tolerance_unit` as `'ppm'` or `'Da'` (same as Cartographer).
- **Sentinel values:** `-1.0` = "not observed" / "not set" for float fields; `-1` in intensity tensors flags loss-masked positions.

## Companion-instrument data formats

Reverse-engineered formats for the lab's bench-side analytical instruments. Authoritative format specs live in their respective reverse-engineering repos; the in-Constellation readers are clean rewrites that produce `core.io` Arrow schemas.

| Instrument | File(s) | Format spec | Domain module |
|---|---|---|---|
| Agilent 1260 HPLC + DAD (OpenLab) | `<run>.dx` (OPC zip: `injection.acmd` XML manifest, `<uuid>.CH` Signal179 chromatograms, `<uuid>.IT` InstrumentTrace179 telemetry, `<uuid>.UV` + `.UVD` Spectra131 2D DAD matrix) | `~/projects/hplc_analysis/FORMAT_NOTES.md` | `constellation.chromatography.readers.agilent_dx` |
| Agilent Fragment Analyzer (nucleic acid CE) | `<run>.raw` (1992 B big-endian header, 12-uint16 capillary table at 0x3E8, 1 Hz frames of 751×uint16 BE pixels) + companion `*.txt` / `*.ANAI` / `*.current` / `method.mthd` | `~/projects/analyzer_analysis/REPORT.md` | `constellation.electrophoresis.readers.agilent_fa` |
| Agilent ProteoAnalyzer (protein CE) | Same `.raw` family as FA; lab cartridges are 12-capillary so constants match. PA-specific test fixture pending. | `~/projects/analyzer_analysis/REPORT.md` (§7) | `constellation.electrophoresis.readers.agilent_fa` (shared) |

Adding a new bench-instrument format follows the same recipe: decode bytes (torch-native if little-endian, numpy boundary if big-endian), stamp `Trace1D` / `SpectraMatrix2D` / `PeakTable` schemas with `x.<domain>.*` metadata, register a `RawReader` subclass in the appropriate domain.

## Ports in flight

Cartographer modules scheduled for rewrite into Constellation. Commit SHAs point at what to read (not what to copy — this is a clean rewrite):

| Source | Destination | Notes |
|---|---|---|
| ~~`cartographer/masses.py`~~ | ~~`core.chem.{elements,composition,isotopes,modifications}` + `core.sequence.alphabets`~~ | **Done** — both `core.chem` and `core.sequence` shipped. Cartographer's `RESIDUE_REGISTRY` / `VOCABULARY` / integer-token-encoding (`TOKEN_TO_INDEX`) is **not** carried forward into `core.sequence` — that's a model-specific encoding decision and will land in `constellation.massspec.tokenize` (or `constellation.models.tokenize`) when the MS port begins. Same for fragment ladders, m/z, and neutral-loss masks (`peptide_ladder_generator`, `generate_neutral_loss_mask`) — MS-specific, slated for `constellation.massspec.peptides`. |
| `cartographer/core_layers.py`, `cartographer/models.py` | `core.nn.{layers,encoders}` + `constellation.models.*` | PascalCase rename |
| ~~`cartographer/error_model.py`~~ | ~~`core.stats.distributions` (Student-t subclass of `Parametric`)~~ | **Partial** — `StudentT(Distribution)` shipped with torch-native `log_prob` (matches Cartographer's `mz_log_likelihood` to atol=1e-10 vs scipy). Recovers `(ν, σ)` from `scipy.stats.t.fit` ground truth to rtol≤0.15 via the LBFGS test shim. Remaining: build the equivalent of `MzErrorModel`/`CalibrationModel` dataclasses + `fit_mz_error_model` driver in `constellation.massspec` once `core.optim` ships the production fit path. |
| `cartographer/data/readers/{__init__,_base}.py` | `core.io.readers` | Registry becomes modality-tagged; `.cif`/`.txt` disambiguation via optional `modality` hint |
| `cartographer/data/spectra.py` (schema + `_cast_to_schema`) | `core.io.schemas` + `constellation.massspec.schemas` | Forward-compat cast generalizes |
| `cartographer/data/nanopore.py` (transcriptome→proteome bridge) | top-level `constellation/transcriptome_to_proteome.py` (eventually) | Deferred until it's a real workflow |
| `cartographer/optimize.py@0cee299` *(recoverable from git)* | `core.optim.de` | DE for nn.Module params; Sobol init; rand/1, best/1, current-to-best/1; optional LBFGS polish folded in |
| ~~`cartographer/peak_shape.py@0cee299`~~ *(recoverable from git)* | ~~`core.stats.peaks`~~ | **Partial** — `GaussianPeak` and `EMGPeak` shipped under the `PeakShape` ABC (split from `Distribution` per Principle 5; both share `Parametric(nn.Module)`). `_erfcx` and `emg_pdf` / `emg_log_pdf` ported with the algebraic-stability identity (`a - b² = -z²/2`). Remaining: `HyperEMGPeak`, `WarpedEMGPeak`, `SplinePeak` — these need `core.optim.DifferentialEvolution` to fit reliably (LBFGS gets stuck in the local minima they create), so they land in the optim-follow-up PR. Also still in Cartographer: `Spectral_Entropy_Loss` (the unfinished one — the Constellation port in `losses.spectral_entropy_loss` completes the truncated formula). |

Retrieve the recoverable files with:

```bash
git -C ~/projects/cartographer show 0cee299:cartographer/optimize.py
git -C ~/projects/cartographer show 0cee299:cartographer/peak_shape.py
```

Also pruned in the same window and worth surfacing: `MS1TensorResult` (chromatogram.py), `rt_range` predicate pushdown (spectra.py), windowed MS1 scoring, `HyperEMGPeak`.
