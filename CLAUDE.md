# CLAUDE.md — Constellation

## Project Purpose

Integrative bioinformatics platform: a shared core of physically grounded primitives (compositions, distributions, optimizers, graphs) with modality modules (mass spectrometry, sequencing, structure, codon optimization, NMR) layered on top. The goal is to connect experimental modalities through common abstractions — mathematical structures exposed once in `core/` and reused everywhere — rather than siloed per-modality code.

Constellation is absorbing four existing lab packages as a **clean rewrite** (no vendored source): **Cartographer** (MS proteomics, first port target), **NanoporeAnalysis** (POD5 → protein), **CoLLAGE** (codon optimization), **Contour** (PDB → MD prep). Chronologer (RT prediction) has already been absorbed into Cartographer.

Current state: **`core.chem` shipped; everything else scaffold.** Functional code lives in `constellation.core.chem` (full periodic table, Composition arithmetic, binned + isotopologue-resolved isotope distributions, UNIMOD vocabulary with subsetting), `constellation.thirdparty.registry` (tool discovery), and `constellation.cli.__main__` (the `constellation doctor` command). All other subpackages are scaffold-only. Core primitives and domain modules are staged for the 90-day roadmap (see `docs/roadmap.md` when written).

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
- **Degenerate-alphabet compatibility rule.** `core.sequence.alphabets` supports IUPAC degenerate codes (N, R, Y, B, Z, X, U, O, ...). Canonical alphabets expose chemical compositions; degenerate alphabets do not. Functions gate via `requires_canonical` / `degenerate_ok`; composition/mass/digest reject degenerate tokens, while kmerize/reverse-complement/ORF tolerate them. `translate()` resolves degenerate codons by enumerating canonical expansions and collapsing the residue set — wobble-position synonymy stays unambiguous, genuinely ambiguous triplets collapse to B/Z/J/X.
- **Pluggable codon tables.** `translate(seq, codon_table=STANDARD)` accepts any entry from `CODON_TABLES` (keyed by NCBI transl_table number: standard, vertebrate mitochondrial, yeast mitochondrial, bacterial, ciliate, ...).
- **Third-party tool env-var convention:** `$CONSTELLATION_<TOOL>_HOME` (uniform across all tools — no `_JAR` / `_DIR` / `_HOME` mixing).

## Import order — project-wide DAG

Load-bearing invariant. Violating it creates circular imports or module-init ordering bugs. Enforced by convention, not linting.

### Inside `core/`

```
chem ─► sequence ─► structure ─► {stats, graph, nn} ─► optim
  │         │            │              │               ▲
  └─────────┴────────────┴──────────────┴─► io ◄────────┘
```

Mirrors the biological hierarchy from building blocks → primary structure → tertiary structure.

- `chem` — never imports from `sequence`, `structure`, or `io`.
- `sequence` — imports `chem`; never imports `structure`.
- `structure` — imports `chem` and `sequence` (tertiary structure sits on primary).
- `stats`, `graph`, `nn` — downstream math; may import from `chem`, `sequence`, `structure`.
- `optim` — furthest downstream; `Parametric.fit()` drives through here.
- `io` — leaf. Format/codec concerns only. Never imports other core.

### Between layers

```
core  ─►  {massspec, sequencing, structure, codon, nmr}  ─►  cli
           (domain modules — never cross-import)
```

Domain modules import from `core` and from `thirdparty` adapters; they **never cross-import to another domain module**. If a workflow legitimately spans modalities (e.g. the transcriptomic → proteomic pipeline), it lives as a thin top-level script under `constellation/` that imports from the relevant domains — no dedicated `bridges/` or `pipelines/` folder until we have ≥2 such workflows to compare.

Adding a new module: confirm its place in this DAG before touching imports. Adding a new reader (Bruker `.d`, Sciex `.wiff`, DCD trajectory, ...) follows the reader-subclasses-core.io.readers rule without special handling.

## Module index

| Module | Role | Status |
|---|---|---|
| `constellation.core.chem` | Elements (full H–Og), Composition (int32 tensor wrapper, plain-Python — hashable, bool `==`), binned + isotopologue-resolved isotope APIs, UNIMOD ModVocab with first-class subsetting | **shipped** |
| `constellation.core.sequence` | Alphabets (canonical + IUPAC), generic ops, nucleic (translate/ORF/rev-comp), protein (digest) | scaffold |
| `constellation.core.structure` | Coords, geometry (Kabsch/RMSD), topology (Arrow-backed graph), ensemble (NMR/cryoEM/MD unified) | scaffold |
| `constellation.core.io` | Schema registry, RawReader ABC, Bundle (primary + companions) | scaffold |
| `constellation.core.stats` | `Parametric` ABC (densities + peak shapes), losses, units | scaffold |
| `constellation.core.optim` | `DifferentialEvolution` (with LBFGS polish), optimizer registry | scaffold |
| `constellation.core.nn` | ResNet blocks, MonotonicMLP, transformer scaffolds (PascalCase) | scaffold |
| `constellation.core.graph` | `Tree[T]`, `Network[NodeT, EdgeT]` | scaffold |
| `constellation.massspec` | MS domain — mzpeak I/O, peptides, chromatograms, scoring ("Counter"), Koina, EncyclopeDIA | scaffold (first port target) |
| `constellation.sequencing` | POD5 → protein pipeline; Phred codec lives here (not in `core.io`) | scaffold |
| `constellation.codon` | CoLLAGE-style codon optimization | scaffold |
| `constellation.structure` | Contour replacement; PDB/mmCIF + MD trajectories unified via `core.structure.Ensemble` | scaffold |
| `constellation.nmr` | Placeholder | scaffold |
| `constellation.models` | NN architectures assembled from `core.nn` | scaffold |
| `constellation.cli` | `constellation <subcommand>` dispatcher; `doctor` wired, others stubbed | partial (`doctor` works) |
| `constellation.thirdparty` | Tool discovery (`registry.find`); EncyclopeDIA adapter registered | partial |
| `constellation.data` | Packaged data: `atoms.json` (NIST AME2020, full periodic table + per-isotope exact masses), `unimod.json` (1560 entries from upstream UNIMOD XML). Raw vendored sources under `data/_raw/` regenerable via `scripts/build-{atoms,unimod}-json.py` | partial |

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
- **Vocabulary vs enablement.** `core.chem.modifications` defines the *universe* (built-in `UNIMOD`); each downstream tool declares its *enabled subset* via `UNIMOD.subset([...])` and ships that with its model checkpoint / pipeline config. The chemistry layer never knows what any specific tool accepts. `register_custom(...)` is the typed escape hatch for genuinely novel non-UNIMOD mods; the legacy `user_modifications.txt` text format from cartographer/Chronologer is **not** carried forward.
- **Composition is plain-Python, not a torch.Tensor subclass.** Hashable, equality-as-bool, only chemistry-meaningful ops exposed. Batched / GPU work uses raw `(B, N_elements)` tensors via `stack()` / `batched_mass()` free functions — `Composition` objects are not in the hot path.
- **Isotope data carries per-isotope exact masses (NIST AME2020), not just abundances.** `isotope_distribution` / `isotope_envelope` use the binned ¹³C-spacing approximation (cartographer-equivalent, fast); `isotopologue_distribution` / `isotope_envelope_exact` resolve ¹⁵N vs ¹³C as distinct peaks for high-res MS, ¹⁵N-labeled samples, and NMR-adjacent work.
- **Composition tensor dtype is int32.** Counts are always whole; `.mass`/`.average_mass` cast to float64 internally for the dot-product. Averaged "compositions" (mixtures) are a downstream concept and are NOT a `Composition`.
- **Tolerance:** default fragment tolerance 20 ppm; functions accept `tolerance_unit` as `'ppm'` or `'Da'` (same as Cartographer).
- **Sentinel values:** `-1.0` = "not observed" / "not set" for float fields; `-1` in intensity tensors flags loss-masked positions.

## Ports in flight

Cartographer modules scheduled for rewrite into Constellation. Commit SHAs point at what to read (not what to copy — this is a clean rewrite):

| Source | Destination | Notes |
|---|---|---|
| ~~`cartographer/masses.py`~~ | ~~`core.chem.{atoms,composition,isotopes,modifications}` + `core.sequence.alphabets`~~ | **Done** — `core.chem` shipped. The `core.sequence.alphabets` half (residue compositions, `RESIDUE_REGISTRY`, vocabulary) is the next ticket. |
| `cartographer/core_layers.py`, `cartographer/models.py` | `core.nn.{layers,encoders}` + `constellation.models.*` | PascalCase rename |
| `cartographer/error_model.py` | `core.stats.distributions` (Student-t subclass of `Parametric`) | `.fit()` via `core.optim`, no scipy |
| `cartographer/data/readers/{__init__,_base}.py` | `core.io.readers` | Registry becomes modality-tagged; `.cif`/`.txt` disambiguation via optional `modality` hint |
| `cartographer/data/spectra.py` (schema + `_cast_to_schema`) | `core.io.schemas` + `constellation.massspec.schemas` | Forward-compat cast generalizes |
| `cartographer/data/nanopore.py` (transcriptome→proteome bridge) | top-level `constellation/transcriptome_to_proteome.py` (eventually) | Deferred until it's a real workflow |
| `cartographer/optimize.py@0cee299` *(recoverable from git)* | `core.optim.de` | DE for nn.Module params; Sobol init; rand/1, best/1, current-to-best/1; optional LBFGS polish folded in |
| `cartographer/peak_shape.py@0cee299` *(recoverable from git)* | `core.stats.distributions` | Merged with density classes under `Parametric` (Principle 5) |

Retrieve the recoverable files with:

```bash
git -C ~/projects/cartographer show 0cee299:cartographer/optimize.py
git -C ~/projects/cartographer show 0cee299:cartographer/peak_shape.py
```

Also pruned in the same window and worth surfacing: `MS1TensorResult` (chromatogram.py), `rt_range` predicate pushdown (spectra.py), windowed MS1 scoring, `HyperEMGPeak`.
