# CLAUDE.md — Constellation

## Project Purpose

Integrative bioinformatics platform: a shared core of physically grounded primitives (compositions, distributions, optimizers, graphs) with modality modules (mass spectrometry, sequencing, structure, codon optimization, NMR) layered on top. The goal is to connect experimental modalities through common abstractions — mathematical structures exposed once in `core/` and reused everywhere — rather than siloed per-modality code.

Constellation is absorbing four existing lab packages as a **clean rewrite** (no vendored source): **Cartographer** (MS proteomics, first port target), **NanoporeAnalysis** (POD5 → protein), **CoLLAGE** (codon optimization), **Contour** (PDB → MD prep). Chronologer (RT prediction) has already been absorbed into Cartographer.

Current state — `core` shipped (`chem`, `sequence`, `io`, `structure`, `stats`, `optim`, partial `graph`); `massspec.{peptide, library, quant, search, acquisitions, annotation, io}` shipped; `sequencing.transcriptome` S1 demux + S1 reference layer + S2 Mode A genome-guided gene counting + Phase 1 alignment-derived intermediates shipped; `viz` PR 1 (IGV-style genome browser via `constellation viz genome`) + PR 2 (lean v1 dashboard via bare `constellation` — dockview-core splittable dock, auto-introspected CLI forms, xterm.js terminal panels) + PR 3 (embedded-in-dashboard genome browser; single `task` panel kind that transitions in-place compute→terminal / viz→widget; toolbar zoom + Fit + Labels toggle + feature-search; IGV-style contig overview bar; per-feature gene-name labels; ResizeObserver-driven redraw; `GET /api/sessions/{id}/search` for substring + `feature_id` lookup over both reference and derived annotation bundles; reference-sequence per-contig cache + writer `row_group_size=1` so cache-miss decode is bounded by one chromosome instead of the whole genome) shipped. Domain-specific status detail lives in the per-module CLAUDE.md files (see Module index below).

## Architecture invariants (load-bearing — design principles from the PI)

1. **Physically grounded statistical models** with explicit priors — preferred over heuristics.
2. **HUPO-PSI standards compliance is the default** for proteomics-domain serialization, identifiers, and metadata keys. Where a HUPO-PSI standard exists for a concept Constellation handles, the PSI form is the canonical *external* representation; relational Arrow tables remain the canonical *in-memory* shape (PSI-bridge, not PSI-first). In-scope: **ProForma 2.0** (modseq strings — `core.sequence.proforma`), **mzPAF v1.0** (peak annotations — `massspec.annotation.mzpaf`), **mzSpecLib v1.0** (spectral library exchange — `massspec.library.readers.mzspeclib`, Phase 3), **USI** (spectrum identifiers — `massspec.annotation.usi`), **PSI-MS CV** (metadata-key vocabulary — `core.ontology.PSI_MS`, Phase 2). PSI-MS CV accessions (`MS:NNNNNNN`) are the preferred metadata-key form anywhere a CV term exists; the `x.<domain>.<key>` namespace remains the fallback for Constellation-specific keys with no PSI equivalent. Deferred until corresponding readers/writers drive them: **mzML**, **mzIdentML**, **mzTab**, **SDRF-Proteomics**, **PEFF**, **mzPeak** (when ratified), **mzQC**, **proBAM / proBed** (lands with the genome→transcriptome→proteome bridge work). Out of scope: **PSI-MI XML**, **TraML**, **FeatureTab**. New PSI ratifications enter the in-scope set unless a written rationale rejects them.
3. **PyArrow tables in memory, Parquet on disk — and partitioned parquet datasets are the canonical internal handoff between pipeline stages.** No pandas inside the package; pandas exits at the port boundary and stays at the edge. When stage A produces N parallel shards (`<output>/<key>/part-NNNNN.parquet`), stage B opens that directory as a `pa.dataset.dataset(...)` and stream-iterates / group-bys / filters via Arrow's compute kernels — `pa.concat_tables(...)` to a single in-memory table is forbidden between stages because it forces O(dataset) RAM and undoes the per-worker parallelism that produced the shards. Single-file aggregation is reserved for **human-facing terminal outputs**: FASTA, TSV summaries, count matrices small enough to load once for export. Cross-modality table joins (transcriptome demux ↔ massspec library, sequencing reads ↔ structure ensembles) are intended to operate on partitioned datasets via Arrow's vectorized join + group_by — never via Python row-loops over `to_pylist()`.
4. **PyTorch for numerical work** wherever possible — multithreaded performance + optimizers + ML.
5. **Generic, nested abstractions** — trees, networks, composition tensors, distributions, optimizers expose statistical similarities across modalities that are otherwise treated as orthogonal.
6. **Unified treatment of optimizers and distributions.** Peak shapes and probability densities share a single `Parametric` ABC; `DifferentialEvolution` and `torch.optim.*` share one registry.
7. **Close-to-instrument data access.** Provide readers that sit directly above the vendor/instrument format; document the assumptions each processing layer adds.
8. **CLI-first** (single `constellation <subcommand>` binary with thin legacy shims for Cartographer muscle memory); GUI later.
9. **Linux primary, WSL second, macOS best-effort.**
10. **Dual audience** — importable library for Jupyter-notebook lab research + polished external-user tool.

## Import order — project-wide DAG

Load-bearing invariant. Violating it creates circular imports or module-init ordering bugs. Enforced by convention, not linting.

### Inside `core/`

```
chem ─► sequence ─► {graph, nn, optim} ─► stats ─► signal ─► structure
                                                                ▲
                                  io ◄── (leaf, consumed by all)─┘
```

Tertiary structure / dynamics are *consequences* of chemistry + sequence + (graph) topology + thermodynamics + optimization, not building blocks for them — so `core.structure` sits at the bottom of the math chain. Bond graphs route through `core.graph`; non-trivial structural manipulation (alignment, fitting, constrained geometry) wants the objective-function + optimizer machinery in `core.stats` + `core.optim`.

- `chem` — never imports from `sequence`, `structure`, or `io`.
- `sequence` — imports `chem`; never imports `structure`.
- `graph` — generic over node/edge types, independent of `chem` and `sequence`.
- `nn` — downstream math primitives; depends on `torch` only.
- `optim` — leaf-tier; depends on `torch` only. Ships the `Optimizer` and `PopulationOptimizer` Protocols, `LBFGSOptimizer` (production wrapper around `torch.optim.LBFGS` with bounds clamping), `DifferentialEvolution` (vectorized via `vmap` + `functional_call`), and `OPTIMIZER_REGISTRY`. Operates generically on any `nn.Module`.
- `stats` — imports `core.optim` for Protocols and dispatches `Parametric.fit()` on optimizer shape (scalar vs population). May import from `chem`, `sequence`. Do NOT import `structure`.
- `signal` — heuristic 1D-trace operations (baseline, smoothing, peak picking, polynomial calibration). Imports `stats` for parametric peak-fitting helpers; otherwise leaf-style. Distinct from `stats` because the operations have imprecisely-specified priors and don't earn membership in the `Parametric` namespace.
- `structure` — most-downstream core module. May import any of `chem`, `sequence`, `graph`, `stats`, `nn`, `optim` as features land. Currently imports just `chem`, `sequence`, `graph`, `io`; `stats` / `optim` enter when Kabsch / RMSD / refinement land in `core.structure.geometry`.
- `io` — leaf. Format/codec concerns only. Never imports other core. The `AtomTable` schema self-registers with the `core.io` schema registry from `core.structure.atoms`.

### Between layers

```
core  ─►  {massspec, sequencing, structure, codon, nmr,
           chromatography, electrophoresis}  ─►  cli
           (domain modules — never cross-import)             ▲
                                                             │
                                       viz ─────────read parquet only,
                                            never imports a domain module
```

Domain modules import from `core` and from `thirdparty` adapters; they **never cross-import to another domain module**. If a workflow legitimately spans modalities (e.g. the transcriptomic → proteomic pipeline), it lives as a thin top-level script under `constellation/` that imports from the relevant domains — no dedicated `bridges/` or `pipelines/` folder until we have ≥2 such workflows to compare.

`viz` is a peer to the domain modules with a stricter constraint: it consumes the parquet outputs the domain pipelines produce (via `pa.dataset.dataset(path)` + filter pushdown) but never imports a domain module's Python API at runtime. The cli dispatcher lazy-imports `viz.cli` so the `[viz]` extras (fastapi / uvicorn / datashader) only become required when the user actually invokes a viz subcommand. The viz layer is then imported by `cli` like any other domain wiring.

Adding a new module: confirm its place in this DAG before touching imports. Adding a new reader (Bruker `.d`, Sciex `.wiff`, DCD trajectory, ...) follows the reader-subclasses-core.io.readers rule without special handling.

## Module index

| Module | Role | Status | Detail |
|---|---|---|---|
| `constellation.core` | Physically-grounded primitives — chem / sequence / io / structure / stats / optim / signal / nn / graph | shipped + partial | [constellation/core/CLAUDE.md](constellation/core/CLAUDE.md) |
| `constellation.massspec` | MS domain — peptide chemistry / library / quant / search / acquisitions / annotation (mzPAF + USI) / io (EncyclopeDIA) / tokenize | partial | [constellation/massspec/CLAUDE.md](constellation/massspec/CLAUDE.md) |
| `constellation.sequencing` | NanoporeAnalysis replacement — S1 demux + reference layer + S2 Mode A gene counting + Phase 1 alignment-derived intermediates + Phase 2 genome-guided fingerprint clustering shipped; Phase 3 de novo / Phase 4 cross-validation / annotation / polish remain | partial | [constellation/sequencing/CLAUDE.md](constellation/sequencing/CLAUDE.md) |
| `constellation.codon` | CoLLAGE-style codon optimization | scaffold | — |
| `constellation.structure` | Contour replacement; PDB/mmCIF + MD trajectories unified via `core.structure.Ensemble` | scaffold | — |
| `constellation.nmr` | Placeholder | scaffold | — |
| `constellation.chromatography` | HPLC-DAD (Agilent OpenLab `.dx` first; LC-MS hyphenation when MS lands). Reader subclass of `core.io.RawReader`; peak workflow via `core.signal` + `core.stats` EMG | scaffold | [docs/plans/companion-instruments-integration.md](docs/plans/companion-instruments-integration.md) |
| `constellation.electrophoresis` | Capillary electrophoresis — Fragment Analyzer + ProteoAnalyzer (`.raw`, same LabVIEW-derived big-endian format). Ladder calibration via `LadderSpec` registry; physically-grounded models (Slater–Noolandi, reptation) reserved for `electrophoresis.physics` | scaffold | [docs/plans/companion-instruments-integration.md](docs/plans/companion-instruments-integration.md) |
| `constellation.models` | NN architectures assembled from `core.nn` | scaffold | — |
| `constellation.viz` | First-party visualization layer. PR 1: `constellation viz genome --session DIR` (IGV-style focused tool) — FastAPI server, Apache Arrow IPC over HTTP, SVG-only client rendering with vector save, six track kernels (`reference_sequence`, `gene_annotation`, `coverage_histogram`, `read_pileup`, `cluster_pileup`, `splice_junctions`), Datashader-backed hybrid mode for dense layers. PR 2 (shipped): bare `constellation` (or `constellation dashboard`) opens a JupyterLab-style dashboard (dockview-core, vanilla TS) wrapping every CLI subcommand as an auto-introspected soft GUI form + xterm.js terminal, with single-compute-job lock + WS stdio stream. PR 3 (shipped): genome browser embeds into the dashboard via a single `task` panel kind that transitions in-place from form → terminal (compute) or widget (viz) via the `viz_registry` descriptor table; toolbar zoom +/–/Fit + Labels toggle + feature-search dropdown; IGV-style contig overview bar above the ruler; per-feature gene-name labels with stroke-halo legibility; `GET /api/sessions/{id}/search` substring + numeric `feature_id` lookup; reference-sequence per-contig cache + writer `row_group_size=1` so a cache miss decodes one chromosome instead of the whole genome. Deferred follow-ups: embedded IPython panel, sandboxed FilePicker, desktop shortcut generator, advanced-search column picker | partial (PR 1 + 2 + 3 shipped) | [constellation/viz/CLAUDE.md](constellation/viz/CLAUDE.md) |
| `constellation.cli` | `constellation <subcommand>` dispatcher; `doctor`, `transcriptome {demultiplex, align, cluster}`, `reference {import, fetch, summary, validate}`, and `viz genome` wired; others stubbed | partial | — |
| `constellation.thirdparty` | Tool discovery (`registry.find`); EncyclopeDIA adapter registered | partial | — |
| `constellation.data` | Packaged data: `elements.json` (NIST AME2020, full periodic table + per-isotope exact masses), `unimod.json` (1560 entries from upstream UNIMOD XML), `codon_tables.json` (7 NCBI tables as overrides on table 1), `proteases.json` (17 enzymes from ExPASy PeptideCutter), `neutral_losses.json` (4 MS neutral-loss rules: H2O / NH3 / HPO3 / H3PO4 with residue + modification triggers). Raw vendored sources under `data/_raw/` regenerable via `scripts/build-{elements,unimod,codon-tables,proteases,neutral-losses}-json.py` | partial | — |

Active plans live under [docs/plans/](docs/plans/); historical design rationale for shipped work lives under [docs/plans/archive/](docs/plans/archive/).

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

## Conventions (project-wide)

Domain-specific conventions live in [constellation/core/CLAUDE.md](constellation/core/CLAUDE.md), [constellation/massspec/CLAUDE.md](constellation/massspec/CLAUDE.md), and [constellation/sequencing/CLAUDE.md](constellation/sequencing/CLAUDE.md).

- **Functions:** `snake_case`.
- **Classes:** `PascalCase` (always — no legacy carryover). Legacy `snake_case` class names from Cartographer (`resnet_block`, `resnet_unit`) do not come across.
- **Third-party tool env-var convention:** `$CONSTELLATION_<TOOL>_HOME` (uniform across all tools — no `_JAR` / `_DIR` / `_HOME` mixing).
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
