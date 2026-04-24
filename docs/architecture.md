# Architecture

Constellation is organized as a shared **core** of modality-agnostic
primitives with **domain modules** layered on top.

## Core (`constellation.core`)

Follows the biological hierarchy from building blocks to tertiary
structure. Each layer depends only on layers to its left:

```
chem ─► sequence ─► structure ─► {stats, graph, nn} ─► optim
  │         │            │              │               ▲
  └─────────┴────────────┴──────────────┴─► io ◄────────┘
```

| Subpackage | What it owns |
|---|---|
| `core.chem` | Atoms, monoisotopic masses, natural abundances, `Composition` atom-count tensor, isotope distributions, UNIMOD modifications as chemical deltas |
| `core.sequence` | `Alphabet` ABC (canonical + IUPAC degenerate), generic ops, nucleic-acid ops (translate with pluggable codon tables + degenerate-codon-aware reverse complement + ORF finding), protein ops (digestion) |
| `core.structure` | Coordinate frames, torch-native geometry (Kabsch/RMSD), Arrow-backed topology, `Ensemble` primitive unifying NMR models, cryo-EM multi-models, and MD trajectory frames |
| `core.io` | Schema registry with forward-compat cast, `RawReader` ABC + suffix dispatch registry, `Bundle` (primary + companions on disk) |
| `core.stats` | `Parametric` ABC uniting probability densities and peak shapes under a single differentiable interface; losses; units |
| `core.optim` | `DifferentialEvolution` with optional LBFGS polish; optimizer registry |
| `core.nn` | ResNet blocks, MonotonicMLP, transformer scaffolds (all PascalCase) |
| `core.graph` | Generic tree and network primitives |

## Domain modules

Each imports from `core` and from `thirdparty` adapters. **Never
cross-imports to another domain module.**

| Domain | Source package | Notes |
|---|---|---|
| `constellation.massspec` | Cartographer (MS proteomics) | First port target; hosts the "Counter" statistical consolidation |
| `constellation.sequencing` | NanoporeAnalysis | POD5 → protein; Phred codec lives here (feature of sequencing, not DNA) |
| `constellation.codon` | CoLLAGE | Codon optimization via transformer |
| `constellation.structure` | Contour + MD trajectories | Unified under `core.structure.Ensemble`; pandas fully replaced with PyArrow+PyTorch during port |
| `constellation.nmr` | (new) | Placeholder |

Cross-modality workflows (e.g. transcriptomic → proteomic) live as
top-level modules under `constellation/` when they exist — no
`bridges/` or `pipelines/` folder until there are ≥2 such workflows.

## Third-party tools

Optional external binaries/jars/DLLs are discovered by
`constellation.thirdparty.registry.find(name)`:

1. `$CONSTELLATION_<TOOL>_HOME` env var (user override)
2. `third_party/<tool>/current/` (installed by `scripts/install-*.sh`)
3. `shutil.which(<bin>)` (conda/system installs)
4. `ToolNotFoundError` with install-script hint

`constellation doctor` prints a tool-status table.
