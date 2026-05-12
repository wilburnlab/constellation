# Constellation Visualization + Dashboard — Implementation Plan

## Status

- **PR 1** (`constellation viz genome` — focused IGV-style genome browser) — **SHIPPED** at commit `bd505d6`. Six track kernels; FastAPI server with Apache Arrow IPC streaming; SVG-only client + Datashader hybrid mode. 46 unit tests + 1 slow uvicorn-boot e2e all green; full repo suite remains 1348 passing.
- **PR 1.5** (build-helper `--pack` mode + `constellation viz install-frontend --from <local>`) — **SHIPPED**. 15 install tests + smoke imports + `constellation doctor` row; full repo suite 1363 passing (was 1348), no regressions, no new ruff errors. End-to-end CLI round-trip verified locally. Workstation-build-then-ship-to-cluster path that bypasses npm-on-HPC entirely.
- **PR 1.6** (release CI + URL-fetch addition) — **SHIPPED** (code; first tag push enables the publishing path). Three things landed together: `.github/workflows/release.yml` (tag-push → frontend pack → wheel + sdist build → GitHub Release attachment → OIDC publish to PyPI or TestPyPI based on tag pattern); `install_frontend_from_url(version, entry, force, verify, dest_root, cache_dir, github_owner_repo)` in `constellation/viz/install.py` with on-disk caching at `~/.cache/constellation/viz-frontend/` and a stdlib-only `urllib.request` fetcher; switch to setuptools-scm with PyPI distribution name `constellation-bio` (import path stays `constellation`). 5 new URL-fetch tests + 17 existing install/imports tests all green; full repo suite 1364 passing (was 1363), no regressions.
- **PR 2** (dashboard shell wrapping the CLI as soft GUI) — pending.

## Context

Constellation has a substantial CLI pipeline producing rich parquet datasets (alignments, coverage, derived exons, transcript clusters, peptides, etc.) but no GUI for exploring those outputs. Existing inspection workflows are either ad-hoc Jupyter notebooks or large precomputed file dumps — neither suits the project's breadth or its dual audience (lab scientists + external collaborators).

This work scaffolds a first-party visualization layer + GUI dashboard onto Constellation. The intent is two-fold:

1. **Make the existing pipeline outputs explorable** via an IGV-style genome browser that reads the existing parquet datasets directly. This validates the core viz architecture (server, kernels, rendering, export) on the load-bearing modality.
2. **Make the CLI accessible to mouse-driven users** via a dashboard that wraps every CLI subcommand as a soft-GUI form-and-terminal panel, preserving the CLI-primary invariant (the dashboard never duplicates compute logic — it constructs CLI invocations and shows their output).

The deliverable lands across four PRs (1, 1.5, 1.6, 2; see Status block above). PR 1 ships the IGV-style focused tool standalone. PR 1.5 + PR 1.6 together close the loop on distributing the prebuilt frontend bundle so HPC / source-checkout users never invoke a JS toolchain. PR 2 wraps the genome browser inside a dashboard shell with auto-introspected forms for every CLI subcommand and an embedded IPython terminal.

## Frontend distribution model

The viz layer is a Python package that also serves a small browser SPA. This crosses a language boundary — Python on one side, TypeScript on the other — so the artifact story has more moving parts than a pure-Python package. Key vocabulary, because the rest of this plan assumes it:

- **Frontend source** — TypeScript files under `constellation/viz/frontend/src/` (`GenomeBrowser.ts`, the six `track_renderers/*.ts`, etc.). Committed to git. **Browsers cannot run this directly**: TS isn't browser-runnable, and library imports (`apache-arrow`, `d3-scale`) have to be resolved against `node_modules/`.
- **Frontend bundle** — the compiled output of running Vite on the source: a few `.html` + `.js` + `.css` files emitted to `constellation/viz/static/<entry>/`. **This is what the FastAPI server mounts and what the browser actually loads.** Platform-independent; the same bundle runs on every OS and any modern browser.
- **pnpm** — JS package manager. Role: fetch the libraries declared in `package.json` into `node_modules/`. Same role as `pip` for Python. Required only when building the frontend from source.
- **Vite** — JS build tool. Role: transpile TS → JS, resolve imports, bundle, minify, emit to disk. Same role as a C compiler+linker for native code. Required only when building the frontend from source.

The bundle is **derived state** — purely a function of `(TS source + lockfile)`. We gitignore it for three reasons: (1) keeping derived state in sync with source is a maintenance tax (every frontend PR would need to include a rebuilt bundle blob); (2) bundles are large + binary-ish (hundreds of KB minified) and pollute git history with diffs that aren't real code changes; (3) merge conflicts on minified JS are unmergeable. This matches every modern JS project and every Python tool that ships a frontend (Jupyter, Streamlit, Bokeh, Marimo, Plotly).

**The bundle's runtime location is always the same:** `constellation/viz/static/<entry>/` inside the installed package. Only how it gets there differs by install path:

| Install path | How the bundle arrives | Time | JS toolchain needed? |
|---|---|---|---|
| `pip install constellation-bio[viz]` from PyPI (release wheel) | Baked into the `.whl` by CI; pip extracts it like any other package data | ~30s | **No** |
| `git clone && pip install -e .[viz]` (developer, frontend work) | Built locally via `python -m constellation.viz.frontend.build` | a few minutes once | Yes |
| `git clone && pip install -e .[viz]` (developer, Python only) | Fetched via `constellation viz install-frontend` | ~30s | **No** |
| Source tarball / HPC offline | `constellation viz install-frontend --from <local-tarball>` | ~5s | **No** |
| CI test runs | Not needed — Python tests skip viz endpoints on missing bundle | n/a | n/a |

**The contract the user community gets:** `pip install constellation-bio[viz]` is one command and works on laptops, workstations, and HPC nodes alike. No node, no npm, no TLS-with-registry fights on the cluster. The PyPI distribution name is `constellation-bio` (the bare name was claimed by an unrelated package); the import path remains `import constellation`.

## Design principles (firm)

1. **Local FastAPI server, Apache Arrow IPC over HTTP** as the canonical wire format. `pa.dataset.dataset(...).to_batches(filter=...)` → `pa.ipc.new_stream(...)` → `StreamingResponse(media_type="application/vnd.apache.arrow.stream")`. Browser decodes via the `apache-arrow` JS package (zero-copy where possible).
2. **SVG-only rendering on the client.** One rendering engine. D3 as the algorithmic toolkit (scales, axes, brushes, transitions, layouts).
3. **Three view classes**: `vector` (pure SVG primitives), `hybrid` (SVG envelope around a Datashader PNG `<image>` for dense layers, vector decorations on top), `raster` (PNG only, thumbnails). Hybrid is the **interactive default** for dense kernels; vector is always available via a save button with a cost-confirmation dialog when expensive.
4. **Per-modality kernels** with mirror-symmetric Python (data) and TS (rendering) modules. Each kernel owns its visual vocabulary, declares its data shape, owns its threshold logic for the per-glyph ↔ hybrid switch.
5. **One server, many frontends.** Focused tool (`constellation viz genome --session ...`) and the dashboard share kernels and FastAPI endpoints. Same kernel mounts as a standalone view, a dashboard panel, or an anywidget in Marimo/Jupyter.
6. **Dashboard is a soft wrapper around the CLI.** Click "Run" → form constructs `constellation <subcmd> --arg value` → `subprocess.Popen` → stream stdout/stderr to xterm.js via WebSocket → read exit code → update session manifest. Discipline: **at most one compute job at a time**; visualization panels are unrestricted.
7. **Auto-introspection of the argparse parser** populates the full CLI tree in the dashboard sidebar. A curated overlay (`introspect/curated.json`) surfaces a smaller "common tasks" list. Sidebar uses a **mode-switch toggle** between Common and All Commands; state persists in localStorage; search filters across both.
8. **Session manifest as the GUI's entry point.** Each pipeline stage already writes `<run>/<stage>/manifest.json`. We add an optional top-level `<run>/session.toml` (additive — never overrides per-stage manifests) that names what's available across all stages. If absent, `Session.discover` walks subdirs containing `manifest.json` and infers stages from directory names.
9. **Vector export is first-class** on every track type. Same render path produces both the on-screen view and the exportable file.
10. **anywidget contract** for kernel embedding so the same TS modules work in Marimo/Jupyter and in the SPA shell.
11. **Frontend ships pre-built in the wheel** (`constellation/viz/static/<entry>/...`); source lives under `constellation/viz/frontend/` and is built via `python -m constellation.viz.frontend.build` (shells out to `pnpm install && pnpm build`). The `static/` tree is gitignored and produced in CI on tag push (or by the build helper for local dev).

## Multi-PR scope

**PR 1 — `constellation viz genome` (focused IGV-style tool)** — SHIPPED
- FastAPI server, Arrow IPC streaming, session discovery, six track-type kernels, SVG-only frontend, vector/hybrid mode switching, vector export.
- Standalone subcommand entry point. No dashboard.

**PR 1.5 — Pack helper + local-file install (this document)** — PLANNED
- Build helper extension: `python -m constellation.viz.frontend.build --pack` produces `dist/constellation-viz-frontend-<X.Y.Z>.tar.gz` + `.sha256` sidecar in addition to the in-tree `static/<entry>/` build. Tarball contains the bundle plus a top-level `bundle.json` (version + sha256-of-contents + build timestamp).
- New CLI subcommand: `constellation viz install-frontend --from <tarball>`. **Single operating mode in PR 1.5**: read a local tarball from disk, verify sha256 against an adjacent `.sha256` sidecar (or `--no-verify` escape hatch), extract to `constellation/viz/static/<entry>/`. URL-fetch is deliberately deferred to PR 1.6.
- `constellation viz genome` startup behavior is **unchanged from PR 1** — server boots, `/api/...` works, `/` returns the JSON pointer suggesting the install command when bundle missing. No auto-fetch (predictable on restricted networks).
- `constellation doctor`: extended with a `frontend bundle` row reporting present/absent + bundle version (read from `static/<entry>/bundle.json` after install).
- Stdlib-only Python: `tarfile` for extract, `hashlib` for sha256. No `urllib.request` use in PR 1.5 (no fetch path yet).
- **Immediate value:** developer builds bundle on a workstation, scps the tarball to OSC, runs `--from` on the cluster. Zero node / npm / pnpm calls anywhere in the install path. Solves the user's current OSC blocker without depending on PR 1.6.

**PR 1.6 — Release CI + URL-fetch addition** — SHIPPED
As built (see plan-pr-1-6-unified-blossom.md for the full design):
1. **`.github/workflows/release.yml`** — fires on `v*` tag push. `classify-tag` extracts the version and routes pre-release tags (`vX.Y.Z(rc|a|b)N`) to TestPyPI vs final tags (`vX.Y.Z`) to real PyPI. `build` runs Node 20 + pnpm + Python 3.12, executes `python -m constellation.viz.frontend.build --pack`, then `python -m build` to produce the wheel + sdist (the wheel embeds the prebuilt static via the existing `[tool.setuptools.package-data]` rule). `gh-release` attaches the frontend tarball + sidecar + wheel + sdist to the GitHub Release. `publish-pypi` uses `pypa/gh-action-pypi-publish@release/v1` with OIDC trusted publishing — no API token, just the `pypi`/`testpypi` GitHub environments matching the PyPI pending publishers.
2. **`install_frontend_from_url`** in `constellation/viz/install.py`. URL constructed from `_RELEASE_BASE_SCHEME://{_RELEASE_BASE_HOST}/{owner_repo}/releases/download/v{version}/constellation-viz-frontend-{version}.tar.gz` (host/scheme split into module constants for testability). Caches at `~/.cache/constellation/viz-frontend/` with sidecar always re-fetched (so a re-pushed release invalidates the cache automatically). Dev/local versions (containing `.dev` or `+`) refused before the network call. Stdlib-only `urllib.request`. CLI: `--from PATH` and `--version X.Y.Z` are mutually exclusive; with neither, defaults to `constellation.__version__`. New flags: `--cache-dir`, `--repo OWNER/REPO`.
3. **Packaging**: PyPI distribution name `constellation-bio`; import name unchanged. Version derived from git tags via setuptools-scm (`fallback_version = "0.0.0"` for source-tarball checkouts without git history). `[project.urls]` populated for PyPI.
4. **`.gitleaks.toml`** allowlists the AWS access-token rule for bio-data paths (`constellation/data/*.json`, `scripts/build-*-json.py`, `tests/data/*.{fasta,fa,json}`) — protein and nucleic-acid one-letter codes alias AWS IAM key prefixes by accident.

Tests: 5 new URL-fetch tests in `tests/test_viz_install_url.py` (happy path, cache reuse, 404, dev-version refusal, sha mismatch); 1364 total passing; no regressions.

Pre-release tags (`v0.0.1rc1`) test the full pipeline against TestPyPI before claiming a real PyPI version slot. Both flavors create a GitHub Release with the frontend tarball attached so `install_frontend_from_url` is testable end-to-end against the pre-release.

**PR 2 — `constellation` / `constellation dashboard` (dashboard shell)** — pending
- Adds bare-`constellation` no-arg dispatch to dashboard.
- Sidebar with curated + auto-introspected CLI tree.
- Schema-driven form generator + xterm.js terminal panels for command runs.
- Subprocess runner with single-compute-job lock and WebSocket stdio streaming.
- Embedded IPython panel (subprocess + xterm.js, reuses runner plumbing).
- Mounts the genome browser from PR 1 as a panel; coordination event bus stub for future cross-modality panels.

## PR 1 — detailed scope

### Module layout (PR 1 only; PR 2 additions noted in the next section)

```
constellation/viz/
├── __init__.py                          # re-exports TRACK_REGISTRY, register_track
├── cli.py                               # _build_viz_parser(subs); _cmd_viz_genome handler
├── server/
│   ├── __init__.py
│   ├── app.py                           # FastAPI factory, static mount, lifespan
│   ├── arrow_stream.py                  # batches → IPC StreamingResponse
│   ├── session.py                       # Session dataclass; discover/load
│   └── endpoints/
│       ├── __init__.py
│       ├── sessions.py                  # GET /api/sessions, /api/sessions/{id}/manifest
│       ├── tracks.py                    # GET /api/tracks/{kind}/{data,metadata}
│       └── export.py                    # POST /api/export/svg (server-side composite, optional)
├── tracks/
│   ├── __init__.py                      # registry; @register_track
│   ├── base.py                          # TrackKernel ABC, TrackQuery, TrackBinding, ThresholdDecision
│   ├── reference_sequence.py
│   ├── gene_annotation.py
│   ├── coverage_histogram.py
│   ├── read_pileup.py
│   ├── cluster_pileup.py
│   └── splice_junctions.py
├── raster/
│   ├── __init__.py
│   └── datashader_png.py                # canvas + transfer functions; returns PNG bytes + extent
├── frontend/                            # SOURCE TREE (committed)
│   ├── package.json
│   ├── vite.config.ts                   # multi-entry: genome (PR1), dashboard (PR2)
│   ├── tsconfig.json
│   ├── index.genome.html
│   └── src/
│       ├── main_genome.ts               # PR1 entry
│       ├── engine/
│       │   ├── arrow_client.ts          # apache-arrow IPC fetch + decode
│       │   ├── svg_layer.ts             # vector glyph primitives
│       │   ├── hybrid_layer.ts          # SVG <image> + vector overlay
│       │   ├── scales.ts                # d3 scale shims, axis builders
│       │   ├── interactions.ts          # zoom / brush / pan
│       │   ├── viewport_bus.ts          # event bus for coordination (used by future panels)
│       │   └── export.ts                # serialize SVG; estimate cost; show confirm
│       ├── track_renderers/
│       │   ├── base.ts                  # TrackRenderer interface
│       │   ├── reference_sequence.ts
│       │   ├── gene_annotation.ts
│       │   ├── coverage_histogram.ts
│       │   ├── read_pileup.ts
│       │   ├── cluster_pileup.ts
│       │   └── splice_junctions.ts
│       ├── widgets/
│       │   └── GenomeBrowser.ts         # locus picker + ruler + track stack
│       └── anywidget/
│           └── genome_widget.ts         # anywidget bundle (Marimo/Jupyter)
└── static/                              # BUILT BUNDLE (gitignored, shipped in wheel)
    └── genome/
```

### CLI changes — `constellation/cli/__main__.py`

Three additions in `_build_parser()`, alongside the existing `_build_transcriptome_parser(subs)` and `_build_reference_parser(subs)` calls:

```python
_build_viz_parser(subs)           # PR 1
_build_dashboard_parser(subs)     # PR 2
```

`_build_viz_parser` adds `viz` with one nested subcommand `genome`. Args: `--session DIR` (required), optional ad-hoc `--reference DIR` / `--align-dir DIR` for sessions without a top-level layout, `--port INT` (default 0 = pick free), `--host` (default 127.0.0.1), `--no-browser`. Handler `_cmd_viz_genome` lazy-imports `constellation.viz.server.app` + `uvicorn`, mounts only the `genome` bundle, opens browser via `webbrowser.open` unless suppressed.

The bare-`constellation` no-arg dispatch lands in PR 2 alongside `_build_dashboard_parser`. In PR 1, no-arg invocation prints help (current behavior).

### Track-kernel contract — `constellation/viz/tracks/base.py`

```python
@dataclass(frozen=True)
class TrackQuery:
    contig: str
    start: int
    end: int
    samples: tuple[str, ...]
    viewport_px: int            # client width in pixels
    max_glyphs: int             # client-supplied ceiling (for tuning)

@dataclass(frozen=True)
class TrackBinding:
    session_id: str
    paths: dict[str, Path]      # resolved parquet paths for this track
    config: dict                # per-track config (palette, height, ...)

class ThresholdDecision(StrEnum):
    VECTOR = "vector"
    HYBRID = "hybrid"
    VECTOR_TOO_EXPENSIVE = "vector_too_expensive"

class TrackKernel(ABC):
    kind: ClassVar[str]
    schema: ClassVar[pa.Schema]                  # wire schema for this kernel's data
    @abstractmethod
    def discover(self, session: "Session") -> list[TrackBinding]: ...
    @abstractmethod
    def metadata(self, binding: TrackBinding) -> dict: ...
    @abstractmethod
    def threshold(self, binding: TrackBinding, q: TrackQuery) -> ThresholdDecision: ...
    @abstractmethod
    def fetch(self, binding: TrackBinding, q: TrackQuery) -> Iterator[pa.RecordBatch]: ...
```

For `HYBRID` decisions, `fetch` routes through `raster/datashader_png.py` and emits a one-row `pa.Table` with columns `(png_bytes: binary, extent_x0, extent_x1, extent_y0, extent_y1, mode)`. The TS renderer mirrors this: each module declares `accept(schema, mode)` and a `render(table) → SVG nodes` function. The threshold decision is also surfaced via response header `X-Track-Mode: vector|hybrid` for client-side awareness and tuning.

### Track-kernel implementations (PR 1)

| Kernel | Data source | Default mode | Threshold |
|---|---|---|---|
| `reference_sequence` | `<root>/genome/SEQUENCE_TABLE` (ParquetDir) | vector | window > 5000 bp → server-side decimation to dashes |
| `gene_annotation` | `<root>/annotation/FEATURE_TABLE` (preferred); falls back to `<root>/S2_align/derived_annotation/` | vector | n_features > 2000 in window → collapse to gene-row |
| `coverage_histogram` | `<root>/S2_align/coverage.parquet` | vector | always (single `<path>` per sample) |
| `read_pileup` | `<root>/S2_align/alignments/` + `alignment_blocks/` | hybrid | n_reads > 4000 OR bp_per_pixel > 50 → datashader PNG |
| `cluster_pileup` | `<root>/S2_cluster/clusters.parquet` + `cluster_membership.parquet` | hybrid | n_reads > 6000 OR bp_per_pixel > 50 |
| `splice_junctions` | `<root>/S2_align/introns.parquet` | vector | n_junctions > 1500 → arc-density heatmap fallback |

All thresholds are kernel-class attributes with sensible initial defaults; tunable per-deployment via a `?force=vector|hybrid` query param during empirical tuning.

### FastAPI endpoints (PR 1)

```
GET  /                                        → 302 /static/genome/
GET  /static/genome/...                       → built SPA bundle (genome entry)
GET  /api/health                              → {ok, version, schema_version}
GET  /api/sessions                            → [{id, root, label, stages_present}]
GET  /api/sessions/{id}/manifest              → resolved manifest JSON
GET  /api/sessions/{id}/contigs               → [{contig, length}] from CONTIG_TABLE
GET  /api/tracks                              → list of available track kinds for a session
GET  /api/tracks/{kind}/metadata?session=&binding=
GET  /api/tracks/{kind}/data?session=&binding=&contig=&start=&end=&samples=&viewport_px=&force=
POST /api/export/svg                          → optional server-side composite (defer if client-side suffices)
```

### Session manifest format

Add `<run>/session.toml` (TOML for consistency with project-level conventions):

```toml
schema_version = 1
label = "pichia run 2026-04"
created_at = "2026-05-09T12:00:00Z"

[reference]
genome = "../refs/Pp_chr"           # ParquetDir
annotation = "../refs/Pp_chr/annotation"

[stages.s2_align]
path = "S2_align"
manifest = "S2_align/manifest.json"
samples = ["sample_1", "sample_2"]

[stages.s2_cluster]
path = "S2_cluster"
manifest = "S2_cluster/manifest.json"
```

When absent, `Session.discover` falls back to walking `<root>` one level deep, classifying subdirs by directory-name heuristics + per-stage `manifest.json` content. Use `tomllib` (stdlib in 3.11+) for parsing — no extra dep.

### Tests (PR 1)

- `tests/test_viz_kernel_base.py` — unit tests for the ABC contract + each track kernel against in-memory fixture parquet datasets. Validates wire schema matches `kernel.schema`.
- `tests/test_viz_threshold.py` — table-driven test of threshold decisions per kernel at varying window sizes / read densities.
- `tests/test_viz_session.py` — `Session.discover` against a temp-dir tree mimicking the real `S2_align/` + `S2_cluster/` layout; with-and-without `session.toml`.
- `tests/test_viz_server.py` — `httpx.AsyncClient(app=app)` against `/api/tracks/{kind}/data`; decode the Arrow IPC stream; assert row count and schema.
- `tests/test_imports.py` — extend with `constellation.viz`, `constellation.viz.tracks`, `constellation.viz.server.app`.
- `tests/test_viz_e2e.py` (`@pytest.mark.slow`) — boots a real uvicorn on a free port in a background thread, hits each endpoint end-to-end against a small fixture session, asserts non-empty Arrow streams.

## PR 1.5 — detailed scope

### Why this PR exists

PR 1's frontend bundle is gitignored and produced by `python -m constellation.viz.frontend.build` (which runs `pnpm install && pnpm build`). On developer workstations with open networks this is fine. On HPC clusters — where users actually run omics workflows — it breaks for two unrelated reasons:

- npm dependency resolution is slow / hangs on shared filesystems and restricted egress.
- node's TLS stack against legacy registry mirrors / corporate proxies frequently fails (`ERR_SSL_NO_CIPHER_MATCH` observed on OSC, 2026-05-09).

The standard answer across Python tools with JS frontends (Jupyter, Streamlit, Bokeh, Marimo, Plotly) is: end users get a prebuilt bundle inside the release wheel and never see a JS toolchain. PR 1.6 makes that real. PR 1.5 ships the workstation-build-then-ship path that works **today**, before any release exists. The CLI surface PR 1.5 introduces (`viz install-frontend`) is forward-compatible — PR 1.6 just adds a URL-fetch mode without breaking the `--from` shape.

### Files to add or modify

**New files**

- `constellation/viz/install.py` — `install_frontend_from_tarball(*, local_path, entry, force, verify, dest_root)` core function. Single Python entry point used by both the CLI handler and tests. Verifies sha256 against an adjacent `.sha256` sidecar, extracts via `tarfile.open(..., "r:gz")` to `<dest_root>/<entry>/`, copies the embedded `bundle.json` next to the extracted tree so `doctor` can probe it. Stdlib-only: `tarfile`, `hashlib`, `shutil`, `pathlib`. ~80 LOC. The module is structured so PR 1.6 can drop in a sibling `install_frontend_from_url(...)` function without refactoring.
- `tests/test_viz_install.py` — covers (a) `--from <local-tarball>` happy path against a fixture tarball built in tmp_path, (b) sha256 mismatch raises with a clear message, (c) missing local file raises, (d) target-directory-non-empty without `--force` raises with hint, (e) `--force` clears existing contents and re-extracts cleanly, (f) bundle.json metadata round-trips correctly into the on-disk install, (g) `--no-verify` skips checksum and emits a warning. No network mocking needed in PR 1.5.

**Modified files**

- `constellation/viz/cli.py` — add a nested subcommand `install-frontend` alongside the existing `genome` parser inside `build_parser(subs)`. Args (PR 1.5): `--from PATH` (**required** in PR 1.5; PR 1.6 makes it optional), `--entry NAME` (default `genome`), `--force`, `--no-verify`. Handler `cmd_viz_install_frontend` lazy-imports `constellation.viz.install.install_frontend_from_tarball` and dispatches.
- `constellation/viz/frontend/build.py` — add `--pack` flag. When set, after the existing `pnpm build` call, the helper:
  1. Reads `constellation.__version__` to determine the artifact name.
  2. Generates `dist/constellation-viz-frontend-<version>.tar.gz` containing the `static/<entry>/` tree, plus a top-level `bundle.json` (`{"constellation_version": "...", "entry": "...", "built_at": "...", "contents_sha256": "..."}`).
  3. Generates `dist/constellation-viz-frontend-<version>.tar.gz.sha256` sidecar (single line: `<hex>  <filename>` — GNU coreutils format, compatible with `sha256sum -c`).
- `constellation/cli/__main__.py::_cmd_doctor` — add a `frontend bundle` row reading `constellation/viz/static/<entry>/bundle.json` if present (status: ok with version), else `not installed` with hint `run: constellation viz install-frontend --from <tarball>`. Default check is the `genome` entry; future entries (PR 2's `dashboard`) get rows added the same way.
- `tests/test_imports.py` — add `import constellation.viz.install`.
- `constellation/viz/CLAUDE.md` — extend the "Frontend distribution" section with the install paths matrix from the Context section above, plus the workstation-build-then-ship workflow concretely.

### Reused existing utilities (do NOT reimplement)

- `scripts/install-encyclopedia.sh` lines 19–101 — sha256-verification pattern; precedent for how third-party asset installs are scripted in this repo. PR 1.5 brings the same shape into Python.
- `constellation/viz/cli.py::build_parser` (lines 21–64) — existing pattern for adding a nested subcommand under `viz`.
- `constellation/viz/frontend/build.py::build` — existing build flow; `--pack` is an additive post-step that reuses the same path resolution (`_FRONTEND_DIR.parent / "static" / entry`).
- `constellation/cli/__main__.py::_cmd_doctor` — existing tool-status table; new row follows the same `(name, status, version, location)` tuple shape used by `thirdparty/registry`.

### Bundle layout

Inside `constellation-viz-frontend-<X.Y.Z>.tar.gz`:

```
constellation-viz-frontend-<X.Y.Z>/
├── bundle.json                    # {constellation_version, entry, built_at, contents_sha256}
└── static/
    └── genome/                    # (or whatever --entry was; usually 'genome')
        ├── index.genome.html
        ├── assets/
        │   ├── main_genome-<hash>.js
        │   └── main_genome-<hash>.css
        └── ...
```

After `install_frontend_from_tarball` extracts, the on-disk layout under `constellation/viz/static/genome/` is the contents of the `static/genome/` subtree; `bundle.json` is copied alongside it (`constellation/viz/static/genome/bundle.json`) so `doctor` can probe what's installed without re-opening the tarball.

The sidecar `<tarball>.sha256` lives next to the tarball on disk (same directory). When the user passes `--from /path/to/foo.tar.gz`, the installer also reads `/path/to/foo.tar.gz.sha256` and verifies the digest. `--no-verify` skips this check and prints a one-line warning to stderr.

### CLI surface (PR 1.5)

```
constellation viz install-frontend --from /path/to/tarball.tar.gz
constellation viz install-frontend --from /path/to/tarball.tar.gz --entry genome --force
constellation viz install-frontend --from /path/to/tarball.tar.gz --no-verify   # warning to stderr
```

(PR 1.6 will add `--version X.Y.Z` and a no-flags default that fetches the matching release.)

Failure modes the PR 1.5 handler must handle gracefully:

- `--from` path doesn't exist → exit non-zero with the path printed.
- sha256 mismatch → exit non-zero with expected vs actual digests + path to the downloaded file (left on disk for inspection).
- sidecar `.sha256` missing → exit non-zero with hint about `--no-verify`.
- target directory non-empty without `--force` → exit non-zero with the existing bundle's version + `--force` hint.

### Verification plan

1. `pip install -e ".[viz,dev]"` (no new extras needed for PR 1.5).
2. `pytest tests/test_viz_install.py` — fully self-contained; uses tmpfile fixtures to build a tiny fake tarball + sha256 sidecar and round-trip them through the installer. No live network.
3. Build a real bundle on a workstation: `python -m constellation.viz.frontend.build --pack` → verify `dist/constellation-viz-frontend-0.0.0.tar.gz` + `.sha256` exist; `tar -tzf dist/*.tar.gz | head` shows the expected layout including `bundle.json`.
4. Local install round-trip on workstation: `rm -rf constellation/viz/static/genome && constellation viz install-frontend --from dist/constellation-viz-frontend-0.0.0.tar.gz` → verify `static/genome/index.genome.html` + `bundle.json` exist.
5. `constellation doctor` — verify `frontend bundle` row reports `ok` + version after install, `not installed` before.
6. `constellation viz genome --session <fixture> --no-browser` — verify `/` now serves the SPA (not the JSON pointer).
7. **OSC validation:** scp `dist/*.tar.gz` + `.sha256` to OSC; run `constellation viz install-frontend --from <path>` on the cluster; confirm no node / npm / pnpm invocations anywhere; open the served URL and confirm the genome browser renders.
8. Negative tests at the CLI: corrupt the tarball, verify sha256-mismatch error fires with both digests printed; delete the sidecar, verify the missing-sidecar error fires with the `--no-verify` hint; run with `--no-verify` on a corrupt tarball, verify warning to stderr + extraction proceeds.

### Out of scope for PR 1.5

- URL fetch (PR 1.6 — lands together with the release CI that makes URLs valid).
- Release CI itself (PR 1.6).
- Auto-fetch on `viz genome` startup — explicitly rejected; predictable behavior on restricted networks.
- Conda-forge packaging.
- Cryptographic signing of the bundle (only sha256 integrity check via the sidecar). If upstream supply-chain requirements emerge, sigstore / minisign verification lands as a later PR.
- Partial / incremental bundle updates — full replace via `--force` is fine for a few-MB asset.

## PR 2 — detailed scope

### Module additions (on top of PR 1)

```
constellation/viz/
├── cli.py                               # add _build_dashboard_parser, _cmd_dashboard
├── server/endpoints/
│   ├── commands.py                      # POST /api/commands; WS /api/commands/{id}/stream; DELETE
│   ├── cli_schema.py                    # GET /api/cli/schema (introspected)
│   └── fs.py                            # GET /api/fs/list (sandboxed file picker; WSL-aware)
├── runner/
│   ├── __init__.py
│   ├── lock.py                          # asyncio.Lock — single-compute-job enforcement
│   ├── runner.py                        # subprocess.Popen + stdio pumps
│   └── registry.py                      # in-memory job registry; cleanup hooks
├── introspect/
│   ├── __init__.py
│   ├── walk.py                          # _build_parser() → JSON schema tree
│   ├── schema.py                        # JSON shape definitions
│   └── curated.json                     # hand-edited "common tasks" overlay
└── frontend/src/
    ├── main_dashboard.ts                # PR2 entry
    └── dashboard/
        ├── App.tsx
        ├── Sidebar.tsx                  # mode-switch toggle Common ↔ All
        ├── CommandForm.tsx              # auto-generated from cli/schema JSON
        ├── Terminal.tsx                 # xterm.js wrapper + WS stdio
        ├── PanelHost.tsx                # tab/dock manager (dockview-react)
        ├── StatusBar.tsx
        ├── FilePicker.tsx               # browser-side over /api/fs/list
        └── NotebookPanel.tsx            # IPython subprocess via reused runner
```

### CLI changes (PR 2)

`_build_dashboard_parser` adds `dashboard` subcommand: `--port`, `--host`, `--no-browser`, `--root DIR` (sandboxed root for the file picker; defaults to `$HOME`).

Modify `main()` to handle bare invocation:

```python
def main(argv=None):
    raw = list(sys.argv[1:] if argv is None else argv)
    if not raw:
        raw = ["dashboard"]
    parser = _build_parser()
    args = parser.parse_args(raw)
    return args.func(args)
```

### CLI introspection — `introspect/walk.py`

Recursive walk of `_build_parser()._actions`. Descend `_SubParsersAction.choices` for nested subcommands. Per `argparse.Action`, extract `dest`, `option_strings`, `type`, `default`, `choices`, `required`, `help`, `nargs`, `metavar`. Map argparse types to a small set of UI types: `int`, `float`, `str`, `path` (dest containing `dir`/`path`/`file`), `flag`, `enum` (when `choices` set), `multi` (when `nargs in ('*','+',int>1)`). Unknown types → `string` with tooltip note. Emits a stable JSON tree:

```json
{
  "name": "constellation",
  "subcommands": [
    {"name": "viz", "subcommands": [{"name": "genome", "args": [...]}]},
    {"name": "transcriptome", "subcommands": [{"name": "align", "args": [...]}, ...]},
    ...
  ]
}
```

A curated overlay (`introspect/curated.json`) marks specific args with richer metadata (file-pickers vs free-text paths, "advanced" group hiding, default-value enrichment) without changing the underlying execution path. Curated entries are deep-links into the same form components.

### Subprocess runner

`runner/runner.py` spawns `python -m constellation.cli.__main__ <subcmd> ...` (NOT the installed entry point — preserves venv correctness). stdout/stderr piped, line-buffered. Two `asyncio.Queue`s feed the WS handler. SIGTERM on cancel; force-kill timer escalates to SIGKILL after grace period.

`runner/lock.py` holds a single global `asyncio.Lock`. Acquired in the POST `/api/commands` handler before spawn; released in `try/finally` regardless of how the WS terminates. Visualization endpoints never touch this lock.

`runner/registry.py` keys jobs by UUID, holds `JobHandle(proc, stdout_q, stderr_q, exit_code, started_at)`. Cleanup on app shutdown via FastAPI lifespan.

### Frontend dashboard

- **Tab/dock manager**: `dockview-react`. Each panel has a `kind` discriminator: `"command" | "viz" | "notebook"`.
- **`Sidebar.tsx`**: mode-switch toggle button at top flips between "Common" (curated) and "All commands" (full tree). Search input filters the visible list. Toggle state persists in `localStorage`. Curated entries deep-link into the same form panel as their full-tree counterparts.
- **`CommandForm.tsx`**: pure schema-driven. Argument groups → collapsible sections. File-path args → `FilePicker`. On Run: POST `/api/commands`, opens a new `Terminal` panel. Run button globally greyed when any compute job is running (subscribes to `/api/commands/active` via WS).
- **`Terminal.tsx`**: `xterm.js` + `xterm-addon-fit`. WS frames pass through directly; ANSI handled natively. Cancel button → DELETE `/api/commands/{id}`. Header shows exit code + elapsed time.
- **`NotebookPanel.tsx`**: opens an IPython subprocess via the same runner module (`python -m IPython --simple-prompt`). Pipes through xterm.js identically. Constellation env preloaded; users can `from constellation.viz.tracks.read_pileup import ReadPileupKernel` directly.
- **`StatusBar.tsx`**: current session, currently-running job + elapsed timer, error toasts.
- **WSL path handling**: `FilePicker` accepts pasted Windows paths (`C:\foo`) and normalizes via `wslpath -u` (when available) or direct rewrite `C:\` → `/mnt/c/`. Server endpoint `/api/fs/list` surfaces `/mnt/c`, `/mnt/d`, ... as sibling roots when `os.uname().release` contains "microsoft".

### Tests (PR 2)

- `tests/test_viz_introspect.py` — round-trip: build the real CLI parser, walk it, assert every shipped subcommand appears with at least its required args.
- `tests/test_viz_runner.py` — spawn a tiny test command through the runner; assert stdout/stderr collected and exit code propagated.
- `tests/test_viz_runner_lock.py` — concurrent attempts to start two jobs; second is rejected with HTTP 409.
- `tests/test_viz_dashboard_endpoints.py` — FastAPI test client against `/api/cli/schema`, `/api/fs/list`, `/api/commands` (with mocked subprocess).

## Critical files to be modified or created

- [constellation/cli/__main__.py](constellation/cli/__main__.py) — extend `_build_parser` and `main`; add `_build_viz_parser` + `_build_dashboard_parser` + handlers
- [constellation/viz/](constellation/viz/) — entire new package (PR 1 then PR 2 additions)
- [pyproject.toml](pyproject.toml) — add `[project.optional-dependencies] viz = ["fastapi", "uvicorn[standard]", "datashader", "websockets", "PyYAML"]`; add `package_data` / `MANIFEST.in` rule for `constellation/viz/static/**/*`
- [tests/test_imports.py](tests/test_imports.py) — add `constellation.viz` and submodule imports
- [tests/test_viz_*.py](tests/) — new test files per the lists above

### Existing utilities to reuse (do NOT reimplement)

- [constellation/sequencing/quant/coverage.py](constellation/sequencing/quant/coverage.py) — `build_pileup` writes the COVERAGE_TABLE consumed by `coverage_histogram` kernel
- [constellation/sequencing/quant/genome_count.py](constellation/sequencing/quant/genome_count.py) — emits ALIGNMENT_TABLE / ALIGNMENT_BLOCK_TABLE consumed by `read_pileup` kernel
- [constellation/sequencing/quant/junctions.py](constellation/sequencing/quant/junctions.py) — emits INTRON_TABLE consumed by `splice_junctions` kernel
- [constellation/sequencing/quant/derived_annotation.py](constellation/sequencing/quant/derived_annotation.py) — emits derived FEATURE_TABLE consumed by `gene_annotation` fallback path
- [constellation/sequencing/transcriptome/cluster_genome.py](constellation/sequencing/transcriptome/cluster_genome.py) — emits TRANSCRIPT_CLUSTER_TABLE / CLUSTER_MEMBERSHIP_TABLE consumed by `cluster_pileup` kernel
- [constellation/sequencing/reference/io.py](constellation/sequencing/reference/io.py) — `load_genome_reference` for the `reference_sequence` kernel
- [constellation/sequencing/annotation/io.py](constellation/sequencing/annotation/io.py) — `load_annotation` for the `gene_annotation` kernel
- [constellation/core/io/](constellation/core/io/) — Arrow schema registry; reuse for schema discovery and validation
- [constellation/thirdparty/registry.py](constellation/thirdparty/registry.py) — registration pattern to mirror in `viz/tracks/__init__.py`

## Verification plan

### PR 1

1. `pip install -e .[viz,sequencing,dev]`
2. Build a fixture session in `/tmp` from existing test data: extend the patterns in `tests/test_quant_coverage.py` to write a tiny `coverage.parquet`, partitioned `alignments/` + `alignment_blocks/`, an `annotation/` ParquetDir, and a top-level `session.toml`.
3. `pytest tests/test_viz_*.py -v` — all green; `pytest -m slow` exercises the e2e uvicorn boot.
4. `constellation viz genome --session /tmp/fixture` — browser opens; bundle loads from `/static/genome/`; coverage track renders within 1s; pileup renders both vector (zoom in <1kb) and hybrid (zoom out to >100kb); the `X-Track-Mode` header flips at the threshold boundary.
5. `constellation viz genome --session <real Pichia run>` — eyeball pileup against IGV on the same BAM; confirm gene annotations align with reference; confirm splice junctions in the introns track match those in `introns.parquet`.
6. Save-as-vector at hybrid threshold; verify the resulting SVG renders correctly in Inkscape and the cost-confirm dialog appeared.
7. `pytest tests/test_imports.py` — confirms smoke imports stay green.

### PR 2

1. `constellation` (no args) → dashboard opens at a free localhost port; URL printed to stdout.
2. Sidebar "Common" mode shows curated entries; toggle to "All commands" → full tree includes every subcommand from `_build_parser`. Search filters across both modes.
3. Click "Run alignment" → form auto-generated; run against test fixture; terminal streams stdout; exit code 0; status bar clears; `Open genome view` action surfaces the new session.
4. Concurrent run attempt → second blocked with toast; HTTP 409 from `/api/commands`.
5. Cancel mid-run → SIGTERM delivered; subprocess exits within grace period; UI re-enables Run buttons.
6. WSL: paste `C:\Users\me\data` into a path field → resolves to `/mnt/c/Users/me/data` and listing populates.
7. Open Notebook panel → IPython prompt; `from constellation.viz.tracks.read_pileup import ReadPileupKernel` works; close panel cleanly stops subprocess.
8. `pytest tests/test_viz_*.py -v` for the new dashboard test files.

## Risks and architectural questions

1. **Frontend distribution.** Pre-built bundle in the wheel is the recommended path; Vite at install time would force every install to have node + pnpm. CI builds `static/` on tag push. Local dev uses `python -m constellation.viz.frontend.build` (shells `pnpm install && pnpm build`). Add a startup gate that prints an actionable error when `static/<entry>/index.html` is missing. The `[viz]` extras gate ensures `fastapi` etc. are installed.
2. **Performance threshold tuning.** Initial defaults are educated guesses; will need empirical tuning against real datasets. `?force=vector|hybrid` query param + `X-Track-Mode` response header support iteration without code changes. Per-kernel defaults are class attributes.
3. **Lock liveness on crash.** Wrap lock acquisition in `try/finally` around the WS lifetime; runner detects orphaned jobs at startup (clears stale registry entries via the FastAPI lifespan hook).
4. **Arrow IPC chunking.** Default `to_batches(batch_size=64K rows)`. Column-prune at the kernel layer (only request columns the renderer uses). Cap max rows per response with a server-side limit + a "result truncated" flag in trailing metadata; the renderer surfaces a small warning indicator.
5. **Cross-modality coordination layer (deferred).** PR 2 ships a `viewport_bus` (`new EventTarget()`) on the frontend; kernels publish locus / selection events. Cross-track sync (peptide selected → genome track scrolls) lands in a later PR but the wire is in place.
6. **Browser autolaunch.** `webbrowser.open(url)` handles Linux/macOS/WSL acceptably (WSL falls through to `wslview` or `cmd.exe /c start`). `--no-browser` always available; URL printed to stdout regardless.
7. **anywidget contract.** `widgets/GenomeBrowser.ts` exposes `mount(el, props)` consumed by both the SPA and the anywidget bundle. Keeps Marimo/Jupyter and the dashboard sharing renderer code verbatim.

## Out of scope for this plan

- Spectrum / structure / phylogeny / NN-activation kernels (separate PRs once IGV architecture proves out)
- Cross-modality coordination types (Vitessce-style; later PR after multiple modality kernels exist)
- Authentication, multi-user hosting (localhost-only)
- Bookmarks, saved views, shareable URLs (defer; URL state machine sufficient for v1)
- proBAM / proBED emission for the genome→proteome bridge (deferred per CLAUDE.md until that bridge work lands)
- Desktop-shortcut auto-installer (`constellation install-shortcut`) — easy follow-up; not blocking v1
