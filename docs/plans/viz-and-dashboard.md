# Constellation Visualization + Dashboard — Implementation Plan

## Status

- **PR 1** (`constellation viz genome` — focused IGV-style genome browser) — **SHIPPED**. Six track kernels (`reference_sequence`, `gene_annotation`, `coverage_histogram`, `read_pileup`, `cluster_pileup`, `splice_junctions`); FastAPI server with Apache Arrow IPC streaming; Datashader-backed hybrid mode; SVG-only client rendering with vector save. 46 unit tests + 1 slow uvicorn-boot e2e — all green; full repo suite remains 1348 passing, no regressions, no new ruff errors.
- **PR 2** (dashboard shell wrapping the CLI as a soft GUI; no-arg `constellation` opens it) — pending. Includes the auto-introspected CLI tree + curated overlay, the subprocess runner with single-compute-job lock, an embedded IPython panel, and the `dockview-react` panel host. The IGV browser from PR 1 mounts as a panel inside the dashboard without changes to its kernel architecture.

## Context

Constellation has a substantial CLI pipeline producing rich parquet datasets (alignments, coverage, derived exons, transcript clusters, peptides, etc.) but no GUI for exploring those outputs. Existing inspection workflows are either ad-hoc Jupyter notebooks or large precomputed file dumps — neither suits the project's breadth or its dual audience (lab scientists + external collaborators).

This work scaffolds a first-party visualization layer + GUI dashboard onto Constellation. The intent is two-fold:

1. **Make the existing pipeline outputs explorable** via an IGV-style genome browser that reads the existing parquet datasets directly. This validates the core viz architecture (server, kernels, rendering, export) on the load-bearing modality.
2. **Make the CLI accessible to mouse-driven users** via a dashboard that wraps every CLI subcommand as a soft-GUI form-and-terminal panel, preserving the CLI-primary invariant (the dashboard never duplicates compute logic — it constructs CLI invocations and shows their output).

The deliverable lands in two PRs. PR 1 ships the IGV-style focused tool standalone. PR 2 wraps it in a dashboard shell with auto-introspected forms for every CLI subcommand and an embedded IPython terminal.

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

## Two-PR scope

**PR 1 — `constellation viz genome` (focused IGV-style tool)**
- FastAPI server, Arrow IPC streaming, session discovery, six track-type kernels, SVG-only frontend, vector/hybrid mode switching, vector export.
- Standalone subcommand entry point. No dashboard.

**PR 2 — `constellation` / `constellation dashboard` (dashboard shell)**
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
