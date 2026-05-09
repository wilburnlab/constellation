# CLAUDE.md — `constellation.viz`

This file extends the project-wide rules in [../../CLAUDE.md](../../CLAUDE.md). Read that first.

`viz/` ships the first-party visualization layer. PR 1 delivers an IGV-style focused tool — `constellation viz genome --session DIR` boots a local FastAPI server backed by the existing pipeline parquet outputs and serves an SVG-rendered genome browser SPA. PR 2 will wrap this as one panel inside a `constellation`-launched dashboard (auto-introspected CLI form + xterm.js terminal + embedded IPython); see [docs/plans/viz-and-dashboard.md](../../docs/plans/viz-and-dashboard.md) for the multi-PR plan.

The `[viz]` extras (`fastapi`, `uvicorn[standard]`, `datashader`, `websockets`, `httpx`) gate the runtime; the package skeleton (`tracks/base.py`, `tracks/<kernel>.py`, `server/session.py`) imports cleanly under the base install and is exercised by `tests/test_imports.py`. Modules that need fastapi / datashader live behind `pytest.importorskip` in their own test files.

## Module index

| Module | Role | Status |
|---|---|---|
| `viz.tracks` | Per-modality kernel registry. `TrackKernel` ABC + `@register_track` decorator + `HYBRID_SCHEMA`. Six kernels shipped: `reference_sequence`, `gene_annotation`, `coverage_histogram`, `read_pileup` (hybrid), `cluster_pileup` (hybrid), `splice_junctions`. | shipped |
| `viz.server` | FastAPI app factory, Arrow IPC streaming helper (`arrow_stream`), `Session` discovery (`session.py`), `endpoints/{sessions, tracks}` routes. | shipped |
| `viz.raster` | Datashader → PNG helper for hybrid-mode payloads + `greedy_row_assign` shared between vector and hybrid renderers. **Only sanctioned pandas boundary in the package** (datashader's aggregation requires a pd.DataFrame input). | shipped |
| `viz.cli` | `_build_viz_parser(subs)` mounted from `constellation/cli/__main__.py`; `cmd_viz_genome` lazy-imports uvicorn + the app factory. | shipped |
| `viz.frontend` | Committed TS/Vite source tree under `frontend/src/` plus the `python -m constellation.viz.frontend.build` helper that runs `pnpm install && pnpm build` and emits to `viz/static/<entry>/`. The `static/` tree is git-ignored; release wheels carry the prebuilt bundle. | shipped |
| `viz.runner`, `viz.introspect` | PR 2: subprocess runner + lock + xterm.js streaming; argparse → JSON-schema introspection; sandboxed file picker. | scaffold (not yet present) |

**Currently registered kinds** (the canonical list — `constellation.viz.registered_kinds()`):
`cluster_pileup`, `coverage_histogram`, `gene_annotation`, `read_pileup`, `reference_sequence`, `splice_junctions`.

## Architecture invariants (load-bearing — these shaped PR 1 and constrain PR 2)

1. **Local FastAPI server, Apache Arrow IPC over HTTP** as the canonical wire format. `pa.dataset.dataset(parquet_dir).to_batches(filter=...)` → `pa.ipc.new_stream(...)` → `StreamingResponse(media_type="application/vnd.apache.arrow.stream")`. The browser-side `apache-arrow` JS package decodes zero-copy. Implementation: [server/arrow_stream.py](server/arrow_stream.py) `encode_ipc_stream` + `batches_to_response`. The IPC writer is backed by `io.BytesIO` (not `pa.BufferOutputStream`, which closes on `getvalue()` and breaks incremental flushing).
2. **SVG-only client rendering.** One rendering engine. D3 is the algorithmic toolkit (scales, axes, brushes, transitions). Three view *classes* (per kernel decision):
   - **vector** — pure SVG primitives, every glyph editable in Illustrator
   - **hybrid** — SVG envelope around a Datashader PNG `<image>` for dense layers, vector decorations on top
   - **raster** — direct PNG, reserved for thumbnails / cached previews

   Hybrid is the **interactive default** for dense kernels (`read_pileup`, `cluster_pileup`); the frontend's `Save SVG` button always offers a vector export with a cost-confirm dialog when the resulting file would be expensive (current threshold: ~50K glyphs). The naming-vs-content gap (a hybrid SVG isn't a true vector graphic) is real — internal language distinguishes "vector view" / "hybrid view" / "raster view" classes.
3. **Per-modality kernels with mirror-symmetric Python (data) and TS (rendering) modules.** Each kernel:
   - Owns its visual vocabulary
   - Declares its wire schema as a class-level `pa.Schema`
   - Owns its threshold logic (`vector_glyph_limit`, `vector_bp_per_pixel_limit`)
   - Has a TS counterpart at [frontend/src/track_renderers/<kind>.ts](frontend/src/track_renderers/) that mirrors the Python emit shape
4. **Read-only over parquet.** The viz server never writes pipeline outputs. Long-running compute stays in CLI / notebook; the GUI consumes parquet datasets via `pa.dataset.dataset(...)` + filter pushdown. PR 2's dashboard preserves this rule: it constructs `constellation <subcmd> ...` invocations and shows their stdout in xterm.js — never duplicates compute logic.
5. **One server, many frontends.** The same FastAPI app serves the focused `constellation viz genome` tool, the PR 2 dashboard panel mount, and (via `anywidget`) Marimo / Jupyter cells. Kernels don't know which shell mounted them.
6. **Frontend ships pre-built in release wheels.** `constellation/viz/static/<entry>/` is git-ignored; the build helper at [frontend/build.py](frontend/build.py) (`python -m constellation.viz.frontend.build`) runs pnpm/npm. Build at install time was rejected — see "Frontend distribution" in the plan. Source-tarball installs without prebuilt assets see an actionable error message when hitting `/`.
7. **Session manifest as the GUI's entry point.** `<run>/session.toml` (TOML, parsed via stdlib `tomllib`) is the canonical, shareable artifact. When absent, `Session.discover` walks `<run>` one level deep and infers stages from directory names (`S2_align/`, `S2_cluster/`, `genome/`, `annotation/`, `derived_annotation/`). Discovery never overrides explicit toml; both forms produce the same `Session` dataclass shape.

## Track kernel contract

Defined in [tracks/base.py](tracks/base.py). Subclasses register via the `@register_track` class decorator at module import; importing `constellation.viz` pulls in every shipped kernel module exactly once.

```python
class TrackKernel(ABC):
    kind: ClassVar[str]                 # registry key, must be unique
    schema: ClassVar[pa.Schema]         # vector-mode wire schema
    vector_glyph_limit: ClassVar[int]   # default threshold knobs
    vector_bp_per_pixel_limit: ClassVar[float]

    def discover(self, session: Session) -> list[TrackBinding]: ...
    def metadata(self, binding: TrackBinding) -> dict: ...
    def threshold(self, binding: TrackBinding, query: TrackQuery) -> ThresholdDecision: ...
    def fetch(self, binding, query, mode) -> Iterator[pa.RecordBatch]: ...
    def estimate_vector_cost(self, binding, query) -> int | None: ...  # optional
```

`TrackQuery` carries `(contig, start, end, samples, viewport_px, max_glyphs, force)`. `TrackBinding` is the kernel's instance-level handle (resolved parquet paths, per-track config, label). For `HYBRID` decisions, `fetch` emits a single one-row table matching `HYBRID_SCHEMA` (PNG bytes + extents + n_items + mode); the server's `/api/tracks/{kind}/data` endpoint surfaces the chosen mode in the `X-Track-Mode` response header so the TS renderer branches without inspecting the schema.

The `?force=vector|hybrid` query parameter overrides threshold logic for empirical tuning. Per-kernel thresholds are class attributes — tunable per-deployment without touching the threshold method body.

## Wire-schema convention

Each vector-mode kernel defines its emit schema in its own module (e.g. `COVERAGE_VECTOR_SCHEMA` in `coverage_histogram.py`). The schemas project away upstream `contig_id` (queries are range-scoped to one contig) and widen integer types where renderer math wants float64. The hybrid-mode schema is shared (`HYBRID_SCHEMA` in `tracks/base.py`) — every dense kernel emits to the same one-row shape so the frontend's hybrid layer is generic.

PR-1 wire schemas:

| Kernel | Vector schema fields |
|---|---|
| `reference_sequence` | `position: i64, base: str, step: i32` |
| `gene_annotation` | `feature_id: i64, start: i64, end: i64, strand: str, type: str, name: str?, parent_id: i64?, source: str?` |
| `coverage_histogram` | `start: i64, end: i64, depth: f64, sample_id: i64` |
| `read_pileup` | `alignment_id: i64, read_id: str, ref_start: i64, ref_end: i64, strand: str, mapq: i32, row: i32` |
| `cluster_pileup` | `cluster_id: i64, span_start: i64, span_end: i64, strand: str, n_reads: i32, row: i32, mode: str` |
| `splice_junctions` | `intron_id: i64, donor_pos: i64, acceptor_pos: i64, strand: str, support: i64, motif: str?, annotated: bool?` |

`row` columns (read_pileup, cluster_pileup) are pre-computed server-side via `raster.datashader_png.greedy_row_assign` so vector and hybrid agree on the layout — zooming between modes preserves visual structure.

## Data-source mapping (PR 1)

| Kernel | Reads from |
|---|---|
| `reference_sequence` | `<root>/genome/sequences.parquet` (`SEQUENCE_TABLE`) |
| `gene_annotation` | `<root>/annotation/features.parquet` (preferred) or `<root>/S2_align/derived_annotation/features.parquet` (fallback). Surfaces both as separate bindings when both exist. |
| `coverage_histogram` | `<root>/S2_align/coverage.parquet` (`COVERAGE_TABLE`) |
| `read_pileup` | `<root>/S2_align/alignments/` partitioned dataset (`ALIGNMENT_TABLE`); optional `<root>/S2_align/alignment_blocks/` |
| `cluster_pileup` | `<root>/S2_cluster/clusters.parquet` + `cluster_membership.parquet` (`TRANSCRIPT_CLUSTER_TABLE`, `CLUSTER_MEMBERSHIP_TABLE`) |
| `splice_junctions` | `<root>/S2_align/introns.parquet` (`INTRON_TABLE`) |

Kernels read parquet files via `pq.read_table(path, columns=[...])` or `pa.dataset.dataset(...).scanner(filter=..., columns=[...])`. They do NOT import from `constellation.sequencing` at runtime — schema constants in tests come from sequencing's schemas module, but that's a fixture-construction convenience, not a runtime dependency. The viz layer is intentionally decoupled from the producing modules at the Python level: communication is via parquet on disk.

## Conventions (viz layer)

- **No pandas inside the package, except in `viz.raster.datashader_png`.** Datashader's `Canvas.line()` requires a pandas DataFrame; we localize the import + DataFrame construction to that one module, analogous to the numpy big-endian-decode boundary in `electrophoresis.readers`. Kernels never see pandas.
- **Kernels don't import each other.** Each kernel module is a self-contained unit; shared helpers (contig-id lookup, greedy row assignment) are duplicated per-kernel rather than introducing a `tracks/_helpers.py`. Small price for module independence at this scale.
- **Frontend renderers receive a `(table, mode, ctx)` tuple and clear-then-rewrite the SVG.** No data-binding / diffing — each render call is full-replacement. The host (`GenomeBrowser` widget) decides re-render cadence (60ms debounce on viewport changes).
- **Thresholds default to class attributes.** `read_pileup.vector_glyph_limit = 4_000`; `cluster_pileup.vector_glyph_limit = 6_000`; `gene_annotation.feature_limit = 2_000`. Empirical tuning lives by adjusting the class attribute, not by editing the threshold method body. `?force=` overrides everything.
- **No mutable shared state across requests.** The FastAPI app keeps a per-process `track_bindings_cache: dict[(session_id, kind), list[TrackBinding]]` for discover-result memoization, but no per-request state. Sessions are immutable frozen dataclasses.
- **Endpoint shape is REST-flat, not GraphQL.** `/api/sessions`, `/api/sessions/{id}/manifest`, `/api/sessions/{id}/contigs`, `/api/tracks?session=...`, `/api/tracks/{kind}/metadata`, `/api/tracks/{kind}/data`. The dashboard PR adds `/api/commands`, `/api/cli/schema`, `/api/fs/list` in the same shape.
- **Coordinates are 0-based half-open** internally (matches the upstream pipeline schemas and BED). Frontend display formats them with `d3-format` thousands separators, but the wire stays in canonical bp.
- **TS source uses ES2022 modules + strict TS**. No bundler-specific globals; `vite build` produces a single multi-asset bundle per entry under `static/<entry>/`.
- **Static-bundle gate.** `create_app(...)` mounts `/static/<entry>/` only when the directory exists; otherwise `/` returns a JSON pointer at the build helper. CLI handlers don't fail on missing assets — the user can still hit `/api/...` endpoints (anywidget/Jupyter use cases).

## Testing

- `tests/test_viz_session.py` — Session discovery (TOML + walk + missing-paths drop). 5 tests.
- `tests/test_viz_kernels.py` — Registry contract + coverage_histogram round-trip. 10 tests.
- `tests/test_viz_kernels_extended.py` — Each of the other 5 kernels + datashader helpers. 14 tests.
- `tests/test_viz_server.py` — FastAPI test client; arrow_stream encode/decode; every endpoint route. 17 tests.
- `tests/test_viz_e2e.py` (`@pytest.mark.slow`) — Real uvicorn boot in a background thread; full HTTP round-trip; Arrow IPC decode by the test client.

The whole viz suite runs in <4s under default `pytest`; the slow e2e takes ~1.4s extra. Tests gate on `pytest.importorskip("fastapi" / "datashader")` so they don't pollute the base-install smoke run.

Fixture sessions are built in tmp_path: write a tiny CONTIG_TABLE + SEQUENCE_TABLE under `<root>/genome/`, optionally a `<root>/annotation/features.parquet`, and the per-stage parquet artifacts. `Session.from_root(<root>)` discovers everything.

## Frontend layout

```
constellation/viz/frontend/
├── package.json                   pnpm/npm dependency manifest
├── vite.config.ts                 multi-entry build, base="/static/<entry>/"
├── tsconfig.json                  ES2022 + strict
├── index.genome.html              shell HTML for the genome entry
├── build.py                       `python -m constellation.viz.frontend.build`
└── src/
    ├── main_genome.ts             entry point (mounts GenomeBrowser)
    ├── engine/
    │   ├── arrow_client.ts        apache-arrow IPC fetch + JSON helpers
    │   ├── scales.ts              d3-scale + d3-axis wrappers
    │   ├── viewport_bus.ts        ViewportBus (locus + selection events)
    │   ├── svg_layer.ts           svgEl/clear/ensureSvg primitives
    │   ├── hybrid_layer.ts        decode HYBRID_SCHEMA + <image> mount
    │   ├── interactions.ts        wheel/drag pan-zoom into ViewportBus
    │   └── export.ts              composite-SVG serializer + download
    ├── track_renderers/
    │   ├── base.ts                TrackRenderer interface
    │   ├── index.ts               kind → renderer registry
    │   └── <kind>.ts              one renderer per Python kernel
    └── widgets/
        └── GenomeBrowser.ts       toolbar + ruler + track stack + export
```

## What's deferred (PR 2 + later)

- `viz.runner` (subprocess runner + asyncio.Lock for the single-compute-job rule)
- `viz.introspect` (argparse → JSON schema for the auto-generated dashboard sidebar)
- `viz.server.endpoints.{commands, cli_schema, fs}`
- `frontend/src/dashboard/` (`Sidebar`, `CommandForm`, `Terminal`, `PanelHost`, `StatusBar`, `FilePicker`, `NotebookPanel`)
- Cross-modality coordination types (Vitessce-style; lands once a non-genome modality kernel exists; the `viewport_bus` is already in place to receive selection events)
- Spectrum / structure / phylogeny / NN-activation kernels (separate PRs)
- proBAM / proBED emission (deferred per the project-wide note pending the genome→proteome bridge work)

## Companion plan

[docs/plans/viz-and-dashboard.md](../../docs/plans/viz-and-dashboard.md) — full multi-PR design, including the rationale for PR 1's six-track cut, the dashboard's auto-introspect + curated-overlay scheme, and the frontend distribution decision.
