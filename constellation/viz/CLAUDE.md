# CLAUDE.md — `constellation.viz`

This file extends the project-wide rules in [../../CLAUDE.md](../../CLAUDE.md). Read that first.

`viz/` ships the first-party visualization layer. The standalone IGV-style genome browser boots via `constellation viz genome --reference <handle> --align-dir DIR [...] [--cluster-dir DIR ...]` or `--saved-session <slug>`; the dashboard shell (bare `constellation`) launches a JupyterLab-style soft GUI that wraps every CLI subcommand as a form + xterm.js terminal, splittable via dockview-core panes. The single `task` panel transitions in-place — sidebar entries open a form; compute commands swap that form for an xterm terminal on Run; viz commands swap it for the actual widget on Open. The genome-browser entry form is reference-cache-first: the user picks a reference from the `~/.constellation/references/` dropdown, then attaches one or more `transcriptome align` / `cluster` output dirs as "results" rows. Once a session is open, a Datasets toolbar popover lets the user toggle per-binding visibility, drag-to-reorder rows, collapse/hide individual tracks, and add or remove sources on the fly without reopening the browser. Each track's gear (⚙) opens a per-binding settings popover for color / palette / font / opacity / dataset-slice filters (publication-quality SVG export), and a sibling toolbar Options dropdown holds browser-wide preferences like SVG clip-to-viewport. Both SPAs share one FastAPI app. See [docs/plans/viz-and-dashboard.md](../../docs/plans/viz-and-dashboard.md) for the multi-PR plan; the reference-cache-first cutover is documented at [docs/plans/reference-first-genome-browser.md](../../docs/plans/reference-first-genome-browser.md) (TBD).

The `[viz]` extras (`fastapi`, `uvicorn[standard]`, `datashader`, `websockets`, `httpx`) gate the runtime; the package skeleton (`tracks/base.py`, `tracks/<kernel>.py`, `server/session.py`, `introspect/`, `runner/`) imports cleanly under the base install and is exercised by `tests/test_imports.py`. Modules that need fastapi / datashader live behind `pytest.importorskip` in their own test files.

## Module index

| Module | Role | Status |
|---|---|---|
| `viz.tracks` | Per-modality kernel registry. `TrackKernel` ABC + `@register_track` decorator + `HYBRID_SCHEMA`. Six kernels shipped: `reference_sequence`, `gene_annotation`, `coverage_histogram`, `read_pileup` (hybrid), `cluster_pileup` (hybrid), `splice_junctions`. | shipped |
| `viz.server` | FastAPI app factory, Arrow IPC streaming helper (`arrow_stream`), `Session` discovery (`session.py`), `endpoints/{sessions, tracks, saved_sessions, references}` routes. `endpoints/sessions.py` also serves `GET /api/sessions/{id}/search?q=…&limit=…` — case-insensitive substring match on `name` (plus numeric `feature_id` when `q` parses as int) across both reference and derived annotation parquets, returning `{feature_id, name, type, strand, contig_name, start, end, source: 'reference'\|'derived'}` rows. PR 5 added the runtime source-mutation endpoints `POST /api/sessions/{id}/sources` + `DELETE /api/sessions/{id}/sources/{source_id}` (rebuild the Session via `Session.with_sources()` under the same `session_id`, then evict the binding cache via `invalidate_binding_cache()`), and `PATCH /api/saved-sessions/{slug}/layout` (rewrites just the `[[track_layout]]` block). PR 6 extended `POST /api/saved-sessions` and `PATCH /api/saved-sessions/{slug}/layout` to additionally accept optional per-entry `style` + `filter` dicts and a top-level `options` dict; `GET /api/saved-sessions/{slug}` surfaces them in the response payload. Missing `options` on the PATCH payload preserves the on-disk value (it's a per-key replace at the block level, not a deep merge). | shipped |
| `viz.raster` | Datashader → PNG helper for hybrid-mode payloads + `greedy_row_assign` shared between vector and hybrid renderers. **Only sanctioned pandas boundary in the package** (datashader's aggregation requires a pd.DataFrame input). | shipped |
| `viz.cli` | `build_parser(subs)` mounts the `viz` subtree from `constellation/cli/__main__.py`. Two subcommands shipped: `viz genome` (lazy-imports uvicorn + the app factory) and `viz install-frontend` (lazy-imports `viz.install`). | shipped |
| `viz.install` | `install_frontend_from_tarball(local_path, entry, force, verify, dest_root)` + `install_frontend_from_url(version, entry, force, verify, dest_root, cache_dir, github_owner_repo)` + `read_bundle_metadata(dest_dir)`. Stdlib-only — `tarfile` / `hashlib` / `shutil` / `urllib.request`. PR 1.6 added URL-fetch (default path; cache at `~/.cache/constellation/viz-frontend/`); PR 1.5's local-tarball path remains as `--from PATH` and the HPC offline escape hatch. | shipped |
| `viz.frontend` | Committed TS/Vite source tree under `frontend/src/` plus the `python -m constellation.viz.frontend.build` helper. Default: builds **every entry discovered under `frontend/index.*.html`** (currently `genome` + `dashboard`); pass `--entry NAME [NAME ...]` to narrow for fast iteration. `--pack` produces `<repo>/dist/constellation-viz-frontend-<version>.tar.gz` + `.sha256` containing every built entry. `known_entries()` is the discovery helper. The `static/` and `dist/` trees are git-ignored. | shipped |
| `viz.introspect` | PR 2: walks the production argparse parser via `walk_parser(_build_parser())` and emits a JSON tree the dashboard renders into auto-generated forms. `curated.json` is the hand-edited "Common" overlay. Type mapping: `BooleanOptionalAction`/`store_true` → flag, `choices=` → enum, `nargs in ('+','*',N>1)` → multi, `dest` path-suffix heuristic → path. | shipped |
| `viz.runner` | PR 2: spawns `python -m constellation.cli.__main__ <argv>` via `asyncio.create_subprocess_exec`, fans stdout/stderr into a per-job history deque + subscriber queues. `lock.py` holds a `threading.Lock` enforcing the single-compute-job rule; `acquire_or_409` is non-blocking and maps to HTTP 409 at the endpoint. Lock release happens in the pump task's `finally` so a stuck subscriber never wedges the dashboard. | shipped |
| `viz.server.endpoints.{cli_schema,commands}` | PR 2: `GET /api/cli/schema` (cached for the process lifetime); `POST /api/commands` (acquires lock, spawns, returns job_id), `GET /api/commands/active`, `GET /api/commands/{id}`, `DELETE /api/commands/{id}` (SIGTERM with 10s grace), `WS /api/commands/{id}/stream` (replays history then live frames; sends `{stream:'exit',line:exit_code}` sentinel before closing). | shipped |
| `viz.server.endpoints.fs` | Deferred: sandboxed file picker + WSL path normalization. | not yet present |

**Currently registered kinds** (the canonical list — `constellation.viz.registered_kinds()`):
`cluster_pileup`, `coverage_histogram`, `gene_annotation`, `read_pileup`, `reference_sequence`, `splice_junctions`.

## Architecture invariants (load-bearing — these shaped PR 1 and constrain PR 2–5)

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
7. **Reference cache as the GUI's entry point.** The dashboard's genome-browser entry form always picks a reference first (from `~/.constellation/references/<organism>/<release>/`, surfaced via `GET /api/references`), then attaches one or more "source" directories that the user has produced with `constellation transcriptome align` / `cluster`. Each source's `manifest.json` (schema v2) names the reference handle it was produced against, so the form can warn on assembly mismatch without needing the user to re-spell the relationship. Sessions are not directory-rooted any more — there is no `session.toml` discovery layer, and the legacy `Session.from_root` / `Session.discover` walks were removed in the reference-cache-first cutover. Saved configurations persist to `~/.constellation/sessions/<slug>.toml` (a sibling of the references and catalogs caches) for one-click resume.
8. **One panel kind per command, in-place phase transitions (PR 3).** Sidebar `command:open` always opens a single `'task'` panel via `dashboard/TaskPanel.ts`. Phase 1 is a form — `CommandForm` for compute, `VizForm` for any path matching `viz_registry`. Phase 2 swaps the form's DOM root for either `TerminalPanel` (compute; streams `/api/commands/{id}/stream`) or the viz widget itself (e.g. `widgets/GenomeBrowser`), and the tab title updates accordingly. The welcome page's quick-launch creates the same panel with `autoSubmit: true` so a session-path entry skips straight to phase 2. New viz tools register one descriptor in `dashboard/viz_registry.ts` and never touch the rest of the shell.
9. **Sessions stay frozen; mutation = atomic rebuild + cache evict (PR 5).** `Session` and `SessionSource` remain `@dataclass(frozen=True, slots=True)`. Source mutation does not touch the existing instance — `Session.with_sources(new_sources)` calls back through the validating `Session.open(...)` path and returns a fresh frozen instance, which the endpoint swaps into `app.state.sessions[session_id]` (the `session_id` derives deterministically from `(reference_path, label)` so it's preserved across rebuilds). `invalidate_binding_cache(cache, session_id)` then evicts every `(session_id, kind)` entry in `app.state.track_bindings_cache` so the next `/api/tracks` discovers against the rebuilt source list. Client-side layout state is keyed by `(source_id, kind)` — never `binding_id` — so persisted state survives the rebuild even though kernel `binding_id`s remain index-based.
10. **Per-binding style/filter are opaque to the layout type; renderers own their schema (PR 6).** `TrackLayoutEntry` carries optional `style?: Record<string, unknown>` and `filter?: Record<string, unknown>` dicts that the layout layer never inspects. Each renderer reads keys it knows about via the `track_renderers/style.ts` helpers (`pickPaletteColor`, `pickNumber`, `pickAllowList`, …) with hardcoded defaults as fallback — unset = pre-PR-6 behavior, fully backward-compatible. New style or filter knobs land as renderer-only changes: add a `pick*(ctx.style, 'my_key', fallback)` call and a row in `widgets/TrackSettingsPanel.ts`; no schema migration, no persistence layer change. Client-side re-render (`MountedTrack.lastFetched` Arrow cache + `restyleTrack()`) means style edits don't re-hit the kernel — color picks are instant. Filters that need pushdown to the kernel (e.g. MAPQ threshold, supplementary-flag toggle on `read_pileup`) require extending `TrackQuery` and round-tripping a refetch; deferred until a concrete need lands. SVG export uses `svg.dataset.naturalHeight` (set by every renderer) for panel y-offsets so overflow content auto-grows the export when clip is off, or wraps each panel in a `<clipPath>` when clip is on (the Options dropdown toggle).

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

## Data-source mapping

Kernels iterate `session.sources` (the list of attached `transcriptome align` / `cluster` output dirs) and emit one binding per source. Reference paths come from the session's `reference_genome` / `reference_annotation` slots, populated by `handle.resolve(reference_handle)` against the cache. Per-stage artifact paths inside each source come from that source's `manifest.json` `outputs` map, joined against well-known relative filenames in `session.py::_ALIGN_DEFAULT_PATHS` / `_CLUSTER_DEFAULT_PATHS`.

| Kernel | Reads from |
|---|---|
| `reference_sequence` | `session.reference_genome / "sequences.parquet"` (`SEQUENCE_TABLE`) — written with `row_group_size=1` by `sequencing.reference.io.ParquetDirWriter` so parquet `contig_id` statistics let a `contig_id == X` predicate skip every row group except the matching one. Loaded sequences cached process-wide in `_SEQUENCE_CACHE` keyed by `(sequences_path, contig_id)`. |
| `gene_annotation` | `session.reference_annotation / "features.parquet"` plus one binding per align source's `derived_annotation/features.parquet` — both surfaced separately so users can compare curated against data-derived. |
| `coverage_histogram` | one binding per align source with `coverage.parquet` (`COVERAGE_TABLE`). |
| `read_pileup` | one binding per align source with `alignments/` (`ALIGNMENT_TABLE`); optional `alignment_blocks/` rides along when present. |
| `cluster_pileup` | one binding per cluster source with `clusters.parquet` + `cluster_membership.parquet` (`TRANSCRIPT_CLUSTER_TABLE`, `CLUSTER_MEMBERSHIP_TABLE`). |
| `splice_junctions` | one binding per align source with `introns.parquet` (`INTRON_TABLE`). |

Kernels iterate via the shared `iter_sources_with(session, *attrs)` helper in `tracks/base.py` (yields `(idx, source)` pairs for sources whose listed slots are non-None). `binding_id` is `f"{kind}-{idx}"` so the per-process binding cache stays stable across re-discovery within a single session. PR 5 added a second-tier identifier: each source-derived kernel stamps `config["source_id"] = src.source_id` on the `TrackBinding` (and `/api/tracks` surfaces it as a top-level field on each binding listing). `source_id` is a `blake2b(path|kind)` hash that's stable across `Session.with_sources()` rebuilds — the client persists per-binding layout state keyed by `(source_id, kind)` so the state survives runtime source add/remove even though the index-based `binding_id` shifts. Reference-only bindings (`reference_sequence`, `gene_annotation:reference`) carry `source_id = null`. Kernels read parquet files via `pq.read_table(path, columns=[...])` or `pa.dataset.dataset(...).scanner(filter=..., columns=[...])`. They do NOT import from `constellation.sequencing` at runtime — schema constants in tests come from sequencing's schemas module, but that's a fixture-construction convenience, not a runtime dependency. The viz layer is intentionally decoupled from the producing modules at the Python level: communication is via parquet on disk.

## Conventions (viz layer)

- **No pandas inside the package, except in `viz.raster.datashader_png`.** Datashader's `Canvas.line()` requires a pandas DataFrame; we localize the import + DataFrame construction to that one module, analogous to the numpy big-endian-decode boundary in `electrophoresis.readers`. Kernels never see pandas.
- **Kernels don't import each other.** Each kernel module is a self-contained unit; shared helpers (contig-id lookup, greedy row assignment) are duplicated per-kernel rather than introducing a `tracks/_helpers.py`. Small price for module independence at this scale.
- **Frontend renderers receive a `(table, mode, ctx)` tuple and clear-then-rewrite the SVG.** No data-binding / diffing — each render call is full-replacement. The host (`GenomeBrowser` widget) decides re-render cadence (60ms debounce on viewport changes). `ctx.showLabels` rides through so kernels can branch on host-level UI toggles without owning state.
- **Genome browser owns its own chrome and styles.** Toolbar (contig select / Go-to / zoom +/–/Fit / Labels toggle / feature search / Datasets popover / Save SVG) + IGV-style contig overview bar + ruler + track stack live in `widgets/GenomeBrowser.ts`; styles ship in `widgets/GenomeBrowser.css` scoped under `.genome-browser-root` so the widget mounts cleanly in either the standalone SPA or a dashboard task panel. A `ResizeObserver` on the host re-renders through the existing 60ms debounce when the pane resizes. Per-track controls (drag-grip reorder via HTML5 DnD, collapse chevron, hide eye, bottom-edge pointer-event resize handle) live in the per-track header. The Datasets popover is its own component (`widgets/DatasetManagerPopover.ts`) anchored to the toolbar button; it lists the reference plus every attached source with per-binding checkboxes, per-source remove (✕), and an inline "Add dataset" form that POSTs to `/api/sessions/{id}/sources`. The empty-state placeholder "All tracks hidden — open the Datasets menu to enable some" appears when every binding is hidden.
- **Per-binding layout state is keyed by `(source_id, kind)`** in the client, not by `binding_id`. State (visible / displayOrder / heightPx / collapsed) persists to `localStorage["constellation.genome.layout.<sessionId>"]` debounced 200 ms; when `session.saved_as` is set, the client also PATCHes `/api/saved-sessions/{slug}/layout` so the configuration survives across machines. On reopen, the saved-session loader prefills the form which then passes `initialLayout` to `GenomeBrowser` — localStorage still takes precedence for last-write-wins per device. Default ordering is kind-grouped: `reference_sequence < gene_annotation < coverage_histogram < read_pileup < cluster_pileup < splice_junctions`, ties broken by source insertion order; runtime adds slot new bindings into the tail of their kind cluster; user reorder via drag reassigns displayOrders sequentially.
- **Thresholds default to class attributes.** `read_pileup.vector_glyph_limit = 4_000`; `cluster_pileup.vector_glyph_limit = 6_000`; `gene_annotation.feature_limit = 2_000`. Empirical tuning lives by adjusting the class attribute, not by editing the threshold method body. `?force=` overrides everything.
- **No mutable shared state across requests.** The FastAPI app keeps a per-process `track_bindings_cache: dict[(session_id, kind), list[TrackBinding]]` for discover-result memoization, but no per-request state. `Session` and `SessionSource` remain frozen dataclasses; runtime source mutation rebuilds the Session via `Session.with_sources()` and atomically swaps the registry entry under the same `session_id` (the source-mutation endpoints invariably call `invalidate_binding_cache(app.state.track_bindings_cache, session_id)` before returning).
- **Endpoint shape is REST-flat, not GraphQL.** `/api/sessions`, `/api/sessions/open`, `/api/sessions/inspect-source`, `/api/sessions/{id}/manifest`, `/api/sessions/{id}/contigs`, `/api/sessions/{id}/search`, `/api/sessions/{id}/sources` (POST), `/api/sessions/{id}/sources/{source_id}` (DELETE), `/api/tracks?session=...` (each listing carries `source_id`), `/api/tracks/{kind}/metadata`, `/api/tracks/{kind}/data`. The dashboard PR adds `/api/commands`, `/api/cli/schema`. The reference-cache-first flow adds `/api/references` (cache enumeration) and `/api/saved-sessions` CRUD; PR 5 added `PATCH /api/saved-sessions/{slug}/layout`. PR 6 extended both `POST /api/saved-sessions` and the PATCH endpoint to accept optional per-entry `style` + `filter` and a top-level `options` field; the PATCH preserves on-disk options when the payload omits the key. `/api/fs/list` (sandboxed file picker) remains deferred.
- **Coordinates are 0-based half-open** internally (matches the upstream pipeline schemas and BED). Frontend display formats them with `d3-format` thousands separators, but the wire stays in canonical bp.
- **TS source uses ES2022 modules + strict TS**. No bundler-specific globals; `vite build` produces a single multi-asset bundle per entry under `static/<entry>/`.
- **Static-bundle gate.** `create_app(...)` mounts `/static/<entry>/` only when the directory exists; otherwise `/` returns a JSON pointer at the build helper. CLI handlers don't fail on missing assets — the user can still hit `/api/...` endpoints (anywidget/Jupyter use cases).

## Testing

- `tests/test_viz_session.py` — Session discovery (TOML + walk + missing-paths drop). 8 tests.
- `tests/test_viz_kernels.py` — Registry contract + coverage_histogram round-trip. 10 tests.
- `tests/test_viz_kernels_extended.py` — Each of the other 5 kernels + datashader helpers + reference_sequence cache hit + single-contig loader. 16 tests.
- `tests/test_viz_server.py` — FastAPI test client; arrow_stream encode/decode; every endpoint route including `/api/sessions/{id}/search`. 29 tests.
- `tests/test_viz_saved_sessions.py` — Saved-session cache CRUD + endpoint lifecycle. PR 6 added round-trips for per-entry `style` / `filter` + browser-wide `[options]`, an empty-state-not-emitted check, and PATCH-extends-style-filter-options + PATCH-preserves-existing-options-when-absent. 12 tests.
- `tests/test_viz_source_mutation.py` (PR 5) — `source_id` stability across add/remove, `Session.with_sources()`, `invalidate_binding_cache()`, `POST/DELETE /api/sessions/{id}/sources`, saved-session v1→v2 migration, `PATCH /api/saved-sessions/{slug}/layout`. 13 tests.
- `tests/test_viz_e2e.py` (`@pytest.mark.slow`) — Real uvicorn boot in a background thread; full HTTP round-trip; Arrow IPC decode by the test client.

The full viz suite (125 tests) runs in ~6s; the slow e2e takes ~1.4s extra. Tests gate on `pytest.importorskip("fastapi" / "datashader")` so they don't pollute the base-install smoke run.

Fixture sessions are built in tmp_path: populate a fake reference cache root (via `monkeypatch.setenv("CONSTELLATION_REFERENCES_HOME", tmp_path / "refs")`) with `<organism>/<release_slug>/genome/{contigs.parquet,sequences.parquet}` + `meta.toml`, then write each source dir with a schema-v2 `manifest.json` referencing the cached handle plus its per-stage parquet artifacts. `Session.open(reference_handle=..., sources=[...])` builds the in-memory Session.

## Frontend distribution

The viz layer crosses a language boundary — Python on one side, TypeScript on the other — so the artifact story has more moving parts than a pure-Python module. The full mental model is in [docs/plans/viz-and-dashboard.md](../../docs/plans/viz-and-dashboard.md) under "Frontend distribution model"; the on-disk consequences are:

- **Frontend source** — committed under `frontend/src/`. TS files. Browsers cannot run these directly.
- **Frontend bundle** — derived: the output of `vite build` against the source. Lives at `viz/static/<entry>/` after the build runs. **This is what the FastAPI server actually mounts** and what the browser loads. Platform-independent.
- **Release artifact** — a single `.tar.gz` of `static/<entry>/` plus a `bundle.json` metadata header, produced by `python -m constellation.viz.frontend.build --pack`. Lands under `<repo>/dist/`. Used to ship the bundle to machines without the JS toolchain.

Both `static/` and `dist/` are git-ignored. The bundle is derived state; gitignoring matches every modern JS project and every Python tool with an SPA frontend.

**Install paths** — the bundle's runtime location is always `constellation/viz/static/<entry>/`; only how it gets there differs:

| Install path | How the bundle arrives | JS toolchain needed? |
|---|---|---|
| `pip install constellation-bio[viz]` from PyPI (release wheel) | Baked into the `.whl` by CI on tag push | No |
| `git clone && pip install -e .[viz]` (frontend dev) | `python -m constellation.viz.frontend.build` (runs pnpm) | Yes |
| `git clone && pip install -e .[viz]` (Python-only dev) | `constellation viz install-frontend` — URL-fetches the matching GitHub release asset | No |
| HPC / restricted-network install | Build bundle on a workstation via `--pack`, scp the tarball + `.sha256` to the cluster, run `install-frontend --from`. Solves the OSC-can't-run-npm case completely. | No |
| CI test runs | Not needed — Python tests skip viz endpoints when the bundle is absent | n/a |

**Install command surface (PR 1.6 + PR 2)**:

```
# URL-fetch (default — uses constellation.__version__)
constellation viz install-frontend                              # installs every entry the release ships
constellation viz install-frontend --version 0.0.1
constellation viz install-frontend --version 0.0.1rc1
constellation viz install-frontend --repo myfork/constellation
constellation viz install-frontend --cache-dir /scratch/$USER/cache

# Local tarball (PR 1.5 path — HPC offline escape hatch)
constellation viz install-frontend --from /path/to/tarball.tar.gz                     # installs every entry IN THE TARBALL (discovered)
constellation viz install-frontend --from /path/to/tarball.tar.gz --entry genome      # narrow to just genome
constellation viz install-frontend --from /path/to/tarball.tar.gz --entry genome dashboard   # explicit subset
constellation viz install-frontend --from /path/to/tarball.tar.gz --no-verify         # stderr warning
```

`--from` and `--version` are mutually exclusive. With neither, `--version` defaults to `constellation.__version__`. With no `--entry`, install dispatches once per entry — discovered from the local tarball directly (`list_tarball_entries`), or from `frontend.build.known_entries()` for URL fetch (the release contract ships every known entry). Dev versions (containing `.dev` or `+`) are refused before any HTTP call with a hint to pass `--version X.Y.Z` or `--from <local>`.

The handler reads the adjacent `.sha256` sidecar (GNU coreutils format: `<hex>  <filename>\n`), verifies the tarball's digest, then extracts `<prefix>/static/<entry>/` into `viz/static/<entry>/` and copies `bundle.json` alongside it so `constellation doctor` can report the installed version. The URL-fetch path additionally caches the downloaded tarball + sidecar under `~/.cache/constellation/viz-frontend/` (or `--cache-dir`); the sidecar is always re-fetched on subsequent invocations so a re-pushed release invalidates the cache automatically.

`pack_bundle()` in `frontend/build.py` is the producer side — public so the release CI workflow (`.github/workflows/release.yml`) can call it directly without re-running `pnpm build` from scratch.

`constellation doctor` adds a `viz frontend (<entry>)` row that reads `static/<entry>/bundle.json` and reports `ok` + version when present, `not installed` with a hint otherwise.

## Frontend layout

```
constellation/viz/frontend/
├── package.json                   pnpm/npm dependency manifest
├── vite.config.ts                 multi-entry build (genome + dashboard), base="/static/<entry>/"
├── tsconfig.json                  ES2022 + strict
├── index.genome.html              shell HTML for the genome entry
├── index.dashboard.html           shell HTML for the dashboard entry (PR 2)
├── build.py                       `python -m constellation.viz.frontend.build` (+ `--pack` for release-style tarballs); `--entry NAME [NAME ...]` builds multiple entries
└── src/
    ├── main_genome.ts             genome entry point (mounts GenomeBrowser)
    ├── main_dashboard.ts          dashboard entry point (PR 2; fetches /api/cli/schema, mounts DashboardShell)
    ├── engine/                    shared with both entries
    │   ├── arrow_client.ts        apache-arrow IPC fetch + JSON helpers
    │   ├── scales.ts              d3-scale + d3-axis wrappers
    │   ├── viewport_bus.ts        ViewportBus (locus + selection events)
    │   ├── svg_layer.ts           svgEl/clear/ensureSvg primitives
    │   ├── hybrid_layer.ts        decode HYBRID_SCHEMA + <image> mount
    │   ├── interactions.ts        wheel/drag pan-zoom into ViewportBus
    │   └── export.ts              composite-SVG serializer + download
    ├── track_renderers/           genome-only
    │   ├── base.ts                TrackRenderer interface (RenderContext.showLabels / style / filter)
    │   ├── style.ts               PR 6 — pick*(ctx.style|filter, key, fallback) coercion helpers
    │   ├── index.ts               kind → renderer registry
    │   └── <kind>.ts              one renderer per Python kernel
    ├── widgets/                   mounted from both entries
    │   ├── GenomeBrowser.ts       toolbar (zoom +/–/Fit / Labels / search / Options popover /
    │   │                          Datasets popover / Save SVG) + overview bar + ruler + track
    │   │                          stack with per-track drag-grip / collapse / gear / hide /
    │   │                          bottom-edge resize controls; owns per-binding
    │   │                          visible/displayOrder/heightPx/collapsed/style/filter +
    │   │                          MountedTrack.lastFetched Arrow cache for restyleTrack(); persists
    │   │                          layout + browser-wide options to localStorage + saved-session TOML
    │   ├── DatasetManagerPopover.ts  PR 5 — popover anchored to the Datasets toolbar button;
    │   │                          reference + per-source binding checkboxes + per-source remove
    │   │                          + inline "Add dataset" form (calls POST /api/sessions/{id}/sources)
    │   ├── OptionsPopover.ts      PR 6 — toolbar Options popover; browser-wide preferences
    │   │                          (v1: "Clip SVG export to viewport"); persists to
    │   │                          localStorage["constellation.genome.options.<sid>"] + PATCH layout
    │   ├── TrackSettingsPanel.ts  PR 6 — per-track gear popover; three sections (General / Style /
    │   │                          Filter) keyed off the kernel `kind`; pickers for color / font /
    │   │                          opacity / row-height bounds / dataset-slice filters (samples /
    │   │                          modes / motifs / strands / min-thresholds); reset-to-defaults
    │   └── GenomeBrowser.css      widget + popover styles scoped under .genome-browser-root
    │                              / .dataset-popover / .options-popover / .track-settings-popover
    └── dashboard/                 dashboard-only (PR 2 + PR 3)
        ├── DashboardShell.ts      DockviewComponent owner; two panel kinds (welcome, task)
        │                          + persistent sidebar rail mount + collapse toggle
        ├── TaskPanel.ts           single-panel phase machine (form → terminal | viz)
        ├── Sidebar.ts             Common ↔ All toggle, search, command tree
        ├── CommandForm.ts         schema-driven argv constructor; onJobStarted callback
        ├── VizForm.ts             descriptor-driven simple form for viz tools
        ├── GenomeBrowserForm.ts   reference-cache-first multi-row entry form
        ├── viz_registry.ts        per-viz-tool descriptors (currently viz/genome)
        ├── Terminal.ts            xterm.js wrapping the /api/commands/{id}/stream WS
        ├── StatusBar.ts           polls /api/commands/active; toast on lock rejection
        ├── state.ts               DashboardState event bus + localStorage helpers
        └── types.ts               TS mirror of introspect/schema.py TypedDicts
```

## Dashboard frontend (`frontend/src/dashboard/`, PR 2 + PR 3)

The dashboard SPA is vanilla TypeScript (no React) — same paradigm as the PR 1 genome browser. Layout uses dockview-core's `DockviewComponent` for splittable docking; xterm.js renders subprocess output.

| File | Role |
|---|---|
| `types.ts` | TypeScript mirror of `introspect/schema.py` TypedDicts (`CliSchema`, `CommandSchema`, `ArgumentSchema`) + the `/api/commands` wire shapes. |
| `state.ts` | `DashboardState` event bus (mirrors `engine/viewport_bus.ts` from PR 1) + `getStored`/`setStored` localStorage helpers namespaced under `constellation.dashboard.*`. |
| `Sidebar.ts` | Mode-switch toggle (Common ↔ All — state persisted), search filter, tree of commands; clicking a leaf emits `command:open`. Public `setMode(mode)` / `focusSearch()` / `getMode()` are called by `DashboardShell` from the collapsed-rail shortcut buttons (PR 7). Mounts into a host element (`#shell-sidebar-body` in the dashboard) — no longer a dockview panel as of PR 7. |
| `viz_registry.ts` | Per-viz-tool descriptors. Two flavours: simple-form descriptors (`{path, label, fields, open}`) render through `VizForm`; rich descriptors set `customForm: (ctx) => CustomFormHandle` to mount their own UI (used by the genome browser entry — see `GenomeBrowserForm.ts`). `findVizDescriptor(path)` resolves sidebar paths starting with `['viz', …]`. New viz tools land as one entry, no shell changes. PR 5 — the genome descriptor's `onSubmit` receives an `initialLayout: TrackLayoutEntry[] \| null` (populated when the form loaded from a saved session) and forwards it to `new GenomeBrowser({ ..., initialLayout })`. |
| `TaskPanel.ts` | Single panel owner. `mount(host, init)` renders phase 1 (`CommandForm` for compute, `VizForm` or descriptor `customForm` for viz). On Run, `CommandForm`'s `onJobStarted` callback transitions the panel to a `TerminalPanel`; on Open, the form invokes `transitionToWidget(mount)` and the panel hosts the widget. dockview-core's `dispose` hook tears down whichever child is active. |
| `GenomeBrowserForm.ts` | Reference-cache-first entry form for the genome browser. Loads `GET /api/references` into a dropdown (with a "Add new genome…" link that emits `command:open` for `reference fetch` in a sibling tab, and a "Refresh catalog" button that POSTs `argv=['catalog','update','--source','all']` to `/api/commands`), renders a repeating list of source rows (each `change` hits `POST /api/sessions/inspect-source` for kind + assembly mismatch detection), and a "Save as…" input that persists the configuration to `~/.constellation/sessions/<slug>.toml` via `POST /api/saved-sessions` on submit. Submit POSTs to `/api/sessions/open` and mounts `GenomeBrowser` on the task panel's host. PR 5 — when the user loads a saved session, the form captures `payload.track_layout` and passes it as the third arg to `onSubmit(result, saved, initialLayout)` so the mounted `GenomeBrowser` can restore per-binding visibility / order / height / collapsed state. |
| `CommandForm.ts` | JSON-schema-driven form generator. Buckets args into Required / Optional / Advanced (heuristic on dest); validates required-but-empty + numeric type mismatches; assembles `argv` and POSTs to `/api/commands`. On 200, invokes the host-provided `onJobStarted({jobId, argv})` callback (no broadcast through `DashboardState`). On 409, surfaces the error inline. |
| `VizForm.ts` | Renders a simple-form descriptor's `fields` list; `remember: true` fields recall from `localStorage` under `constellation.dashboard.viz.<name>`. Submit calls `descriptor.open(host, values)` via the task panel. Descriptors that need richer UI (e.g. the genome browser's multi-row form) skip VizForm via `customForm` instead. |
| `Terminal.ts` | xterm.js + `@xterm/addon-fit` wrapping a WS to `/api/commands/{id}/stream`. Frames are newline-delimited JSON `{stream, line}`; stderr renders red; the final `{stream:'exit',line:N}` frame closes the timer. |
| `StatusBar.ts` | Polls `/api/commands/active` every 2s; emits `job:active`. Toasts on `job:rejected`. |
| `DashboardShell.ts` | Owns the `DockviewComponent` for the right-hand workspace and the persistent left rail outside dockview. dockview-core 3.x uses a `createComponent(opts)` factory (not a `components: {name: Class}` map); we switch on `opts.name` to instantiate the right renderer. PR 7 dropped the `sidebar` factory case — the rail is no longer a dockview panel — leaving two kinds: `welcome` + `task` (always backed by `TaskPanel`). The rail (`#shell-sidebar`) toggles between 280 px expanded / 48 px collapsed via the logo header; collapse state persists at `localStorage["constellation.dashboard.sidebar.collapsed.v1"]`. Workspace layout persists at `localStorage["constellation.dashboard.layout.v2"]` (bumped from `.v1` when the sidebar moved out — stale v1 blobs that referenced a `sidebar` panel are ignored on read). When collapsed, the rail surfaces ★ Common / ☰ All / ⌕ Search shortcut buttons that expand the sidebar and set the relevant mode / focus the search input via `Sidebar.setMode()` / `Sidebar.focusSearch()`. |

dockview-core theming uses ~40 `--dv-*` CSS variables; v1 ships the dark theme and softens corners via `--dv-tab-border-radius: 6px` etc. — full override surface lives in `index.dashboard.html`'s `<style>` block. PR 7 swapped the underlying palette to an Aura-inspired set (`--bg #15141b`, `--accent #a277ff` purple, `--accent-alt #61ffca`, recessed `--bg-rail #110f18` for the persistent left sidebar); the `--dv-*` overrides remain mapped through `--bg`/`--fg`/`--border`/`--accent` so dockview auto-adopts the palette. See the project-root [`NOTICE.md`](../../NOTICE.md) + Acknowledgements below for attribution.

## What's deferred (later)

- **Kernel-pushdown filters** — PR 6 ships dataset-slice filters that all apply client-side (the renderer iterates fetched rows and skips ones the filter rejects). Filters that change *what the kernel queries* — MAPQ threshold and supplementary-flag toggle on `read_pileup`, anything that affects row-packing or the top-N truncation on `splice_junctions` — would need new `TrackQuery` fields, a refetch path on filter change (invalidate `MountedTrack.lastFetched`), and per-kernel `fetch()` predicate plumbing. Deferred until a concrete need lands; the all-client v1 covers publication-export use cases.
- **Source-grouped layout option** — current rendering is kind-grouped (all coverage rows together, then all pileup rows, etc.), then fully user-reorderable. A toggle to switch the default to source-grouped (all of sample-A's tracks together, then sample-B's) was discussed and deferred — drag-to-reorder already gives users the manual escape hatch.
- **Per-track Options follow-ons** — the PR 6 Options dropdown currently holds one toggle (clip SVG export). Natural future occupants without dedicated UI today: default label-visibility, axis-font globals, color-blind palette swap. Each lands as one new row in `OptionsPopover.build()` plus an entry in the `BrowserOptions` interface.
- `viz.server.endpoints.fs` — sandboxed file picker + WSL path normalization; the Datasets popover's "Add dataset" form takes plain text path input today.
- `frontend/src/dashboard/FilePicker.ts`, `NotebookPanel.ts` — IPython-in-dock and tree-based file picker; IPython would reuse the `viz.runner` plumbing.
- Desktop shortcut generator (`constellation install-shortcut` writing `.desktop` / `.lnk` / `.app`).
- Advanced-search column picker — v1 `/api/sessions/{id}/search` hardcodes `(table=annotation, column=name)`; v2 will accept `table`/`column` query params and a Basic ↔ Advanced toggle in the toolbar dropdown.
- Frontend render-pileup guard — concurrent `render()` calls in `GenomeBrowser.ts` are not currently guarded by a generation counter. The 60 ms debounce + per-track AbortController cover the common case; needed if/when a fetchTrackData call takes longer than one user-pan cadence.
- Cross-modality coordination types (Vitessce-style; lands once a non-genome modality kernel exists; the `viewport_bus` from PR 1 is already in place to receive selection events).
- Spectrum / structure / phylogeny / NN-activation kernels (separate PRs).
- proBAM / proBED emission (deferred per the project-wide note pending the genome→proteome bridge work).

## Companion plan

[docs/plans/viz-and-dashboard.md](../../docs/plans/viz-and-dashboard.md) — full multi-PR design, including the rationale for PR 1's six-track cut, the dashboard's auto-introspect + curated-overlay scheme, and the frontend distribution decision.

## Acknowledgements

The dashboard and genome browser color palette is inspired by the [Aura theme](https://github.com/daltonmenezes/aura-theme) by Dalton Menezes (MIT). Accent selection draws additional inspiration from the [Aura port for Raycast](https://ray.so/themes/shubh_porwal/aura) by Shubh Porwal. Values are adapted for our elevation needs, not copied verbatim — full attribution lives in [`NOTICE.md`](../../NOTICE.md) at the repo root, with header comments at the top of the `:root` blocks in [`index.dashboard.html`](frontend/index.dashboard.html) and [`index.genome.html`](frontend/index.genome.html).
