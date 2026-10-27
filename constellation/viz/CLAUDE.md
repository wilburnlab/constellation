# CLAUDE.md — `constellation.viz`

This file extends the project-wide rules in [../../CLAUDE.md](../../CLAUDE.md). Read that first.

`viz/` ships the first-party visualization layer. The standalone IGV-style genome browser boots via `constellation viz genome --reference <handle> --align-dir DIR [...] [--cluster-dir DIR ...]` or `--saved-session <slug>`; the dashboard shell (bare `constellation`) launches a JupyterLab-style soft GUI that wraps every CLI subcommand as a form + xterm.js terminal, splittable via dockview-core panes. The single `task` panel transitions in-place — sidebar entries open a form; compute commands swap that form for an xterm terminal on Run; viz commands swap it for the actual widget on Open. The genome-browser entry form is reference-cache-first: the user picks a reference from the `~/.constellation/references/` dropdown, then attaches one or more `transcriptome align` / `cluster` output dirs as "results" rows. Once a session is open, a Datasets toolbar popover lets the user toggle per-binding visibility, drag-to-reorder rows, collapse/hide individual tracks, and add or remove sources on the fly without reopening the browser. Each track's gear (⚙) opens a per-binding settings popover for color / palette / font / opacity / dataset-slice filters (publication-quality SVG export), and a sibling toolbar Options dropdown holds browser-wide preferences like SVG clip-to-viewport. The mass-spec browser is the second modality (`constellation viz massspec --acquisition DIR --experiment DIR`, or *Mass-spec browser* in the dashboard): chromatograms and spectra of converted acquisitions in a grid of MS levels by chromatogram and spectrum, with what search experiments identified in them (a search box, an identifications drawer, and targets traced and annotated across the grid); its plan and status are in [docs/plans/ms-browser.md](../../docs/plans/ms-browser.md). The three SPAs (genome, mass spec, dashboard) share one FastAPI app. See [docs/plans/viz-and-dashboard.md](../../docs/plans/viz-and-dashboard.md) for the multi-PR plan that built it, and [docs/plans/viz-modality-generalization.md](../../docs/plans/viz-modality-generalization.md) for the work that made genome one modality among several, on both the server and the frontend (pre-work for a mass-spec data browser).

The `[viz]` extras (`fastapi`, `pydantic`, `uvicorn[standard]`, `datashader`, `websockets`, `httpx`) gate the runtime; the package skeleton (`tracks/base.py`, `modalities/` including each modality's kernels, sessions and query models, `server/session.py`, `introspect/`, `runner/`) imports cleanly under the base install and is exercised by `tests/test_imports.py` and `tests/test_viz_base_install.py`. Modules that need fastapi / datashader live behind `pytest.importorskip` in their own test files.

## Module index

| Module | Role | Status |
|---|---|---|
| `viz.tracks` | The modality-neutral kernel contract: `TrackKernel` ABC, the base `TrackQuery`, `TrackBinding`, `HYBRID_SCHEMA`, and the `@register_track` registry (`tracks/base.py`). No kernels live here. | shipped |
| `viz.modalities` | Modality registry (`Modality`, `register_modality`, `get_modality`, `DEFAULT_MODALITY`). A modality is one kind of browser; it owns its session classes, kernels, query models, request models and extra routes, and the server core reaches it only through its descriptor. Stdlib-only, because `import constellation.viz` runs it. | shipped |
| `viz.modalities.genome` | The genome browser: `session.py` (`GenomeSession` / `GenomeSource`, `inspect_source`, `session_from_saved`), `query.py` (`GenomeQuery` and its three subclasses), `kernel.py` (`GenomeTrackKernel` base), `requests.py`, `endpoints.py` (contigs, feature search, `/api/references`), and `tracks/` — the six kernels (`reference_sequence`, `gene_annotation`, `coverage_histogram`, `read_pileup` (hybrid), `cluster_pileup` (hybrid), `splice_junctions`) plus `_alignment_view.py`. Declares its `DOMAIN_IMPORTS`. | shipped |
| `viz.modalities.massspec` | The mass-spec browser (steps B–G of [docs/plans/ms-browser.md](../../docs/plans/ms-browser.md)): `session.py` (`MassSpecSession` / `MsSource` / `MsAcquisition`; a source is a converted acquisition (bundle) or a search experiment, its kind read from its manifest; an experiment's runs are linked to their bundles by `massspec.results.acquisition_bundles`, and a run reached twice (same `source_sha256`) is one acquisition), `query.py` (`ChromatogramQuery`, `SpectrumQuery`, `XicQuery`), `kernel.py` (`MassSpecTrackKernel`), `tracks/` (`ms_chromatogram`, `ms_spectrum`, `ms_xic`; `ms_xic` has no renderer of its own, its traces are drawn over the chromatogram), `endpoints.py` (routes under `/api/massspec/sessions/{sid}/`), `cache.py` (per-process LRU caches of scan tables and identifications, keyed by file path and modification time) and `requests.py`. Declares its `DOMAIN_IMPORTS`. | partial (plan steps H1–H3 to come) |
| `viz.server` | FastAPI app factory (mounts each modality's routers), Arrow IPC streaming helper (`arrow_stream`), the session contract (`session.py`: `SessionLike` / `SourceLike` Protocols + the persisted-id helpers), request validation against stdlib-dataclass models (`validation.py`), and the modality-neutral `endpoints/{sessions, tracks, saved_sessions}` routes. The genome modality serves `GET /api/sessions/{id}/search?q=…&limit=…` — case-insensitive substring match on `name` (plus numeric `feature_id` when `q` parses as int) across both reference and derived annotation parquets, returning `{feature_id, name, type, strand, contig_name, start, end, source: 'reference'\|'derived'}` rows. PR 5 added the runtime source-mutation endpoints `POST /api/sessions/{id}/sources` + `DELETE /api/sessions/{id}/sources/{source_id}` (rebuild the Session via `Session.with_sources()` under the same `session_id`, then evict the binding cache via `invalidate_binding_cache()`), and `PATCH /api/saved-sessions/{slug}/layout` (rewrites just the `[[track_layout]]` block). PR 6 extended `POST /api/saved-sessions` and `PATCH /api/saved-sessions/{slug}/layout` to additionally accept optional per-entry `style` + `filter` dicts and a top-level `options` dict; `GET /api/saved-sessions/{slug}` surfaces them in the response payload. Missing `options` on the PATCH payload preserves the on-disk value (it's a per-key replace at the block level, not a deep merge). | shipped |
| `viz.raster` | Datashader → PNG helper for hybrid-mode payloads + `greedy_row_assign` shared between vector and hybrid renderers. **Only sanctioned pandas boundary in the package** (datashader's aggregation requires a pd.DataFrame input). | shipped |
| `viz.cli` | `build_parser(subs)` mounts the `viz` subtree from `constellation/cli/__main__.py`. Three subcommands shipped: `viz genome` and `viz massspec` (`--acquisition DIR` / `--experiment DIR`, each repeatable; both lazy-import uvicorn + the app factory and share the server flags `--host` / `--port` / `--no-browser` / `--root`), and `viz install-frontend` (lazy-imports `viz.install`). | shipped |
| `viz.install` | `install_frontend_from_tarball(local_path, entry, force, verify, dest_root)` + `install_frontend_from_url(version, entry, force, verify, dest_root, cache_dir, github_owner_repo)` + `read_bundle_metadata(dest_dir)`. Stdlib-only — `tarfile` / `hashlib` / `shutil` / `urllib.request`. PR 1.6 added URL-fetch (default path; cache at `~/.cache/constellation/viz-frontend/`); PR 1.5's local-tarball path remains as `--from PATH` and the HPC offline escape hatch. | shipped |
| `viz.frontend` | Committed TS/Vite source tree under `frontend/src/` plus the `python -m constellation.viz.frontend.build` helper. Default: builds **every entry discovered under `frontend/index.*.html`** (currently `genome`, `massspec` and `dashboard`); pass `--entry NAME [NAME ...]` to narrow for fast iteration. `--pack` produces `<repo>/dist/constellation-viz-frontend-<version>.tar.gz` + `.sha256` containing every built entry. `known_entries()` is the discovery helper. The `static/` and `dist/` trees are git-ignored. | shipped |
| `viz.introspect` | PR 2: walks the production argparse parser via `walk_parser(_build_parser())` and emits a JSON tree the dashboard renders into auto-generated forms. `curated.json` is the hand-edited "Common" overlay (a `curated` array for the sidebar + an `arg_hints` array that refines per-argument path classification). Type mapping: `BooleanOptionalAction`/`store_true` → flag, `choices=` → enum, `nargs in ('+','*',N>1)` → multi, `dest` path-suffix heuristic → path. **File-picker layer:** path args also carry `path_kind` (`dir`/`file`/`either`, dir-vs-file split from the same dest heuristic, dir-first) so the frontend FilePicker opens in the right mode; repeatable path args stay `multi` but keep `path_kind` for a Browse-to-append affordance; `dest == "reference"` stamps `widget="reference"` (editable dropdown from `/api/references`, never a filesystem picker — guarded by `_NOT_PATH_DEST`); `arg_hints` (matched by `(path, dest)`) overrides `path_kind`/`widget`/`glob`/`is_path` for genuine heuristic misses (`--samples`→file `*.tsv`, `--reads`→either `*.bam,*.sam`, `--junc-bed`→file `*.bed`). | shipped |
| `viz.runner` | PR 2: spawns `python -m constellation.cli.__main__ <argv>` via `asyncio.create_subprocess_exec`, fans stdout/stderr into a per-job history deque + subscriber queues. `lock.py` holds a `threading.Lock` enforcing the single-compute-job rule; `acquire_or_409` is non-blocking and maps to HTTP 409 at the endpoint. Lock release happens in the pump task's `finally` so a stuck subscriber never wedges the dashboard. | shipped |
| `viz.server.endpoints.{cli_schema,commands}` | PR 2: `GET /api/cli/schema` (cached for the process lifetime); `POST /api/commands` (acquires lock, spawns, returns job_id), `GET /api/commands/active`, `GET /api/commands/{id}`, `DELETE /api/commands/{id}` (SIGTERM with 10s grace), `WS /api/commands/{id}/stream` (replays history then live frames; sends `{stream:'exit',line:exit_code}` sentinel before closing). | shipped |
| `viz.server.endpoints.fs` | `GET /api/fs/list` — sandboxed directory listing for the frontend FilePicker. Every path is confined to `app.state.fs_roots` (home + cwd + WSL `/mnt/*` drives by default; widened with `constellation dashboard --root DIR` / `viz genome --root DIR` / `viz massspec --root DIR`). `_resolve_within_roots` normalizes pasted Windows paths (`C:\…`→`/mnt/c/…` via `wslpath`/regex under WSL), `resolve()`s to collapse `..` + deref symlinks, then a containment check rejects escapes (403); symlinks pointing out of the sandbox are dropped from listings. Query params: `path`, `dirs_only`, `globs`. | shipped |

**Currently registered kinds** (the canonical list — `constellation.viz.registered_kinds()`):
`cluster_pileup`, `coverage_histogram`, `gene_annotation`, `ms_chromatogram`, `ms_spectrum`, `ms_xic`, `read_pileup`, `reference_sequence`, `splice_junctions`.

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
   - Declares its `modality` and the query it takes (`query_model`)
   - Has a TS counterpart at [frontend/src/modalities/<name>/renderers/<kind>.ts](frontend/src/modalities/genome/renderers/) that mirrors the Python emit shape and declares what its host needs to know about the kind (default order, unit noun, server-side filters, settings)

   Kernels live with their modality, under `modalities/<name>/tracks/`; renderers under the frontend's `src/modalities/<name>/renderers/`.
4. **Read-only over parquet.** The viz server never writes pipeline outputs. Long-running compute stays in CLI / notebook; the GUI consumes parquet datasets via `pa.dataset.dataset(...)` + filter pushdown. PR 2's dashboard preserves this rule: it constructs `constellation <subcmd> ...` invocations and shows their stdout in xterm.js — never duplicates compute logic.
5. **One server, many frontends.** The same FastAPI app serves the focused `constellation viz genome` tool and the PR 2 dashboard panel mount. Kernels don't know which shell mounted them. (An `anywidget` mount for Marimo / Jupyter cells was part of this design but has not been built — there is no anywidget code in the tree.)
6. **Frontend ships pre-built in release wheels.** `constellation/viz/static/<entry>/` is git-ignored; the build helper at [frontend/build.py](frontend/build.py) (`python -m constellation.viz.frontend.build`) runs pnpm/npm. Build at install time was rejected — see "Frontend distribution" in the plan. Source-tarball installs without prebuilt assets see an actionable error message when hitting `/`.
7. **Reference cache as the GUI's entry point.** The dashboard's genome-browser entry form always picks a reference first (from `~/.constellation/references/<organism>/<release>/`, surfaced via `GET /api/references`), then attaches one or more "source" directories that the user has produced with `constellation transcriptome align` / `cluster`. Each source's `manifest.json` (the transcriptome manifest's current schema — v5 at the time of writing; `read_manifest_dir` rejects any other version) names the reference handle it was produced against, so the form can warn on assembly mismatch without needing the user to re-spell the relationship. Sessions are not directory-rooted any more — there is no `session.toml` discovery layer, and the legacy `Session.from_root` / `Session.discover` walks were removed in the reference-cache-first cutover. Saved configurations persist to `~/.constellation/sessions/<slug>.toml` (a sibling of the references and catalogs caches) for one-click resume.
8. **One panel kind per command, in-place phase transitions (PR 3).** Sidebar `command:open` always opens a single `'task'` panel via `dashboard/TaskPanel.ts`. Phase 1 is a form — `CommandForm` for compute, `VizForm` for any path matching `viz_registry`. Phase 2 swaps the form's DOM root for either `TerminalPanel` (compute; streams `/api/commands/{id}/stream`) or the viz widget itself (e.g. `modalities/genome/GenomeBrowser`), and the tab title updates accordingly. The welcome page's quick-launch creates the same panel with `autoSubmit: true` so a session-path entry skips straight to phase 2. New viz tools register one descriptor in `dashboard/viz_registry.ts` — the only dashboard file allowed to import a modality — and never touch the rest of the shell.
9. **Sessions stay frozen; mutation = atomic rebuild + cache evict (PR 5).** A modality's session and source classes (`GenomeSession`, `GenomeSource`) are `@dataclass(frozen=True, slots=True)`. Source mutation does not touch the existing instance — `session.with_sources(new_sources)` calls back through the validating `open(...)` path and returns a fresh frozen instance, which the endpoint swaps into `app.state.sessions[session_id]` (the `session_id` derives deterministically from `(reference_path, label)` so it's preserved across rebuilds). `invalidate_binding_cache(cache, session_id)` then evicts every `(session_id, kind)` entry in `app.state.track_bindings_cache` so the next `/api/tracks` discovers against the rebuilt source list. Client-side layout state is keyed by `(source_id, kind)` — never `binding_id` — so persisted state survives the rebuild even though kernel `binding_id`s remain index-based.
10. **Per-binding style/filter are opaque to the layout type; each kind declares its own controls (PR 6, restated by the modality work).** A layout entry (`panels/layout.ts::LayoutEntry`) carries optional `style?: Record<string, unknown>` and `filter?: Record<string, unknown>` dicts that the layout layer never inspects. Each renderer reads the keys it knows through the `panels/style.ts` helpers (`pickPaletteColor`, `pickNumber`, `pickAllowList`, …) with its `DEFAULTS` constant as fallback — unset = the renderer's literal default — and declares the matching controls as a `SettingsSchema` (`panels/settings_schema.ts`) built from the same constant, which the generic `panels/SettingsPanel.ts` renders. A new knob is a renderer-only change: one `pick*(ctx.style, 'my_key', DEFAULTS.my_key)` call and one control in that renderer's schema; no shared code, no schema migration, no persistence change. `renderers/descriptors.test.ts` draws every kind through recording dicts and fails if a popover offers a key the drawing code never looks up. Style and client-side filter edits redraw from the table already fetched (`Panel.lastFetched` + `PanelStack.restyle()`), so colour picks are instant. A filter the kernel applies is declared in the renderer's `pushdown` map (key → query-parameter encoder): `min_mapq` (`read_pileup`, and `cluster_pileup`'s members view) and `cluster_view` (`cluster_pileup`; the wire schema differs between its two views), which are fields of the kernels' query models (`ReadPileupQuery.min_mapq`, `ClusterPileupQuery.cluster_view`). A change to a declared key drops the cache and refetches (every visible track) rather than restyling. Further pushdown filters (a supplementary-flag toggle, anything that changes row packing) are one field on the query model plus one `pushdown` entry. SVG export uses `svg.dataset.naturalHeight` (set by every renderer) for panel y-offsets so overflow content auto-grows the export when clip is off, or wraps each panel in a `<clipPath>` when clip is on (the Options dropdown toggle).
11. **Genome is one modality; the server core names none.** `viz/server/` and `viz/tracks/base.py` know nothing about loci, references or contigs. A session is anything satisfying `SessionLike` (`session_id`, `label`, `modality`, `sources`, `warnings`, `saved_as`, `with_sources`, `summary`, `to_manifest`) — Protocols, not base classes, because the concrete classes are slots dataclasses and zero-argument `super()` fails in a slots-dataclass subclass on Python 3.12. Requests name a modality (`"genome"` when omitted, which is what every pre-modality client and saved file means); `POST /api/sessions/open`, `/inspect-source` and `POST /api/saved-sessions` validate the body against that modality's own stdlib-dataclass model and dispatch to it. A kernel only ever sees sessions of its own modality (the guard is in `endpoints/tracks.py::_bindings_for`). Adding a modality is: one `viz/modalities/<name>/` package that calls `register_modality`, an import of it in `constellation/viz/__init__.py`, a `viz <name>` CLI subcommand, and the frontend counterpart. `tests/test_viz_modalities.py` proves it by registering a toy modality entirely inside the test.
12. **The frontend has the same shape: generic layers below, one folder per modality above.** `src/engine/` (transport, SVG, export, popover mechanics), `src/panels/` (panel chrome, the reorderable stack, layout persistence, the schema-driven settings popover) and `src/widgets/` (path inputs) never import `src/modalities/`; a modality never imports another; `src/dashboard/` reaches a modality only through `viz_registry.ts`. `src/boundary.test.ts` enforces all three. `panels/` is a set of components a host drives, not a framework: a host builds a `PanelStack`, gives it a three-method `PanelDriver` (the current view; fetch one panel; draw one panel) and calls `render()`. The stack asks each kind's `PanelKind` descriptor for anything kind-specific and treats the host's view as opaque beyond a cache key and a width — which is what lets a browser with a different viewport (a retention-time or m/z window instead of a locus) reuse it unchanged. `Panel` is separate from `PanelStack` so a different container (a grid) can hold the same panels. `panels/PanelStack.test.ts` drives the stack under a non-genome host as the acceptance test. The frontend counterpart of a new modality is one `src/modalities/<name>/` folder plus one descriptor in `dashboard/viz_registry.ts` (and an `index.<name>.html` shell if it also ships standalone — entries are discovered from the shells by both `vite.config.ts` and `build.py`).

## Track kernel contract

Defined in [tracks/base.py](tracks/base.py). Subclasses register via the `@register_track` class decorator at module import; importing `constellation.viz` imports every shipped modality, which pulls in its kernel modules exactly once.

```python
class TrackKernel(ABC):
    kind: ClassVar[str]                        # registry key, must be unique
    modality: ClassVar[str]                    # owning modality; required
    schema: ClassVar[pa.Schema]                # vector-mode wire schema
    query_model: ClassVar[type[TrackQuery]]    # the query this kernel takes
    vector_glyph_limit: ClassVar[int]          # default threshold knob

    def discover(self, session: SessionLike) -> list[TrackBinding]: ...
    def metadata(self, binding: TrackBinding) -> dict: ...
    def threshold(self, binding: TrackBinding, query: TrackQuery) -> ThresholdDecision: ...
    def fetch(self, binding, query, mode) -> Iterator[pa.RecordBatch]: ...
    def schema_for(self, query, mode) -> pa.Schema: ...                 # optional
    def response_headers(self, query, mode) -> dict[str, str]: ...      # optional
    def estimate_vector_cost(self, binding, query) -> int | None: ...   # optional
```

The genome kernels subclass `GenomeTrackKernel` ([modalities/genome/kernel.py](modalities/genome/kernel.py)), which sets `modality = "genome"`, defaults `query_model` to `GenomeQuery`, adds the `vector_bp_per_pixel_limit` knob and emits the `X-Track-View` header.

The mass-spec kernels subclass `MassSpecTrackKernel` ([modalities/massspec/kernel.py](modalities/massspec/kernel.py)): one binding per acquisition (`binding_id = "{kind}:{acquisition key}"`, the bundle in `paths`, the acquisition's description in `config`, since `fetch` never sees the session), always vector. A trace or spectrum with more than `decimate_above` (20,000) points is reduced to `viewport_px` columns (M4, `core.signal.decimate`), which in the measured payloads happens only to a DDA run's all-MS2 trace and to profile spectra; the rows say so in a `decimated` column, because `response_headers` never sees the binding. They build their table when `fetch` is called and return an iterator rather than being generators, so a refusal (no base peaks recorded, an unknown modification) is a `ValueError`, which the tracks endpoint answers with HTTP 400.

**Queries are per-kernel.** The base `TrackQuery` carries only what every kernel shares: `viewport_px` and `force`, plus a `check()` hook. A kernel declares the query it takes as `query_model`, a subclass adding the fields that say what to read. For the genome kernels ([modalities/genome/query.py](modalities/genome/query.py)):

| Model | Fields | Used by |
|---|---|---|
| `GenomeQuery` | `contig`, `start`, `end` (`end > start` in `check()`) | `reference_sequence`, `gene_annotation`, `splice_junctions` |
| `CoverageQuery` | + `samples` | `coverage_histogram` |
| `ReadPileupQuery` | + `samples`, `min_mapq` | `read_pileup` |
| `ClusterPileupQuery` | + `cluster_view` (extends `ReadPileupQuery`; the members view reads `min_mapq`) | `cluster_pileup` |

For the mass-spec kernels ([modalities/massspec/query.py](modalities/massspec/query.py)); retention times are seconds and m/z Thomson, as in the bundle:

| Model | Fields | Used by |
|---|---|---|
| `ChromatogramQuery` | `level`, `trace` (`tic` / `bpc` / `scans`: one point per scan with its precursor m/z, a DDA level's scan events), `window` (a DIA window id), `rt_start` / `rt_end` (what a reduced trace is reduced for) | `ms_chromatogram` |
| `SpectrumQuery` | `scan`, `mz_start` / `mz_end`; to annotate: `modified_sequence` (ProForma), `charge`, `tolerance`, `tolerance_unit`, `ion_types`, `max_fragment_charge`, `neutral_losses` | `ms_spectrum` |
| `XicQuery` | `level`, `window`, `mz` and `label` (repeated, at most 64), `tolerance`, `tolerance_unit`, `rt_start` / `rt_end` | `ms_xic` |

`/api/tracks/{kind}/data` names no kernel-specific parameter: it resolves the kernel, builds an instance of its model from the query string and validates it (`endpoints/tracks.py::_parse_query`). Type and range failures are HTTP 422, a failed `check()` is 400, and a parameter the kernel does not declare is ignored. Models are stdlib keyword-only frozen dataclasses with range constraints in `field(metadata={"ge": …})` — pydantic enforces them in the endpoint layer, so nothing reached by `import constellation.viz` needs pydantic. They do not use `slots` (subclasses call `super().check()`), and may not declare fields named `session` or `binding`. A new kernel-specific knob is therefore one field on one model — no shared code changes. A kernel with more than one vector-mode schema overrides `schema_for(query, mode)` (`cluster_pileup` does, per view). `TrackBinding` is the kernel's instance-level handle (resolved parquet paths, per-track config, label). For `HYBRID` decisions, `fetch` emits a single one-row table matching `HYBRID_SCHEMA` (PNG bytes + extents + n_items + mode); the server's `/api/tracks/{kind}/data` endpoint surfaces the chosen mode in the `X-Track-Mode` response header so the TS renderer branches without inspecting the schema. It also sends `X-Track-Kind`, plus whatever the kernel's `response_headers` adds — for every genome kernel, `X-Track-View` (the requested `cluster_view`, empty when unset).

The `?force=vector|hybrid` query parameter overrides threshold logic for empirical tuning. Per-kernel thresholds are class attributes — tunable per-deployment without touching the threshold method body.

## Wire-schema convention

Each vector-mode kernel defines its emit schema in its own module (e.g. `COVERAGE_VECTOR_SCHEMA` in `coverage_histogram.py`). The schemas project away upstream `contig_id` (queries are range-scoped to one contig) and widen integer types where renderer math wants float64. The hybrid-mode schema is shared (`HYBRID_SCHEMA` in `tracks/base.py`) — every dense kernel emits to the same one-row shape so the frontend's hybrid layer is generic.

Vector-mode wire schemas:

| Kernel | Vector schema fields |
|---|---|
| `reference_sequence` | `position: i64, base: str, step: i32` |
| `gene_annotation` | `feature_id: i64, start: i64, end: i64, strand: str, type: str, name: str?, parent_id: i64?, source: str?` |
| `coverage_histogram` | `start: i64, end: i64, depth: f64, sample_id: i64` |
| `read_pileup` | `alignment_id: i64, read_id: str, ref_start: i64, ref_end: i64, strand: str, mapq: i32, row: i32, blocks: list<struct<ref_start: i64, ref_end: i64, n_match: i32, n_mismatch: i32>>, mismatch_positions: list<i64>, sample_id: i64?, sample_name: str?` |
| `cluster_pileup` (clusters view) | `cluster_id: i64, span_start: i64, span_end: i64, strand: str, n_reads: i32, row: i32, mode: str` |
| `cluster_pileup` (members view) | `alignment_id: i64, read_id: str, cluster_id: i64, ref_start: i64, ref_end: i64, strand: str, mapq: i32, row: i32, blocks: list<struct<…>>, mismatch_positions: list<i64>, role: str` |
| `splice_junctions` | `intron_id: i64, donor_pos: i64, acceptor_pos: i64, strand: str, support: i64, motif: str?, annotated: bool?` |
| `ms_chromatogram` | `scan: i32, rt: f64, intensity: f64?, mz: f64?, decimated: bool` |
| `ms_spectrum` | `mz: f64, intensity: f64, annotation: str?, ion_type: str?, ion_number: i16?, fragment_charge: i8?, loss: str?, mz_theoretical: f64?, error_da: f64?, error_ppm: f64?, decimated: bool` (the annotation columns null unless a peptide was given; `annotation` is mzPAF) |
| `ms_xic` | `trace: i16, label: str?, mz_target: f64, scan: i32, rt: f64, intensity: f64, mz_observed: f64?` (one row per trace and scan; 0 intensity where the ion is absent) |

`row` columns (read_pileup, cluster_pileup) are pre-computed server-side via `raster.datashader_png.greedy_row_assign` so vector and hybrid agree on the layout — zooming between modes preserves visual structure.

## Data-source mapping

The genome kernels iterate `session.sources` (the list of attached `transcriptome align` / `cluster` output dirs) and emit one binding per source. Reference paths come from the `GenomeSession`'s `reference_genome` / `reference_annotation` slots, populated by `handle.resolve(reference_handle)` against the cache. Per-stage artifact paths inside each source come from that source's `manifest.json` `outputs` map, joined against well-known relative filenames in `modalities/genome/session.py::_ALIGN_DEFAULT_PATHS` / `_CLUSTER_DEFAULT_PATHS`.

| Kernel | Reads from |
|---|---|
| `reference_sequence` | `session.reference_genome / "sequences.parquet"` (`SEQUENCE_TABLE`) — written with `row_group_size=1` by `sequencing.reference.io.ParquetDirWriter` so parquet `contig_id` statistics let a `contig_id == X` predicate skip every row group except the matching one. Loaded sequences cached process-wide in `_SEQUENCE_CACHE` keyed by `(sequences_path, contig_id)`. |
| `gene_annotation` | `session.reference_annotation / "features.parquet"` plus one binding per align source's `derived_annotation/features.parquet` — both surfaced separately so users can compare curated against data-derived. |
| `coverage_histogram` | one binding per align source with `coverage.parquet` (`COVERAGE_TABLE`). |
| `read_pileup` | one binding per align source with `alignments/` (`ALIGNMENT_TABLE`); optional `alignment_blocks/` rides along when present. |
| `cluster_pileup` | one binding per cluster source with `clusters.parquet` + `cluster_membership.parquet` (`TRANSCRIPT_CLUSTER_TABLE`, `CLUSTER_MEMBERSHIP_TABLE`). |
| `splice_junctions` | one binding per align source with `introns.parquet` (`INTRON_TABLE`). |

Kernels iterate via the shared `iter_sources_with(session, *attrs)` helper in `tracks/base.py` (yields `(idx, source)` pairs for sources whose listed slots are non-None). `binding_id`s are index-based so the per-process binding cache stays stable across re-discovery within a single session — but they are not uniformly `{kind}-{idx}`: `reference_sequence`, `reference` / `derived-{idx}` (gene_annotation), `coverage-{idx}`, `read_pileup-{idx}`, `cluster_pileup-{idx}`, `splice_junctions-{idx}`. PR 5 added a second-tier identifier: each source-derived kernel stamps `config["source_id"] = src.source_id` on the `TrackBinding` (and `/api/tracks` surfaces it as a top-level field on each binding listing). `source_id` is a `blake2b(path|kind)` hash (`viz.server.session.derive_source_id`, shared by every modality and pinned by a literal-value test) that's stable across `with_sources()` rebuilds — the client persists per-binding layout state keyed by `(source_id, kind)` so the state survives runtime source add/remove even though the index-based `binding_id` shifts. Reference-only bindings (`reference_sequence`, `gene_annotation:reference`) carry `source_id = null`. Kernels read parquet files via `pq.read_table(path, columns=[...])` or `pa.dataset.dataset(...).scanner(filter=..., columns=[...])`. Kernels reach their data through parquet on disk, not through the producing modules' Python API. The viz layer does make a small number of function-level imports from `constellation.sequencing` (listed under Conventions below); none is reached by `import constellation.viz`.

The mass-spec kernels each read one acquisition's bundle through `massspec.scans`, with the scan table read once per process (`cache.scan_table_for`): `ms_chromatogram` from `scan_metadata.parquet`; `ms_spectrum` one scan of the bundle's peaks (`massspec.bundle`; row groups pruned on `scan`), annotated by `massspec.spectra.annotate`; `ms_xic` from the bundle's peak index (`mz_index/`) when it has one, else from the bundle's peaks within the RT window asked for (0.25–0.45 s for ±2 min of a 100-min run; the whole run takes seconds). The modality's routes read an experiment through `massspec.results`: identifications (cached per process), one precursor's detail and trace targets, the searched library's entry and the run's RT mapping. Unlike the genome kernels these reach their data through the domain's Python read API, which was written for the browser and is usable on its own from a notebook.

## Conventions (viz layer)

- **No pandas inside the package, except in `viz.raster.datashader_png`.** Datashader's `Canvas.line()` requires a pandas DataFrame; we localize the import + DataFrame construction to that one module, analogous to the numpy big-endian-decode boundary in `electrophoresis.readers`. Kernels never see pandas.
- **Kernels don't import each other.** Each kernel module is a self-contained unit; shared helpers (contig-id lookup, greedy row assignment) are duplicated per-kernel rather than introducing a `tracks/_helpers.py`. Small price for module independence at this scale. **Exception: `tracks/_alignment_view.py`** ships the block-attach + cs:long mismatch-position helpers shared by `read_pileup` (PR 2) and (after PR 5) `cluster_pileup`'s member-view path — both kernels need the same ~150 LOC of nested-list assembly + cs grammar walk, so the trade flipped from "duplicate it" to "extract once." The helpers are underscore-prefixed (`_alignment_view`) to signal they're not part of the public kernel surface.
- **Domain imports go through the modality boundary.** The viz core imports no domain module. A modality package may import a small pure-function surface from one, declared as `DOMAIN_IMPORTS` in its `__init__` and imported inside function bodies; `tests/test_viz_import_boundary.py` checks the declaration against the code in both directions and admits functions and exception classes only. The genome modality's surface is eight names behind six call sites:
  - `modalities/genome/tracks/_alignment_view.py` → `sequencing.align.cigar.parse_cs_long_mismatch_positions` (per-base substitution positions for the pile-up's X glyphs; the cs grammar is non-trivial enough that duplicating the walk would invite drift).
  - `modalities/genome/session.py` → `sequencing.reference.handle.{parse_handle, resolve, read_meta_toml, ReferenceNotInstalledError}` and `sequencing.transcriptome.manifest.read_manifest_dir` (source loading, the cluster → align back-reference, and `inspect_source`).
  - `modalities/genome/endpoints.py` → `sequencing.reference.handle.{list_installed, read_defaults}`.

  The mass-spec modality's surface is 23 names from `massspec.scans`, `massspec.results`, `massspec.spectra.annotate`, `massspec.library.ids` and `massspec.peptide.mz`, listed in its `DOMAIN_IMPORTS`.

  Widening the surface means adding to `DOMAIN_IMPORTS`, which makes it a reviewed change rather than a quiet one. The bar for a new name: pure function, narrow surface, no upstream state.
- **Frontend renderers receive a `(table, mode, ctx)` tuple and clear-then-rewrite the SVG.** No data-binding / diffing — each render call is full-replacement. The host (`GenomeBrowser` widget) decides re-render cadence (60ms debounce on viewport changes); `PanelStack.render()` then hands the visible panels to `panels/scheduler.ts`, which fetches them in parallel (requests start in display order), draws a panel again from what it already has when its data is for the current view, and drops a response for a view that has since changed (latest wins). A request is keyed by the view and by the panel's pushed-down filter values, so one still running for an old filter value is never taken for the new one. `ctx.showLabels` rides through so kernels can branch on host-level UI toggles without owning state.
- **Genome browser owns its own chrome; the track stack is the generic one.** Toolbar (contig select / Go-to / zoom +/–/Fit / Labels toggle / feature search / Options / Datasets popover / Save SVG) + IGV-style contig overview bar + ruler live in `modalities/genome/GenomeBrowser.ts`, styled by `modalities/genome/genome.css` scoped under `.genome-browser-root` so the widget mounts cleanly in either the standalone SPA or a dashboard task panel (the standalone page's HTML supplies only the palette variables and the full-height frame). The track stack below them is a `panels/PanelStack`: per-track header controls (drag-grip reorder via HTML5 DnD, collapse chevron, gear, hide eye), the bottom-edge pointer-event resize handle, the empty-state placeholder and layout persistence all live in `panels/` and are styled by `panels/panels.css`, whose rules hang off the panel's own root (`.track`) and the stack's container (`.track-stack`) rather than off any host. A `ResizeObserver` on the host re-renders through the browser's 60 ms debounce when the pane resizes. The Datasets popover is genome's own component (`modalities/genome/DatasetManagerPopover.ts`) anchored to the toolbar button; it lists the reference plus every attached source with per-binding checkboxes, per-source remove (✕), and an inline "Add dataset" form that POSTs to `/api/sessions/{id}/sources`. `dispose()` releases everything the browser registered outside its own DOM (bus subscriptions, document / window listeners).
- **The mass-spec browser owns its chrome; the grid is the generic one.** `modalities/massspec/MassSpecBrowser.ts` holds the toolbar (back / forward, Run, the search box and its suggestions, identification and scan steppers, trace and intensity, the RT readout, Build index, the drawer toggle, Save SVG), the chip bar (the target and added traces), the drawer (`drawer.ts`: Identifications, Ions, Scan) and the view (`MsView`: acquisition, one RT window shared by the chromatograms, per row its level, DIA window, scan and m/z window, the target, added traces). The grid below is a `panels/PanelGrid` driven through a `GridDriver` whose `dataKey` says what a cell's data depends on, so a pan or a style change redraws without a request; its layout (weights, collapse, focus, per-cell style) persists under `localStorage["constellation.massspec.grid.v1"]`, per modality rather than per acquisition. Scan navigation runs on the client from the acquisition's scan table (`scan_index.ts`). A target (`target.ts`) comes from the precursor or resolve route and reaches the kernels as query parameters, never as server state. Replies are checked against a generation counter (acquisition, target), so a slow one for a view already left is dropped. The renderers (`renderers/{chromatogram,spectrum}.ts`) draw on the generic axes of `engine/axis.ts`, and their data colours are in `palette.ts`, checked with the dataviz palette validator against the dark surface.
- **Per-binding layout state is keyed by `(source_id, kind)`** in the client, not by `binding_id`. State (visible / displayOrder / heightPx / collapsed) persists to `localStorage["constellation.genome.layout.<sessionId>"]` debounced 200 ms; when `session.saved_as` is set, the client also PATCHes `/api/saved-sessions/{slug}/layout` so the configuration survives across machines. The entry shape, the merge-after-reload rule and the single writer are `panels/layout.ts` (`LayoutStore`); the genome browser supplies only its storage namespace. On reopen, the saved-session loader prefills the form which then passes `initialLayout` to `GenomeBrowser`. An `initialLayout` takes precedence: the `localStorage` layout is read only when none is supplied. Browser-wide options are always read from `localStorage` (the saved session's `[options]` block is written but not read back into the browser). Default ordering is kind-grouped by each renderer's declared `order`: `reference_sequence < gene_annotation < coverage_histogram < read_pileup < cluster_pileup < splice_junctions`, ties broken by source insertion order; runtime adds slot new bindings into the tail of their kind cluster; user reorder via drag reassigns displayOrders sequentially.
- **Thresholds default to class attributes.** `read_pileup.vector_glyph_limit = 4_000`; `cluster_pileup.vector_glyph_limit = 6_000`; `gene_annotation.feature_limit = 2_000`. Empirical tuning lives by adjusting the class attribute, not by editing the threshold method body. `?force=` overrides everything.
- **No mutable shared state across requests.** The FastAPI app keeps a per-process `track_bindings_cache: dict[(session_id, kind), list[TrackBinding]]` for discover-result memoization, but no per-request state. Session and source classes are frozen dataclasses; runtime source mutation rebuilds the session via its `with_sources()` and atomically swaps the registry entry under the same `session_id` (the source-mutation endpoints invariably call `invalidate_binding_cache(app.state.track_bindings_cache, session_id)` before returning).
- **Endpoint shape is REST-flat, not GraphQL.** Modality-neutral routes live in `server/endpoints/`; routes that only make sense for one modality live with it and are mounted through its `routers()` factory (genome: `/api/sessions/{id}/contigs`, `/api/sessions/{id}/search`, `/api/references`, which answer 404 for a non-genome session). `/api/sessions`, `/api/sessions/open`, `/api/sessions/inspect-source`, `/api/sessions/{id}/manifest`, `/api/sessions/{id}/contigs`, `/api/sessions/{id}/search`, `/api/sessions/{id}/sources` (POST), `/api/sessions/{id}/sources/{source_id}` (DELETE), `/api/tracks?session=...` (each listing carries `source_id`), `/api/tracks/{kind}/metadata`, `/api/tracks/{kind}/data`. The dashboard PR adds `/api/commands`, `/api/cli/schema`. The reference-cache-first flow adds `/api/references` (cache enumeration) and `/api/saved-sessions` CRUD; PR 5 added `PATCH /api/saved-sessions/{slug}/layout`. PR 6 extended both `POST /api/saved-sessions` and the PATCH endpoint to accept optional per-entry `style` + `filter` and a top-level `options` field; the PATCH preserves on-disk options when the payload omits the key. The file-picker layer adds `GET /api/fs/list` (sandboxed directory listing confined to `app.state.fs_roots`). The mass-spec modality's routes are under `/api/massspec/sessions/{sid}/`, since genome's router is mounted first and owns `/api/sessions/{id}/search`: `acquisitions`, `acquisitions/{key}/{windows, scans, scans/{n}, identifications, precursor}` and `resolve`.
- **Coordinates are 0-based half-open** internally (matches the upstream pipeline schemas and BED). Frontend display formats them with `d3-format` thousands separators, but the wire stays in canonical bp.
- **TS source uses ES2022 modules + strict TS**. No bundler-specific globals; `vite build` produces a single multi-asset bundle per entry under `static/<entry>/`.
- **Static-bundle gate.** `create_app(...)` mounts `/static/<entry>/` only when the directory exists; otherwise `/` returns a JSON pointer at the build helper. CLI handlers don't fail on missing assets — the user can still hit `/api/...` endpoints (anywidget/Jupyter use cases).

## Testing

### Backend (pytest)

`tests/test_viz_*.py` — about 235 tests, ~13 s. Files gate on `pytest.importorskip("fastapi" / "datashader")` so they don't pollute the base-install smoke run.

| File | Covers |
|---|---|
| `test_viz_session.py` | `GenomeSession.open` against a fake reference cache; slot resolution; assembly-mismatch warnings; **literal-value pins for `session_id` and `source_id`** (saved layouts and `localStorage` keys depend on them). |
| `test_viz_kernels.py`, `test_viz_kernels_extended.py` | Registry contract; each kernel's vector output; hybrid emit for both pile-up kernels; datashader helpers; the reference-sequence cache. |
| `test_viz_alignment_view.py` | Block attach + cs:long mismatch positions. |
| `test_viz_server.py` | Every endpoint over a FastAPI test client, including pushdown (`min_mapq`, `cluster_view`), hybrid mode (`force=hybrid`), status codes (400 inverted window, 422 range) and response headers. |
| `test_viz_saved_sessions.py`, `test_viz_source_mutation.py` | Saved-session cache + endpoints; `source_id` stability; runtime add/remove source; layout PATCH. |
| `test_viz_fs_endpoint.py`, `test_viz_references_endpoint.py` | File-picker sandbox; reference-cache listing. |
| `test_viz_introspect.py`, `test_viz_runner.py`, `test_viz_runner_lock.py`, `test_viz_dashboard_endpoints.py` | CLI introspection; subprocess runner; single-job lock; command endpoints. |
| `test_viz_install.py`, `test_viz_install_url.py`, `test_viz_entry_defaults.py` | Frontend bundle pack / install / entry discovery. |
| `test_viz_base_install.py` | **Base-install guard.** In a subprocess with fastapi / pydantic / starlette / uvicorn / datashader blocked, `constellation.viz`, `constellation.viz.cli`, `constellation.viz.frontend.build` and the CLI parser must import, and no domain module may be loaded by importing viz. The release workflow builds the frontend from a base install; CI installs `.[viz,ms]`, so nothing else exercises that path. |
| `test_viz_import_boundary.py` | **Import boundary.** AST walk of every viz source file: the core imports only `constellation.viz` / `constellation.core`; a modality's domain imports must be declared in `DOMAIN_IMPORTS`, sit inside a function, and be functions or exception classes. Rejects stale grants and dynamic imports, and checks itself against a planted violation. |
| `test_viz_modalities.py` | **Modality acceptance test.** Registers a toy modality defined entirely inside the test (session, float-window kernel, request models, router) and drives it over HTTP; pins that kernels and genome-only routes never serve another modality's session. |
| `test_viz_massspec.py` | The mass-spec modality over HTTP on synthetic bundles and stand-in-jar experiments: sessions (kind detection, linking runs to bundles, warnings, one run reached twice), the three kernels (traces, reduction, annotation, refusals as 400), every route, and the CLI refusing to start with nothing to open. The read API underneath is tested by `test_massspec_{scans,results,spectra_annotate,peak_index}.py`. |
| `test_viz_frontend_fixtures.py` | **Fixture drift guard** — see below. |
| `test_viz_e2e.py` (`slow`) | Real uvicorn boot; full HTTP round-trip; Arrow IPC decode. |

Fixture sessions are built in tmp_path by `tests/_viz_fixtures.py`: populate a fake reference cache root (via `monkeypatch.setenv("CONSTELLATION_REFERENCES_HOME", tmp_path / "refs")`) with `<organism>/<release_slug>/genome/{contigs.parquet,sequences.parquet}` + `meta.toml`, then write each source dir with a current-schema `manifest.json` referencing the cached handle plus its per-stage parquet artifacts. `GenomeSession.open(reference_handle=..., sources=[...])` builds the in-memory session. Tests reach the query and session types through `kernel_query(kind, ...)` / `open_session(...)` in `_viz_fixtures.py` rather than naming them.

### Frontend (vitest)

`pnpm test` in `constellation/viz/frontend/` (vitest + jsdom; config in `vitest.config.ts`, separate from the bundle's `vite.config.ts`). About 330 tests in 25 files, ~12 s. Test files sit beside the code as `*.test.ts`. CI runs `pnpm typecheck` + `pnpm test` in the `Frontend (typecheck + vitest)` job.

| File | Covers |
|---|---|
| `boundary.test.ts` | **Layering.** `engine/`, `panels/` and `widgets/` import no modality; a modality imports no other; `dashboard/` names a modality only in `viz_registry.ts`. Checks itself against planted violations. |
| `engine/{export,arrow_client,popover}.test.ts` | Composite- and grid-SVG sizing, clipping and id prefixing, and axes that keep their look without a stylesheet; the exact `/api/tracks/{kind}/data` URL (`buildTrackDataUrl`); popover dismissal and anchoring. |
| `engine/{axis,interval,pan_zoom,bus}.test.ts` | Ticks and formats in display units; interval clamp / zoom / pan; pointer pan, wheel zoom, shift-drag brush, click versus drag, the ignore predicate; the typed bus. |
| `panels/{style,layout,kind}.test.ts` | Every `pick*` helper; layout snapshot / apply / merge-after-reload / reorder and the `LayoutStore`; unit nouns and pushdown encoding. |
| `panels/SettingsPanel.test.ts` | The schema interpreter on a schema that belongs to no real kind: every control type, defaults versus stored values, computed defaults and option lists, conditional controls, in-place rebuild when an edit changes which controls apply, reset. |
| `panels/PanelStack.test.ts` | **Generic-stack acceptance test.** The stack under a host that is not a genome browser (a time-window view, numeric data, two toy kinds): chrome, order, hide / collapse, debounced persistence, fetch-and-draw, status line, restyle-versus-refetch routing, rebuild after a source change, a request for an old filter value not reused. |
| `panels/scheduler.test.ts` | Parallel fetch, redraw from the cache, a running request kept for the same key, latest wins, failures reported only for the current key, the concurrency cap. |
| `panels/PanelGrid.test.ts` | Track sizes from weights, collapse and focus; explicit placement; only cells whose key changed fetched; the placeholder; persistence; export rectangles; rebuilding for new rows. |
| `modalities/genome/{interactions,scales}.test.ts` | Zoom / clamp math; scale + unit formatting. |
| `modalities/genome/renderers/renderers.test.ts` | **Renderer parity snapshots.** Each renderer draws from the Arrow table its kernel really emits, at default and non-default `style` / `filter`, plus empty and hybrid paths. Snapshots are standalone files under `renderers/__snapshots__/*.svg` — open one in a browser to see the track. |
| `modalities/genome/renderers/settings.test.ts` | **Form-model snapshots** (`renderers/__snapshots__/settings.*.json`): the sections, controls, labels, initial values, bounds and options each kind's schema produces. Plus edit propagation, the data-driven mode / motif pickers, the cluster view switch, "Show labels" following the toolbar, reset and dismissal. |
| `modalities/genome/renderers/descriptors.test.ts` | What each kind declares (order, unit, pushdown), and **no dead controls**: each kind is drawn from its fixture through `style` / `filter` dicts that record every key looked up, and everything its popover offers must be among them or be a server-side filter. |
| `modalities/genome/GenomeBrowser.test.ts` | **Black-box host test.** Mounts the real widget against an in-memory fake server (`__fixtures__/fake_server.ts`) and pins request order and URLs, the track stack, status text, `localStorage` + the layout PATCH, restyle-versus-refetch routing, drag-reorder, resize, add/remove source, navigation, search, export, and that `dispose()` leaves no document / window listener or bus subscription behind. |
| `modalities/massspec/MassSpecBrowser.test.ts` | **Black-box host test** against an in-memory fake server (a DIA and a DDA run, identifications, the precursor, resolve and XIC routes): requests and URLs, scan linking between levels, targets and their traces, annotation, search, history, added traces, DDA scan events, reduced spectra, replies arriving out of order, Build index, CSV. Its Arrow tables are built in the test; kernel-generated fixtures like genome's are not built yet. |
| `modalities/massspec/{drawer,search}.test.ts` | The drawer (ion table and coverage, sorting and filtering, empty state); the search grammar and matching. |
| `dashboard/VizForm.test.ts` | Remembered fields are stored per tool; the older shared key is still read. |

**Fixtures come from the kernels.** `src/modalities/genome/__fixtures__/data/*.arrow` are Arrow IPC streams produced by the real Python kernels over a small synthetic session (`tests/_viz_frontend_fixtures.py`), so the renderers are tested against the true wire types (Int64, nested `list<struct>`, nullable strings) rather than tables hand-built in JS. `tests/test_viz_frontend_fixtures.py` rebuilds them on every pytest run and fails if the committed files differ, so a kernel change that alters the wire shape surfaces as a fixture diff and then a renderer snapshot diff. Regenerate with:

```
python scripts/build-viz-frontend-fixtures.py
pnpm -C constellation/viz/frontend test -- -u     # then review the snapshot diff
```

**What jsdom cannot check:** CSS (it applies none), real layout (element sizes are stubbed), native drag-and-drop and pointer capture (driven by synthetic events). Those need a browser. A cheap way to cover the static half without adding a browser to the test stack: `constellation viz genome --no-browser --port N` over a small session, then headless Chrome (`chrome --headless=new --screenshot=out.png --window-size=1400,1600 --virtual-time-budget=9000 http://localhost:N/`) before and after a change, and a pixel diff of the two PNGs. Repeat shots differ by ~76 px at delta 1 (anti-aliasing), so compare with that tolerance. Under WSL the Windows Chrome binary works against the WSL server on `localhost`. To shoot a state other than the opening one (an identification picked, a cell maximized), copy the built `static/<entry>/index.<entry>.html` to a `probe.html` beside it with a small script that clicks things on timers (`static/` is git-ignored; remove the copy afterwards); the mass-spec browser was checked this way with `constellation viz massspec --experiment DIR --no-browser --port N`.

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
├── package.json                   pnpm dependency manifest (lockfile: pnpm-lock.yaml, pnpm 9)
├── vite.config.ts                 multi-entry build; entries discovered from index.<entry>.html,
│                                  base="/static/<entry>/"
├── vitest.config.ts               unit-test config (jsdom); `pnpm test`
├── tsconfig.json                  ES2022 + strict
├── index.genome.html              shell HTML for the genome entry (palette + frame only)
├── index.dashboard.html           shell HTML for the dashboard entry
├── index.massspec.html            shell HTML for the mass-spec entry
├── build.py                       `python -m constellation.viz.frontend.build` (+ `--pack` for release-style tarballs); `--entry NAME [NAME ...]` builds multiple entries
└── src/                           (`*.test.ts` files sit beside the code they test)
    ├── boundary.test.ts           the layering rule below, enforced
    ├── main_genome.ts             genome entry point (mounts GenomeBrowser)
    ├── main_dashboard.ts          dashboard entry point (fetches /api/cli/schema, mounts DashboardShell)
    ├── main_massspec.ts           mass-spec entry point (mounts MassSpecBrowser on the server's session)
    │
    │   ── generic layers: never import modalities/ ──
    ├── engine/                    transport + drawing primitives
    │   ├── arrow_client.ts        apache-arrow IPC fetch + JSON helpers; the data URL carries
    │   │                          whatever parameters the caller passes and names none
    │   ├── svg_layer.ts           svgEl / clear / ensureSvg primitives
    │   ├── hybrid_layer.ts        decode HYBRID_SCHEMA + <image> mount
    │   ├── export.ts              composite-SVG serializers (a stack; a grid, ids prefixed per
    │   │                          cell) + download
    │   ├── axis.ts                continuous axes: plot frame, ticks in display units, formats
    │   │                          (RT in minutes, m/z, intensity, percent), drawAxes
    │   ├── interval.ts            clamp / zoom-about / pan over an interval with limits
    │   ├── pan_zoom.ts            attachPanZoom1D: wheel, drag, shift-drag brush, click, hover
    │   ├── bus.ts                 TypedBus<E>
    │   ├── plot.css               axis theme colours (the elements carry fallbacks)
    │   └── popover.ts             attachDismiss (outside click / Escape) + positionBelowRight
    ├── panels/                    modality-neutral panel chrome
    │   ├── Panel.ts               one panel: header (grip / collapse / label / status / gear /
    │   │                          hide), body, resize handle; its layout state and the last
    │   │                          data it was drawn from
    │   ├── PanelStack.ts          a vertical, reorderable stack of panels: order, drag-reorder,
    │   │                          resize, hide / collapse, gear popover, restyle-vs-refetch,
    │   │                          status line, debounced persistence; renders through the
    │   │                          scheduler. Driven by its host through `PanelDriver`
    │   ├── PanelGrid.ts           rows × columns of panels: focus (cell / row / column),
    │   │                          collapse, splitters, own persisted state, grid export rects.
    │   │                          Driven by its host through `GridDriver`
    │   ├── scheduler.ts           RenderScheduler: parallel fetch, redraw from cache,
    │   │                          latest wins
    │   ├── grid.css               the grid's headers, cells, splitters
    │   ├── kind.ts                `PanelKind` — what a kind declares: order, unit, pushdown
    │   │                          encoders, settings schema
    │   ├── layout.ts              the persisted `LayoutEntry`, apply / merge-after-reload /
    │   │                          reorder, and the single-writer `LayoutStore`
    │   │                          (localStorage + saved-session PATCH)
    │   ├── settings_schema.ts     the declarative control vocabulary: number / text / select /
    │   │                          toggle / allow-list / palette / note, with values that may be
    │   │                          functions of `{meta, style, filter, host}`
    │   ├── SettingsPanel.ts       renders a schema as the gear popover
    │   ├── OptionsPopover.ts      toolbar Options popover (v1: "Clip SVG export to viewport")
    │   ├── style.ts               pick*(style|filter, key, fallback) coercion helpers
    │   └── panels.css             panel chrome + the two popovers
    ├── widgets/                   reusable inputs
    │   ├── PathInput.ts           editable path text box + "Browse…" button; onChange + onCommit
    │   ├── FilePicker.ts          directory/file browser popover over GET /api/fs/list; roots
    │   │                          dropdown + breadcrumb + dir-first list; dir/file/either modes
    │   │                          (new-subfolder name for output dirs — never mkdir's on disk)
    │   └── FilePicker.css         PathInput row + .file-picker-popover chrome
    │
    │   ── one folder per modality: may import the layers above, never another modality ──
    ├── modalities/genome/
    │   ├── GenomeBrowser.ts       session, locus, toolbar, overview bar, ruler, feature search,
    │   │                          SVG export; composes a PanelStack and implements its driver
    │   │                          (fetch a track for a locus; draw it with the kind's renderer)
    │   ├── GenomeBrowserForm.ts   reference-cache-first multi-row entry form
    │   ├── DatasetManagerPopover.ts  reference + per-source binding checkboxes, per-source
    │   │                          remove, inline "Add dataset" form
    │   ├── viewport_bus.ts        ViewportBus (locus events on a TypedBus)
    │   ├── interactions.ts        engine/pan_zoom with whole-base-pair rounding, into the bus
    │   ├── scales.ts              d3-scale + d3-axis wrappers, bp formatting
    │   ├── types.ts               wire shapes of the genome session endpoints
    │   ├── genome.css             the browser's own chrome, scoped under .genome-browser-root
    │   ├── renderers/
    │   │   ├── base.ts            TrackRenderer = PanelKind + render(table, mode, ctx)
    │   │   ├── index.ts           kind → renderer registry
    │   │   ├── <kind>.ts          one per Python kernel: DEFAULTS, settings schema, drawing
    │   │   ├── _alignment_view.ts alignment-row drawing + its controls, shared by read_pileup
    │   │   │                      and cluster_pileup's members view
    │   │   ├── pushdown.ts        encoders for the server-side filters
    │   │   └── settings_common.ts schema helpers the kinds share
    │   └── __fixtures__/          test-only: kernel-generated Arrow fixtures (`data/*.arrow`),
    │                              their loaders, and the in-memory fake server
    ├── modalities/massspec/
    │   ├── MassSpecBrowser.ts     toolbar, search, chip bar, history, the view; composes a
    │   │                          PanelGrid and implements its driver; export
    │   ├── drawer.ts              Identifications / Ions / Scan tabs
    │   ├── scan_index.ts          the acquisition's scan table, for navigation on the client
    │   ├── search.ts              the search grammar and matching identifications
    │   ├── target.ts              the precursor being looked at, from the precursor / resolve
    │   │                          routes
    │   ├── palette.ts             series, isotope and ion-type colours
    │   ├── types.ts               wire shapes of the mass-spec routes
    │   ├── ms.css                 the browser's chrome, scoped under .ms-browser
    │   └── renderers/
    │       ├── chromatogram.ts    traces, sticks, target overlays, marks, median, legend
    │       └── spectrum.ts        sticks, annotation, labels, isotopes, error strip
    │
    └── dashboard/                 the shell; names a modality only in viz_registry.ts
        ├── DashboardShell.ts      DockviewComponent owner; two panel kinds (welcome, task)
        │                          + persistent sidebar rail mount + collapse toggle
        ├── TaskPanel.ts           single-panel phase machine (form → terminal | viz)
        ├── Sidebar.ts             Common ↔ All toggle, search, command tree
        ├── CommandForm.ts         schema-driven argv constructor; onJobStarted callback
        ├── VizForm.ts             descriptor-driven simple form for viz tools
        ├── viz_registry.ts        per-viz-tool descriptors (viz/genome, viz/massspec)
        ├── Terminal.ts            xterm.js wrapping the /api/commands/{id}/stream WS
        ├── StatusBar.ts           polls /api/commands/active; toast on lock rejection
        ├── state.ts               DashboardState event bus + localStorage helpers
        └── types.ts               TS mirror of introspect/schema.py TypedDicts
```

**A host's side of `panels/`** — everything the generic layer asks of a browser:

```ts
interface PanelView { key: string; widthPx: number }        // extend with a locus, a time window, …
interface PanelDriver<V extends PanelView, D> {
  view(): V | null;                                          // null = nothing to show yet
  fetch(panel: Panel<D>, view: V, signal: AbortSignal): Promise<D>;
  draw(panel: Panel<D>, data: D, view: V, svg: SVGSVGElement,
       size: { widthPx: number; heightPx: number }): number; // items drawn, for the status line
}
```

plus `kindOf(kind)` returning each kind's `PanelKind`, a `LayoutStore`, and two notifications (`requestRender`, `onChanged`). The view is taken once per render so every panel of a pass is fetched and drawn for the same one, and `key` is what cached data is checked against before a redraw.

**A host's side of `PanelGrid`** is the same idea with a per-cell key, since cells of one view depend on different things (a chromatogram on the RT window only when its trace was reduced, a spectrum on its scan):

```ts
interface GridDriver<V, D> {
  view(): V | null;
  dataKey(cell: GridCell<D>, view: V): string | null;        // what the cell's data depends on;
                                                             // null = show the placeholder
  fetch(cell: GridCell<D>, view: V, signal: AbortSignal): Promise<D>;
  draw(cell: GridCell<D>, data: D, view: V, svg: SVGSVGElement,
       size: { widthPx: number; heightPx: number }): number;
  placeholder?(cell: GridCell<D>, view: V): string;
}
```

The grid adds each kind's pushed-down filter values to the host's key itself.

## Dashboard frontend (`frontend/src/dashboard/`, PR 2 + PR 3)

The dashboard SPA is vanilla TypeScript (no React) — same paradigm as the PR 1 genome browser. Layout uses dockview-core's `DockviewComponent` for splittable docking; xterm.js renders subprocess output.

| File | Role |
|---|---|
| `types.ts` | TypeScript mirror of `introspect/schema.py` TypedDicts (`CliSchema`, `CommandSchema`, `ArgumentSchema`) + the `/api/commands` wire shapes + `InstalledReference` (`/api/references`, read by `CommandForm` for any reference-handle argument). A browser's own session shapes live with it (`modalities/genome/types.ts`). |
| `state.ts` | `DashboardState` event bus (same pattern as the genome browser's `viewport_bus.ts`) + `getStored`/`setStored` localStorage helpers namespaced under `constellation.dashboard.*`. |
| `Sidebar.ts` | Mode-switch toggle (Common ↔ All — state persisted), search filter, tree of commands; clicking a leaf emits `command:open`. Public `setMode(mode)` / `focusSearch()` / `getMode()` are called by `DashboardShell` from the collapsed-rail shortcut buttons (PR 7). Mounts into a host element (`#shell-sidebar-body` in the dashboard) — no longer a dockview panel as of PR 7. |
| `viz_registry.ts` | Per-viz-tool descriptors. Two flavours: simple-form descriptors (`{path, label, fields, open}`) render through `VizForm`; rich descriptors set `customForm: (ctx) => CustomFormHandle` to mount their own UI (used by the genome browser entry — see `GenomeBrowserForm.ts`). `findVizDescriptor(path)` resolves sidebar paths starting with `['viz', …]`. New viz tools land as one entry, no shell changes; the mass-spec browser is a simple-form one (an acquisition or experiment directory, more comma-separated). PR 5 — the genome descriptor's `onSubmit` receives an `initialLayout: LayoutEntry[] \| null` (populated when the form loaded from a saved session) and forwards it to `new GenomeBrowser({ ..., initialLayout })`. |
| `TaskPanel.ts` | Single panel owner. `mount(host, init)` renders phase 1 (`CommandForm` for compute, `VizForm` or descriptor `customForm` for viz). On Run, `CommandForm`'s `onJobStarted` callback transitions the panel to a `TerminalPanel`; on Open, the form invokes `transitionToWidget(mount)` and the panel hosts the widget. dockview-core's `dispose` hook tears down whichever child is active. |
| `../modalities/genome/GenomeBrowserForm.ts` | (Lives with the genome modality; listed here because the task panel hosts it.) Reference-cache-first entry form for the genome browser. Loads `GET /api/references` into a dropdown (with a "Add new genome…" link that emits `command:open` for `reference fetch` in a sibling tab, and a "Refresh catalog" button that POSTs `argv=['catalog','update','--source','all']` to `/api/commands`), renders a repeating list of source rows (each `change` hits `POST /api/sessions/inspect-source` for kind + assembly mismatch detection), and a "Save as…" input that persists the configuration to `~/.constellation/sessions/<slug>.toml` via `POST /api/saved-sessions` on submit. Submit POSTs to `/api/sessions/open` and mounts `GenomeBrowser` on the task panel's host. PR 5 — when the user loads a saved session, the form captures `payload.track_layout` and passes it as the third arg to `onSubmit(result, saved, initialLayout)` so the mounted `GenomeBrowser` can restore per-binding visibility / order / height / collapsed state. |
| `CommandForm.ts` | JSON-schema-driven form generator. Buckets args into Required / Optional / Advanced (heuristic on dest); validates required-but-empty + numeric type mismatches; assembles `argv` and POSTs to `/api/commands`. On 200, invokes the host-provided `onJobStarted({jobId, argv})` callback (no broadcast through `DashboardState`). On 409, surfaces the error inline. |
| `VizForm.ts` | Renders a simple-form descriptor's `fields` list; `remember: true` fields recall from `localStorage` under `constellation.dashboard.viz.<tool>.<field>` (`<tool>` = the descriptor's command path without its leading `viz`; a value under the older shared `constellation.dashboard.viz.<field>` key is read as a fallback). Submit calls `descriptor.open(host, values)` via the task panel. Descriptors that need richer UI (e.g. the genome browser's multi-row form) skip VizForm via `customForm` instead. |
| `Terminal.ts` | xterm.js + `@xterm/addon-fit` wrapping a WS to `/api/commands/{id}/stream`. Frames are newline-delimited JSON `{stream, line}`; stderr renders red; the final `{stream:'exit',line:N}` frame closes the timer. |
| `StatusBar.ts` | Polls `/api/commands/active` every 2s; emits `job:active`. Toasts on `job:rejected`. |
| `DashboardShell.ts` | Owns the `DockviewComponent` for the right-hand workspace and the persistent left rail outside dockview. dockview-core 3.x uses a `createComponent(opts)` factory (not a `components: {name: Class}` map); we switch on `opts.name` to instantiate the right renderer. PR 7 dropped the `sidebar` factory case — the rail is no longer a dockview panel — leaving two kinds: `welcome` + `task` (always backed by `TaskPanel`). The rail (`#shell-sidebar`) toggles between 280 px expanded / 48 px collapsed via the logo header; collapse state persists at `localStorage["constellation.dashboard.sidebar.collapsed.v1"]`. Workspace layout persists at `localStorage["constellation.dashboard.layout.v2"]` (bumped from `.v1` when the sidebar moved out — stale v1 blobs that referenced a `sidebar` panel are ignored on read). When collapsed, the rail surfaces ★ Common / ☰ All / ⌕ Search shortcut buttons that expand the sidebar and set the relevant mode / focus the search input via `Sidebar.setMode()` / `Sidebar.focusSearch()`. |

dockview-core theming uses ~40 `--dv-*` CSS variables; v1 ships the dark theme and softens corners via `--dv-tab-border-radius: 6px` etc. — full override surface lives in `index.dashboard.html`'s `<style>` block. PR 7 swapped the underlying palette to an Aura-inspired set (`--bg #15141b`, `--accent #a277ff` purple, `--accent-alt #61ffca`, recessed `--bg-rail #110f18` for the persistent left sidebar); the `--dv-*` overrides remain mapped through `--bg`/`--fg`/`--border`/`--accent` so dockview auto-adopts the palette. See the project-root [`NOTICE.md`](../../NOTICE.md) + Acknowledgements below for attribution.

## Known issues

Found while writing the characterization tests and doing the modality work. None is fixed; the first two are pinned as current behavior by tests so they cannot change by accident. (Six others found the same way were fixed as the plan's "parity exceptions" — see [docs/plans/viz-modality-generalization.md](../../docs/plans/viz-modality-generalization.md) — and the entry form's class staying on the embedded browser was fixed with the mass-spec browser, step D of [docs/plans/ms-browser.md](../../docs/plans/ms-browser.md).)

- **"Visible sources" filter matches the wrong thing.** The `gene_annotation` popover offers `reference` / `derived`, but the renderer compares against each feature's own `source` column (the GFF source, e.g. `RefSeq`). Unticking either box hides every feature whose source column is non-empty.
- **Duplicate `display_order` after adding a source.** `computeInsertOrder` shifts not-yet-restored siblings, so the persisted layout can carry ties (e.g. `0,1,2,3,3,4,4,5`). The on-screen order is still right because the sort is stable over the panel list.
- **Default swatch can differ from the drawn colour.** The popover indexes the colour cycle by position in the metadata list; the renderers by encounter order in the current window. Left alone deliberately: fixing it changes drawn output.
- **Saved `[options]` are write-only**, and `_normalize_options` keeps only `clip_svg`.
- **The members view counts reads as clusters.** In `cluster_pileup`'s members view the status line reads "showing N clusters" where N is the number of member reads drawn; a kind's unit is one fixed pair.
- **A cluster source repeats its align source's tracks.** `_attach_align_slots_from_cluster` copies every align slot through the cluster manifest's `align_dir` back-reference (its docstring names four), so attaching an align dir and the cluster dir built from it lists coverage, reads and splice junctions once per source.
- **Clearing a number field stores 0** (`Number('') === 0`) instead of removing the key; a text field removes it.
- **Entry-form selects and link buttons are unstyled.** `index.dashboard.html` has no rule for `.viz-form select` or `.link-btn`, so the saved-session and reference dropdowns and the "Add new genome… / Refresh catalog / + Add row" buttons render with the browser's light defaults against the dark theme.
- **The predicted retention time needs the searched library where the search found it.** The precursor route maps the library's retention time onto the run from the library file at the path the search recorded; for a copied or moved experiment, or a library moved since, the browser shows no predicted time. (An experiment's own bundles are found wherever it is.)
- **Knobs with no control.** The drawing code reads `letter_font_family` (reference_sequence), `strand_chevron_min_width_px` (gene_annotation), `palette.exon` (read_pileup), `palette.default` (splice_junctions, for junctions with no motif), and per-cluster colours and `visible_clusters` in the members view; no popover offers them. They can only be set by editing a saved session.

## What's deferred (later)

- **More kernel-pushdown filters** — `min_mapq` and `cluster_view` are pushed down today (see invariant 10). Still client-side or absent: a supplementary-flag toggle on `read_pileup`, and anything that affects row-packing or the top-N truncation on `splice_junctions`. Each needs a field on the kernel's query model, `fetch()` predicate plumbing, and an entry in the renderer's `pushdown` map.
- **anywidget / Jupyter mount** — designed (invariant 5), never built.
- **Source-grouped layout option** — current rendering is kind-grouped (all coverage rows together, then all pileup rows, etc.), then fully user-reorderable. A toggle to switch the default to source-grouped (all of sample-A's tracks together, then sample-B's) was discussed and deferred — drag-to-reorder already gives users the manual escape hatch.
- **Per-track Options follow-ons** — the PR 6 Options dropdown currently holds one toggle (clip SVG export). Natural future occupants without dedicated UI today: default label-visibility, axis-font globals, color-blind palette swap. Each lands as one new row in `OptionsPopover.build()` plus an entry in the `BrowserOptions` interface.
- `frontend/src/dashboard/NotebookPanel.ts` — IPython-in-dock; would reuse the `viz.runner` plumbing.
- **File picker follow-ons** — the shipped `widgets/FilePicker.ts` + `widgets/PathInput.ts` (over `GET /api/fs/list`) replace every plain-text path box across the GUI (auto-introspected `CommandForm` `path` args, `GenomeBrowserForm` source rows, the Datasets "Add dataset" form, `VizForm` path fields). Deferred: retrofitting the existing unsandboxed path-reading endpoints (`/api/sessions/inspect-source`, `/api/sessions/open`) onto the same `_resolve_within_roots` sandbox; a picker that can `mkdir` (kept read-only by design — the CLI creates output dirs); and converting `--reads` into repeating `PathInput` rows (kept as a textarea + Browse-to-append to avoid touching argv assembly).
- Desktop shortcut generator (`constellation install-shortcut` writing `.desktop` / `.lnk` / `.app`).
- Advanced-search column picker — v1 `/api/sessions/{id}/search` hardcodes `(table=annotation, column=name)`; v2 will accept `table`/`column` query params and a Basic ↔ Advanced toggle in the toolbar dropdown.
- A client cache with overscan — not built, and not needed so far: the mass-spec payload survey (`docs/plans/ms-browser.md`) found whole traces and scans small enough to fetch once and zoom locally, with only a DDA run's all-MS2 trace reduced (M4) on the server.
- Cross-modality coordination (Vitessce-style). Each browser has its own `engine/bus.ts` `TypedBus` (genome's `ViewportBus` is a locus on top of one); nothing links two browsers, and genome's `selection:feature` event is still unused. The generic pieces that coordination within one browser needed now exist — continuous axes and y-axes (`engine/axis.ts`), 1D pan/zoom (`engine/pan_zoom.ts`, which genome's `interactions.ts` now wraps), the grid container (`panels/PanelGrid.ts`) and grid export (`engine/export.ts::buildGridSvg`) — and are used by the mass-spec browser.
- **The rest of the mass-spec browser's first generation** (plan steps H1–H3): the library mirror plot and Δm/z ruler, the multi-run XIC overlay with RT alignment, and the RT × m/z map (a points rasterizer and float extents in `hybrid_layer.ts`).
- **Mass-spec saved views.** `POST /api/saved-sessions` keeps a mass-spec session's source list and `viz massspec --saved-session` reopens it, but the browser has no Save control, and the target, scans and m/z windows are not kept anywhere (the grid layout is in `localStorage`). Saving them needs `sessions/cache.py` to keep opaque modality options.
- **Kernel-generated Arrow fixtures for the mass-spec renderers**, with a drift test, as genome has.
- Structure / phylogeny / NN-activation kernels (separate PRs).
- proBAM / proBED emission (deferred per the project-wide note pending the genome→proteome bridge work).

## Companion plan

[docs/plans/viz-and-dashboard.md](../../docs/plans/viz-and-dashboard.md) — full multi-PR design, including the rationale for PR 1's six-track cut, the dashboard's auto-introspect + curated-overlay scheme, and the frontend distribution decision.

## Acknowledgements

The dashboard and genome browser color palette is inspired by the [Aura theme](https://github.com/daltonmenezes/aura-theme) by Dalton Menezes (MIT). Accent selection draws additional inspiration from the [Aura port for Raycast](https://ray.so/themes/shubh_porwal/aura) by Shubh Porwal. Values are adapted for our elevation needs, not copied verbatim — full attribution lives in [`NOTICE.md`](../../NOTICE.md) at the repo root, with header comments at the top of the `:root` blocks in [`index.dashboard.html`](frontend/index.dashboard.html), [`index.genome.html`](frontend/index.genome.html) and [`index.massspec.html`](frontend/index.massspec.html).
