# Viz foundation: make genome one modality (pre-work for the MS browser)

## Status

- **Stage 0 (PR A) — housekeeping and safety net:** implemented on
  `chore/viz-foundation-housekeeping` (2026-10-09).
- **Stage 1 (PR B) — backend contracts:** not started.
- **Stage 2 (PR C) — frontend factoring:** not started.

Plan approved 2026-10-09. Companion to
[viz-and-dashboard.md](viz-and-dashboard.md), which records how the viz layer
was built; the layer's current state lives in
[constellation/viz/CLAUDE.md](../../constellation/viz/CLAUDE.md).

## Context

The GUI (`constellation/viz/`) shipped as eight PRs between May and July 2026 and
has not been worked on since. The next feature is a mass-spec data browser, but
the layers the genome browser calls "generic" are genome-shaped one level down:
the kernel query is `contig/start/end`, the session requires a reference genome,
the shared data endpoint names `min_mapq` and `cluster_view`, and a 1,678-line
`GenomeBrowser.ts` fuses reusable panel chrome with genome logic. A second
modality added today would mean editing shared code for every kernel-specific knob.

This plan covers the modality-neutral foundation only (steps 1–3 of the agreed
order). Outcome: the genome browser behaves as it does now, but sits behind
contracts a second modality can implement without touching shared code.

**Decisions already made**
- Scope is steps 1–3. The engine (generic axes, y-axes, selection bus, fetch
  cache, decimation), the massspec query API and the detections view are planned
  later with the MS browser. This plan only leaves seams for them.
- viz may import a small, declared, pure-function surface from a domain module.
  The rule is amended and enforced by a test instead of by convention.
- Frontend gets vitest tests before the restructure; no headless browser.

**Ground rules**
- **Wire-compatible.** Every existing URL, query-parameter name, response
  shape, Arrow schema, saved-session TOML and `localStorage` key keeps working.
- **Base-install importable.** `import constellation.viz` and `constellation.cli`
  must work without fastapi, pydantic or datashader. The release workflow builds
  the frontend from a base install, and CI never exercises that path today.
- **Moves are pure `git mv` commits**, separate from edits, with no re-export
  shims at old paths (a test monkeypatches a kernel module's attributes).

Three PRs. B and C are independent of each other once A lands.

---

## Stage 0 — Housekeeping and safety net (PR A)

1. **Dead dependencies.** Remove `d3-array`, `d3-brush`, `d3-zoom` and their
   `@types/*` from `constellation/viz/frontend/package.json` (zero imports).
   Regenerate `pnpm-lock.yaml` with pnpm 9 via corepack. Do not use npm here:
   the local `node_modules` is a pnpm layout.
2. **Frontend tests** (vitest on a major that supports the pinned vite 5.4, plus
   jsdom; `"test": "vitest run"`), all written against today's code:
   - Helpers: `engine/interactions.ts`, `engine/scales.ts`,
     `track_renderers/style.ts`, `engine/export.ts`.
   - `buildTrackDataUrl()`, extracted as a pure function from
     `engine/arrow_client.ts::fetchTrackData`, with a URL test.
   - One SVG file snapshot per renderer (six kinds, vector mode; default and
     non-default `style`/`filter`). Fixtures are small `.arrow` files generated
     from the real Python kernels by a script, with a pytest that fails if a
     kernel's wire schema drifts from its fixture.
   - A form-model snapshot of the current `TrackSettingsPanel` for each kind.
   - A black-box test of `new GenomeBrowser(...).mount()` under jsdom with a
     recording `fetch` and fake timers. It pins request URLs, track order,
     status text, `localStorage` keys and JSON, the layout PATCH body,
     restyle-versus-refetch routing and the add-source merge. This goes a step
     past "pure logic", but it is the only net under the host rewrite and still
     needs no browser.
3. **Backend tests added on today's code:**
   - A subprocess test that blocks fastapi, pydantic, starlette, uvicorn and
     datashader, imports `constellation.viz`, `constellation.viz.cli`,
     `constellation.viz.frontend.build` and `constellation.cli.__main__`, and
     asserts no `constellation.sequencing*` module was loaded.
   - Literal-value pins for `session_id` and `source_id` (saved layouts and
     `localStorage` keys depend on them).
   - Hybrid mode over HTTP (`force=hybrid`) and `ClusterPileupKernel._emit_hybrid`,
     neither of which has a test today.
4. **CI.** Add a `Frontend (typecheck + vitest)` job to `.github/workflows/ci.yml`:
   Node 20, pnpm 9, `pnpm install --frozen-lockfile`, `pnpm typecheck`,
   `pnpm test`. Confirm the action versions in `release.yml` are still current
   before mirroring them. Making the job a *required* check is a manual
   branch-protection change.
5. **Packaging.** Add `pydantic>=2` to the `[viz]` extra in `pyproject.toml`; it
   is only transitive via fastapi today and Stage 1 relies on v2.
6. **Docs and stale references.** Bring `constellation/viz/CLAUDE.md` in line
   with the code: pushdown filters shipped; `read_pileup` wire schema; manifest
   schema v5; test counts; layout precedence; binding-id formats; the missing
   `reference-first-genome-browser.md`; anywidget, notebook panel and
   cross-panel coordination listed as not built. Fix the stale docstrings in
   `viz/server/session.py` and `viz/cli.py`, the manifest version in the root
   `CLAUDE.md`, and the `reference link` fallback in `constellation/cli/__main__.py`
   that writes a `session.toml` for a `Session.from_root` that no longer exists.
   Record this plan as `docs/plans/viz-modality-generalization.md` (this file).
7. **Rebuild the local bundle**; the one on disk predates the 2026-09-15 change.

---

## Stage 1 — Generalize the backend contracts (PR B)

```
constellation/viz/
  tracks/base.py            generic: TrackQuery base, TrackKernel ABC, registry
  server/session.py         generic: SessionLike / SourceLike Protocols + id helpers
  server/endpoints/         generic routes only
  modalities/__init__.py    NEW: Modality descriptor + registry
  modalities/genome/        NEW: query.py, session.py, endpoints.py,
                            tracks/ (six kernels + _alignment_view.py), DOMAIN_IMPORTS
```

1. **Tests only.** Route the 27 `TrackQuery(...)` call sites and the direct
   `Session.open(...)` calls through two helpers in `tests/_viz_fixtures.py`, so
   later commits touch the helpers, not 80 tests.
2. **Move.** `git mv` the six kernels and `_alignment_view.py` into
   `viz/modalities/genome/tracks/`, and the session dataclasses into
   `modalities/genome/session.py` as `GenomeSession` / `GenomeSource` with
   bodies unchanged. `server/session.py` shrinks to two Protocols
   (`session_id, label, modality, sources, warnings, saved_as`, `with_sources`,
   `summary`, `to_manifest`) and the id helpers. Protocols, not base
   dataclasses: zero-argument `super()` fails inside a `slots=True` dataclass
   subclass on Python 3.12. Update `viz/__init__.py`, `tests/test_imports.py`
   and test imports. Add the import-boundary test (step 7) in this commit.
3. **Modality registry.** `Modality(name, open_session, session_from_saved,
   inspect_source, request models, routers)`, where `routers` is a
   zero-argument factory that imports its FastAPI module only when `create_app`
   calls it. `server/endpoints/sessions.py` keeps list / open / manifest /
   add-source / delete-source / inspect-source, dispatching on a `modality`
   field that defaults to `"genome"`; request bodies validate against the
   modality's own model so a missing `reference_handle` stays a 422. `contigs`,
   `search` and `/api/references` move to `modalities/genome/endpoints.py` at
   the same URLs. Summary and manifest JSON gain `"modality": "genome"`. The
   modality guard goes in `endpoints/tracks.py::_bindings_for`, so a kernel
   never sees another modality's session on any route.
4. **Query models.** `TrackQuery` shrinks to `viewport_px` and `force`;
   `mode_extra` and `max_glyphs` (read by no kernel, sent by no client) are
   deleted. Models are stdlib keyword-only frozen dataclasses with range
   constraints in `field(metadata={"ge": …})`:
   - `GenomeQuery`: `contig, start, end`, with `end > start` in `check()`.
   - `CoverageQuery`: adds `samples`.
   - `ReadPileupQuery`: adds `samples, min_mapq`.
   - `ClusterPileupQuery(ReadPileupQuery)`: adds `cluster_view`, default `None`.

   A `GenomeTrackKernel` base holds `modality`, the default `query_model`,
   `vector_bp_per_pixel_limit`, and `response_headers`, which keeps
   `X-Track-View` on all six kernels as today. `get_data` resolves the kernel,
   collects `request.query_params` against `kernel.query_model` (`getlist` for
   sequence fields), validates with a cached `pydantic.TypeAdapter` (422), then
   calls `check()` (400). pydantic 2 enforces the stdlib field
   metadata (verified against pydantic 2.13), so the models import nothing
   outside the standard library.
5. **Saved sessions.** `SavedSession` gains `modality` (default `"genome"` when
   absent; no `schema_version` bump, following the `[options]` precedent),
   threaded through `write_saved`, `read_saved`, the POST and PATCH endpoints
   (PATCH rewrites the whole file, so an unthreaded field is silently dropped)
   and the CLI. `viz genome --saved-session` rejects other modalities and
   reports a bad file instead of raising.
6. **Acceptance test.** `tests/test_viz_modalities.py` registers a toy modality
   inside the test (a session class and one kernel with `ToyQuery(x0: float,
   x1: float)`) and drives open → list tracks → fetch data over HTTP, plus
   cross-modality 404s. If it passes without editing shared code, genome is one
   modality.
7. **Import boundary** (`tests/test_viz_import_boundary.py`, AST scan of
   `constellation/viz/`): `constellation.core.*` and `constellation.viz.*` are
   allowed everywhere. A domain import is legal only under
   `viz/modalities/<name>/`, inside a function, as `from … import name`, and
   only if `(module, name)` is listed in that package's `DOMAIN_IMPORTS`. For
   genome that is the six sites that exist today: `sequencing.reference.handle`
   (`parse_handle`, `resolve`, `read_meta_toml`, `list_installed`,
   `read_defaults`, `ReferenceNotInstalledError`),
   `sequencing.transcriptome.manifest.read_manifest_dir` and
   `sequencing.align.cigar.parse_cs_long_mismatch_positions`. Amend the DAG text
   in the root `CLAUDE.md` and the invariants in `viz/CLAUDE.md`.

---

## Stage 2 — Factor the genome-specific frontend out (PR C)

```
src/
  engine/     arrow_client, svg_layer, hybrid_layer, export, popover (NEW)
  panels/     NEW generic layer: Panel, PanelStack, layout, settings_schema,
              SettingsPanel, DatasetManagerPopover, OptionsPopover, style, panels.css
  widgets/    PathInput, FilePicker (unchanged)
  modalities/genome/
              GenomeBrowser, GenomeBrowserForm, viewport (Locus, ViewportBus,
              zoom/pan, scales), renderers/, types, genome.css
  dashboard/  shell only; viz_registry keeps its dynamic import of the widget
```

The generic layer is a component the host drives, not a framework that calls
back through sixteen hooks. Track listing, the toolbar, overview and ruler,
export decorations and the dataset model stay ordinary `GenomeBrowser` code.
`panels/` needs three small contracts:

```ts
interface PanelDriver {            // the only callbacks panels/ makes
  viewKey(): string | null;        // null = nothing to render
  contentWidth(): number;
  fetch(p: PanelState, signal: AbortSignal): Promise<{ data: unknown; count: number }>;
  draw(p: PanelState, data: unknown, svg: SVGSVGElement, size: Size): void;
}
interface LayoutStore { load(): Persisted | null; save(layout, options): void }  // one writer
interface PanelKind { kind: string; order: number; unit: [string, string];
  settings?: SettingsSchema; pushdown?: Record<string, (v: unknown) => string | undefined> }
```

Commits, each with `tsc`, vitest and all Stage 0 snapshots green:

1. **Moves**, plus a TS boundary test: `engine/`, `panels/` and `widgets/` never
   import `modalities/`. Genome types leave `engine/` first so the compiler
   lists the remaining seams.
2. **`engine/popover.ts`.** One helper for the four dismiss blocks and three
   position blocks, with the existing clamp difference kept as a parameter.
3. **Layout logic** lifted verbatim out of `GenomeBrowser.ts` behind one
   `LayoutStore`. Layout and options stay a single PATCH; the server rewrites
   the whole file, so two writers would lose updates.
4. **`PanelKind` descriptors.** Each renderer exports `order`, `unit` and
   per-key `pushdown` encoders that reproduce today's URLs exactly. This deletes
   `KIND_ORDER`, `pluralUnit`, `PUSHDOWN_FILTER_KEYS` and the
   `min_mapq`/`cluster_view` special cases.
5. **Settings schema**, ported with zero diff in the form-model snapshots.
   Controls: number, text, select, toggle, color, allow-list, palette. Labels,
   defaults and option lists may be functions of `{meta, style, filter, host}`,
   which covers paired metadata arrays, fallback lists, conditional visibility
   and inherited defaults without an escape hatch. Each renderer exports one
   `DEFAULTS` constant used by both its drawing code and its schema. The 600
   lines of `if (kind === …)` in `TrackSettingsPanel.ts` go away.
6. **`Panel` and `PanelStack`.** `Panel` owns one panel's chrome, state, fetched
   cache and restyle; `PanelStack` owns order, drag-reorder, resize, visibility
   and the empty state. Kept separate on purpose: the MS browser's 2×2 grid
   reuses `Panel` under a different container.
7. **CSS split** into `panels.css` and `genome.css`, class names kept.
   `vite.config.ts` discovers entries by glob, matching `build.py::known_entries()`.
8. **Parity exceptions**, one commit each with its snapshot diff, then the
   dashboard tidy (genome-only types out of `dashboard/types.ts`; `VizForm`
   remember-keys namespaced per tool with the old key as fallback).

**Parity exceptions (the only intended behavior changes).** Drawn track output
stays byte-identical.
1. Mode and motif pickers read `modes_in_data` / `motifs_in_data`. Today the
   panel reads `meta.modes` / `meta.motifs`, which the kernels never send, so
   the hard-coded fallback lists always win.
2. Controls no renderer reads are removed: General → Opacity and Show legend,
   and unread `palette.default` rows. The two label-font rows move to
   `gene_annotation`, their only reader.
3. `cluster_pileup` members view shows the alignment controls and defaults it
   actually uses, and the panel rebuilds when the view changes.
4. The "Show labels" checkbox reflects the toolbar toggle when unset.
5. Document and window listeners are released on `dispose`.
6. Removing the duplicated inline styles in `index.genome.html` drops two rules
   that exist only there (`.track` bottom border, `.track-header` spacing), so
   the standalone page matches the embedded browser.

**Known issues left alone.** The default colour swatch can differ from the drawn
colour (fixing it changes drawn output). Saved `[options]` are written but never
read back into the browser. `_normalize_options` keeps only `clip_svg`.

**Found during Stage 0, not yet scheduled.** Two more defects surfaced while
writing the characterization tests. Both are pinned as current behavior; decide
before PR C whether they join the parity exceptions.
- The gear popover's "Visible sources" filter offers `reference` / `derived`,
  but the renderer compares against each feature's own `source` column (the GFF
  source, e.g. `RefSeq`), so unticking either box hides every feature with a
  non-empty source.
- Adding a source can persist duplicate `display_order` values
  (`computeInsertOrder` shifts siblings whose saved order has not been restored
  yet). The on-screen order is still correct because the sort is stable.

---

## Seams deliberately left open (not built here)

- **Viewport model.** `PanelDriver.viewKey()` and `draw()` are opaque to
  `panels/`, so a linked RT × m/z viewport and y-axes need no change there.
- **Fetch scheduling.** The serial loop is one `PanelStack` method over
  `PanelDriver.fetch`. Parallel fetch, client cache, overscan and server-side
  decimation replace that method.
- **Layout containers and export.** A grid container is a sibling of
  `PanelStack`; `buildCompositeSvg` stays a vertical stack.
- **Hybrid schema.** Extents stay int64. Widening them to float is a wire change
  with no consumer yet, so it belongs to the first float-axis kernel.
- **Saved sessions.** Only the `modality` discriminator is added.
- **Dashboard.** One tab per command path, and restored panels restart at the form.

After this plan, adding a modality is one `viz/modalities/<name>/` package, one
`src/modalities/<name>/` folder, one `viz <name>` CLI subcommand and one
registry descriptor.

---

## Verification

Per PR:
- `pytest tests/test_viz_*.py tests/test_imports.py` and `ruff check .`
- `pnpm typecheck && pnpm test` in `constellation/viz/frontend/`
- `python -m constellation.viz.frontend.build` (both entries build)
- `pytest tests/test_viz_e2e.py` (real uvicorn boot, Arrow round-trip)

Stage-specific:
- **PR B:** the toy-modality, import-boundary and base-install tests pass; the
  nine existing data-endpoint tests pass with no URL, status-code or header change.
- **PR C:** renderer, form-model and host snapshots are unchanged through
  commit 7; each parity exception shows exactly its own diff.

Manual, once per PR. jsdom cannot exercise real drag-and-drop, pointer capture
or CSS, so each PR ends with a check in a browser against a real saved session.
Confirm: every track renders; reorder, resize,
collapse and hide survive a reload; the gear popover works for each kind; add
and remove a dataset; Save SVG with clip on and off; the standalone
`constellation viz genome` page still looks right.
