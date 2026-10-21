# Mass-spec browser, first generation

## Status

- **Plan approved** by the PI on 2026-10-19. The approved original is
  `~/.claude/plans/the-next-major-feature-cached-hearth.md`. Work is on
  `feat/ms-browser`.
- **PR A (payload survey):** done; the results are under "Payload survey"
  at the end of this file. Two of them change later PRs: the in-memory peak
  index cannot be built for a real run on this machine, so PR F's streaming
  build is required; and an XIC bounded to ±2 min needs no index. Bundles
  were measured on ext4 only: putting one on a Windows drive (`/mnt/e`)
  means writing to the PI's data drive, which waits for the PI's go-ahead.
- **PR B (domain, slice 1):** implemented. `massspec/scans.py`
  (`open_acquisition`, `scan_table`, `isolation_window_table`,
  `chromatogram`, `nearest_scan`, `parent_scan`, `dependent_scans`,
  `spectrum`, `spectra`, `default_tolerance`), `peptide.ions.ion_number` /
  `bond_position` (the MSP and Koina readers now use them), the mzPAF
  projection fix (`length=` is now required), base peaks derived on import,
  `core.signal.decimate.decimate_trace`, and `quant.peak_index.find_peak_index`.
  Departures:
  - Decimation is M4 (first, last, minimum and maximum per pixel column,
    returned as indices) in torch, not min/max with Arrow `group_by`: the
    core layer's numerics are torch by rule, Arrow has no argmin aggregate,
    and indices let a caller keep every column of its table.
  - Only the synthetic-run builders moved to `tests/_massspec_fixtures.py`;
    the fake-jar experiment helpers stay in `test_massspec_search_workflow.py`
    until PR F needs them.
- **PR C (server, slice 1):** implemented. `viz/modalities/massspec/`
  with `ms_chromatogram` and `ms_spectrum`, the acquisitions, windows,
  scans and scan-header routes, and `constellation viz massspec`.
  Departures:
  - The flag is `--acquisition DIR`, not `--bundle`: it names the concept,
    not Constellation's internal term for the directory.
  - One viz-core change: the tracks endpoint turns a `ValueError` from
    `kernel.fetch` into HTTP 400. The mass-spec kernels build their table
    when `fetch` is called (they return an iterator rather than being
    generators), so a refusal ("this bundle records no base peaks") reaches
    the client as an error instead of an empty stream. Genome's generator
    kernels are unaffected.
- **PR D (engine + host, slice 1):** implemented. `engine/{axis, interval,
  pan_zoom, bus}.ts`, `panels/{PanelGrid, scheduler}.ts`, `buildGridSvg`,
  `fetchArrow`, optional `Panel` chrome with `addHeaderButton`, the
  `massspec` modality (browser, scan index, two renderers), the standalone
  `massspec` entry, the dashboard descriptor, and the `viz-form` class fix
  in `TaskPanel`. Checked against the real wide-window run in headless
  Chrome. Departures:
  - The grid persists its own state (`constellation.massspec.grid.v1`:
    weights, collapse, focus, per-cell style) instead of adding a `slot` to
    `panels/layout.ts`. Grid cells are fixed by the data's shape, so the
    stack's layout model (order, visibility, per-source keys) does not
    apply, and genome's layout code stays untouched.
  - Axes are drawn as plain SVG rather than through d3-axis, so they render
    the same under jsdom and in an export; a seconds axis shown in minutes
    ticks on round minutes (`AxisFormat.per`).
- **PR E (genome onto the engine):** implemented. Genome's `interactions.ts`
  wraps `attachPanZoom1D` (`zoomLocus` is `zoomAbout` with whole-base-pair
  rounding), `ViewportBus` is a locus on a `TypedBus`, and
  `PanelStack.render` goes through the `RenderScheduler`: panels are fetched
  in parallel, a panel whose data is for the current view is drawn again
  without a request, and a late response for a view already left is dropped
  (a new host test holds one back to check). Checked by headless-Chrome
  screenshots of the synthetic genome session (`tests/_viz_frontend_fixtures.py`)
  before and after, at the default view and after a Go-to plus a drag pan:
  no pixel differs by more than one level. What changes for a user:
  - Fewer requests. Expanding a collapsed track, showing a hidden one,
    toggling Labels or changing one track's server-side filter used to
    refetch every visible track; now only tracks whose data is stale are
    fetched. Seven host tests that pinned the old refetches were updated.
  - Panning uses pointer events with pointer capture, so no window
    listeners are held. Presses on the browser's own controls (track
    headers, resize handles, the overview bar, form fields) are left to
    them: before, a drag on the overview both centred and panned the view,
    and a resize drag could pan too. Shift-drag still pans (the engine's
    brush is off for genome), double-click does nothing, as before.
  - A mostly horizontal scroll with Shift held now pans; before, Shift
    turned it into a zoom in, whatever its direction.
  Departure: `zoomLocus` and `attachPanZoom` stay as thin genome-specific
  wrappers rather than being removed, so their callers and tests are
  unchanged.
- **PR F (domain, slice 2):** implemented.
  - The peak index is built streaming (`quant.peak_index`): the windows are
    found from three columns, each batch is appended to its partition's
    spill file, and each partition is then sorted alone. On synthetic DIA
    and binned DDA runs it writes exactly what the in-memory build wrote
    (manifest and every partition), reading seven peaks per batch or all at
    once. On the reference runs: DDA, 195M peaks, 6 min and 6.8 GB peak
    (the in-memory build was killed at 23.2 GB); wide-window DIA, 294M
    peaks, 10 min and 9.6 GB, 3.5 GB of index. Also `index_slice` (an m/z
    slice of one partition, row-group pruned), `index_partition` (a window's
    partition by its bounds) and a `row_group_size` option.
  - `scans.xic` (one zero-filled trace per m/z over a level and DIA window,
    from the index or from `peaks.parquet` within an RT window) and
    `scans.locate_precursor` (the covering window and the apex scans).
  - `spectra.annotate`: `annotate_spectrum` (mzPAF labels, ion number,
    charge, loss and error per peak; the intact precursor as `p`) and
    `ion_ladder` (the theoretical ions, numbered as labels are).
  - `massspec.results`: `search_units`, `identifications` (peptide-centric
    quantities or spectrum-centric matches with their scans, joined with
    peptides, scores and proteins; partition-filtered), `acquisition_bundles`
    (the plan's linking rules), `precursor_detail`, `library_spectrum`,
    `searched_library_spectrum` (EncyclopeDIA's own index on sequence and
    charge, the entry recognised by its content-derived precursor id) and
    `rt_mapping`.
  - Checked on eight random identifications of the searched wide-window
    run: the apex scan is within 0.0001 s of `rt_observed`; `xic` from the
    index equals `xic` from `peaks.parquet` (35–50 ms against 250–340 ms
    for ±2 min) and, on the scans `extract_xic_scan_major` reads (the
    integration window), equals it; annotating the apex MS2 scan labels
    12–53 peaks; and no b/y label of the stored result spectra disagrees
    with the ion ladder. `identifications` of the run: 31,836 rows in
    1.2 s; `precursor_detail` 0.1–0.2 s; `library_spectrum` under 0.1 s;
    a searched-library lookup 20 ms.
  Departures:
  - `xic` takes, in each scan, the peak within tolerance nearest the
    target (the more intense of two equally near), as the existing XIC
    extraction does, so the two agree; "most intense" would pull in an
    interferer sharing the window.
  - `quant.targets.targets_from_experiment` was left as it is rather than
    delegating to `identifications`: for a spectrum-centric search it lists
    the quantified precursors, while `identifications` lists the scored
    matches, and its callers want the former.
  - `searched_library_spectrum` does not re-hash the library (gigabytes);
    a file whose size differs from what the search recorded is not read.
- **PR G (identifications and targets):** implemented, in two commits.
  - Server: the `ms_xic` kernel (one trace per m/z, labelled; at most 64
    per request), annotation on `ms_spectrum` (a peptide, charge,
    tolerance, ion series, fragment charges and losses as query fields),
    and three routes: `…/acquisitions/{key}/identifications?max_q=`
    (Arrow, every linked experiment's, with an `experiment` column),
    `…/acquisitions/{key}/precursor?experiment=&unit=&precursor_id=` (what
    to trace for one identification: window and apex scans, isotopes, the
    most intense stored fragments with their quantified flags, the ion
    ladder, the median chromatogram, the search's tolerances and the
    library's retention time mapped onto the run) and `resolve?q=` (a
    peptide typed by hand, ProForma with an optional `/charge`).
    `viz massspec --experiment DIR` attaches searches. Measured on the
    wide-window run: identifications 0.3 s the first time, then 30 ms;
    the precursor route 1.2 s; an XIC 0.2–0.4 s; an annotated spectrum
    0.13 s.
  - Frontend: the search box with suggestions (identifications by peptide
    or protein, a peptide to trace when nothing identified it, `scan N`,
    `rt M`, an m/z), the identifications drawer (sortable, filterable,
    300 rows drawn at a time) with an Ions tab (the b/y table lit with
    what the shown spectrum matched, and a coverage line) and a Scan tab
    (the scan's header), the target chip, back and forward, `[` / `]` to
    step identifications, target traces with the integration window, the
    reported retention time, the predicted time and the engine's median
    chromatogram, annotated MS2 spectra with a mass-error strip, isotope
    markers on MS1, a peak tooltip, traces added by clicking a peak, CSV
    and spectrum-reference buttons on every cell, and "Build index", which
    runs `massspec chromatogram build-index` through the dashboard's
    command runner and polls it. Checked in headless Chrome on the
    searched wide-window run (31,836 identifications) and on the DDA run
    alone with a typed peptide.
  Departures:
  - The precursor route takes the experiment and the search as well as
    the id, since a content-derived precursor id names the same precursor
    in every search; the masses route moves to H1 with the ruler that
    uses it, and the RT-mapping route to H2.
  - Identifications are sent at q ≤ 0.01, not every scored precursor, and
    the drawer shows q-value rather than score. Their 64-bit ids stay
    strings on the client; a JavaScript number would round them.
  - With a target, a row's traces replace its TIC rather than sitting on
    a faint one. On one intensity axis the TIC (up to 1.1e9 in the
    wide-window run) would flatten a precursor's isotopes (6.5e5 for
    HAVSEGTK/2), and a second axis is not an option.
  - Without a peak index, traces are extracted over the target's
    integration window ±2 min (at most 10 min around the time shown
    otherwise), and the chip bar says so. On a DDA MS2 row they always
    cover only the target's time or the time shown: tracing an ion over
    every MS2 scan of the reference DDA run took 9.7 s and 9 MB, against
    0.25–0.5 s for 4–10 min.
  - A row holds at most two added traces. The series palette has eight
    colours and a target's fragments take six, so no colour is used
    twice; a third replaces the oldest.
  - On a DDA run the MS2 cell of a typed peptide opens on the scan that
    isolated it (or its M+1) with the most signal, and ← / → step among
    those scans while the target is shown, as Skyline steps through a
    precursor's MS/MS.
  - "Copy spectrum reference" copies `run:scan:N`; a USI waits for a
    dataset accession.
  - Found on the way: in the wide-window result, EncyclopeDIA's reported
    retention time (`entries.RTInSeconds`) lies outside its own
    integration window (`peptidequants.RTInSecondsStart` / `Stop`) for
    2,731 of 31,079 precursors (the window's centre never does). The
    browser draws both as the jar recorded them.
- **Review of steps D–G (Codex, 2026-10-21):** thirteen findings, each
  reproduced by a test that fails on the code before the fix.
  - `results.identifications`: a ScribeTwo match without a q-value (it
    scores only the best match of each precursor) passed any q filter; it
    is now accepted with its precursor's best match. A quantitative
    report's rows were listed as the run's identifications, and could not
    be opened; only searches are read now, unless `unit` names a report.
  - Tolerances come from what the engine recorded running with
    (`engine_settings`; ScribeTwo adjusts what it is given, 0.35 → 0.42 Da
    in the stand-in), else from what was given.
  - The precursor route takes the match's `scan`: a later match of a
    precursor showed its own spectrum at the first match's time and q.
  - `scans.xic` from an index with float32 times dropped the first or last
    scan; its time bounds are widened past that rounding (scan membership
    still decides).
  - `PanelStack` and `PanelGrid` key a request by the panel's pushed-down
    filter values too, so a request still running for the old value is
    not taken for the new one (genome's MAPQ and cluster view).
  - `PanelGrid` places every header and cell explicitly, and `[hidden]`
    wins over their display rules: maximizing a column or cell had put
    cells in the wrong tracks (checked in headless Chrome).
  - Axes carry their stroke, fill and font size as attributes, and a grid
    export sets the page's typeface on its root: an exported SVG has no
    stylesheet, so its axis lines were not drawn.
  - The browser drops a slower reply for an acquisition already left;
    draws a target's DDA scan events from the whole scan table (the
    server's reduction could drop them); fetches a reduced (profile)
    spectrum again for the m/z window shown; restores each row's m/z
    window with Back / Forward; and stops lighting ions once the MS2 scan
    shown did not isolate the target.

## Context

The viz layer was generalised (PR #98, 2026-10-10) so genome became one *modality* among many. The engine work a mass-spec browser needs was deferred on purpose: parallel fetch, generic axes, y-axes, linked viewports, caching and decimation. Designing it before real MS payloads existed would have made the "generic" layer MS-shaped by guesswork, the way it had become genome-shaped. Since then, PR #100 (EncyclopeDIA phases 0–7) produced the data such a browser reads: converted runs (bundles), experiment directories with content-keyed identification tables, and spectrum annotation primitives.

This plan builds the first-generation browser:
- an **n × 2 grid**: rows are MS levels (MS1, MS2, …), columns are chromatogram and spectrum;
- the grid collapses or maximizes by cell, row or column;
- a **navigation and search bar** sits above it;
- it includes the common features of Skyline, Xcalibur Qual Browser, TOPPView/MZmine, Lorikeet/IPSA and EncyclopeDIA's viewer, prioritising the cheap ones.

The survey results from PR A are recorded under "Payload survey" at the end of this file.

### Decisions (PI, 2026-10-19)
- **Input:** a converted run (bundle) is enough to browse raw data. Adding an experiment directory brings identifications, annotation and peak boundaries. Several acquisitions can share one session.
- **Acquisition types:** DIA and DDA, both in v1.
- **Extras in the first generation:** library mirror plot, RT × m/z 2D map, multi-run XIC overlay, and a Δm/z ruler.
- **Genome browser:** moves onto the new engine within this effort, right after the engine lands.

### What the measurements already settle
These numbers come from read-only SQLite queries of the reference spectra files in `/mnt/e/261010_u2os` (Stellar, ion trap, 100-min runs).

| Run | MS1 scans (avg peaks) | MS2 scans (avg / max peaks) | Windows |
|---|---|---|---|
| GPF 400–500 | 3,715 (1,143) | 182,035 (820 / 1,530) | 49 |
| Wide window 350–1200, 8 m/z | 2,242 (1,148) | 237,651 (1,227 / 1,692) | 106 |
| DDA | 8,977 (1,129) | 179,532 (1,031 / 1,665) | ~160k distinct |

What follows from them:
- **The client fetches whole traces and whole scans, then zooms locally.** A DIA chromatogram is at most about 4k points, a DDA MS1 trace about 9k, and a spectrum at most about 1.7k peaks. So no client cache or overscan gets built: the measurements make them unnecessary.
- **Only the DDA all-MS2 trace (~180k points) is decimated**, by min/max per pixel on the server. A 2D map rasterizes ~4M MS1 points per run, which is cheap.
- **The default tolerance is not 20 ppm.** These are ion-trap scans. The tolerance comes from the search that made the identification; failing that, from the scan's analyzer (ITMS → 0.35 Da, FTMS → 10 ppm).
- **Two feasibility risks remain for PR A to measure:**
  - `build_peak_index` loads and sorts everything in memory. The wide-window run has about 292M MS2 peaks, roughly 15 GB plus copies.
  - Cold reads from `/mnt/e` measured 9–17 MB/s.

---

## The browser

### Layout
```
┌ nav: ◀ ▶ │ acquisition ▾ [compare] │ search……………… │ target chip ✕ │ ◀ id ▶ │ ◀ scan# ▶ │ RT [..] m/z [..] │ ⚙ ⤓ ┐
├────────────┬──────── Chromatogram ⤢ ▸ ─────────┬──────── Spectrum ⤢ ▸ ────────────┬─ Identifications ─┐
│ MS1 ⤢ ▸    │ TIC/BPC · precursor isotope XICs   │ scan · isolation band · isotopes │ sortable table     │
│ MS2 ⤢ ▸    │ window TIC · fragment XICs / DDA   │ annotated · mirror · error strip │ (collapsible)      │
│ window ▾   │ scan-event sticks                  │                                  │                    │
└────────────┴────────────────────────────────────┴──────────────────────────────────┴────────────────────┘
```
- **Rows:** one per MS level present in the scan table. The row header holds the DIA window or DDA scan-filter picker.
- **Focus:** maximize a cell, a row or a column (header buttons; double-clicking a cell header; Esc restores).
  - A collapsed row shrinks to its header.
  - A collapsed column shrinks to a 28 px strip.
  - Splitters set row and column weights.
- **Layout persistence:** per modality, not per acquisition, so switching runs keeps the layout.
- **Identifications drawer:** a client-side table of q-value, score, RT, intensity, charge and protein. Selecting a row sets the target.

### Linking model
- The chromatogram column shares one **RT window** (pan and zoom are linked) and an RT crosshair on hover.
- Clicking a chromatogram picks the nearest scan of that row's level and window, and shows it in the row's spectrum cell.
- Scans in different rows follow each other:
  - Choosing an MS2 scan shows its parent MS1 scan: the master or precursor scan, else the preceding MS1.
  - Choosing an MS1 scan shows the next MS2 scan in the current DIA window, or for DDA the first scan it triggered.
- Spectrum cells have independent m/z windows. The MS1 spectrum shades the current MS2 isolation window and marks the target's isotopes.
- **Selecting a target** (a precursor):
  - RT zooms to its peak boundaries ± margin.
  - The MS1 chromatogram shows isotope XICs (M, M+1, M+2) over a faint TIC.
  - The MS2 row switches to the covering DIA window and shows the top-N fragment XICs. The fragments come from the library, the quantified set or the theoretical ladder.
  - The chromatograms shade the boundaries and draw lines at the apex and the predicted RT.
  - Both spectrum cells jump to the apex scans, and the MS2 spectrum is annotated.
- **DDA:** the MS2 chromatogram row shows scan-event sticks. With a target set, only sticks whose precursor m/z is within tolerance are shown. A PSM selects its exact scan.
- **Ad-hoc traces:** clicking a spectrum peak adds that m/z's XIC to the row's chromatogram as a removable chip (Xcalibur's habit).
- **History and keys:** back/forward history records target, scan and acquisition changes, not pans. Keys:
  - `←`/`→` step the scan in the focused row;
  - `[`/`]` step to the previous or next identification;
  - `/` focuses search, `f` fits the view, `Esc` clears focus.

### Search grammar
The parsing happens on the client; only chemistry goes to the server.

| Input | Result |
|---|---|
| `PEPTIDEK`, `PEPTIDEK/2`, ProForma (`EM[Oxidation]EVTSESK/2`) | Matching identified precursors, through content-derived `precursor_id`. If none, an **ad-hoc target** whose m/z comes from the server's `resolve` route. Works without an experiment. |
| Accession, gene or description text | Proteins, then their precursors |
| `512.27`, `mz 512.27` | An ad-hoc MS1 XIC at that m/z |
| `scan 12345`, `#12345` | Jump to that scan |
| `rt 35.2` | Centre the RT window there (minutes) |

### Features
**Core (v1):**
- The grid, linking, search and drawer described above.
- A hover tooltip showing m/z, intensity, ion label and ppm error.
- Absolute, relative or log intensity.
- The top-N peak labels, with annotated ions labelled first and collisions avoided.
- A scan header strip: filter string, analyzer, activation, collision energy, injection time, AGC and precursor.
- Toggles for ion type, charge and loss.
- A mass-error strip under the annotated spectrum, and a b/y ion table with a sequence coverage map.
- An overlay of EncyclopeDIA's median chromatogram (`precursor_chromatogram`), and highlighting of the quantified ions (`fragment_quant`).
- A scan-filter dropdown for DDA and mixed methods.
- CSV export of a cell; SVG export of a cell or of the whole grid.
- "Copy spectrum reference" (run:scan; a full USI waits for a dataset accession).
- An "index missing → build" action that launches `massspec chromatogram build-index` through the dashboard's command runner. The server never writes.

**The four extras the PI chose:**
1. **Library mirror plot:** a toggle on the MS2 spectrum cell.
   - The reference is either the **searched** library, looked up per precursor through EncyclopeDIA's own `entries(PeptideModSeq, PrecursorCharge, SourceFile)` index, or the experiment's result library (`fragment` table).
   - Similarity comes from `massspec.spectra` (`fragment_basis` → `align_to_basis` → `compare_spectra`), shown as cosine and spectral angle.
2. **Δm/z ruler:** pick two peaks, or shift-drag between them, to see Δm/z and Δmass at charges 1–3.
   - It matches the difference against residue and common-modification masses. The server provides these once, from `core.chem` and `unimod.json`.
3. **Multi-run XIC overlay:** a "compare" mode in the nav bar with several acquisitions selected.
   - The MS1 row shows the target's monoisotopic XIC per run; the MS2 row shows the summed quantified-fragment XIC per run.
   - The RT axis is either raw or **aligned to library RT** through each run's `rt_alignment` fit (a monotone interpolation of the fitted pairs).
   - A per-run quantity bar chart comes from `precursor_quant`.
4. **RT × m/z map:** a Trace | Map switch on chromatogram cells, using MS1, or the current window's MS2.
   - The x axis is the column's shared RT axis and the y axis is m/z.
   - Datashader rasterizes the points on the server. The target's m/z band and the isolation windows are drawn on top.

**Deferred past the first generation:**
- MS saved sessions; these need viz-core changes (below).
- QC panels for RT alignment and mass calibration.
- Chromatogram smoothing.
- Spectra averaged over an RT range.
- dotp and idotp scores.
- Live Koina prediction as a mirror source.
- Migrating `extract_xic_indexed` onto the new index slice.

---

## Architecture

### Domain read API (massspec; usable from a notebook)
Every public surface is a function. The viz import boundary admits only functions and exception classes from domain modules.

- **`massspec/scans.py`** (new), the acquisition side:
  - `open_acquisition(bundle)` handles both bundle kinds and returns:
    - paths, plus the index root if there is one;
    - DIA or DDA and whether windows are staggered (reusing `acquisition_structure`);
    - profile mode and the analyzers per level;
    - the RT range and the scan counts.
  - `scan_table(bundle, columns=NAV_COLUMNS)`, with a derived window id and parent scan.
  - `isolation_window_table(bundle)`.
  - `spectrum(bundle, scan)` and `spectra(bundle, scans)`, reading through row-group pruning on `scan`.
  - `chromatogram(bundle, level, kind="tic"|"bpc", window=None)`, from `scan_metadata`.
  - `xic(bundle, mz, level, window, tolerance, unit, rt_range=None, index=None)`, which uses the index when present and otherwise an RT-pruned read of `peaks.parquet`.
  - `nearest_scan`, `parent_scan` and `locate_precursor` (the covering window and the apex MS1/MS2 scans, derived from `rt_observed`; nothing stores the apex scan).
- **`massspec/results.py`** (new), the experiment side:
  - `identifications(exp, unit=, acquisition_id=, max_qvalue=0.01)` joins precursor_quant, precursor, peptide, precursor_score and protein. For spectrum-centric searches it uses PSM rows with their scan.
  - `acquisition_bundles(exp, candidates=())` links acquisitions to bundles (rules below).
  - `precursor_detail(...)` returns scores, proteins, boundaries, `fragment_quant`, the median chromatogram and the search's tolerances.
  - `library_spectrum(...)` returns the result library's spectrum.
  - `searched_library_spectrum(...)` makes one indexed SQLite query (`?mode=ro&immutable=1`) on the library path recorded in the search manifest.
  - `rt_mapping(exp, unit, acquisition_id)`.
  - `quant/targets.py::targets_from_experiment` then delegates to `identifications`.
- **Linking bundles to acquisitions**, in order:
  1. The search manifest's `inputs.source` is a bundle path (checked by `source_sha256`).
  2. A `.raw` input → `<exp>/bundles/<stem>`, checked by SHA-256.
  3. Searched from a spectra file → an attached bundle whose `source_file` resolves through `Acquisitions.id_for_source`. If nothing matches, the identifications still show, and the raw-data cells say "attach a bundle (`massspec encyclopedia import-spectra`)".
  4. Merged GPF → `merged_from`, each fraction resolved by rules 1–3. `locate_precursor` picks the fraction whose windows cover the precursor.
- **`massspec/quant/peak_index.py`:** add `find_peak_index`, `index_slice(index_root, level, window_id, mz_lo, mz_hi, rt_lo, rt_hi)` (row-group pruned) and a `row_group_size` parameter.
  - **Conditional on PR A:** a streaming build (spill to per-partition writers, then sort each partition) if building the wide-window index is infeasible in memory.
- **`massspec/peptide/ions.py`:** a public `ion_number(ion_type, position, length)` / `bond_position(...)`, moved out of `io/msp/_annotate.py`.
- **`massspec/annotation/mzpaf.py`:** fix the latent bug in `fragment_row_to_mzpaf` and `fragment_table_to_mzpaf`, which treat the 0-based bond `position` as a 1-based ion number. Every label would come out wrong; nothing calls them in production yet.
- **`massspec/spectra/annotate.py`** (new): `annotate_spectrum(mz, intensity, modseq, charge, tolerance, unit, ion_types, max_fragment_charge, neutral_losses)` returns one row per peak, with an mzPAF label, ion type and number, charge, loss and error. It wraps `peptide.match.assign_fragments`.
- **`core/signal/decimate.py`** (new): min/max decimation per x bin, written with Arrow `group_by`. It is generic, so HPLC and electrophoresis traces can use it later. It is imported inside functions to keep torch out of CLI startup.
- **`massspec/io/encyclopedia/_dia_import.py`:** fill `base_peak_mz/intensity` from the stored peaks, so BPC works for imported bundles.

### Server modality (`viz/modalities/massspec/`)
Modelled on genome and on the toy modality in `tests/test_viz_modalities.py`. **The viz core needs no change for v1.**

- **Session:**
  - `MassSpecSession(sources: MsSource[], acquisitions: MsAcquisition[])`.
  - Sources are `kind ∈ {acquisition, experiment}`; both are directories.
  - `inspect_source` dispatches on `core.manifests` manifest kind.
  - Experiments resolve their bundles automatically, deduplicated by path.
  - The session anchor is stored and carried through `with_sources`, so removing the first source does not change `session_id`.
- **Bindings:** one per kernel and acquisition that has a bundle.
  - The id is `"{kind}:{acq.key}"`, stable when sources are added or removed.
  - `config` carries the experiment links, because `fetch` never sees the session.
- **Kernels:** all `VECTOR` except the map. Query models are frozen keyword-only dataclasses, with no field named `session` or `binding`.

| Kind | Key query fields | Reads |
|---|---|---|
| `ms_chromatogram` | `level, trace∈{tic,bpc,scans}, window, rt_start/end, max_points` | `scan_metadata.parquet` |
| `ms_xic` | `level, window`, `mz[]` **or** `modified_sequence+charge`, `n_isotopes, ion_types, max_fragment_charge, top_n, fragment_source∈{library,quantified,theoretical}, unit, precursor_id, tolerance(+unit; None → search/analyzer default)` | index, or RT-pruned peaks |
| `ms_spectrum` | `scan`, `mz_start/end` (profile only), annotation params | `peaks.parquet` |
| `ms_reference_spectrum` | `source∈{searched,result}, unit, precursor_id` | library SQLite or the experiment `fragment` table |
| `ms_map` | `level, window, rt_start/end, mz_start/end`; HYBRID with a float64 extent schema the kernel declares itself (no core change) | MS1 / window peaks |

- **Trace wire schema:** includes a `decimated` column. It cannot be a header, because `response_headers` never sees the binding.
- **Routes** sit under `/api/massspec/sessions/{sid}/…`. Genome already owns `/api/sessions/{id}/search` and its router is mounted first, so the MS routes need their own prefix.
  - `acquisitions` (JSON: levels, DIA/DDA, RT range, default tolerances, `has_index`, experiment links)
  - `…/windows`
  - `…/scans` (Arrow, for nearest-scan lookup and keyboard stepping on the client)
  - `…/scans/{scan}` (header JSON)
  - `identifications` (Arrow, sent whole: 32k–76k rows is about 3–7 MB)
  - `resolve?q=`
  - `precursors/{id}`
  - `rt-mapping`
  - `masses`
- **The selected target travels as query parameters,** not as server-side state.
- **Caches:** process-level LRU caches with a lock and a byte cap, keyed by `(path, mtime)`. They hold scan tables, decoded peak row groups, index partitions and identifications.
- **New rasterizer:** `viz/raster/` gains `rasterize_points(x, y, w, x_range, y_range, …)` with float extents for `ms_map`. The client's `engine/hybrid_layer.ts` then has to honour `extent_y0/y1`.
- **CLI:** `viz massspec [--experiment DIR …] [--bundle DIR …] [--root] [--host/--port/--no-browser]` in `viz/cli.py`, which also makes the dashboard sidebar list it. Kernel modules import nothing heavy at module level.

### Frontend engine (generic; `engine/`, `panels/`)
- **`engine/axis.ts`:**
  - `plotFrame(size, margins)`;
  - `xAxis` / `yAxis` → `ContinuousAxis{domain, scale, invert, ticks, format}`;
  - `drawAxes(svg, frame, axes, uid)` returns a clipped plot group;
  - `autoYDomain`;
  - `AXIS_FORMATS`: RT in minutes, m/z (decimals follow the span), intensity in SI or scientific notation, percent.
  - The left margin is fixed per column so the RT plot areas line up across rows. RT stays in seconds on the wire.
- **`engine/interval.ts` + `engine/pan_zoom.ts`:**
  - `clampInterval`, `zoomAbout` and `panBy` over `IntervalLimits{min, max, minSpan, round?}`;
  - `attachPanZoom1D({surface, plotRect, get, set, limits, onClick, onHover})`: wheel zooms, drag pans, shift-drag zooms to a range, double-click fits, and a click selects (under 4 px of movement); uses pointer capture.
- **`engine/bus.ts`:** `TypedBus<E>` with `on`, `emit` and `dispose`. The event vocabulary belongs to the modality.
- **`panels/scheduler.ts`:** `RenderScheduler` runs bounded parallel fetches with a generation counter. Latest wins: a stale result is dropped, and a panel whose data key is unchanged redraws without a fetch. This also fixes the existing `PanelStack.render` race.
- **`panels/PanelGrid.ts`:**
  - a sibling of `PanelStack` that reuses `Panel`;
  - rows × columns with `GridFocus` (none/row/col/cell), collapse, splitters, and `layoutRects()` (arithmetic, so jsdom can test it);
  - a `GridDriver{view, dataKey, fetch, draw, placeholder}`, which separates the **data key** (what to fetch) from the view (how to draw), so an RT pan only redraws.
- **`engine/export.ts`:** `buildGridSvg(cells, …)` gives each cell's ids a unique prefix and clips each cell. `buildCompositeSvg` is unchanged.
- **Changes to existing files:**
  - `Panel.ts`: optional chrome (reorder, resize, hide), `addHeaderButton`, and an overlay SVG for the crosshair.
  - `layout.ts`: an optional `slot` in the layout key. With no slot the key string is unchanged, so genome's localStorage stays valid.
  - `arrow_client.ts`: `fetchArrow(url, signal)`.
  - `dashboard/viz_registry.ts`: one new descriptor.
  - `TaskPanel.ts`: drop the `viz-form` class when switching to the widget.

### Frontend modality (`src/modalities/massspec/`)
- `MassSpecBrowser.ts` is the host: nav bar, grid, drawer and history.
- `state.ts` holds `MsViewState` and the `MsEvents` bus vocabulary.
- `search.ts` parses search input; `nav.ts` handles keys and history.
- `IdentificationsDrawer.ts`.
- `renderers/`: `chromatogram.ts`, `spectrum.ts` (sticks, labels, annotation colours, mirror, error strip, ruler), `map.ts`, `ion_table.ts`.
- `ms.css`.
- Standalone entry: `index.massspec.html` + `main_massspec.ts`.
- Colours: a categorical palette for fragment and run traces. The dataviz skill applies when these are written.

---

## PR sequence
Each step is a thin vertical slice, so real payloads shape the engine.

| PR | Contents |
|---|---|
| **A — Payload survey** | `scripts/bench-ms-browser-access.py`, which reads only the inputs it is given and writes to scratch. It produces two bundles from the spectra files (`import-spectra` the wide-window and DDA spectra files into ext4 and onto `/mnt/e` scratch) and converts one `.raw`, so both bundle kinds are covered. It searches the wide-window bundle (8–10 min; Percolator workaround per the notes), then measures: <br>• `build_peak_index` time and peak RSS (`/usr/bin/time -v`); <br>• cold and warm latency of a spectrum by scan, an index m/z slice and an RT-pruned XIC, on ext4 and on `/mnt/e`; <br>• RT bin width 10 vs 60 s, and index row-group sizes; <br>• master-scan coverage; <br>• the `rt_observed` → apex-scan distance; <br>• whether the 3.1 GB predicted library has the `entries` index; <br>• `rt_alignment` coverage per run. <br>It also checks the linking rules on real manifests. Results go into `docs/plans/ms-browser.md`. |
| **B — Domain, slice 1** | `massspec/scans.py` (without `xic` and `locate_precursor`), `ion_number` + the mzPAF fix, `core/signal/decimate.py`, the base-peak fill on import. The synthetic-bundle and fake-jar experiment helpers move out of `tests/test_massspec_search_workflow.py` into `tests/_massspec_fixtures.py`. |
| **C — Server, slice 1** | The modality package: session, inspect, the bundle-only path; `ms_chromatogram` and `ms_spectrum`; the acquisitions, windows and scans routes; the `viz massspec` CLI. Includes HTTP tests and Arrow fixtures generated by the kernels (an MS sibling of `_viz_frontend_fixtures.py` plus the drift test). |
| **D — Engine + host, slice 1** (D1 engine / D2 modality, stacked) | The axis, interval, pan/zoom, bus, scheduler, `PanelGrid` and grid export, and the `Panel`/`layout` changes, with toy vitest tests in the style of `PanelStack.test.ts`. The MS host: grid, linked RT, click → scan, linking between levels, nav bar (acquisition, scan stepper, RT and m/z inputs), focus and collapse, export. Plus the standalone entry, the dashboard descriptor and the `TaskPanel` fix. |
| **E — Genome onto the engine** | `zoomLocus`/`attachPanZoom` → `interval` + `attachPanZoom1D` with `round: Math.round`. `ViewportBus` → `TypedBus`. `PanelStack.render` → the scheduler, which fixes the race; `GenomeBrowser.test.ts` moves from pinning serial request order to latest-wins. Verified by a headless-Chrome pixel diff before and after. |
| **F — Domain, slice 2** | `massspec/results.py` (identifications; linking, including spectra-file and merged GPF; precursor detail; both library lookups; RT mapping), `index_slice`/`find_peak_index` (plus the streaming build if A calls for it), `scans.xic` + `locate_precursor`, `spectra/annotate.py`, `targets_from_experiment` delegating. |
| **G — Identifications and targets** | `ms_xic`, annotation on `ms_spectrum`; the identifications, resolve, precursor and masses routes. The search bar, identifications drawer, target chip, history, target traces, boundaries and apex, annotation, mass-error strip, ion table and coverage, ad-hoc XIC chips, and the remaining core toggles. |
| **H1 — Mirror + ruler** | `ms_reference_spectrum`, the mirror renderer and similarity score, the Δm/z ruler. |
| **H2 — Multi-run overlay** | Compare mode, RT alignment to library RT, per-run quantity bars. |
| **H3 — RT × m/z map** | `rasterize_points`, `ms_map` with float extents, `decodeHybrid` reading the y extents, the Trace \| Map switch. |
| **I — Docs** | The root, `viz/` and `massspec/` CLAUDE.md files; plan status. |

Later, outside this plan: MS saved sessions. These need `sessions/cache.py` to keep `slot` and opaque options, plus a modality-owned `[view]` block in `write_saved`/`read_saved` and the saved-session endpoints. QC panels and the other deferred items listed above also wait.

## Risks, and how each is retired
1. **The peak index can't be built in memory for 292M-peak runs** → measured in A; the streaming build in F is the fallback.
2. **Latency from `/mnt`** → measured in A; the fallbacks are the row-group LRU cache, smaller RT bins and index row groups, and documenting "keep bundles on ext4".
3. **`rt_observed` doesn't map to an apex scan** → one query in A.
4. **Stellar trailers lack a master scan** → a count in A; the fallback is the preceding MS1 scan.
5. **Linking fails for experiments searched from spectra files, or for merged GPF** → real manifests inspected in A; synthetic tests in F.
6. **Grid CSS behaves differently in a real browser than in jsdom** → headless-Chrome screenshots, as in the viz foundation PR C.
7. **mzPAF labels are wrong** → fixed and tested in B.
8. **The predicted library lacks the per-precursor index** → checked in A. Without it, the mirror falls back to the result library.

## Verification
- **Per PR:**
  - pytest in parallel chunks, per the local notes (`tests/test_massspec_*`, `tests/test_viz_*`, `tests/test_imports.py`);
  - `ruff check .`;
  - `corepack pnpm@9 typecheck && … test` in `viz/frontend/`;
  - `python -m constellation.viz.frontend.build` for every entry, including `massspec`;
  - `tests/test_viz_e2e.py`;
  - the import-boundary, base-install and frontend-fixture drift tests.
- **Correctness cross-checks** on the reference data, using scratch copies and never running the jar inside `/mnt/e`:
  - a browser XIC for a known precursor equals `extract_xic_scan_major`;
  - TIC equals `scan_metadata.tic`;
  - annotation labels agree with the b/y annotations EncyclopeDIA stored in the `fragment` table;
  - the apex scan lies within half a cycle of `rt_observed`;
  - the mirror cosine is sensible on a high-scoring identification.
- **Real-browser check:**
  - run `constellation viz massspec --experiment <exp> --bundle <bundle>` (and the same through the dashboard);
  - take headless-Chrome screenshots of: the default grid, a maximized row, a collapsed column, a selected target (DIA and DDA), mirror, overlay, map;
  - for E, pixel-diff the genome browser before and after within the tolerance in the notes.

## Critical files
- **New, domain:**
  - `constellation/massspec/{scans,results}.py`
  - `massspec/spectra/annotate.py`
  - `core/signal/decimate.py`
  - `scripts/bench-ms-browser-access.py`
  - `tests/_massspec_fixtures.py`
- **Modified, domain:**
  - `massspec/quant/peak_index.py`
  - `massspec/peptide/ions.py`
  - `massspec/annotation/mzpaf.py`
  - `massspec/io/encyclopedia/_dia_import.py`
  - `massspec/quant/targets.py`
- **New, server:**
  - `viz/modalities/massspec/{__init__,session,requests,query,kernel,endpoints}.py`
  - `viz/modalities/massspec/tracks/{chromatogram,xic,spectrum,reference,map}.py`
  - `viz/raster/` (points rasterizer)
- **Modified, server:**
  - `viz/__init__.py` (one import)
  - `viz/cli.py`
- **New, frontend:**
  - `src/engine/{axis,interval,pan_zoom,bus}.ts`
  - `src/panels/{PanelGrid,scheduler}.ts`
  - `src/modalities/massspec/…`
  - `index.massspec.html`
  - `src/main_massspec.ts`
- **Modified, frontend:**
  - `src/panels/{Panel,layout,PanelStack}.ts`
  - `src/engine/{export,arrow_client,hybrid_layer}.ts`
  - `src/dashboard/{viz_registry,TaskPanel}.ts`
  - `src/modalities/genome/{GenomeBrowser,interactions,viewport_bus}.ts` (PR E)
- **Reused as is:**
  - `massspec.manifest.open_table`
  - `workflows.encyclopedia.read_experiment_table`
  - `library.ids.precursor_id`
  - `peptide.match.assign_fragments`
  - `peptide.mz.precursor_mz` and `peptide.envelope.peptide_envelope`
  - `acquisition_structure.isolation_windows`
  - `spectra.{fragment_basis, align_to_basis, compare_spectra}`
  - `acquisitions.Acquisitions.id_for_source`
  - `experiment.summarize`
  - `core.manifests`

---

## Payload survey (PR A, 2026-10-19)

Measured with `scripts/bench-ms-browser-access.py` on this machine (12 cores,
31 GB RAM, WSL2, bundles on ext4) against the reference data in
`/mnt/e/261010_u2os`: the wide-window DIA run (350–1200, 8 m/z windows) and
the DDA run, each as a bundle imported from its EncyclopeDIA spectra file
and as a Thermo conversion, plus a peptide-centric search of the wide-window
run against the six-fraction chromatogram library (`massspec search`,
`--percolator-version v3-05`, 0.35 Da: 31,836 precursors and 5,168 proteins;
the GUI reference has 31,833, within repeat-search variation). "Cold" means
the bundle's files were evicted from the page cache before the read.

| | Wide-window DIA | DDA |
|---|---|---|
| Peaks (rows of `peaks.parquet`) | 294,141,177 (2.29 GB) | 195,219,042 (1.55 GB) |
| Row groups | 309, ~20 s of RT each (pyarrow's 1Mi-row cap splits the 60 s bins) | 238, ~28 s each |
| MS1 scans / MS2 scans | 2,242 / 237,651 | 8,977 / 179,532 |
| Stored peaks per scan, MS1 / MS2 | ~1,150 / ~1,390 (max 1,690) | ~1,140 / ~1,100 (max 1,670) |
| Windows; scans per window | 106; 2,242 | 159,789 distinct (no windows) |
| MS1 cycle | 2.58 s | 0.67 s |
| Scan table (navigation columns) | 239,893 rows, 14.8 MB; 22–40 ms cold | 188,509 rows, 11.6 MB |
| Spectrum by scan number, cold / warm (median) | 50–70 / 30–50 ms | 50–55 / 29–33 ms |
| …with 10 s RT bins | 41–48 / 24–30 ms | — |
| MS1 XIC from `peaks.parquet`, RT ±2 min, cold | 400–450 ms (13 row groups) | 250–290 ms (9 row groups) |
| MS1 XIC from `peaks.parquet`, whole run, cold | 7–9 s | ~5 s |
| `build_peak_index` | not attempted | **killed at 23.2 GB** resident, before writing |

What it settles:

1. **Payloads are small; nothing needs a client cache.** A DIA chromatogram is
   2,242 points, a spectrum under 1,700 peaks. Only the DDA all-MS2 trace
   (~180,000 scans) is reduced (M4) on the server.
2. **The in-memory peak index cannot be built here** for a real run: the
   build loads every peak and sorts the table, and the smaller of the two
   runs passed 23 GB. The streaming build (spill to per-partition files,
   sort each) is required in PR F, not conditional.
3. **A peptide view does not need the index.** An XIC bounded to ±2 min
   reads 9–25 row groups in 0.25–0.45 s cold. A whole-run XIC (5–9 s) does,
   so `ms_xic` reads RT-bounded without an index and offers the index build
   for the rest.
4. **Spectrum reads are fast enough at the default 60 s bins;** 10 s bins
   save ~20 ms and double the row-group count. No change.
5. **Apex scans are exact.** For every identification (31,836), the nearest
   MS2 scan of the window covering the precursor is at exactly `rt_observed`
   (distance 0.0 s); the nearest MS1 scan is within 0.8 s (median; 4.7 s at
   most). Every precursor falls in exactly one window of this scheme.
6. **Parent scans.** DDA MS2 scans carry their precursor scan in both bundle
   kinds (100%). Stellar DIA MS2 scans carry none (0%), so for DIA the parent
   is the preceding MS1 scan, which `scan_table` derives.
7. **Library lookups are cheap through EncyclopeDIA's own index**
   (`entries(PeptideModSeq, PrecursorCharge, SourceFile)`, present in both
   libraries): 0.04 ms on the 169 MB chromatogram library on ext4, 14 ms
   median (40 ms at most) on the 3.1 GB predicted library read in place on
   `/mnt/e`. The mirror plot can use the searched library.
8. **The identifications drawer's table** is 31,836 rows, 4.7 MB.
9. **Thermo bundles' `peak_count` is not the stored peak count**: ~9,400 per
   scan against ~1,150 peaks stored (imported bundles agree with what is
   stored). Anything that sizes a read by it would be wrong; noted for the
   reader's owner, not changed here.
10. **Reading from `/mnt/e`:** sequential copies ran at 64–70 MB/s (a 2.7 GB
    `.raw` in 38 s); random SQLite reads at 9–17 MB/s. Bundles on a Windows
    drive were not measured.

Timings for the record: importing a spectra file 1.5–2.5 min (1.2–1.3 GB
resident); converting the DDA `.raw` 6.6 min (6.4 GB); `massspec search` of
the wide-window `.raw` 30.5 min end to end (conversion, spectra file, search,
tables; 8.4 GB), with other work competing for the cores.
