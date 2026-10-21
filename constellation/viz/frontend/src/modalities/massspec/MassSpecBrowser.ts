// MassSpecBrowser — chromatograms and spectra of one acquisition, by MS
// level, and what its searches identified.
//
// The layout is a `PanelGrid`: one row per MS level the run has (MS1, MS2,
// …), a chromatogram column and a spectrum column, with a drawer beside it
// listing the identifications. Above them a toolbar picks the acquisition,
// searches (a peptide, a protein, a scan, a time, an m/z), steps between
// scans and identifications, and moves back and forward through what was
// shown.
//
// How the panels are linked:
// - the chromatograms share one retention-time window (pan and zoom any of
//   them) and one hover crosshair;
// - clicking a chromatogram picks the nearest scan of that row's level
//   (and DIA window), which its spectrum then shows;
// - a scan picked on one level carries to the others: down to the scan it
//   was taken from, up to the first scan taken after it in the row's
//   window (DIA) or the first it triggered (DDA);
// - an MS1 spectrum shades the isolation window of the MS2 scan shown
//   below it.
// Each spectrum keeps its own m/z window.
//
// A target is the precursor being looked at: an identification picked in
// the drawer or the search box, or a peptide typed by hand. With one, the
// chromatograms trace its ions (its isotopes in MS1, its most intense
// fragments in the DIA window that isolated it) with the engine's
// integration window, apex and the library's predicted time marked, a DDA
// level's scan events keep only the scans that isolated it, MS2 spectra of
// scans that isolated it are annotated with its ions, and the drawer's Ions
// tab shows which of them the spectrum matched. Clicking a spectrum's peak
// adds a trace of that m/z to its row.
//
// Data comes from the `ms_chromatogram` / `ms_xic` / `ms_spectrum` kernels
// and the modality's own routes; scan navigation runs on the client from
// the acquisition's scan table.

import type { Table } from 'apache-arrow';
import { fetchArrow, fetchJson, fetchJsonMethod, fetchTrackData } from '../../engine/arrow_client';
import { buildGridSvg, downloadSvg } from '../../engine/export';
import { Interval, IntervalLimits, sameInterval } from '../../engine/interval';
import { attachPanZoom1D } from '../../engine/pan_zoom';
import { PanelKind } from '../../panels/kind';
import { GridAxisSpec, GridCell, GridDriver, PanelGrid } from '../../panels/PanelGrid';
import { SettingsSchema } from '../../panels/settings_schema';
import { Drawer, IdRow, idRows } from './drawer';
import { ISOTOPES, SERIES, seriesColor } from './palette';
import {
  CHROMATOGRAM_KIND,
  CHROM_MARGIN,
  IntensityMode,
  Overlay,
  drawChromatogram,
} from './renderers/chromatogram';
import {
  SPECTRUM_KIND,
  SPECTRUM_MARGIN,
  annotationSettings,
  drawSpectrum,
  nearestPeak,
  spectrumExtent,
} from './renderers/spectrum';
import { ScanIndex } from './scan_index';
import { matchIdentifications, parseQuery } from './search';
import { Target, ToleranceUnit, targetFromPrecursor, targetFromResolve } from './target';
import type { AcquisitionInfo, IsolationWindow } from './types';
import './ms.css';

export type TraceKind = 'tic' | 'bpc';

/** One MS level's row. */
export interface RowState {
  level: number;
  /** DIA window shown in this row; null for MS1 and DDA levels. */
  window: number | null;
  scan: number | null;
  /** The spectrum's m/z window; null fits the spectrum. */
  mz: Interval | null;
}

/** Everything the panels are drawn from. */
export interface MsView {
  acquisition: string;
  rt: Interval;
  rows: Record<string, RowState>;
  trace: TraceKind;
  intensity: IntensityMode;
  /** The precursor being looked at. */
  target: Target | null;
  /** Traces the user added, by row: m/z values. */
  adhoc: Record<string, number[]>;
  /** Retention times the traces are extracted over; null for the whole
   *  run (with a peak index, which makes that cheap). */
  xicRt: Interval | null;
}

/** What a cell holds: its kernel's table, and a chromatogram's traces. */
export interface CellData {
  table: Table;
  xic: Table | null;
}

export interface MassSpecBrowserOptions {
  host: HTMLElement;
  sessionId: string;
}

/** One trace drawn in a chromatogram row. */
interface RowTrace {
  mz: number;
  label: string;
  color: string;
  dashed: boolean;
}

interface HistoryEntry {
  acquisition: string;
  target: Target | null;
  idKey: string | null;
  rows: Record<string, { scan: number | null; window: number | null }>;
  rt: Interval;
  adhoc: Record<string, number[]>;
}

interface Suggestion {
  label: string;
  detail: string;
  run: () => void;
}

const KINDS: Record<string, PanelKind> = {
  [CHROMATOGRAM_KIND.kind]: CHROMATOGRAM_KIND,
  [SPECTRUM_KIND.kind]: SPECTRUM_KIND,
};
const FALLBACK_SETTINGS: SettingsSchema = { sections: [] };
const GRID_STORAGE_KEY = 'constellation.massspec.grid.v1';
const DRAWER_STORAGE_KEY = 'constellation.massspec.drawer.v1';
const COLUMNS: GridAxisSpec[] = [
  { id: 'chrom', label: 'Chromatogram', weight: 3 },
  { id: 'spec', label: 'Spectrum', weight: 2 },
];
/** Fragments traced for a target (and so its series colours 0–5). */
const TARGET_FRAGMENTS = 6;
/** Traces a user may add per row; they take the last series colours, so
 *  with a target's six fragments no colour is used twice. */
const MAX_ADHOC = SERIES.length - TARGET_FRAGMENTS;
/** Without a peak index, traces are extracted over this much either side
 *  of a target (seconds), and over at most this span otherwise. */
const XIC_MARGIN_S = 120;
const XIC_MAX_SPAN_S = 600;
/** Identifications at this q-value or better are listed. */
const MAX_Q = 0.01;
/** Pixels within which a pointer is over a peak. */
const PEAK_HIT_PX = 6;
const C13 = 1.0033548;

export const rowId = (level: number): string => `ms${level}`;

function readStored(key: string): string | null {
  try {
    return window.localStorage.getItem(key);
  } catch {
    return null;
  }
}

function writeStored(key: string, value: string): void {
  try {
    window.localStorage.setItem(key, value);
  } catch {
    /* storage unavailable: the choice lasts this page */
  }
}

/** Convert a tolerance at `mz` to m/z units. */
export function toleranceDa(mz: number, tolerance: [number, ToleranceUnit]): number {
  return tolerance[1] === 'Da' ? tolerance[0] : (mz * tolerance[0]) / 1e6;
}

export class MassSpecBrowser {
  private readonly opts: MassSpecBrowserOptions;
  private readonly root: HTMLElement;
  private readonly toolbar: HTMLElement;
  private readonly chips: HTMLElement;
  private readonly main: HTMLElement;
  private readonly gridHost: HTMLElement;
  private readonly statusEl: HTMLElement;
  private readonly drawer: Drawer;
  private acquisitionSelect!: HTMLSelectElement;
  private traceSelect!: HTMLSelectElement;
  private intensitySelect!: HTMLSelectElement;
  private searchInput!: HTMLInputElement;
  private suggestBox!: HTMLElement;
  private backBtn!: HTMLButtonElement;
  private forwardBtn!: HTMLButtonElement;
  private indexBtn!: HTMLButtonElement;
  private drawerBtn!: HTMLButtonElement;
  private rtLabel!: HTMLElement;
  private scanLabel!: HTMLElement;
  private grid: PanelGrid<MsView, CellData> | null = null;
  private acquisitions: AcquisitionInfo[] = [];
  private info: AcquisitionInfo | null = null;
  private scans: ScanIndex | null = null;
  private windows: IsolationWindow[] = [];
  private identifications: IdRow[] = [];
  private view: MsView | null = null;
  /** The drawer row the target came from. */
  private selectedIdKey: string | null = null;
  /** The row the user last picked a scan in (where arrow keys step). */
  private activeRow: string | null = null;
  /** Chromatogram keys whose last response was reduced (they refetch on zoom). */
  private decimated = new Set<string>();
  /** Traces fetched, by request, so a chromatogram whose base trace is
   *  refetched on zoom does not extract its traces again. */
  private xicCache = new Map<string, Table>();
  private suggestions: Suggestion[] = [];
  private suggestAt = 0;
  private history: HistoryEntry[] = [];
  private historyAt = -1;
  private restoring = false;
  /** Bumped on every target request; a reply for an older one is dropped. */
  private targetSeq = 0;
  /** The spectrum the Ions tab was last lit from. */
  private ionsFrom: { target: string; table: Table } | null = null;
  private targetScanCache: { key: string; scans: number[] } | null = null;
  private scanHeaders = new Map<string, Record<string, unknown>>();
  private scanHeaderTimer: ReturnType<typeof setTimeout> | null = null;
  private indexPoll: ReturnType<typeof setTimeout> | null = null;
  private detachers: Array<() => void> = [];
  private renderQueued = false;
  private resizeObserver: ResizeObserver | null = null;
  private disposed = false;

  constructor(opts: MassSpecBrowserOptions) {
    this.opts = opts;
    this.root = document.createElement('div');
    this.root.className = 'ms-browser';
    this.root.tabIndex = 0;
    this.toolbar = document.createElement('div');
    this.toolbar.className = 'ms-toolbar';
    this.chips = document.createElement('div');
    this.chips.className = 'ms-chips';
    this.main = document.createElement('div');
    this.main.className = 'ms-main';
    this.gridHost = document.createElement('div');
    this.gridHost.className = 'ms-grid-host';
    this.statusEl = document.createElement('span');
    this.statusEl.className = 'ms-status';
    this.drawer = new Drawer({ onSelect: (row) => void this.selectIdentification(row) });
    this.drawer.element.hidden = readStored(DRAWER_STORAGE_KEY) === 'closed';
    this.main.append(this.gridHost, this.drawer.element);
    this.root.append(this.toolbar, this.chips, this.main);
  }

  /** Current view (for tests and the host page). */
  get currentView(): MsView | null {
    return this.view;
  }

  get panelGrid(): PanelGrid<MsView, CellData> | null {
    return this.grid;
  }

  get identificationRows(): readonly IdRow[] {
    return this.identifications;
  }

  async mount(): Promise<void> {
    this.opts.host.appendChild(this.root);
    this.buildToolbar();
    this.root.addEventListener('keydown', (e) => this.onKey(e));
    try {
      this.acquisitions = await fetchJson<AcquisitionInfo[]>(`${this.sessionBase()}/acquisitions`);
    } catch (err) {
      this.setStatus(`could not load the session: ${(err as Error).message}`, true);
      return;
    }
    for (const a of this.acquisitions) {
      const option = document.createElement('option');
      option.value = a.key;
      option.textContent = a.label;
      this.acquisitionSelect.appendChild(option);
    }
    if (this.acquisitions.length === 0) {
      this.setStatus('this session has no acquisitions');
      return;
    }
    if (typeof ResizeObserver !== 'undefined') {
      this.resizeObserver = new ResizeObserver(() => this.onResize());
      this.resizeObserver.observe(this.gridHost);
    }
    await this.showAcquisition(this.acquisitions[0].key);
  }

  private sessionBase(): string {
    return `/api/massspec/sessions/${encodeURIComponent(this.opts.sessionId)}`;
  }

  private acquisitionBase(key: string): string {
    return `${this.sessionBase()}/acquisitions/${encodeURIComponent(key)}`;
  }

  // ── toolbar ─────────────────────────────────────────────────────────

  private buildToolbar(): void {
    const field = (label: string, control: HTMLElement): HTMLElement => {
      const wrap = document.createElement('label');
      wrap.className = 'ms-field';
      const text = document.createElement('span');
      text.textContent = label;
      wrap.append(text, control);
      return wrap;
    };
    const select = (options: Array<[string, string]>): HTMLSelectElement => {
      const el = document.createElement('select');
      for (const [value, label] of options) {
        const option = document.createElement('option');
        option.value = value;
        option.textContent = label;
        el.appendChild(option);
      }
      return el;
    };
    const button = (text: string, title: string, onClick: () => void): HTMLButtonElement => {
      const el = document.createElement('button');
      el.type = 'button';
      el.className = 'ms-btn';
      el.textContent = text;
      el.title = title;
      el.addEventListener('click', onClick);
      return el;
    };

    this.backBtn = button('←', 'Back (Alt+←)', () => void this.go(-1));
    this.forwardBtn = button('→', 'Forward (Alt+→)', () => void this.go(1));
    this.acquisitionSelect = select([]);
    this.acquisitionSelect.addEventListener('change', () => void this.showAcquisition(this.acquisitionSelect.value));

    const search = document.createElement('div');
    search.className = 'ms-search';
    this.searchInput = document.createElement('input');
    this.searchInput.type = 'search';
    this.searchInput.className = 'ms-search-input';
    this.searchInput.placeholder = 'Peptide, protein, scan 123, rt 35.2, 512.27  (/)';
    this.searchInput.setAttribute('aria-label', 'Search');
    this.suggestBox = document.createElement('div');
    this.suggestBox.className = 'ms-suggest';
    this.suggestBox.hidden = true;
    search.append(this.searchInput, this.suggestBox);
    this.searchInput.addEventListener('input', () => this.updateSuggestions());
    this.searchInput.addEventListener('focus', () => this.updateSuggestions());
    this.searchInput.addEventListener('blur', () => {
      this.suggestBox.hidden = true;
    });
    this.searchInput.addEventListener('keydown', (e) => this.onSearchKey(e));

    this.traceSelect = select([
      ['tic', 'Total ion current'],
      ['bpc', 'Base peak'],
    ]);
    this.traceSelect.addEventListener('change', () => {
      if (!this.view) return;
      this.view.trace = this.traceSelect.value as TraceKind;
      this.requestRender();
    });
    this.intensitySelect = select([
      ['abs', 'Absolute'],
      ['rel', 'Relative (%)'],
      ['log', 'Log'],
    ]);
    this.intensitySelect.addEventListener('change', () => {
      if (!this.view) return;
      this.view.intensity = this.intensitySelect.value as IntensityMode;
      this.requestRender();
    });
    this.rtLabel = document.createElement('span');
    this.rtLabel.className = 'ms-readout';
    this.scanLabel = document.createElement('span');
    this.scanLabel.className = 'ms-readout';
    this.indexBtn = button('Build index', 'Build a peak index, so traces cover the whole run', () => void this.buildIndex());
    this.indexBtn.hidden = true;
    this.drawerBtn = button('Identifications', 'Show or hide the drawer', () => this.toggleDrawer());
    this.drawerBtn.classList.toggle('active', !this.drawer.element.hidden);

    const spacer = document.createElement('span');
    spacer.className = 'ms-spacer';
    this.toolbar.append(
      this.backBtn,
      this.forwardBtn,
      field('Run', this.acquisitionSelect),
      search,
      button('◀', 'Previous identification ([)', () => void this.stepIdentification(-1)),
      button('▶', 'Next identification (])', () => void this.stepIdentification(1)),
      field('Trace', this.traceSelect),
      field('Intensity', this.intensitySelect),
      field('RT', this.rtLabel),
      button('Fit', 'Show the whole run (f)', () => this.fitRt()),
      button('◀', 'Previous scan (←)', () => this.stepScan(-1)),
      this.scanLabel,
      button('▶', 'Next scan (→)', () => this.stepScan(1)),
      spacer,
      this.statusEl,
      this.indexBtn,
      this.drawerBtn,
      button('Save SVG', 'Export the panels shown', () => this.exportSvg()),
    );
    this.updateHistoryButtons();
  }

  private setStatus(text: string, error = false): void {
    this.statusEl.textContent = text;
    this.statusEl.classList.toggle('ms-status-error', error);
  }

  private updateReadouts(): void {
    const view = this.view;
    if (!view) return;
    this.rtLabel.textContent = `${(view.rt[0] / 60).toFixed(2)}–${(view.rt[1] / 60).toFixed(2)} min`;
    const row = this.activeRow ? view.rows[this.activeRow] : null;
    if (row?.scan != null && this.scans) {
      const rt = this.scans.rtOf(row.scan);
      this.scanLabel.textContent = `MS${row.level} scan ${row.scan}${rt === null ? '' : ` · ${(rt / 60).toFixed(2)} min`}`;
    } else {
      this.scanLabel.textContent = 'no scan';
    }
  }

  private toggleDrawer(): void {
    this.drawer.element.hidden = !this.drawer.element.hidden;
    this.drawerBtn.classList.toggle('active', !this.drawer.element.hidden);
    writeStored(DRAWER_STORAGE_KEY, this.drawer.element.hidden ? 'closed' : 'open');
    this.onResize();
  }

  /** The target chip and the added traces, under the toolbar. */
  private updateChips(): void {
    const view = this.view;
    this.chips.replaceChildren();
    const chip = (text: string, title: string, onRemove: () => void, cls = ''): void => {
      const el = document.createElement('span');
      el.className = `ms-chip ${cls}`.trim();
      el.title = title;
      const label = document.createElement('span');
      label.textContent = text;
      const remove = document.createElement('button');
      remove.type = 'button';
      remove.className = 'ms-chip-remove';
      remove.textContent = '✕';
      remove.title = 'Remove';
      remove.addEventListener('click', onRemove);
      el.append(label, remove);
      this.chips.appendChild(el);
    };
    const t = view?.target;
    if (t) {
      const parts = [`${t.label}`, `m/z ${t.precursorMz.toFixed(4)}`];
      if (t.rt !== null) parts.push(`${(t.rt / 60).toFixed(2)} min`);
      if (t.qvalue !== null) parts.push(`q ${t.qvalue.toExponential(1)}`);
      if (!t.source) parts.push('not identified');
      chip(parts.join(' · '), t.proteins ?? t.modifiedSequence, () => this.clearTarget(), 'ms-chip-target');
    }
    for (const [id, mzs] of Object.entries(view?.adhoc ?? {})) {
      const row = view?.rows[id];
      mzs.forEach((mz, i) => {
        chip(`MS${row?.level ?? '?'} ${mz.toFixed(4)}`, 'A trace you added', () => this.removeAdhoc(id, mz), 'ms-chip-adhoc');
        (this.chips.lastElementChild as HTMLElement).style.borderColor = SERIES[SERIES.length - 1 - i];
      });
    }
    const note = (text: string): void => {
      const el = document.createElement('span');
      el.className = 'ms-chip-note';
      el.textContent = text;
      this.chips.appendChild(el);
    };
    if (view && !this.info?.has_index && view.xicRt && this.chips.childElementCount) {
      note(`traces cover ${(view.xicRt[0] / 60).toFixed(1)}–${(view.xicRt[1] / 60).toFixed(1)} min (no peak index)`);
    }
    if (view && Object.keys(view.adhoc).some((id) => view.rows[id]?.level > 1 && view.rows[id]?.window === null)) {
      note('added MS2 traces cover the time shown');
    }
    this.chips.hidden = this.chips.childElementCount === 0;
  }

  // ── acquisition ─────────────────────────────────────────────────────

  async showAcquisition(key: string, opts: { record?: boolean } = {}): Promise<void> {
    const info = this.acquisitions.find((a) => a.key === key);
    if (!info) return;
    this.acquisitionSelect.value = key;
    const base = this.acquisitionBase(key);
    this.setStatus('loading scans…');
    let table: Table;
    try {
      [table, this.windows] = await Promise.all([
        fetchArrow(`${base}/scans`),
        fetchJson<IsolationWindow[]>(`${base}/windows`),
      ]);
    } catch (err) {
      this.setStatus(`could not load ${info.label}: ${(err as Error).message}`, true);
      return;
    }
    if (this.disposed) return;
    const previous = this.view;
    this.info = info;
    this.scans = new ScanIndex(table);
    this.decimated.clear();
    this.xicCache.clear();
    this.indexBtn.hidden = info.has_index;

    const rows: Record<string, RowState> = {};
    for (const level of info.levels) {
      const windows = this.windowsOf(level);
      rows[rowId(level)] = {
        level,
        window: windows.length ? windows[Math.floor(windows.length / 2)].window : null,
        scan: null,
        mz: null,
      };
    }
    this.view = {
      acquisition: key,
      rt: [info.rt_range[0], Math.max(info.rt_range[1], info.rt_range[0] + 1)],
      rows,
      trace: (this.traceSelect.value as TraceKind) || 'tic',
      intensity: (this.intensitySelect.value as IntensityMode) || 'abs',
      target: null,
      adhoc: {},
      xicRt: null,
    };
    const rowSpecs = info.levels.map((level) => ({ id: rowId(level), label: `MS${level}`, weight: 1 }));
    if (!this.grid) this.grid = this.createGrid(rowSpecs);
    else this.grid.setRows(rowSpecs);
    this.decorateRows();
    this.decorateCells();
    this.attachInteractions();
    this.onResize();

    // Open on the most intense MS1 scan, so every panel shows something.
    const first = this.scans.mostIntense(info.levels[0]);
    if (first !== null) this.selectScan(rowId(info.levels[0]), first, { record: false });
    this.setStatus('');
    this.updateReadouts();
    this.updateChips();
    this.requestRender();

    await this.loadIdentifications(info);
    // A target carries over: the same precursor where this run's searches
    // found it, otherwise as a peptide.
    const carried = previous?.target;
    if (carried && !this.restoring) {
      const found = carried.source
        ? this.identifications.find((r) => r.precursor_id === carried.source!.precursorId && r.experiment === carried.source!.experiment)
        : undefined;
      if (found) await this.selectIdentification(found, { record: false });
      else this.applyTarget(asPeptide(carried), { record: false });
    }
    if (opts.record !== false) this.record();
  }

  private async loadIdentifications(info: AcquisitionInfo): Promise<void> {
    this.identifications = [];
    this.drawer.setIdentifications([]);
    if (!info.searches?.length) {
      this.drawer.setEmpty('No search of this run is attached. Add an experiment directory to list what was identified.');
      return;
    }
    try {
      const table = await fetchArrow(`${this.acquisitionBase(info.key)}/identifications?max_q=${MAX_Q}`);
      if (this.disposed || this.info !== info) return;
      this.identifications = idRows(table);
      this.drawer.setIdentifications(this.identifications);
      this.setStatus(`${this.identifications.length.toLocaleString()} identifications at q ≤ ${MAX_Q}`);
    } catch (err) {
      this.setStatus(`could not load identifications: ${(err as Error).message}`, true);
    }
  }

  private windowsOf(level: number): IsolationWindow[] {
    return this.windows.filter((w) => w.level === level);
  }

  /** The window of `level` that isolates `mz`, the one centred nearest it. */
  private windowFor(level: number, mz: number): number | null {
    let best: IsolationWindow | null = null;
    for (const w of this.windowsOf(level)) {
      if (mz < w.isolation_lower || mz > w.isolation_upper) continue;
      if (!best || Math.abs(w.center - mz) < Math.abs(best.center - mz)) best = w;
    }
    return best ? best.window : null;
  }

  private windowCovers(row: RowState, mz: number): boolean {
    const w = this.windows.find((x) => x.level === row.level && x.window === row.window);
    return !!w && mz >= w.isolation_lower && mz <= w.isolation_upper;
  }

  private createGrid(rows: GridAxisSpec[]): PanelGrid<MsView, CellData> {
    return new PanelGrid<MsView, CellData>({
      host: this.gridHost,
      rows,
      cols: COLUMNS,
      cellEntry: (row, col) => {
        const kind = col === 'chrom' ? CHROMATOGRAM_KIND.kind : SPECTRUM_KIND.kind;
        const level = row.replace(/^ms/, '');
        return {
          kind,
          binding_id: `${kind}:${this.view?.acquisition ?? ''}`,
          label: `MS${level} ${col === 'chrom' ? 'chromatogram' : 'spectrum'}`,
          source_id: null,
        };
      },
      driver: this.driver(),
      kindOf: (kind) => KINDS[kind] ?? null,
      fallbackSettings: FALLBACK_SETTINGS,
      storageKey: GRID_STORAGE_KEY,
    });
  }

  /** A window picker in each DIA row's header. */
  private decorateRows(): void {
    if (!this.grid || !this.view) return;
    for (const [id, row] of Object.entries(this.view.rows)) {
      const slot = this.grid.rowSlot(id);
      if (!slot) continue;
      slot.replaceChildren();
      const windows = this.windowsOf(row.level);
      if (windows.length === 0) continue;
      const select = document.createElement('select');
      select.className = 'ms-window-select';
      select.title = 'Isolation window';
      for (const w of windows) {
        const option = document.createElement('option');
        option.value = String(w.window);
        option.textContent = `${w.isolation_lower.toFixed(1)}–${w.isolation_upper.toFixed(1)}`;
        select.appendChild(option);
      }
      select.value = String(row.window);
      select.addEventListener('change', () => this.setWindow(id, Number(select.value)));
      slot.appendChild(select);
    }
  }

  private syncWindowSelect(id: string): void {
    const select = this.grid?.rowSlot(id)?.querySelector('select');
    const row = this.view?.rows[id];
    if (select && row) select.value = String(row.window);
  }

  /** Copy and download buttons in each cell's header. */
  private decorateCells(): void {
    for (const cell of this.grid?.cells ?? []) {
      const add = (text: string, title: string, onClick: () => void): void => {
        const b = document.createElement('button');
        b.type = 'button';
        b.className = 'panel-grid-maximize ms-cell-btn';
        b.textContent = text;
        b.title = title;
        b.addEventListener('click', onClick);
        cell.panel.addHeaderButton(b);
      };
      if (cell.col === 'spec') add('⧉', 'Copy a reference to this spectrum (run:scan)', () => this.copySpectrumRef(cell));
      add('⤓', 'Download what this panel shows as CSV', () => this.downloadCsv(cell));
    }
  }

  /** Show another DIA window in a row, keeping the time the user was at. */
  setWindow(id: string, window: number): void {
    const row = this.view?.rows[id];
    if (!row || !this.scans) return;
    const rt = row.scan !== null ? this.scans.rtOf(row.scan) : null;
    row.window = window;
    if (rt !== null) {
      const scan = this.scans.nearest(row.level, window, rt);
      if (scan !== null) this.selectScan(id, scan, { record: false });
    }
    this.syncWindowSelect(id);
    this.requestRender();
  }

  // ── scans ───────────────────────────────────────────────────────────

  /** Show `scan` in row `id` and carry the choice to the other levels. */
  selectScan(id: string, scan: number, opts: { record?: boolean } = {}): void {
    const view = this.view;
    const scans = this.scans;
    if (!view || !scans || !view.rows[id]) return;
    view.rows[id].scan = scan;
    this.activeRow = id;
    const level = view.rows[id].level;
    // Down: the scan each lower level's scan was taken from.
    let current: number | null = scan;
    for (let l = level - 1; l >= 1 && current !== null; l--) {
      current = scans.parentOf(current);
      while (current !== null && (scans.levelOf(current) ?? 0) > l) current = scans.parentOf(current);
      const row = view.rows[rowId(l)];
      if (row && current !== null) row.scan = current;
    }
    // Up: the first scan taken after it, in the row's window or as its child.
    current = scan;
    for (let l = level + 1; view.rows[rowId(l)] && current !== null; l++) {
      const row = view.rows[rowId(l)];
      const children: number[] = row.window === null ? scans.children(current) : [];
      const next: number | null = children.length ? children[0] : scans.firstAfter(current, l, row.window);
      if (next !== null) {
        row.scan = next;
        const w = scans.windowOf(next);
        if (w !== null && row.window !== null && w !== row.window) {
          row.window = w;
          this.syncWindowSelect(rowId(l));
        }
      }
      current = next;
    }
    this.updateReadouts();
    this.scheduleScanHeader();
    if (opts.record !== false) this.record();
    this.requestRender();
  }

  /** Step the active row's scan within its level and window. */
  stepScan(delta: number): void {
    const view = this.view;
    if (!view || !this.scans || !this.activeRow) return;
    const row = view.rows[this.activeRow];
    if (row?.scan == null) return;
    // A DDA level with a target steps among the scans that isolated it.
    const own = row.level > 1 && row.window === null ? this.targetScans(row.level) : [];
    if (own.length) {
      const at = own.findIndex((s) => s >= row.scan!);
      const from = at < 0 ? own.length : own[at] === row.scan ? at : delta > 0 ? at - 1 : at;
      const scan = own[Math.min(own.length - 1, Math.max(0, from + delta))];
      if (scan !== undefined && scan !== row.scan) {
        this.selectScan(this.activeRow, scan, { record: false });
        this.keepScanInView(scan);
      }
      return;
    }
    const next = this.scans.step(row.scan, delta, row.window);
    if (next !== null && next !== row.scan) {
      this.selectScan(this.activeRow, next, { record: false });
      this.keepScanInView(next);
    }
  }

  /** Show the scan, whatever its level (search `scan 123`). */
  goToScan(scan: number): boolean {
    const view = this.view;
    const scans = this.scans;
    if (!view || !scans || !scans.has(scan)) return false;
    const level = scans.levelOf(scan) ?? 1;
    const id = rowId(level);
    const row = view.rows[id];
    if (!row) return false;
    const w = scans.windowOf(scan);
    if (w !== null && row.window !== null) {
      row.window = w;
      this.syncWindowSelect(id);
    }
    this.selectScan(id, scan);
    this.keepScanInView(scan);
    return true;
  }

  private keepScanInView(scan: number): void {
    const rt = this.scans?.rtOf(scan);
    const view = this.view;
    if (!view || rt == null || (rt >= view.rt[0] && rt <= view.rt[1])) return;
    const span = view.rt[1] - view.rt[0];
    view.rt = this.clampRt([rt - span / 2, rt + span / 2]);
  }

  /** Centre the retention-time window on `rt` (seconds). */
  centerRt(rt: number): void {
    const view = this.view;
    if (!view) return;
    const span = view.rt[1] - view.rt[0];
    view.rt = this.clampRt([rt - span / 2, rt + span / 2]);
    this.requestRender();
  }

  fitRt(): void {
    if (!this.view || !this.info) return;
    this.view.rt = [this.info.rt_range[0], this.info.rt_range[1]];
    this.requestRender();
  }

  private rtLimits(): IntervalLimits {
    const range = this.info?.rt_range ?? [0, 1];
    return { min: range[0], max: Math.max(range[1], range[0] + 1), minSpan: 2 };
  }

  private clampRt(iv: Interval): Interval {
    const l = this.rtLimits();
    const span = Math.min(iv[1] - iv[0], l.max - l.min);
    let lo = Math.max(l.min, iv[0]);
    if (lo + span > l.max) lo = l.max - span;
    return [lo, lo + span];
  }

  private scheduleScanHeader(): void {
    if (this.scanHeaderTimer) clearTimeout(this.scanHeaderTimer);
    this.scanHeaderTimer = setTimeout(() => void this.loadScanHeader(), 150);
  }

  private async loadScanHeader(): Promise<void> {
    const view = this.view;
    const row = this.activeRow ? view?.rows[this.activeRow] : null;
    if (!view || row?.scan == null) return;
    const key = `${view.acquisition}|${row.scan}`;
    let header = this.scanHeaders.get(key);
    if (!header) {
      try {
        header = await fetchJson<Record<string, unknown>>(`${this.acquisitionBase(view.acquisition)}/scans/${row.scan}`);
      } catch {
        return;
      }
      if (this.scanHeaders.size > 64) this.scanHeaders.clear();
      this.scanHeaders.set(key, header);
    }
    if (!this.disposed && this.activeRow && this.view?.rows[this.activeRow]?.scan === row.scan) this.drawer.setScan(header);
  }

  // ── targets ─────────────────────────────────────────────────────────

  /** Look at one identification: fetch what to trace for it, then show it. */
  async selectIdentification(row: IdRow, opts: { record?: boolean } = {}): Promise<void> {
    const view = this.view;
    if (!view) return;
    const acquisition = view.acquisition;
    const seq = ++this.targetSeq;
    const label = `${row.modified_sequence ?? '?'}/${row.charge ?? '?'}`;
    this.setStatus(`loading ${label}…`);
    const params = new URLSearchParams({
      experiment: row.experiment,
      unit: row.unit,
      precursor_id: row.precursor_id,
      top_n: String(TARGET_FRAGMENTS),
    });
    let json: Record<string, unknown>;
    try {
      json = await fetchJson<Record<string, unknown>>(`${this.acquisitionBase(acquisition)}/precursor?${params}`);
    } catch (err) {
      if (seq === this.targetSeq) this.setStatus(`could not load ${label}: ${(err as Error).message}`, true);
      return;
    }
    if (seq !== this.targetSeq || this.disposed || this.view?.acquisition !== acquisition) return;
    const target = targetFromPrecursor(json, row.precursor_id);
    if (row.scan !== null) {
      // A spectrum-centric match: the scan it was matched in.
      target.apexScans = { ms2: row.scan, ms1: this.scans?.parentOf(row.scan) ?? target.apexScans.ms1 };
    }
    this.selectedIdKey = row.key;
    this.drawer.setSelected(row.key);
    this.applyTarget(target, opts);
    this.setStatus('');
  }

  /** Look at a peptide typed by hand. */
  async tracePeptide(text: string, charge: number | null): Promise<void> {
    const seq = ++this.targetSeq;
    const params = new URLSearchParams({ q: text, top_n: String(TARGET_FRAGMENTS) });
    if (charge !== null) params.set('charge', String(charge));
    let json: Record<string, unknown>;
    try {
      json = await fetchJson<Record<string, unknown>>(`${this.sessionBase()}/resolve?${params}`);
    } catch (err) {
      if (seq === this.targetSeq) this.setStatus(`not a peptide: ${text} (${(err as Error).message})`, true);
      return;
    }
    if (seq !== this.targetSeq || this.disposed) return;
    this.selectedIdKey = null;
    this.drawer.setSelected(null);
    this.applyTarget(targetFromResolve(json));
    this.setStatus('');
  }

  /** Show `target`: its time, its window, its apex scans, its traces. */
  applyTarget(target: Target, opts: { record?: boolean } = {}): void {
    const view = this.view;
    const scans = this.scans;
    if (!view || !scans || !this.info) return;
    view.target = target;
    const ms1 = view.rows[rowId(1)];
    const ms2 = view.rows[rowId(2)];
    // The MS2 row moves to the window that isolated the precursor.
    if (ms2 && ms2.window !== null) {
      const w = target.window ?? this.windowFor(2, target.precursorMz);
      if (w !== null) {
        ms2.window = w;
        this.syncWindowSelect(rowId(2));
      }
    }
    if (target.rt !== null) {
      const lo = target.rtStart ?? target.rt - 15;
      const hi = target.rtEnd ?? target.rt + 15;
      const margin = Math.max(30, (hi - lo) * 1.5);
      view.rt = this.clampRt([lo - margin, hi + margin]);
    }
    view.xicRt = this.info.has_index ? null : this.xicRange(target);
    if (ms1) {
      const z = Math.max(1, target.charge);
      ms1.mz = [target.precursorMz - 1 - 2 / z, target.precursorMz + 1 + 4 / z];
    }
    if (ms2) ms2.mz = null;

    if (target.apexScans.ms2 !== null && ms2 && scans.has(target.apexScans.ms2)) {
      this.selectScan(rowId(2), target.apexScans.ms2, { record: false });
    } else if (target.apexScans.ms1 !== null && ms1 && scans.has(target.apexScans.ms1)) {
      this.selectScan(rowId(1), target.apexScans.ms1, { record: false });
    } else if (ms2 && ms2.window !== null) {
      // A typed peptide: the scan of its window nearest the time shown.
      const at = (view.rt[0] + view.rt[1]) / 2;
      const scan = scans.nearest(2, ms2.window, at);
      if (scan !== null) this.selectScan(rowId(2), scan, { record: false });
    } else if (ms2) {
      // A typed peptide, DDA: of the scans that isolated it, the one with
      // the most signal.
      let best: number | null = null;
      for (const scan of this.targetScans(2)) {
        const r = scans.row(scan)!;
        if (best === null || scans.tics[r] > scans.tics[scans.row(best)!]) best = scan;
      }
      if (best !== null) {
        this.selectScan(rowId(2), best, { record: false });
        this.keepScanInView(best);
      }
    }
    this.ionsFrom = null;
    this.drawer.setIons(target, new Set());
    this.updateChips();
    if (opts.record !== false) this.record();
    this.requestRender();
  }

  clearTarget(): void {
    const view = this.view;
    if (!view?.target) return;
    view.target = null;
    this.targetSeq++;
    this.selectedIdKey = null;
    this.drawer.setSelected(null);
    this.drawer.setIons(null, new Set());
    if (this.info?.has_index || !Object.values(view.adhoc).some((l) => l.length)) view.xicRt = null;
    const ms1 = view.rows[rowId(1)];
    if (ms1) ms1.mz = null;
    this.updateChips();
    this.record();
    this.requestRender();
  }

  /** Whether a DDA scan with precursor m/z `mz` isolated the target: the
   *  target's m/z or one of its first isotopes (an instrument may pick
   *  M+1), within the precursor tolerance. */
  private isolatesTarget(mz: number): boolean {
    const t = this.view?.target;
    if (!t || !Number.isFinite(mz)) return false;
    const tol = toleranceDa(t.precursorMz, this.tolerance(1, 'precursor'));
    const z = Math.max(1, t.charge);
    return [-1, 0, 1, 2].some((k) => Math.abs(mz - (t.precursorMz + (k * C13) / z)) <= tol);
  }

  /** The DDA scans of `level` that isolated the target, in scan order. */
  private targetScans(level: number): number[] {
    const scans = this.scans;
    if (!scans || !this.view?.target) return [];
    const key = `${this.view.acquisition}|${this.view.target.key}|${level}`;
    if (this.targetScanCache?.key === key) return this.targetScanCache.scans;
    const out: number[] = [];
    for (const r of scans.rows(level, null)) if (this.isolatesTarget(scans.precursorMz[r])) out.push(scans.scans[r]);
    this.targetScanCache = { key, scans: out };
    return out;
  }

  /** Without a peak index: the times a target's traces are extracted over. */
  private xicRange(target: Target | null): Interval {
    const view = this.view!;
    if (target?.rt != null) {
      return this.clampRt([(target.rtStart ?? target.rt) - XIC_MARGIN_S, (target.rtEnd ?? target.rt) + XIC_MARGIN_S]);
    }
    const mid = (view.rt[0] + view.rt[1]) / 2;
    const half = Math.min(XIC_MAX_SPAN_S, (view.rt[1] - view.rt[0]) * 2) / 2;
    return this.clampRt([mid - half, mid + half]);
  }

  /** The time shown, widened and snapped to whole minutes so that small
   *  pans do not ask again; at most XIC_MAX_SPAN_S. */
  private followRange(view: MsView): Interval {
    const span = Math.min(XIC_MAX_SPAN_S, (view.rt[1] - view.rt[0]) * 2);
    const mid = (view.rt[0] + view.rt[1]) / 2;
    return this.clampRt([Math.floor((mid - span / 2) / 60) * 60, Math.ceil((mid + span / 2) / 60) * 60]);
  }

  /** Step to the previous or next identification in the drawer's order. */
  async stepIdentification(delta: number): Promise<void> {
    const order = this.drawer.order;
    if (order.length === 0) return;
    const at = this.selectedIdKey ? order.findIndex((r) => r.key === this.selectedIdKey) : -1;
    const next = at < 0 ? (delta > 0 ? 0 : order.length - 1) : Math.min(order.length - 1, Math.max(0, at + delta));
    if (next !== at) await this.selectIdentification(order[next]);
  }

  /** Trace `mz` in row `id` (a peak clicked, an m/z searched). */
  addAdhoc(id: string, mz: number): void {
    const view = this.view;
    if (!view?.rows[id]) return;
    const list = view.adhoc[id] ?? [];
    if (list.some((m) => Math.abs(m - mz) < 1e-4)) return;
    list.push(mz);
    if (list.length > MAX_ADHOC) {
      list.shift();
      this.setStatus(`at most ${MAX_ADHOC} added traces per row: the oldest was removed`);
    }
    view.adhoc[id] = list;
    if (!this.info?.has_index && !view.xicRt) view.xicRt = this.xicRange(view.target);
    this.updateChips();
    this.requestRender();
  }

  removeAdhoc(id: string, mz: number): void {
    const view = this.view;
    if (!view) return;
    view.adhoc[id] = (view.adhoc[id] ?? []).filter((m) => m !== mz);
    if (view.adhoc[id].length === 0) delete view.adhoc[id];
    this.updateChips();
    this.requestRender();
  }

  // ── search ──────────────────────────────────────────────────────────

  private updateSuggestions(): void {
    this.suggestions = this.suggestionsFor(this.searchInput.value);
    this.suggestAt = 0;
    this.suggestBox.replaceChildren();
    this.suggestions.forEach((s, i) => {
      const item = document.createElement('div');
      item.className = 'ms-suggest-item';
      item.classList.toggle('active', i === this.suggestAt);
      const label = document.createElement('span');
      label.className = 'ms-suggest-label';
      label.textContent = s.label;
      const detail = document.createElement('span');
      detail.className = 'ms-suggest-detail';
      detail.textContent = s.detail;
      item.append(label, detail);
      // mousedown, not click: the input's blur would hide the list first.
      item.addEventListener('mousedown', (e) => {
        e.preventDefault();
        this.runSuggestion(i);
      });
      this.suggestBox.appendChild(item);
    });
    this.suggestBox.hidden = this.suggestions.length === 0 || document.activeElement !== this.searchInput;
  }

  /** What the search box offers for `text`, best first. */
  suggestionsFor(text: string): Suggestion[] {
    const q = parseQuery(text);
    const out: Suggestion[] = [];
    const idSuggestion = (r: IdRow): Suggestion => ({
      label: `${r.modified_sequence}/${r.charge}`,
      detail: [
        r.precursor_mz !== null ? `m/z ${r.precursor_mz.toFixed(3)}` : '',
        r.rt !== null ? `${(r.rt / 60).toFixed(2)} min` : '',
        r.qvalue !== null ? `q ${r.qvalue.toExponential(1)}` : '',
        r.proteins ?? '',
      ]
        .filter(Boolean)
        .join(' · '),
      run: () => void this.selectIdentification(r),
    });
    switch (q.kind) {
      case 'scan':
        out.push({ label: `Scan ${q.scan}`, detail: this.scans?.has(q.scan) ? 'go to it' : 'not in this run', run: () => this.runScan(q.scan) });
        break;
      case 'rt':
        out.push({ label: `${q.minutes} min`, detail: 'centre the time window here', run: () => this.centerRt(q.minutes * 60) });
        break;
      case 'mz':
        out.push({ label: `m/z ${q.mz}`, detail: 'trace it in MS1', run: () => this.addAdhoc(rowId(1), q.mz) });
        break;
      case 'peptide': {
        for (const r of matchIdentifications(this.identifications, q, 12)) out.push(idSuggestion(r));
        const exact = out.length > 0 && this.identifications.some((r) => r.modified_sequence === q.text);
        if (!exact) {
          out.push({
            label: `${q.text}${q.charge ? `/${q.charge}` : ''}`,
            detail: 'trace this peptide (not identified)',
            run: () => void this.tracePeptide(q.text, q.charge),
          });
        }
        // A gene name made of amino-acid letters reads as a peptide too.
        if (q.charge === null) {
          for (const r of matchIdentifications(this.identifications, { kind: 'text', text: q.text }, 6)) {
            if (!out.some((s) => s.label === `${r.modified_sequence}/${r.charge}`)) out.push(idSuggestion(r));
          }
        }
        break;
      }
      case 'text':
        for (const r of matchIdentifications(this.identifications, q, 12)) out.push(idSuggestion(r));
        if (out.length === 0) out.push({ label: 'Nothing found', detail: this.identifications.length ? text : 'no search attached', run: () => undefined });
        break;
      default:
        break;
    }
    return out;
  }

  private runScan(scan: number): void {
    if (!this.goToScan(scan)) this.setStatus(`scan ${scan} is not in this run`, true);
  }

  private runSuggestion(i: number): void {
    const s = this.suggestions[i];
    if (!s) return;
    this.suggestBox.hidden = true;
    this.searchInput.blur();
    this.root.focus();
    s.run();
  }

  private onSearchKey(e: KeyboardEvent): void {
    if (e.key === 'ArrowDown' || e.key === 'ArrowUp') {
      if (!this.suggestions.length) return;
      this.suggestAt = (this.suggestAt + (e.key === 'ArrowDown' ? 1 : -1) + this.suggestions.length) % this.suggestions.length;
      Array.from(this.suggestBox.children).forEach((el, i) => el.classList.toggle('active', i === this.suggestAt));
      e.preventDefault();
    } else if (e.key === 'Enter') {
      if (!this.suggestions.length) this.updateSuggestions();
      this.runSuggestion(this.suggestAt);
      e.preventDefault();
    } else if (e.key === 'Escape') {
      this.suggestBox.hidden = true;
      this.searchInput.blur();
      this.root.focus();
      e.preventDefault();
    }
    e.stopPropagation();
  }

  // ── history ─────────────────────────────────────────────────────────

  private snapshotView(): HistoryEntry | null {
    const view = this.view;
    if (!view) return null;
    const rows: HistoryEntry['rows'] = {};
    for (const [id, r] of Object.entries(view.rows)) rows[id] = { scan: r.scan, window: r.window };
    const adhoc: Record<string, number[]> = {};
    for (const [id, l] of Object.entries(view.adhoc)) adhoc[id] = [...l];
    return { acquisition: view.acquisition, target: view.target, idKey: this.selectedIdKey, rows, rt: [...view.rt], adhoc };
  }

  /** Remember what is shown, for Back and Forward. Pans are not recorded. */
  private record(): void {
    if (this.restoring) return;
    const entry = this.snapshotView();
    if (!entry) return;
    const same = (a: HistoryEntry, b: HistoryEntry): boolean =>
      a.acquisition === b.acquisition &&
      a.target?.key === b.target?.key &&
      JSON.stringify(a.rows) === JSON.stringify(b.rows) &&
      JSON.stringify(a.adhoc) === JSON.stringify(b.adhoc);
    const current = this.history[this.historyAt];
    if (current && same(current, entry)) {
      this.history[this.historyAt] = entry;
      return;
    }
    this.history = this.history.slice(0, this.historyAt + 1);
    this.history.push(entry);
    if (this.history.length > 100) this.history.shift();
    this.historyAt = this.history.length - 1;
    this.updateHistoryButtons();
  }

  private updateHistoryButtons(): void {
    this.backBtn.disabled = this.historyAt <= 0;
    this.forwardBtn.disabled = this.historyAt >= this.history.length - 1;
  }

  async go(delta: number): Promise<void> {
    const at = this.historyAt + delta;
    const entry = this.history[at];
    if (!entry) return;
    this.historyAt = at;
    this.restoring = true;
    try {
      if (entry.acquisition !== this.view?.acquisition) await this.showAcquisition(entry.acquisition, { record: false });
      const view = this.view;
      if (!view) return;
      for (const [id, r] of Object.entries(entry.rows)) {
        if (!view.rows[id]) continue;
        view.rows[id].scan = r.scan;
        view.rows[id].window = r.window;
        this.syncWindowSelect(id);
      }
      view.rt = [...entry.rt];
      view.adhoc = {};
      for (const [id, l] of Object.entries(entry.adhoc)) view.adhoc[id] = [...l];
      view.target = entry.target;
      view.xicRt = this.info?.has_index ? null : view.target || Object.keys(view.adhoc).length ? this.xicRange(view.target) : null;
      this.selectedIdKey = entry.idKey;
      this.drawer.setSelected(entry.idKey);
      this.ionsFrom = null;
      this.drawer.setIons(entry.target, new Set());
      this.updateChips();
      this.updateReadouts();
      this.scheduleScanHeader();
      this.requestRender();
    } finally {
      this.restoring = false;
      this.updateHistoryButtons();
    }
  }

  // ── grid driver ─────────────────────────────────────────────────────

  private traceOf(row: RowState): 'tic' | 'bpc' | 'scans' {
    if (row.level > 1 && row.window === null) return 'scans'; // DDA scan events
    return this.view?.trace ?? 'tic';
  }

  private chromKey(view: MsView, row: RowState): string {
    return `${view.acquisition}|L${row.level}|w${row.window ?? '*'}|${this.traceOf(row)}`;
  }

  private tolerance(level: number, kind: 'precursor' | 'fragment'): [number, ToleranceUnit] {
    const t = this.view?.target;
    const fromTarget = kind === 'fragment' ? t?.fragmentTolerance : t?.precursorTolerance;
    if (fromTarget) return fromTarget;
    const d = this.info?.default_tolerance[String(level)] ?? this.info?.default_tolerance['1'];
    return d ? [d[0], d[1] === 'Da' ? 'Da' : 'ppm'] : [20, 'ppm'];
  }

  /** What a chromatogram row traces: the target's ions it can see, then
   *  the user's own. */
  private tracesOf(view: MsView, id: string): RowTrace[] {
    const row = view.rows[id];
    const out: RowTrace[] = [];
    const t = view.target;
    if (t && row.level === 1) {
      t.ms1.forEach((x, i) => out.push({ mz: x.mz, label: x.label, color: ISOTOPES[Math.min(i, ISOTOPES.length - 1)], dashed: false }));
    } else if (t && row.level === 2 && row.window !== null && this.windowCovers(row, t.precursorMz)) {
      t.ms2.slice(0, TARGET_FRAGMENTS).forEach((x, i) =>
        out.push({ mz: x.mz, label: x.label, color: seriesColor(i), dashed: x.quantified === false }),
      );
    }
    (view.adhoc[id] ?? []).forEach((mz, i) =>
      out.push({ mz, label: mz.toFixed(3), color: SERIES[SERIES.length - 1 - i], dashed: false }),
    );
    return out;
  }

  private xicRequest(view: MsView, id: string): { key: string; params: Record<string, string | number | string[] | undefined> } | null {
    const row = view.rows[id];
    const traces = this.tracesOf(view, id);
    if (traces.length === 0) return null;
    const tol = this.tolerance(row.level, row.level === 1 ? 'precursor' : 'fragment');
    // A DDA level's MS2 scans isolate many precursors: tracing an ion over
    // all of them reads the whole run (10 s on the reference DDA run), so
    // its traces cover the target's time, or follow the time shown.
    const dda = row.level > 1 && row.window === null;
    const rt = dda ? (view.target?.rt != null ? this.xicRange(view.target) : this.followRange(view)) : view.xicRt;
    const params = {
      level: row.level,
      window: row.window ?? undefined,
      mz: traces.map((t) => String(t.mz)),
      label: traces.map((t) => t.label),
      tolerance: tol[0],
      tolerance_unit: tol[1],
      rt_start: rt ? rt[0] : undefined,
      rt_end: rt ? rt[1] : undefined,
    };
    const key = `${view.acquisition}|L${row.level}|w${row.window ?? '*'}|${params.mz.join(',')}|${tol.join('')}|${rt ? rt.map((v) => v.toFixed(1)).join('-') : 'all'}`;
    return { key, params };
  }

  /** Whether a row's spectrum is annotated with the target's ions, and how. */
  private annotationOf(cell: GridCell<CellData>, view: MsView): Record<string, string | number | string[] | undefined> | null {
    const row = view.rows[cell.row];
    const t = view.target;
    if (!t || row.level < 2 || row.scan === null) return null;
    if (this.scans?.isolates(row.scan, t.precursorMz) === false) return null;
    const settings = annotationSettings(cell.panel.filter);
    const tol = this.tolerance(row.level, 'fragment');
    return {
      modified_sequence: t.modifiedSequence,
      charge: t.charge,
      tolerance: tol[0],
      tolerance_unit: tol[1],
      ion_types: settings.ion_types,
      max_fragment_charge: settings.max_fragment_charge ?? undefined,
      neutral_losses: settings.neutral_losses,
    };
  }

  private driver(): GridDriver<MsView, CellData> {
    return {
      view: () => this.view,
      dataKey: (cell, view) => {
        const row = view.rows[cell.row];
        if (!row) return null;
        if (cell.col === 'chrom') {
          const key = this.chromKey(view, row);
          const xic = this.xicRequest(view, cell.row)?.key ?? '';
          // A reduced trace was reduced for a window: fetch again when it moves.
          const base = this.decimated.has(key) ? `${key}|${view.rt[0].toFixed(1)}-${view.rt[1].toFixed(1)}` : key;
          return `${base}|${xic}`;
        }
        if (row.scan === null) return null;
        const annotation = this.annotationOf(cell, view);
        return `${view.acquisition}|scan${row.scan}${annotation ? `|${JSON.stringify(annotation)}` : ''}`;
      },
      fetch: async (cell, view, signal) => {
        const row = view.rows[cell.row];
        const session = this.opts.sessionId;
        const widthPx = this.grid?.cellSize(cell).widthPx ?? 800;
        const viewport_px = Math.max(1, Math.min(8192, Math.round(widthPx)));
        if (cell.col === 'chrom') {
          const key = this.chromKey(view, row);
          const xicReq = this.xicRequest(view, cell.row);
          const [base, xic] = await Promise.all([
            fetchTrackData(
              'ms_chromatogram',
              {
                session,
                binding: `ms_chromatogram:${view.acquisition}`,
                level: row.level,
                trace: this.traceOf(row),
                window: row.window ?? undefined,
                viewport_px,
                rt_start: this.decimated.has(key) ? view.rt[0] : undefined,
                rt_end: this.decimated.has(key) ? view.rt[1] : undefined,
              },
              signal,
            ),
            xicReq ? this.fetchXic(view.acquisition, xicReq, viewport_px, signal) : Promise.resolve(null),
          ]);
          const reduced = base.table.numRows > 0 && Boolean(base.table.getChild('decimated')?.get(0));
          if (reduced) this.decimated.add(key);
          return { table: base.table, xic };
        }
        const { table } = await fetchTrackData(
          'ms_spectrum',
          {
            session,
            binding: `ms_spectrum:${view.acquisition}`,
            scan: row.scan ?? 0,
            viewport_px,
            ...(this.annotationOf(cell, view) ?? {}),
          },
          signal,
        );
        return { table, xic: null };
      },
      draw: (cell, data, view, svg, size) => this.drawCell(cell, data, view, svg, size),
      placeholder: () => 'Click a chromatogram to pick a scan.',
    };
  }

  private async fetchXic(
    acquisition: string,
    req: { key: string; params: Record<string, string | number | string[] | undefined> },
    viewport_px: number,
    signal: AbortSignal,
  ): Promise<Table> {
    const cached = this.xicCache.get(req.key);
    if (cached) return cached;
    const { table } = await fetchTrackData(
      'ms_xic',
      { session: this.opts.sessionId, binding: `ms_xic:${acquisition}`, viewport_px, ...req.params },
      signal,
    );
    if (this.xicCache.size >= 32) this.xicCache.delete(this.xicCache.keys().next().value as string);
    this.xicCache.set(req.key, table);
    return table;
  }

  private drawCell(
    cell: GridCell<CellData>,
    data: CellData,
    view: MsView,
    svg: SVGSVGElement,
    size: { widthPx: number; heightPx: number },
  ): number {
    const row = view.rows[cell.row];
    const uid = `ms-${cell.row}-${cell.col}`;
    const t = view.target;
    if (cell.col === 'chrom') {
      const selectedRt = row.scan !== null ? (this.scans?.rtOf(row.scan) ?? null) : null;
      const traces = this.tracesOf(view, cell.row);
      const overlays = data.xic ? overlaysFrom(data.xic, traces) : [];
      const showsTarget = !!t && traces.length > (view.adhoc[cell.row]?.length ?? 0);
      const sticks = this.traceOf(row) === 'scans';
      const stickFilter = t && sticks ? (mz: number) => this.isolatesTarget(mz) : null;
      return drawChromatogram(svg, size, data.table, {
        rt: view.rt,
        intensity: view.intensity,
        selectedRt,
        sticks,
        stickFilter,
        overlays,
        marks: t ? { start: t.rtStart, end: t.rtEnd, apex: t.rt, predicted: t.predictedRt } : null,
        median:
          showsTarget && row.level > 1 && t && t.median.length
            ? { xs: Float64Array.from(t.median, (p) => p.rt), ys: Float64Array.from(t.median, (p) => p.intensity) }
            : null,
        style: cell.panel.style,
        uid,
      });
    }
    // An MS1 spectrum shades what the scan shown one level up isolated.
    const above = Object.values(view.rows).find((r) => r.level === row.level + 1);
    let isolation: Interval | null = null;
    let marker: number | null = null;
    if (above?.scan != null && this.scans) {
      const r = this.scans.row(above.scan);
      if (r !== undefined) {
        const mz = this.scans.precursorMz[r];
        marker = Number.isFinite(mz) ? mz : null;
        const lo = this.scans.isolationLower[r];
        const hi = this.scans.isolationUpper[r];
        if (Number.isFinite(lo) && Number.isFinite(hi) && hi > lo) isolation = [lo, hi];
        else {
          const w = this.windows.find((x) => x.level === above.level && x.window === this.scans!.windows[r]);
          isolation = w ? [w.isolation_lower, w.isolation_upper] : null;
        }
      }
    }
    if (t && row.level === 1) marker = t.precursorMz;
    const rt = row.scan !== null ? this.scans?.rtOf(row.scan) : null;
    cell.panel.headerLabelEl.textContent =
      row.scan === null ? `MS${row.level} spectrum` : `MS${row.level} scan ${row.scan}${rt == null ? '' : ` · ${(rt / 60).toFixed(2)} min`}`;
    const annotated = this.annotationOf(cell, view) !== null;
    const tol = this.tolerance(row.level, 'fragment');
    const shown = drawSpectrum(svg, size, data.table, {
      mz: row.mz,
      intensity: view.intensity,
      isolation,
      target: marker,
      isotopes: t && row.level === 1 ? t.ms1 : [],
      errorStrip: annotated ? { tolerance: tol[0], unit: tol[1] } : null,
      style: cell.panel.style,
      uid,
    });
    if (annotated && t) this.updateIons(t, data.table);
    return shown;
  }

  /** The drawer's ion table lit with what the shown spectrum matched. */
  private updateIons(target: Target, table: Table): void {
    if (this.ionsFrom?.target === target.key && this.ionsFrom.table === table) return;
    this.ionsFrom = { target: target.key, table };
    const labels = table.getChild('annotation');
    const matched = new Set<string>();
    for (let i = 0; i < table.numRows; i++) {
      const l = labels?.get(i);
      if (l) matched.add(String(l));
    }
    this.drawer.setIons(target, matched);
  }

  // ── interactions ────────────────────────────────────────────────────

  private attachInteractions(): void {
    for (const off of this.detachers) off();
    this.detachers = [];
    const grid = this.grid;
    if (!grid) return;
    for (const cell of grid.cells) {
      const body = cell.panel.bodyHost;
      if (cell.col === 'chrom') {
        const cursor = document.createElement('div');
        cursor.className = 'ms-cursor';
        cursor.hidden = true;
        body.appendChild(cursor);
        this.detachers.push(
          attachPanZoom1D({
            surface: body,
            plotRect: () => this.plotRect(cell, CHROM_MARGIN),
            get: () => this.view?.rt ?? [0, 1],
            set: (next) => {
              if (!this.view || sameInterval(this.view.rt, next)) return;
              this.view.rt = next;
              this.requestRender();
            },
            limits: () => this.rtLimits(),
            onClick: (rt) => {
              const row = this.view?.rows[cell.row];
              const scan = row && this.scans?.nearest(row.level, row.window, rt);
              if (scan != null) this.selectScan(cell.row, scan);
            },
            onHover: (rt) => this.showCursor(rt),
          }),
        );
      } else {
        const tip = document.createElement('div');
        tip.className = 'ms-tooltip';
        tip.hidden = true;
        body.appendChild(tip);
        this.detachers.push(
          attachPanZoom1D({
            surface: body,
            plotRect: () => this.plotRect(cell, SPECTRUM_MARGIN),
            get: () => this.mzWindowOf(cell),
            set: (next, cause) => {
              const row = this.view?.rows[cell.row];
              if (!row) return;
              row.mz = cause === 'reset' ? null : next;
              this.requestRender();
            },
            limits: () => {
              const [, hi] = this.spectrumExtentOf(cell);
              return { min: 0, max: Math.max(hi, 50), minSpan: 0.05 };
            },
            fit: () => this.spectrumExtentOf(cell),
            onClick: (mz) => {
              const peak = this.peakAt(cell, mz);
              if (peak) this.addAdhoc(cell.row, peak.mz);
            },
            onHover: (mz) => this.showTooltip(cell, tip, mz),
          }),
        );
      }
    }
  }

  private mzWindowOf(cell: GridCell<CellData>): Interval {
    return this.view?.rows[cell.row]?.mz ?? this.spectrumExtentOf(cell);
  }

  private spectrumExtentOf(cell: GridCell<CellData>): Interval {
    const table = cell.panel.lastFetched?.data.table;
    const mzs = table?.getChild('mz')?.toArray() as Float64Array | undefined;
    return spectrumExtent(mzs ?? new Float64Array(0));
  }

  /** The peak of a spectrum cell under the pointer at `mz`. */
  private peakAt(cell: GridCell<CellData>, mz: number): { index: number; mz: number; table: Table } | null {
    const table = cell.panel.lastFetched?.data.table;
    const mzs = table?.getChild('mz')?.toArray() as Float64Array | undefined;
    if (!table || !mzs || !this.grid) return null;
    const domain = this.mzWindowOf(cell);
    const width = Math.max(1, this.grid.cellSize(cell).widthPx - SPECTRUM_MARGIN.left - SPECTRUM_MARGIN.right);
    const index = nearestPeak(mzs, mz, (PEAK_HIT_PX * (domain[1] - domain[0])) / width);
    return index === null ? null : { index, mz: mzs[index], table };
  }

  private showTooltip(cell: GridCell<CellData>, tip: HTMLElement, mz: number | null): void {
    const peak = mz === null ? null : this.peakAt(cell, mz);
    if (!peak || !this.grid) {
      tip.hidden = true;
      return;
    }
    const { table, index } = peak;
    const intensity = Number(table.getChild('intensity')?.get(index));
    const label = table.getChild('annotation')?.get(index) as string | null | undefined;
    const ppm = table.getChild('error_ppm')?.get(index) as number | null | undefined;
    const parts = [`m/z ${peak.mz.toFixed(4)}`, intensity.toExponential(2)];
    if (label) parts.push(label);
    if (ppm !== null && ppm !== undefined) parts.push(`${ppm >= 0 ? '+' : ''}${ppm.toFixed(1)} ppm`);
    tip.textContent = parts.join(' · ');
    const domain = this.mzWindowOf(cell);
    const width = this.grid.cellSize(cell).widthPx - SPECTRUM_MARGIN.left - SPECTRUM_MARGIN.right;
    const x = SPECTRUM_MARGIN.left + ((peak.mz - domain[0]) / (domain[1] - domain[0])) * width;
    tip.hidden = false;
    tip.style.left = `${Math.max(0, Math.min(x + 8, this.grid.cellSize(cell).widthPx - 200))}px`;
  }

  private plotRect(cell: GridCell<CellData>, margin: { left: number; right: number }): { left: number; width: number } {
    const rect = cell.panel.bodyHost.getBoundingClientRect();
    const width = this.grid?.cellSize(cell).widthPx ?? rect.width;
    return { left: rect.left + margin.left, width: Math.max(1, width - margin.left - margin.right) };
  }

  /** The hover crosshair, drawn on every chromatogram without redrawing them. */
  private showCursor(rt: number | null): void {
    const view = this.view;
    if (!this.grid || !view) return;
    for (const cell of this.grid.cells) {
      if (cell.col !== 'chrom') continue;
      const cursor = cell.panel.bodyHost.querySelector<HTMLElement>('.ms-cursor');
      if (!cursor) continue;
      if (rt === null || rt < view.rt[0] || rt > view.rt[1]) {
        cursor.hidden = true;
        continue;
      }
      const width = this.grid.cellSize(cell).widthPx - CHROM_MARGIN.left - CHROM_MARGIN.right;
      cursor.hidden = false;
      cursor.style.left = `${CHROM_MARGIN.left + ((rt - view.rt[0]) / (view.rt[1] - view.rt[0])) * width}px`;
      cursor.title = `${(rt / 60).toFixed(2)} min`;
    }
  }

  private onKey(e: KeyboardEvent): void {
    const target = e.target as HTMLElement | null;
    if (target && ['INPUT', 'SELECT', 'TEXTAREA'].includes(target.tagName)) return;
    if (e.altKey && (e.key === 'ArrowLeft' || e.key === 'ArrowRight')) {
      void this.go(e.key === 'ArrowLeft' ? -1 : 1);
      e.preventDefault();
    } else if (e.key === 'ArrowLeft' || e.key === 'ArrowRight') {
      this.stepScan(e.key === 'ArrowLeft' ? -1 : 1);
      e.preventDefault();
    } else if (e.key === '[' || e.key === ']') {
      void this.stepIdentification(e.key === '[' ? -1 : 1);
      e.preventDefault();
    } else if (e.key === '/') {
      this.searchInput.focus();
      e.preventDefault();
    } else if (e.key === 'Escape') {
      this.grid?.setFocus({ type: 'none' });
    } else if (e.key === 'f') {
      this.fitRt();
    }
  }

  // ── peak index ──────────────────────────────────────────────────────

  /** Build the acquisition's peak index through the dashboard's command
   *  runner (the server itself never writes), then trace whole runs. */
  async buildIndex(): Promise<void> {
    const info = this.info;
    if (!info || info.has_index) return;
    const out = `${info.bundle.replace(/\/+$/, '')}/mz_index`;
    if (
      typeof window.confirm === 'function' &&
      !window.confirm(
        `Build a peak index for ${info.label}?\n\nIt is written to ${out}, takes about as much disk as the run's peaks, and minutes to build.`,
      )
    ) {
      return;
    }
    let job: { job_id: string };
    try {
      job = await fetchJsonMethod<{ job_id: string }>('/api/commands', 'POST', {
        argv: ['massspec', 'chromatogram', 'build-index', '--peaks-dir', info.bundle, '-o', out],
      });
    } catch (err) {
      this.setStatus(`could not start the index build: ${(err as Error).message}`, true);
      return;
    }
    this.indexBtn.disabled = true;
    this.indexBtn.textContent = 'Building index…';
    const poll = async (): Promise<void> => {
      this.indexPoll = null;
      if (this.disposed) return;
      let state: { state: string; exit_code: number | null };
      try {
        state = await fetchJson(`/api/commands/${encodeURIComponent(job.job_id)}`);
      } catch {
        this.indexPoll = setTimeout(() => void poll(), 5000);
        return;
      }
      if (state.state === 'running' || state.state === 'pending') {
        this.indexPoll = setTimeout(() => void poll(), 3000);
        return;
      }
      this.indexBtn.disabled = false;
      this.indexBtn.textContent = 'Build index';
      if (state.exit_code === 0) {
        info.has_index = true;
        this.indexBtn.hidden = this.info === info;
        if (this.info === info && this.view) {
          this.view.xicRt = null;
          this.xicCache.clear();
          this.updateChips();
          this.requestRender();
        }
        this.setStatus(`peak index built for ${info.label}`);
      } else {
        this.setStatus(`the index build ${state.state} (exit ${state.exit_code ?? '?'}); its output is in the dashboard's command log`, true);
      }
    };
    this.indexPoll = setTimeout(() => void poll(), 2000);
  }

  // ── rendering ───────────────────────────────────────────────────────

  private onResize(): void {
    if (!this.grid) return;
    const rect = this.gridHost.getBoundingClientRect();
    this.grid.resize({ widthPx: rect.width || this.gridHost.clientWidth, heightPx: rect.height || this.gridHost.clientHeight });
    this.requestRender();
  }

  /** Render on the next frame; calls in between are merged. */
  requestRender(): void {
    if (this.renderQueued || this.disposed) return;
    this.renderQueued = true;
    const run = (): void => {
      this.renderQueued = false;
      if (this.disposed) return;
      this.updateReadouts();
      void this.grid?.render();
    };
    if (typeof requestAnimationFrame === 'function') requestAnimationFrame(run);
    else setTimeout(run, 16);
  }

  /** Render now and wait for every fetch (tests, export). */
  async renderNow(): Promise<void> {
    this.updateReadouts();
    await this.grid?.render();
  }

  // ── export and teardown ─────────────────────────────────────────────

  private copySpectrumRef(cell: GridCell<CellData>): void {
    const row = this.view?.rows[cell.row];
    if (!row || row.scan === null || !this.info) return;
    const ref = `${this.info.label}:scan:${row.scan}`;
    const done = (): void => this.setStatus(`copied ${ref}`);
    if (navigator.clipboard?.writeText) navigator.clipboard.writeText(ref).then(done, () => this.setStatus(ref));
    else this.setStatus(ref);
  }

  /** The cell's data as CSV: a spectrum's peaks, or a chromatogram's
   *  points (its traces, when it shows any). */
  cellCsv(cell: GridCell<CellData>): string | null {
    const data = cell.panel.lastFetched?.data;
    if (!data) return null;
    if (cell.col === 'chrom' && data.xic && data.xic.numRows > 0) {
      return tableCsv(data.xic, ['label', 'mz_target', 'scan', 'rt', 'intensity', 'mz_observed']);
    }
    const columns =
      cell.col === 'chrom'
        ? ['rt', 'intensity', 'scan', 'mz']
        : ['mz', 'intensity', 'annotation', 'ion_type', 'ion_number', 'fragment_charge', 'loss', 'mz_theoretical', 'error_ppm'];
    return tableCsv(data.table, columns);
  }

  private downloadCsv(cell: GridCell<CellData>): void {
    const csv = this.cellCsv(cell);
    if (!csv) return;
    const row = this.view?.rows[cell.row];
    const what = cell.col === 'chrom' ? `ms${row?.level}-chromatogram` : `scan-${row?.scan}`;
    const name = `${this.info?.label ?? 'run'}-${what}.csv`;
    const url = URL.createObjectURL(new Blob([csv], { type: 'text/csv' }));
    const a = document.createElement('a');
    a.href = url;
    a.download = name;
    a.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }

  exportSvg(): void {
    if (!this.grid) return;
    const cells = this.grid.layoutRects().flatMap((rect) => {
      const svg = rect.cell.panel.bodyHost.querySelector('svg.track-canvas') as SVGSVGElement | null;
      return svg ? [{ svg, x: rect.x, y: rect.y, width: rect.width, height: rect.height, title: rect.cell.panel.headerLabelEl.textContent ?? '' }] : [];
    });
    if (cells.length === 0) return;
    const width = Math.max(...cells.map((c) => c.x + c.width));
    const height = Math.max(...cells.map((c) => c.y + c.height));
    const title = this.view?.target ? `${this.info?.label ?? ''} · ${this.view.target.label}` : (this.info?.label ?? 'mass spectrometry');
    const { svg } = buildGridSvg(cells, { title, widthPx: width, heightPx: height });
    downloadSvg(`${this.info?.label ?? 'spectra'}.svg`, svg);
  }

  dispose(): void {
    this.disposed = true;
    for (const off of this.detachers) off();
    this.detachers = [];
    if (this.scanHeaderTimer) clearTimeout(this.scanHeaderTimer);
    if (this.indexPoll) clearTimeout(this.indexPoll);
    this.resizeObserver?.disconnect();
    this.grid?.dispose();
    this.root.remove();
  }
}

/** A target as a peptide: what to trace, without where it was found. */
function asPeptide(t: Target): Target {
  return {
    ...t,
    key: `peptide|${t.modifiedSequence}/${t.charge}`,
    rt: null,
    rtStart: null,
    rtEnd: null,
    predictedRt: null,
    window: null,
    apexScans: { ms1: null, ms2: null },
    median: [],
    qvalue: null,
    source: null,
  };
}

/** One overlay per trace of an `ms_xic` table, in the row's trace order. */
function overlaysFrom(xic: Table, traces: RowTrace[]): Overlay[] {
  const trace = xic.getChild('trace');
  const rt = xic.getChild('rt');
  const intensity = xic.getChild('intensity');
  const points: Array<Array<[number, number]>> = traces.map(() => []);
  for (let i = 0; i < xic.numRows; i++) {
    const k = Number(trace?.get(i));
    points[k]?.push([Number(rt?.get(i)), Number(intensity?.get(i))]);
  }
  return traces.map((t, k) => {
    const p = points[k].sort((a, b) => a[0] - b[0]);
    return {
      label: t.label,
      color: t.color,
      dashed: t.dashed,
      xs: Float64Array.from(p, (v) => v[0]),
      ys: Float64Array.from(p, (v) => v[1]),
    };
  });
}

function csvCell(v: unknown): string {
  if (v === null || v === undefined) return '';
  const s = typeof v === 'bigint' ? v.toString() : String(v);
  return /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
}

function tableCsv(table: Table, columns: string[]): string {
  const present = columns.filter((c) => table.getChild(c));
  const lines = [present.join(',')];
  const cols = present.map((c) => table.getChild(c)!);
  for (let i = 0; i < table.numRows; i++) lines.push(cols.map((c) => csvCell(c.get(i))).join(','));
  return `${lines.join('\n')}\n`;
}
