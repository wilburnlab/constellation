// MassSpecBrowser — chromatograms and spectra of one acquisition, by MS level.
//
// The layout is a `PanelGrid`: one row per MS level the run has (MS1, MS2,
// …), a chromatogram column and a spectrum column. Above it a toolbar
// picks the acquisition, the trace and the intensity scale, steps between
// scans and exports the grid.
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
// Data comes from the `ms_chromatogram` / `ms_spectrum` kernels; scan
// navigation runs on the client from the acquisition's scan table.

import type { Table } from 'apache-arrow';
import { fetchArrow, fetchJson, fetchTrackData } from '../../engine/arrow_client';
import { buildGridSvg, downloadSvg } from '../../engine/export';
import { Interval, IntervalLimits, sameInterval } from '../../engine/interval';
import { attachPanZoom1D } from '../../engine/pan_zoom';
import { PanelKind } from '../../panels/kind';
import { GridAxisSpec, GridCell, GridDriver, PanelGrid } from '../../panels/PanelGrid';
import { SettingsSchema } from '../../panels/settings_schema';
import {
  CHROMATOGRAM_KIND,
  CHROM_MARGIN,
  IntensityMode,
  drawChromatogram,
} from './renderers/chromatogram';
import { SPECTRUM_KIND, SPECTRUM_MARGIN, drawSpectrum, spectrumExtent } from './renderers/spectrum';
import { ScanIndex } from './scan_index';
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
}

export interface MassSpecBrowserOptions {
  host: HTMLElement;
  sessionId: string;
}

const KINDS: Record<string, PanelKind> = {
  [CHROMATOGRAM_KIND.kind]: CHROMATOGRAM_KIND,
  [SPECTRUM_KIND.kind]: SPECTRUM_KIND,
};
const FALLBACK_SETTINGS: SettingsSchema = { sections: [] };
const GRID_STORAGE_KEY = 'constellation.massspec.grid.v1';
const COLUMNS: GridAxisSpec[] = [
  { id: 'chrom', label: 'Chromatogram', weight: 3 },
  { id: 'spec', label: 'Spectrum', weight: 2 },
];

export const rowId = (level: number): string => `ms${level}`;

export class MassSpecBrowser {
  private readonly opts: MassSpecBrowserOptions;
  private readonly root: HTMLElement;
  private readonly toolbar: HTMLElement;
  private readonly gridHost: HTMLElement;
  private readonly statusEl: HTMLElement;
  private acquisitionSelect!: HTMLSelectElement;
  private traceSelect!: HTMLSelectElement;
  private intensitySelect!: HTMLSelectElement;
  private rtLabel!: HTMLElement;
  private scanLabel!: HTMLElement;
  private grid: PanelGrid<MsView, Table> | null = null;
  private acquisitions: AcquisitionInfo[] = [];
  private info: AcquisitionInfo | null = null;
  private scans: ScanIndex | null = null;
  private windows: IsolationWindow[] = [];
  private view: MsView | null = null;
  /** The row the user last picked a scan in (where arrow keys step). */
  private activeRow: string | null = null;
  /** Chromatogram keys whose last response was reduced (they refetch on zoom). */
  private decimated = new Set<string>();
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
    this.gridHost = document.createElement('div');
    this.gridHost.className = 'ms-grid-host';
    this.statusEl = document.createElement('span');
    this.statusEl.className = 'ms-status';
    this.root.append(this.toolbar, this.gridHost);
  }

  /** Current view (for tests and the host page). */
  get currentView(): MsView | null {
    return this.view;
  }

  get panelGrid(): PanelGrid<MsView, Table> | null {
    return this.grid;
  }

  async mount(): Promise<void> {
    this.opts.host.appendChild(this.root);
    this.buildToolbar();
    this.root.addEventListener('keydown', (e) => this.onKey(e));
    const base = `/api/massspec/sessions/${encodeURIComponent(this.opts.sessionId)}`;
    try {
      this.acquisitions = await fetchJson<AcquisitionInfo[]>(`${base}/acquisitions`);
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

    this.acquisitionSelect = select([]);
    this.acquisitionSelect.addEventListener('change', () => void this.showAcquisition(this.acquisitionSelect.value));
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

    const spacer = document.createElement('span');
    spacer.className = 'ms-spacer';
    this.toolbar.append(
      field('Run', this.acquisitionSelect),
      field('Trace', this.traceSelect),
      field('Intensity', this.intensitySelect),
      field('RT', this.rtLabel),
      button('Fit', 'Show the whole run (f)', () => this.fitRt()),
      button('◀', 'Previous scan (←)', () => this.stepScan(-1)),
      this.scanLabel,
      button('▶', 'Next scan (→)', () => this.stepScan(1)),
      spacer,
      this.statusEl,
      button('Save SVG', 'Export the panels shown', () => this.exportSvg()),
    );
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

  // ── acquisition ─────────────────────────────────────────────────────

  async showAcquisition(key: string): Promise<void> {
    const info = this.acquisitions.find((a) => a.key === key);
    if (!info) return;
    this.acquisitionSelect.value = key;
    const base = `/api/massspec/sessions/${encodeURIComponent(this.opts.sessionId)}/acquisitions/${encodeURIComponent(key)}`;
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
    this.info = info;
    this.scans = new ScanIndex(table);
    this.decimated.clear();

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
    };
    const rowSpecs = info.levels.map((level) => ({ id: rowId(level), label: `MS${level}`, weight: 1 }));
    if (!this.grid) this.grid = this.createGrid(rowSpecs);
    else this.grid.setRows(rowSpecs);
    this.decorateRows();
    this.attachInteractions();
    this.onResize();

    // Open on the most intense MS1 scan, so every panel shows something.
    const first = this.scans.mostIntense(info.levels[0]);
    if (first !== null) this.selectScan(rowId(info.levels[0]), first);
    this.setStatus('');
    this.updateReadouts();
    this.requestRender();
  }

  private windowsOf(level: number): IsolationWindow[] {
    return this.windows.filter((w) => w.level === level);
  }

  private createGrid(rows: GridAxisSpec[]): PanelGrid<MsView, Table> {
    return new PanelGrid<MsView, Table>({
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

  /** Show another DIA window in a row, keeping the time the user was at. */
  setWindow(id: string, window: number): void {
    const row = this.view?.rows[id];
    if (!row || !this.scans) return;
    const rt = row.scan !== null ? this.scans.rtOf(row.scan) : null;
    row.window = window;
    if (rt !== null) {
      const scan = this.scans.nearest(row.level, window, rt);
      if (scan !== null) this.selectScan(id, scan);
    }
    const select = this.grid?.rowSlot(id)?.querySelector('select');
    if (select) select.value = String(window);
    this.requestRender();
  }

  // ── scans ───────────────────────────────────────────────────────────

  /** Show `scan` in row `id` and carry the choice to the other levels. */
  selectScan(id: string, scan: number): void {
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
        if (w !== null && row.window !== null) row.window = w;
      }
      current = next;
    }
    this.updateReadouts();
    this.requestRender();
  }

  /** Step the active row's scan within its level and window. */
  stepScan(delta: number): void {
    const view = this.view;
    if (!view || !this.scans || !this.activeRow) return;
    const row = view.rows[this.activeRow];
    if (row?.scan == null) return;
    const next = this.scans.step(row.scan, delta, row.window);
    if (next !== null && next !== row.scan) {
      this.selectScan(this.activeRow, next);
      this.keepScanInView(next);
    }
  }

  private keepScanInView(scan: number): void {
    const rt = this.scans?.rtOf(scan);
    const view = this.view;
    if (!view || rt == null || (rt >= view.rt[0] && rt <= view.rt[1])) return;
    const span = view.rt[1] - view.rt[0];
    view.rt = this.clampRt([rt - span / 2, rt + span / 2]);
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

  // ── grid driver ─────────────────────────────────────────────────────

  private traceOf(row: RowState): 'tic' | 'bpc' | 'scans' {
    if (row.level > 1 && row.window === null) return 'scans'; // DDA scan events
    return this.view?.trace ?? 'tic';
  }

  private chromKey(view: MsView, row: RowState): string {
    return `${view.acquisition}|L${row.level}|w${row.window ?? '*'}|${this.traceOf(row)}`;
  }

  private driver(): GridDriver<MsView, Table> {
    return {
      view: () => this.view,
      dataKey: (cell, view) => {
        const row = view.rows[cell.row];
        if (!row) return null;
        if (cell.col === 'chrom') {
          const key = this.chromKey(view, row);
          // A reduced trace was reduced for a window: fetch again when it moves.
          return this.decimated.has(key) ? `${key}|${view.rt[0].toFixed(1)}-${view.rt[1].toFixed(1)}` : key;
        }
        return row.scan === null ? null : `${view.acquisition}|scan${row.scan}`;
      },
      fetch: async (cell, view, signal) => {
        const row = view.rows[cell.row];
        const session = this.opts.sessionId;
        const widthPx = this.grid?.cellSize(cell).widthPx ?? 800;
        if (cell.col === 'chrom') {
          const key = this.chromKey(view, row);
          const params = {
            session,
            binding: `ms_chromatogram:${view.acquisition}`,
            level: row.level,
            trace: this.traceOf(row),
            window: row.window ?? undefined,
            viewport_px: Math.max(1, Math.min(8192, Math.round(widthPx))),
            rt_start: this.decimated.has(key) ? view.rt[0] : undefined,
            rt_end: this.decimated.has(key) ? view.rt[1] : undefined,
          };
          const { table } = await fetchTrackData('ms_chromatogram', params, signal);
          const reduced = table.numRows > 0 && Boolean(table.getChild('decimated')?.get(0));
          if (reduced) this.decimated.add(key);
          return table;
        }
        const { table } = await fetchTrackData(
          'ms_spectrum',
          {
            session,
            binding: `ms_spectrum:${view.acquisition}`,
            scan: row.scan ?? 0,
            viewport_px: Math.max(1, Math.min(8192, Math.round(widthPx))),
          },
          signal,
        );
        return table;
      },
      draw: (cell, data, view, svg, size) => this.drawCell(cell, data, view, svg, size),
      placeholder: () => 'Click a chromatogram to pick a scan.',
    };
  }

  private drawCell(
    cell: GridCell<Table>,
    data: Table,
    view: MsView,
    svg: SVGSVGElement,
    size: { widthPx: number; heightPx: number },
  ): number {
    const row = view.rows[cell.row];
    const uid = `ms-${cell.row}-${cell.col}`;
    if (cell.col === 'chrom') {
      const selectedRt = row.scan !== null ? (this.scans?.rtOf(row.scan) ?? null) : null;
      return drawChromatogram(svg, size, data, {
        rt: view.rt,
        intensity: view.intensity,
        selectedRt,
        sticks: this.traceOf(row) === 'scans',
        style: cell.panel.style,
        uid,
      });
    }
    // An MS1 spectrum shades what the scan shown one level up isolated.
    const above = Object.values(view.rows).find((r) => r.level === row.level + 1);
    let isolation: Interval | null = null;
    let target: number | null = null;
    if (above?.scan != null && this.scans) {
      const r = this.scans.row(above.scan);
      if (r !== undefined) {
        const mz = this.scans.precursorMz[r];
        target = Number.isFinite(mz) ? mz : null;
        const w = this.windows.find((x) => x.level === above.level && x.window === this.scans!.windows[r]);
        isolation = w ? [w.isolation_lower, w.isolation_upper] : null;
      }
    }
    const rt = row.scan !== null ? this.scans?.rtOf(row.scan) : null;
    cell.panel.headerLabelEl.textContent =
      row.scan === null ? `MS${row.level} spectrum` : `MS${row.level} scan ${row.scan}${rt == null ? '' : ` · ${(rt / 60).toFixed(2)} min`}`;
    return drawSpectrum(svg, size, data, { mz: row.mz, intensity: view.intensity, isolation, target, style: cell.panel.style, uid });
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
        this.detachers.push(
          attachPanZoom1D({
            surface: body,
            plotRect: () => this.plotRect(cell, SPECTRUM_MARGIN),
            get: () => this.view?.rows[cell.row]?.mz ?? this.spectrumExtentOf(cell),
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
          }),
        );
      }
    }
  }

  private spectrumExtentOf(cell: GridCell<Table>): Interval {
    const table = cell.panel.lastFetched?.data;
    const mzs = table?.getChild('mz')?.toArray() as Float64Array | undefined;
    return spectrumExtent(mzs ?? new Float64Array(0));
  }

  private plotRect(cell: GridCell<Table>, margin: { left: number; right: number }): { left: number; width: number } {
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
    if (e.key === 'ArrowLeft' || e.key === 'ArrowRight') {
      this.stepScan(e.key === 'ArrowLeft' ? -1 : 1);
      e.preventDefault();
    } else if (e.key === 'Escape') {
      this.grid?.setFocus({ type: 'none' });
    } else if (e.key === 'f') {
      this.fitRt();
    }
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

  exportSvg(): void {
    if (!this.grid) return;
    const cells = this.grid.layoutRects().flatMap((rect) => {
      const svg = rect.cell.panel.bodyHost.querySelector('svg.track-canvas') as SVGSVGElement | null;
      return svg ? [{ svg, x: rect.x, y: rect.y, width: rect.width, height: rect.height, title: rect.cell.panel.headerLabelEl.textContent ?? '' }] : [];
    });
    if (cells.length === 0) return;
    const width = Math.max(...cells.map((c) => c.x + c.width));
    const height = Math.max(...cells.map((c) => c.y + c.height));
    const { svg } = buildGridSvg(cells, { title: this.info?.label ?? 'mass spectrometry', widthPx: width, heightPx: height });
    downloadSvg(`${this.info?.label ?? 'spectra'}.svg`, svg);
  }

  dispose(): void {
    this.disposed = true;
    for (const off of this.detachers) off();
    this.detachers = [];
    this.resizeObserver?.disconnect();
    this.grid?.dispose();
    this.root.remove();
  }
}
