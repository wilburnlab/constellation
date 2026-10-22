// PanelGrid — panels in rows and columns, any of which can be focused
// (maximized) or collapsed.
//
// A sibling of `PanelStack` for a browser whose panels are fixed by the
// data's shape rather than chosen by the user: rows and columns are
// declared by the host, each cell is one `Panel`, and the grid owns the
// rest: arranging the cells, focusing a cell, a row or a column, collapsing
// a row or a column, dragging the boundaries between them, the gear
// popover, persisting all of it, and getting each visible cell fetched and
// drawn through a `RenderScheduler`.
//
// Sizes are arithmetic, not CSS: the host tells the grid its size, the
// grid gives every track (row or column) a pixel size from the weights
// and the focus, and sets them on the CSS grid. So each cell's drawing
// size is known exactly, the same numbers serve an SVG export, and the
// layout can be tested without a browser.
//
// The host drives it through `GridDriver`. The view is opaque here: the
// driver says, for one cell and the current view, which data the cell
// should show (`dataKey`), how to fetch it and how to draw it. A key that
// has not changed is drawn again from the cache, so a host can call
// `render()` on every pan and pay only for the cells whose data moved.

import { ensureSvg } from '../engine/svg_layer';
import { Panel, PanelEntry } from './Panel';
import { PanelKind, encodePushdown, pushdownChanged, unitFor } from './kind';
import { RenderScheduler } from './scheduler';
import { SettingsPanel } from './SettingsPanel';
import { Bag, SettingsSchema } from './settings_schema';
import './grid.css';

export interface GridAxisSpec {
  id: string;
  label: string;
  /** Share of the space among its siblings (default 1). */
  weight?: number;
}

export type GridFocus =
  | { type: 'none' }
  | { type: 'row'; row: string }
  | { type: 'col'; col: string }
  | { type: 'cell'; row: string; col: string };

/** What the grid persists. */
export interface GridState {
  rowWeights: Record<string, number>;
  colWeights: Record<string, number>;
  collapsedRows: string[];
  collapsedCols: string[];
  focus: GridFocus;
  /** Per-cell style and filter, by cell id (`row/col`). */
  cells: Record<string, { style?: Record<string, unknown>; filter?: Record<string, unknown> }>;
}

export interface GridCell<D> {
  row: string;
  col: string;
  /** `row/col`. */
  id: string;
  panel: Panel<D>;
}

export interface GridDriver<V, D> {
  /** The current view, or null when there is nothing to show yet. */
  view(): V | null;
  /** Which data the cell shows for `view`; null for none (a placeholder). */
  dataKey(cell: GridCell<D>, view: V): string | null;
  fetch(cell: GridCell<D>, view: V, signal: AbortSignal): Promise<D>;
  /** Draw into the cell's SVG; returns the number of items, for the status line. */
  draw(
    cell: GridCell<D>,
    data: D,
    view: V,
    svg: SVGSVGElement,
    size: { widthPx: number; heightPx: number },
  ): number;
  /** Text shown in a cell whose key is null. */
  placeholder?(cell: GridCell<D>, view: V): string;
}

export interface PanelGridConfig<V, D> {
  host: HTMLElement;
  rows: GridAxisSpec[];
  cols: GridAxisSpec[];
  /** The kind and label of the panel in one cell. */
  cellEntry(row: string, col: string): PanelEntry;
  driver: GridDriver<V, D>;
  kindOf(kind: string): PanelKind | null;
  fallbackSettings: SettingsSchema;
  settingsHost?(): Bag;
  /** localStorage key the grid state is kept under; not persisted when omitted. */
  storageKey?: string;
  /** Requests running at once (default 4). */
  concurrency?: number;
  onFocusChange?(focus: GridFocus): void;
}

/** Width of the row-header strip and height of the column-header strip. */
export const ROW_HEADER_PX = 30;
export const COL_HEADER_PX = 24;
/** A cell's own header (label, status, buttons). */
export const CELL_HEADER_PX = 24;
/** What a collapsed column or row keeps. */
export const COLLAPSED_COL_PX = 28;
export const COLLAPSED_ROW_PX = CELL_HEADER_PX;
const MIN_TRACK_PX = 60;
const PERSIST_DEBOUNCE_MS = 200;

export interface CellRect {
  cell: GridCell<unknown>;
  x: number;
  y: number;
  width: number;
  height: number;
}

function emptyState(): GridState {
  return { rowWeights: {}, colWeights: {}, collapsedRows: [], collapsedCols: [], focus: { type: 'none' }, cells: {} };
}

function readState(key: string | undefined): GridState {
  if (!key) return emptyState();
  try {
    const raw = window.localStorage.getItem(key);
    if (!raw) return emptyState();
    const parsed = JSON.parse(raw) as Partial<GridState>;
    return { ...emptyState(), ...parsed, cells: { ...(parsed.cells ?? {}) } };
  } catch {
    return emptyState();
  }
}

/** Pixel sizes of a list of tracks sharing `total` pixels. A collapsed
 *  track gets `collapsedPx`, a track the focus hides gets 0, and the rest
 *  share what is left by weight (never below a usable minimum). */
export function trackSizes(
  ids: readonly string[],
  total: number,
  weights: Record<string, number>,
  collapsed: ReadonlySet<string>,
  shown: ReadonlySet<string>,
  collapsedPx: number,
): Record<string, number> {
  const sizes: Record<string, number> = {};
  let fixed = 0;
  let weightSum = 0;
  for (const id of ids) {
    if (!shown.has(id)) sizes[id] = 0;
    else if (collapsed.has(id)) {
      sizes[id] = collapsedPx;
      fixed += collapsedPx;
    } else weightSum += weights[id] ?? 1;
  }
  const free = Math.max(0, total - fixed);
  for (const id of ids) {
    if (id in sizes) continue;
    sizes[id] = weightSum > 0 ? Math.max(MIN_TRACK_PX, Math.floor((free * (weights[id] ?? 1)) / weightSum)) : 0;
  }
  return sizes;
}

export class PanelGrid<V, D> {
  private readonly config: PanelGridConfig<V, D>;
  private rows: GridAxisSpec[];
  private cols: GridAxisSpec[];
  private state: GridState;
  private readonly root: HTMLElement;
  private readonly scheduler: RenderScheduler<GridCell<D>, D>;
  private cellMap = new Map<string, GridCell<D>>();
  private rowHeaders = new Map<string, { el: HTMLElement; slot: HTMLElement }>();
  private colHeaders = new Map<string, { el: HTMLElement; slot: HTMLElement }>();
  private size = { widthPx: 0, heightPx: 0 };
  private rowPx: Record<string, number> = {};
  private colPx: Record<string, number> = {};
  private persistTimer: number | null = null;
  private settingsPopover: SettingsPanel | null = null;
  private settingsCell: GridCell<D> | null = null;
  private endDrag: (() => void) | null = null;

  constructor(config: PanelGridConfig<V, D>) {
    this.config = config;
    this.rows = [...config.rows];
    this.cols = [...config.cols];
    this.state = readState(config.storageKey);
    this.scheduler = new RenderScheduler({ concurrency: config.concurrency });
    this.root = document.createElement('div');
    this.root.className = 'panel-grid';
    config.host.appendChild(this.root);
    this.build();
  }

  // ── structure ───────────────────────────────────────────────────────

  get cells(): GridCell<D>[] {
    return [...this.cellMap.values()];
  }

  cell(row: string, col: string): GridCell<D> | undefined {
    return this.cellMap.get(`${row}/${col}`);
  }

  /** Element in a row's header for the host's own controls. */
  rowSlot(row: string): HTMLElement | undefined {
    return this.rowHeaders.get(row)?.slot;
  }

  colSlot(col: string): HTMLElement | undefined {
    return this.colHeaders.get(col)?.slot;
  }

  /** Replace the rows (an acquisition with other MS levels). Cells are
   *  rebuilt; their style survives through the persisted state. */
  setRows(rows: GridAxisSpec[]): void {
    this.rows = [...rows];
    this.build();
  }

  private build(): void {
    this.closeSettings();
    this.scheduler.cancelAll();
    this.cellMap.clear();
    this.rowHeaders.clear();
    this.colHeaders.clear();
    this.root.replaceChildren();

    // Every item has an explicit place: hidden items (a focused row or
    // column hides the others) leave no hole for auto-placement to fill
    // with the wrong cell.
    const place = (el: HTMLElement, row: number, col: number): HTMLElement => {
      el.style.gridRow = String(row);
      el.style.gridColumn = String(col);
      return el;
    };
    const corner = document.createElement('div');
    corner.className = 'panel-grid-corner';
    this.root.appendChild(place(corner, 1, 1));
    this.cols.forEach((col, i) => this.root.appendChild(place(this.buildColHeader(col, i), 1, i + 2)));
    this.rows.forEach((row, i) => {
      this.root.appendChild(place(this.buildRowHeader(row, i), i + 2, 1));
      for (const [j, col] of this.cols.entries()) {
        const entry = this.config.cellEntry(row.id, col.id);
        const id = `${row.id}/${col.id}`;
        const saved = this.state.cells[id] ?? {};
        const panel = new Panel<D>({
          entry,
          chrome: { reorder: false, resize: false, hide: false, collapse: false },
          meta: {},
          visible: true,
          collapsed: false,
          heightPx: 0,
          displayOrder: 0,
          style: saved.style ?? {},
          filter: saved.filter ?? {},
        });
        panel.element.classList.add('panel-grid-cell');
        place(panel.element, i + 2, j + 2);
        panel.element.dataset.row = row.id;
        panel.element.dataset.col = col.id;
        const cell: GridCell<D> = { row: row.id, col: col.id, id, panel };
        const maximize = document.createElement('button');
        maximize.type = 'button';
        maximize.className = 'panel-grid-maximize';
        maximize.textContent = '⤢';
        maximize.title = 'Maximize this panel (Esc restores)';
        maximize.addEventListener('click', () => this.toggleFocus({ type: 'cell', row: row.id, col: col.id }));
        panel.addHeaderButton(maximize);
        panel.headerLabelEl.addEventListener('dblclick', () =>
          this.toggleFocus({ type: 'cell', row: row.id, col: col.id }),
        );
        panel.settingsBtn.addEventListener('click', () => this.toggleSettings(cell));
        this.cellMap.set(id, cell);
        this.root.appendChild(panel.element);
      }
    });
    this.applySizes();
  }

  private buildColHeader(col: GridAxisSpec, index: number): HTMLElement {
    const el = document.createElement('div');
    el.className = 'panel-grid-col-header';
    el.dataset.col = col.id;
    const collapse = this.headerButton('▾', 'Collapse this column', () => this.setColCollapsed(col.id, !this.isColCollapsed(col.id)));
    const label = document.createElement('span');
    label.className = 'panel-grid-header-label';
    label.textContent = col.label;
    const slot = document.createElement('span');
    slot.className = 'panel-grid-header-slot';
    const maximize = this.headerButton('⤢', 'Maximize this column', () => this.toggleFocus({ type: 'col', col: col.id }));
    el.append(collapse, label, slot, maximize);
    if (index < this.cols.length - 1) el.appendChild(this.splitter('col', col.id, index));
    this.colHeaders.set(col.id, { el, slot });
    return el;
  }

  private buildRowHeader(row: GridAxisSpec, index: number): HTMLElement {
    const el = document.createElement('div');
    el.className = 'panel-grid-row-header';
    el.dataset.row = row.id;
    const collapse = this.headerButton('▾', 'Collapse this row', () => this.setRowCollapsed(row.id, !this.isRowCollapsed(row.id)));
    const label = document.createElement('span');
    label.className = 'panel-grid-header-label';
    label.textContent = row.label;
    const maximize = this.headerButton('⤢', 'Maximize this row', () => this.toggleFocus({ type: 'row', row: row.id }));
    const slot = document.createElement('span');
    slot.className = 'panel-grid-header-slot';
    el.append(collapse, label, maximize, slot);
    if (index < this.rows.length - 1) el.appendChild(this.splitter('row', row.id, index));
    this.rowHeaders.set(row.id, { el, slot });
    return el;
  }

  private headerButton(text: string, title: string, onClick: () => void): HTMLButtonElement {
    const button = document.createElement('button');
    button.type = 'button';
    button.className = 'panel-grid-header-btn';
    button.textContent = text;
    button.title = title;
    button.addEventListener('click', onClick);
    return button;
  }

  // ── focus and collapse ──────────────────────────────────────────────

  get focus(): GridFocus {
    return this.state.focus;
  }

  setFocus(focus: GridFocus): void {
    this.state.focus = focus;
    this.applySizes();
    this.config.onFocusChange?.(focus);
    this.schedulePersist();
    void this.render();
  }

  /** Focus `focus`, or return to the full grid when it is already focused. */
  toggleFocus(focus: GridFocus): void {
    this.setFocus(sameFocus(this.state.focus, focus) ? { type: 'none' } : focus);
  }

  isRowCollapsed(row: string): boolean {
    return this.state.collapsedRows.includes(row);
  }

  isColCollapsed(col: string): boolean {
    return this.state.collapsedCols.includes(col);
  }

  setRowCollapsed(row: string, collapsed: boolean): void {
    this.state.collapsedRows = toggleIn(this.state.collapsedRows, row, collapsed);
    this.applySizes();
    this.schedulePersist();
    void this.render();
  }

  setColCollapsed(col: string, collapsed: boolean): void {
    this.state.collapsedCols = toggleIn(this.state.collapsedCols, col, collapsed);
    this.applySizes();
    this.schedulePersist();
    void this.render();
  }

  private shownRows(): Set<string> {
    const f = this.state.focus;
    const all = this.rows.map((r) => r.id);
    if ((f.type === 'row' || f.type === 'cell') && all.includes(f.row)) return new Set([f.row]);
    return new Set(all);
  }

  private shownCols(): Set<string> {
    const f = this.state.focus;
    const all = this.cols.map((c) => c.id);
    if ((f.type === 'col' || f.type === 'cell') && all.includes(f.col)) return new Set([f.col]);
    return new Set(all);
  }

  /** Whether a cell is drawn (shown by the focus, its row and column open). */
  isCellOpen(cell: GridCell<D>): boolean {
    return cell.panel.visible && !cell.panel.collapsed;
  }

  // ── sizes ───────────────────────────────────────────────────────────

  /** The host's size changed. */
  resize(size: { widthPx: number; heightPx: number }): void {
    this.size = { widthPx: Math.max(0, Math.floor(size.widthPx)), heightPx: Math.max(0, Math.floor(size.heightPx)) };
    this.applySizes();
  }

  private applySizes(): void {
    const collapsedRows = new Set(this.state.collapsedRows);
    const collapsedCols = new Set(this.state.collapsedCols);
    // A focused row or column is shown open even if it was collapsed.
    const f = this.state.focus;
    if (f.type === 'row' || f.type === 'cell') collapsedRows.delete(f.row);
    if (f.type === 'col' || f.type === 'cell') collapsedCols.delete(f.col);
    this.colPx = trackSizes(
      this.cols.map((c) => c.id),
      this.size.widthPx - ROW_HEADER_PX,
      this.state.colWeights,
      collapsedCols,
      this.shownCols(),
      COLLAPSED_COL_PX,
    );
    this.rowPx = trackSizes(
      this.rows.map((r) => r.id),
      this.size.heightPx - COL_HEADER_PX,
      this.state.rowWeights,
      collapsedRows,
      this.shownRows(),
      COLLAPSED_ROW_PX,
    );

    this.root.style.gridTemplateColumns = [`${ROW_HEADER_PX}px`, ...this.cols.map((c) => `${this.colPx[c.id]}px`)].join(' ');
    this.root.style.gridTemplateRows = [`${COL_HEADER_PX}px`, ...this.rows.map((r) => `${this.rowPx[r.id]}px`)].join(' ');
    for (const [id, { el }] of this.colHeaders) {
      el.hidden = this.colPx[id] === 0;
      el.classList.toggle('collapsed', collapsedCols.has(id));
    }
    for (const [id, { el }] of this.rowHeaders) {
      el.hidden = this.rowPx[id] === 0;
      el.classList.toggle('collapsed', collapsedRows.has(id));
    }
    for (const cell of this.cellMap.values()) {
      const hidden = this.rowPx[cell.row] === 0 || this.colPx[cell.col] === 0;
      const rowShut = collapsedRows.has(cell.row);
      const colShut = collapsedCols.has(cell.col);
      cell.panel.element.hidden = hidden;
      cell.panel.visible = !hidden;
      cell.panel.collapsed = rowShut || colShut;
      cell.panel.element.classList.toggle('collapsed-row', rowShut);
      cell.panel.element.classList.toggle('collapsed-col', colShut);
    }
    this.root.dataset.focus = f.type;
  }

  /** Drawing size of a cell's SVG. */
  cellSize(cell: GridCell<D>): { widthPx: number; heightPx: number } {
    return {
      widthPx: Math.max(1, (this.colPx[cell.col] ?? 0) - 2),
      heightPx: Math.max(1, (this.rowPx[cell.row] ?? 0) - CELL_HEADER_PX - 2),
    };
  }

  /** Where each open cell's drawing sits, relative to the grid's top-left
   *  data corner (headers excluded): for an export of the grid. */
  layoutRects(): CellRect[] {
    const rects: CellRect[] = [];
    let y = 0;
    for (const row of this.rows) {
      let x = 0;
      for (const col of this.cols) {
        const cell = this.cell(row.id, col.id);
        if (cell && this.isCellOpen(cell)) {
          rects.push({ cell: cell as GridCell<unknown>, x, y, ...sizeOf(this.cellSize(cell)) });
        }
        x += this.colPx[col.id] ?? 0;
      }
      y += this.rowPx[row.id] ?? 0;
    }
    return rects;
  }

  /** Track sizes, for tests. */
  get trackPx(): { rows: Record<string, number>; cols: Record<string, number> } {
    return { rows: { ...this.rowPx }, cols: { ...this.colPx } };
  }

  // ── splitters ───────────────────────────────────────────────────────

  private splitter(axis: 'row' | 'col', id: string, index: number): HTMLElement {
    const handle = document.createElement('div');
    handle.className = `panel-grid-splitter panel-grid-splitter-${axis}`;
    handle.title = 'Drag to resize';
    handle.addEventListener('pointerdown', (e) => {
      if (e.button !== 0) return;
      e.preventDefault();
      const list = axis === 'col' ? this.cols : this.rows;
      const next = list[index + 1];
      if (!next) return;
      const px = axis === 'col' ? this.colPx : this.rowPx;
      const a = px[id] ?? 0;
      const b = px[next.id] ?? 0;
      const start = axis === 'col' ? e.clientX : e.clientY;
      const weights = axis === 'col' ? this.state.colWeights : this.state.rowWeights;
      const wa = weights[id] ?? 1;
      const wb = weights[next.id] ?? 1;
      const onMove = (ev: PointerEvent): void => {
        const delta = (axis === 'col' ? ev.clientX : ev.clientY) - start;
        const na = Math.max(MIN_TRACK_PX, a + delta);
        const nb = Math.max(MIN_TRACK_PX, a + b - na);
        if (a + b <= 0) return;
        weights[id] = ((wa + wb) * na) / (na + nb);
        weights[next.id] = ((wa + wb) * nb) / (na + nb);
        this.applySizes();
      };
      const stop = (): void => {
        handle.removeEventListener('pointermove', onMove);
        handle.removeEventListener('pointerup', stop);
        handle.removeEventListener('pointercancel', stop);
        this.endDrag = null;
        this.schedulePersist();
        void this.render();
      };
      handle.setPointerCapture?.(e.pointerId);
      handle.addEventListener('pointermove', onMove);
      handle.addEventListener('pointerup', stop);
      handle.addEventListener('pointercancel', stop);
      this.endDrag = stop;
    });
    return handle;
  }

  // ── drawing ─────────────────────────────────────────────────────────

  /** Fetch what changed and draw every open cell for the current view. */
  async render(): Promise<void> {
    const view = this.config.driver.view();
    if (view === null) return;
    const plans = [];
    for (const cell of this.cellMap.values()) {
      if (!this.isCellOpen(cell)) {
        if (cell.panel.collapsed) cell.panel.statusEl.textContent = 'collapsed';
        continue;
      }
      // The kind's server-side filter values are part of what is fetched,
      // whether or not the host's key names them.
      const hostKey = this.config.driver.dataKey(cell, view);
      const pushed = encodePushdown(this.config.kindOf(cell.panel.entry.kind), cell.panel.filter);
      const key = hostKey !== null && Object.keys(pushed).length ? `${hostKey}|${JSON.stringify(pushed)}` : hostKey;
      plans.push({
        target: cell,
        key,
        fetch: (signal: AbortSignal) => this.config.driver.fetch(cell, view, signal),
      });
    }
    await this.scheduler.pass(plans, {
      cached: (cell) => {
        const last = cell.panel.lastFetched;
        return last ? { key: last.viewKey, data: last.data } : undefined;
      },
      store: (cell, key, data) => {
        cell.panel.lastFetched = { data, viewKey: key };
      },
      draw: (cell, data) => this.drawCell(cell, data),
      loading: (cell) => {
        cell.panel.statusEl.textContent = 'loading…';
      },
      failed: (cell, error) => {
        console.warn(`render failed for ${cell.id}`, error);
        cell.panel.statusEl.textContent = 'failed';
        cell.panel.statusEl.title = error.message;
      },
      empty: (cell) => this.showPlaceholder(cell),
    });
  }

  /** Draw one cell from what it last fetched (after a style change). */
  redraw(cell: GridCell<D>): void {
    const last = cell.panel.lastFetched;
    if (last && this.isCellOpen(cell)) this.drawCell(cell, last.data);
  }

  private drawCell(cell: GridCell<D>, data: D): void {
    const view = this.config.driver.view();
    if (view === null) return;
    cell.panel.bodyHost.querySelector('.panel-grid-placeholder')?.remove();
    const size = this.cellSize(cell);
    const svg = ensureSvg(cell.panel.bodyHost, size.widthPx, size.heightPx);
    while (svg.firstChild) svg.removeChild(svg.firstChild);
    try {
      const count = this.config.driver.draw(cell, data, view, svg, size);
      const kind = this.config.kindOf(cell.panel.entry.kind);
      cell.panel.statusEl.textContent = count === 0 ? 'no data' : `${count.toLocaleString()} ${unitFor(kind, count)}`;
      cell.panel.statusEl.title = '';
    } catch (err) {
      console.warn(`draw failed for ${cell.id}`, err);
      cell.panel.statusEl.textContent = 'draw failed';
    }
  }

  private showPlaceholder(cell: GridCell<D>): void {
    const view = this.config.driver.view();
    const host = cell.panel.bodyHost;
    host.replaceChildren();
    const note = document.createElement('div');
    note.className = 'panel-grid-placeholder';
    note.textContent = (view !== null && this.config.driver.placeholder?.(cell, view)) || '';
    host.appendChild(note);
    cell.panel.lastFetched = undefined;
    cell.panel.statusEl.textContent = '—';
  }

  /** Forget a cell's data so the next render fetches it. */
  invalidate(cell?: GridCell<D>): void {
    for (const c of cell ? [cell] : this.cellMap.values()) c.panel.lastFetched = undefined;
  }

  // ── settings ────────────────────────────────────────────────────────

  private toggleSettings(cell: GridCell<D>): void {
    const same = this.settingsCell === cell;
    this.closeSettings();
    if (!same) this.openSettings(cell);
  }

  private openSettings(cell: GridCell<D>): void {
    const kind = this.config.kindOf(cell.panel.entry.kind);
    const panel = cell.panel;
    const remember = (): void => {
      const entry: GridState['cells'][string] = {};
      if (Object.keys(panel.style).length) entry.style = { ...panel.style };
      if (Object.keys(panel.filter).length) entry.filter = { ...panel.filter };
      this.state.cells[cell.id] = entry;
      this.schedulePersist();
    };
    const filterChanged = (before: Record<string, unknown>): void => {
      if (pushdownChanged(kind, before, panel.filter)) {
        panel.lastFetched = undefined;
        void this.render();
      } else this.redraw(cell);
    };
    this.settingsCell = cell;
    this.settingsPopover = new SettingsPanel({
      anchor: panel.settingsBtn,
      kind: panel.entry.kind,
      label: panel.entry.label,
      schema: kind?.settings ?? this.config.fallbackSettings,
      meta: panel.meta,
      style: { ...panel.style },
      filter: { ...panel.filter },
      host: this.config.settingsHost?.(),
      onStyleChange: (style) => {
        panel.style = { ...style };
        this.redraw(cell);
        remember();
      },
      onFilterChange: (filter) => {
        const before = panel.filter;
        panel.filter = { ...filter };
        filterChanged(before);
        remember();
      },
      onReset: () => {
        const before = panel.filter;
        panel.style = {};
        panel.filter = {};
        filterChanged(before);
        remember();
      },
      onClose: () => this.closeSettings(),
    });
    this.settingsPopover.mount(document.body);
  }

  closeSettings(): void {
    this.settingsPopover?.dispose();
    this.settingsPopover = null;
    this.settingsCell = null;
  }

  // ── persistence and teardown ────────────────────────────────────────

  /** The persisted state (a copy). */
  snapshot(): GridState {
    return JSON.parse(JSON.stringify(this.state)) as GridState;
  }

  private schedulePersist(): void {
    const key = this.config.storageKey;
    if (!key) return;
    if (this.persistTimer !== null) window.clearTimeout(this.persistTimer);
    this.persistTimer = window.setTimeout(() => {
      this.persistTimer = null;
      try {
        window.localStorage.setItem(key, JSON.stringify(this.state));
      } catch {
        /* storage full or blocked: the layout is a convenience */
      }
    }, PERSIST_DEBOUNCE_MS);
  }

  dispose(): void {
    if (this.persistTimer !== null) window.clearTimeout(this.persistTimer);
    this.endDrag?.();
    this.closeSettings();
    this.scheduler.cancelAll();
    this.root.remove();
  }
}

function sameFocus(a: GridFocus, b: GridFocus): boolean {
  return JSON.stringify(a) === JSON.stringify(b);
}

function toggleIn(list: string[], id: string, on: boolean): string[] {
  const rest = list.filter((x) => x !== id);
  return on ? [...rest, id] : rest;
}

function sizeOf(size: { widthPx: number; heightPx: number }): { width: number; height: number } {
  return { width: size.widthPx, height: size.heightPx };
}
