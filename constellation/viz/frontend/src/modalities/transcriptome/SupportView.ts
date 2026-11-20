// The support view: one node's reads against its consensus.
//
//   ┌ toolbar: node · space · compare · order · Save SVG ───────────────┐
//   │ labels │ ruler · consensus · coverage · minor fraction · candidates │
//   │        │ covariance arcs                                           │
//   ├────────┼───────────────────────────────────────────────────────────┤
//   │ per-   │ the read matrix (canvas; engine/matrix at the level of    │
//   │ read   │ detail each axis's zoom calls for)                        │
//   │ columns│                                                           │
//   └ status: what was replayed, its verdict, the level of detail ─────┘
//
// The wheel zooms columns, shift + wheel zooms reads, a drag pans both
// (engine/pan_zoom attachPanZoom2D). Clicking an arc sorts the reads by
// their joint state at its two columns. Compare restricts the columns to
// those where the template's nodes differ, on their shared frame.
//
// Everything comes from one replay on the server (a second for most
// clusters, a minute for 20,000 reads), fetched as a summary and six
// tables; a newer request makes an older one's reply be dropped.

import { tableFromIPC } from 'apache-arrow';
import { createCanvasLayer, type CanvasLayer } from '../../engine/canvas_layer';
import { buildGridSvg, downloadSvg, layerSvg, type ExportCell } from '../../engine/export';
import { zoomAbout, type IntervalLimits } from '../../engine/interval';
import { attachPanZoom2D, type View2D } from '../../engine/pan_zoom';
import {
  DEFAULT_MATRIX_STYLE,
  drawMatrix,
  matrixGlyphCost,
  matrixLod,
  type MatrixViewport,
} from '../../engine/matrix/draw';
import { indexMatrix, type MatrixIndex } from '../../engine/matrix/model';
import { canvasSink, svgSink } from '../../engine/matrix/sinks';
import { clear, svgEl } from '../../engine/svg_layer';
import { ARC_NEGATIVE, ARC_POSITIVE, arcGeometry, drawArcs } from './arcs';
import {
  SupportData,
  defaultOrder,
  departureExcess,
  differingColumns,
  orderByPair,
  restrictColumns,
  supportFromTables,
} from './support_data';
import {
  NodeChoice,
  SUPPORT_TABLES,
  Space,
  SupportMeta,
  SupportTarget,
  routeName,
  templateLabel,
} from './types';

export const SIDE_W = 100;
export const TRACKS_H = 176;
const FALLBACK = { w: 800, h: 360 };

const ROW = {
  ruler: [0, 16],
  consensus: [18, 32],
  coverage: [34, 62],
  minor: [64, 92],
  candidates: [94, 100],
  /** Arcs rise from arcsBase to at most arcsTop, below the candidates. */
  arcsBase: TRACKS_H - 4,
  arcsTop: 104,
} as const;

const SAMPLE_COLOURS = ['#a277ff', '#61ffca', '#ffca85', '#f694ff', '#82e2ff', '#ff6767', '#c3e88d', '#f7c67f'];
const ROUTE_COLOURS: Record<number, string> = { 1: '#ffca85', 2: '#82e2ff', 3: '#f694ff' };
const SIDE_COLS = [
  { key: 'sample', label: 'smp', title: 'sample' },
  { key: 'identity', label: 'id', title: 'identity to the template (E-step)' },
  { key: 'quality', label: 'Q', title: 'Dorado mean quality' },
  { key: 'margin', label: 'mrg', title: "the M-step's pattern-assignment margin (nats)" },
  { key: 'excess', label: '×err', title: 'departures over those expected at 1.2% error' },
] as const;
const SIDE_COL_W = 16;
/** Rows at least this tall when a view opens, so each read is a row. */
const OPEN_PX_PER_ROW = 2;
/** Rows this tall are read letter by letter. */
const READABLE_PX_PER_ROW = 12;

export interface SupportViewOptions {
  host: HTMLElement;
  /** `/api/transcriptome/sessions/{sid}/runs/{key}` */
  base: string;
  /** Sample id → name. */
  samples: Record<string, string>;
}

function mix(a: string, b: string, t: number): string {
  const p = (h: string) => [1, 3, 5].map((i) => parseInt(h.slice(i, i + 2), 16));
  const [x, y] = [p(a), p(b)];
  const k = Math.max(0, Math.min(1, t));
  return `rgb(${x.map((v, i) => Math.round(v + (y[i] - v) * k)).join(',')})`;
}

function niceStep(span: number, px: number, minPx = 70): number {
  const raw = (span * minPx) / Math.max(1, px);
  const pow = 10 ** Math.floor(Math.log10(Math.max(raw, 1e-9)));
  for (const m of [1, 2, 5, 10]) if (m * pow >= raw) return Math.max(1, m * pow);
  return Math.max(1, 10 * pow);
}

export class SupportView {
  readonly root: HTMLDivElement;
  private readonly opts: SupportViewOptions;
  private readonly title: HTMLSpanElement;
  private readonly nodeSel: HTMLSelectElement;
  private readonly spaceSel: HTMLSelectElement;
  private readonly compareBtn: HTMLButtonElement;
  private readonly resetBtn: HTMLButtonElement;
  private readonly saveBtn: HTMLButtonElement;
  private readonly body: HTMLDivElement;
  private readonly corner: SVGSVGElement;
  private readonly tracks: SVGSVGElement;
  private readonly side: SVGSVGElement;
  private readonly matrixHost: HTMLDivElement;
  private readonly tooltip: HTMLDivElement;
  private readonly loading: HTMLDivElement;
  private readonly foot: HTMLDivElement;
  private readonly summary: HTMLDivElement;
  private readonly layer: CanvasLayer;
  private readonly detachPanZoom: () => void;
  private readonly resizeObs: ResizeObserver | null = null;

  private target: SupportTarget | null = null;
  private node: NodeChoice = 'cluster';
  private space: Space = 'consensus';
  private compare = false;
  private data: SupportData | null = null;
  private shownIndex: MatrixIndex | null = null;
  /** Display column → data column, when the columns are restricted. */
  private shownCols: Int32Array | null = null;
  private order: Int32Array = new Int32Array(0);
  private orderNote: string | null = null;
  private pair: [number, number] | null = null;
  private excess: Float64Array = new Float64Array(0);
  private view: View2D = { x: [0, 1], y: [0, 1] };
  private size = { ...FALLBACK };
  private generation = 0;
  private ticker: ReturnType<typeof setInterval> | null = null;
  private pendingRender: ReturnType<typeof setTimeout> | null = null;
  private pointer = { x: 0, y: 0 };

  constructor(opts: SupportViewOptions) {
    this.opts = opts;
    this.root = document.createElement('div');
    this.root.className = 'tx-support';

    const bar = document.createElement('div');
    bar.className = 'tx-toolbar';
    this.title = document.createElement('span');
    this.title.className = 'tx-title';
    this.nodeSel = document.createElement('select');
    this.nodeSel.className = 'tx-select';
    this.nodeSel.title = 'whose consensus the reads are read against';
    this.spaceSel = document.createElement('select');
    this.spaceSel.className = 'tx-select';
    for (const [v, l] of [['consensus', 'consensus'], ['frame', 'frame (every insertion column)']]) {
      const o = document.createElement('option');
      o.value = v;
      o.textContent = l;
      this.spaceSel.appendChild(o);
    }
    this.compareBtn = this.button('Compare nodes', 'only the columns where the template’s nodes differ');
    this.resetBtn = this.button('Default order', 'node, then pattern, then start, then identity');
    this.saveBtn = this.button('Save SVG', 'vector where under 50,000 glyphs, else an image of the matrix');
    bar.append(this.title, this.nodeSel, this.spaceSel, this.compareBtn, this.resetBtn, this.saveBtn);

    // What is shown, in a sentence; then how to move around it.
    this.summary = document.createElement('div');
    this.summary.className = 'tx-summary';
    const viewbar = document.createElement('div');
    viewbar.className = 'tx-viewbar';
    const group = (label: string, buttons: Array<[string, string, () => void]>) => {
      const g = document.createElement('span');
      g.className = 'tx-field';
      g.append(label);
      for (const [text, title, act] of buttons) {
        const b = this.button(text, title);
        b.addEventListener('click', act);
        g.appendChild(b);
      }
      return g;
    };
    const hint = document.createElement('span');
    hint.className = 'tx-hint';
    hint.textContent = 'wheel: zoom along the transcript · shift + wheel: zoom reads · drag: pan · double-click: everything';
    viewbar.append(
      group('Reads', [
        ['−', 'show more reads (shorter rows)', () => this.zoom('y', 1.5)],
        ['+', 'show fewer reads (taller rows)', () => this.zoom('y', 1 / 1.5)],
        ['All', 'every read in the height of the view', () => this.fitRows(true)],
        ['Readable', `rows ${READABLE_PX_PER_ROW} px tall: letters once columns are wide enough`, () => this.readableRows()],
      ]),
      group('Columns', [
        ['−', 'zoom out along the transcript', () => this.zoom('x', 1.5)],
        ['+', 'zoom in along the transcript', () => this.zoom('x', 1 / 1.5)],
        ['Whole', 'the whole length', () => this.fitColumns()],
      ]),
      hint,
    );

    this.body = document.createElement('div');
    this.body.className = 'tx-body';
    this.body.style.gridTemplateColumns = `${SIDE_W}px 1fr`;
    this.body.style.gridTemplateRows = `${TRACKS_H}px 1fr`;
    this.corner = svgEl('svg', { class: 'tx-corner', width: SIDE_W, height: TRACKS_H }) as SVGSVGElement;
    this.tracks = svgEl('svg', { class: 'tx-tracks', height: TRACKS_H }) as SVGSVGElement;
    this.side = svgEl('svg', { class: 'tx-side', width: SIDE_W }) as SVGSVGElement;
    this.matrixHost = document.createElement('div');
    this.matrixHost.className = 'tx-matrix';
    this.tooltip = document.createElement('div');
    this.tooltip.className = 'tx-tooltip';
    this.tooltip.hidden = true;
    this.loading = document.createElement('div');
    this.loading.className = 'tx-loading';
    this.loading.textContent = 'Pick a cluster on the left, or open a template by round and id.';
    this.body.append(this.corner, this.tracks, this.side, this.matrixHost, this.tooltip, this.loading);

    this.foot = document.createElement('div');
    this.foot.className = 'tx-foot';
    this.root.append(bar, this.summary, viewbar, this.body, this.legend(), this.foot);
    opts.host.appendChild(this.root);

    this.layer = createCanvasLayer(this.matrixHost, (ctx, s) => this.paint(ctx, s.widthPx, s.heightPx));

    this.nodeSel.addEventListener('change', () => {
      const v = this.nodeSel.value;
      this.node = v === 'cluster' || v === 'pooled' ? v : Number(v);
      this.compare = false;
      void this.load();
    });
    this.spaceSel.addEventListener('change', () => {
      this.space = this.spaceSel.value as Space;
      this.compare = false;
      void this.load();
    });
    this.compareBtn.addEventListener('click', () => void this.toggleCompare());
    this.resetBtn.addEventListener('click', () => this.resetOrder());
    this.saveBtn.addEventListener('click', () => {
      const svg = this.exportSvg();
      if (svg) downloadSvg(`support-${this.describeTarget().replace(/\s+/g, '-')}.svg`, svg);
    });
    this.body.addEventListener('pointermove', (e) => {
      this.pointer = { x: e.clientX, y: e.clientY };
    });
    this.body.addEventListener('pointerleave', () => (this.tooltip.hidden = true));

    this.detachPanZoom = attachPanZoom2D({
      surface: this.body,
      plotRect: () => {
        const r = this.matrixHost.getBoundingClientRect();
        return { left: r.left, top: r.top, width: r.width || this.size.w, height: r.height || this.size.h };
      },
      get: () => this.view,
      set: (v) => {
        this.view = v;
        this.scheduleRender();
      },
      limitsX: () => this.limitsX(),
      limitsY: () => this.limitsY(),
      onHover: (at) => this.hover(at),
      ignore: (t) => t instanceof Element && t.closest('.tx-arc, button, select, input') !== null,
    });
    if (typeof ResizeObserver === 'function') {
      this.resizeObs = new ResizeObserver(() => this.scheduleRender());
      this.resizeObs.observe(this.matrixHost);
    }
  }

  private button(label: string, title: string): HTMLButtonElement {
    const b = document.createElement('button');
    b.className = 'tx-btn';
    b.textContent = label;
    b.title = title;
    return b;
  }

  // ── what is shown ─────────────────────────────────────────────────

  async show(target: SupportTarget, node?: NodeChoice, space?: Space): Promise<void> {
    this.target = target;
    this.node = node ?? (target.kind === 'cluster' ? 'cluster' : 'pooled');
    this.space = space ?? 'consensus';
    this.compare = false;
    await this.load();
  }

  url(table?: string): string {
    const t = this.target!;
    const path =
      t.kind === 'cluster'
        ? `/clusters/${t.clusterId}/support`
        : `/rounds/${t.round}/templates/${t.templateId}/support`;
    const q = new URLSearchParams({ node: String(this.node), space: this.space });
    return `${this.opts.base}${path}${table ? `/${table}` : ''}?${q.toString()}`;
  }

  describeTarget(): string {
    const t = this.target;
    if (!t) return '';
    return t.kind === 'cluster' ? `cluster ${t.clusterId}` : templateLabel(t.templateId);
  }

  private async load(): Promise<void> {
    if (!this.target) return;
    const gen = ++this.generation;
    this.startProgress();
    try {
      const response = await fetch(this.url());
      if (!response.ok) {
        let detail = response.statusText;
        try {
          detail = String((await response.json()).detail ?? detail);
        } catch {
          /* not JSON */
        }
        throw new Error(detail);
      }
      const meta = (await response.json()) as SupportMeta;
      if (gen !== this.generation) return;
      const tables = await Promise.all(
        SUPPORT_TABLES.map(async (name) => {
          const response = await fetch(this.url(name));
          if (!response.ok) throw new Error(`${name}: ${response.status}`);
          return [name, tableFromIPC(await response.arrayBuffer())] as const;
        }),
      );
      if (gen !== this.generation) return;
      this.data = supportFromTables(meta, Object.fromEntries(tables));
      this.excess = departureExcess(this.data);
      this.order = defaultOrder(this.data);
      this.orderNote = null;
      this.pair = null;
      this.rebuildShown();
      this.fit();
      this.updateToolbar();
      this.stopProgress();
      this.renderNow();
    } catch (err) {
      if (gen !== this.generation) return;
      this.stopProgress(`Could not show ${this.describeTarget()}: ${(err as Error).message}`);
    }
  }

  private startProgress(): void {
    const started = Date.now();
    const say = () => {
      const s = Math.round((Date.now() - started) / 1000);
      this.loading.textContent =
        `Replaying the M-step behind ${this.describeTarget()}… ${s} s` +
        (s >= 5 ? ' — a node of 20,000 reads takes about a minute' : '');
    };
    this.loading.hidden = false;
    say();
    if (this.ticker) clearInterval(this.ticker);
    this.ticker = setInterval(say, 1000);
  }

  private stopProgress(error?: string): void {
    if (this.ticker) clearInterval(this.ticker);
    this.ticker = null;
    this.loading.hidden = !error;
    if (error) this.loading.textContent = error;
  }

  private async toggleCompare(): Promise<void> {
    if (!this.data) return;
    if (this.compare) {
      this.compare = false;
    } else {
      if (this.node !== 'pooled' || this.space !== 'frame') {
        this.node = 'pooled';
        this.space = 'frame';
        await this.load();
      }
      if (!this.data || this.data.differences.n === 0) return;
      this.compare = true;
    }
    this.rebuildShown();
    this.fit();
    this.updateToolbar();
    this.renderNow();
  }

  private rebuildShown(): void {
    const data = this.data!;
    if (this.compare) {
      this.shownCols = differingColumns(data);
      this.shownIndex = indexMatrix(restrictColumns(data.matrix, this.shownCols));
    } else {
      this.shownCols = null;
      this.shownIndex = data.index;
    }
  }

  private nCols(): number {
    return this.shownIndex ? this.shownIndex.data.nCols : 1;
  }

  /** Data column of a display column. */
  private dataCol(j: number): number {
    return this.shownCols ? (this.shownCols[j] ?? -1) : j;
  }

  /** Display column of a data column, or -1. */
  private displayCol(c: number): number {
    if (!this.shownCols) return c;
    let lo = 0;
    let hi = this.shownCols.length;
    while (lo < hi) {
      const mid = (lo + hi) >>> 1;
      if (this.shownCols[mid] < c) lo = mid + 1;
      else hi = mid;
    }
    return this.shownCols[lo] === c ? lo : -1;
  }

  private limitsX(): IntervalLimits {
    return { min: 0, max: Math.max(1, this.nCols()), minSpan: Math.min(this.nCols(), 4) || 1 };
  }

  private limitsY(): IntervalLimits {
    return { min: 0, max: Math.max(1, this.order.length), minSpan: Math.min(this.order.length, 3) || 1 };
  }

  private heightPx(): number {
    return this.matrixHost.clientHeight || this.size.h || FALLBACK.h;
  }

  /** The whole length, and every read if each still gets
   *  {@link OPEN_PX_PER_ROW} pixels — else the first reads at that height. */
  private fit(): void {
    this.view = { x: [0, Math.max(1, this.nCols())], y: this.view.y };
    this.fitRows(this.order.length * OPEN_PX_PER_ROW <= this.heightPx(), false);
  }

  private fitRows(all: boolean, draw = true): void {
    const n = Math.max(1, this.order.length);
    const span = all ? n : Math.max(1, Math.min(n, this.heightPx() / OPEN_PX_PER_ROW));
    this.view = { x: this.view.x, y: [0, span] };
    if (draw) this.renderNow();
  }

  private fitColumns(): void {
    this.view = { x: [0, Math.max(1, this.nCols())], y: this.view.y };
    this.renderNow();
  }

  private readableRows(): void {
    const span = Math.max(1, this.heightPx() / READABLE_PX_PER_ROW);
    const y0 = Math.min(this.view.y[0], Math.max(0, this.order.length - span));
    this.view = { x: this.view.x, y: [y0, Math.min(this.order.length, y0 + span)] };
    this.renderNow();
  }

  /** Zoom one axis about the middle of the view (> 1 zooms out). */
  private zoom(axis: 'x' | 'y', factor: number): void {
    if (!this.data) return;
    const limits = axis === 'x' ? this.limitsX() : this.limitsY();
    this.view = { ...this.view, [axis]: zoomAbout(this.view[axis], factor, 0.5, limits) };
    this.renderNow();
  }

  private resetOrder(): void {
    if (!this.data) return;
    this.order = defaultOrder(this.data);
    this.orderNote = null;
    this.pair = null;
    this.renderNow();
  }

  /** Sort the reads by their joint state at two data columns. */
  sortByPair(u: number, v: number): void {
    if (!this.data) return;
    this.order = orderByPair(this.data, u, v);
    this.pair = [u, v];
    this.orderNote = `rows by their state at columns ${u + 1} and ${v + 1}`;
    this.renderNow();
  }

  private updateToolbar(): void {
    const data = this.data!;
    const meta = data.meta;
    this.title.textContent = this.describeTarget();
    this.nodeSel.innerHTML = '';
    const add = (value: string, label: string) => {
      const o = document.createElement('option');
      o.value = value;
      o.textContent = label;
      this.nodeSel.appendChild(o);
    };
    if (this.target?.kind === 'cluster') add('cluster', `this cluster (node ${meta.haplotype_id ?? '?'})`);
    if (meta.kind !== 'rebuilt') {
      add('pooled', 'pooled: every read of the template');
      for (const n of meta.nodes) add(String(n.haplotype_id), `node ${n.haplotype_id} · ${n.n_reads.toLocaleString('en-US')} reads`);
    }
    this.nodeSel.value = String(this.node);
    this.spaceSel.value = this.space;
    this.compareBtn.disabled = meta.kind === 'rebuilt' || (meta.nodes?.length ?? 0) < 2;
    this.compareBtn.classList.toggle('active', this.compare);
    this.compareBtn.textContent = this.compare ? 'All columns' : 'Compare nodes';
    this.summary.textContent = this.describeView();
  }

  /** What is on screen, in a sentence or two. */
  describeView(): string {
    const data = this.data;
    if (!data) return '';
    const m = data.meta;
    const n = (v: number) => v.toLocaleString('en-US');
    const reads = data.reads.n;
    const tmpl =
      m.template_id !== null && m.template_id !== undefined
        ? templateLabel(m.template_id)
        : this.target?.kind === 'template'
          ? templateLabel(this.target.templateId)
          : 'its template';
    const nodes = m.nodes ?? [];
    const split =
      nodes.length > 1
        ? `, which the M-step split into ${nodes.length} nodes (${nodes.map((x) => n(x.n_reads)).join(' · ')} reads)`
        : '';
    let candidates = 0;
    for (let i = 0; i < data.columns.n; i += 1) candidates += data.columns.candidate[i];
    let what: string;
    if (m.kind === 'rebuilt') {
      what =
        `Cluster ${m.cluster_id} was made by the final merge: its ${n(reads)} reads ` +
        `(${n(m.n_absorbed_reads)} of them from the nodes it absorbed) re-aligned to the survivor's consensus ` +
        `(${n(m.n_columns)} nt), one read per row`;
    } else if (this.compare) {
      what = `The ${n(reads)} reads of ${tmpl}${split}, at the ${n(this.nCols())} columns where the nodes' consensus sequences differ`;
    } else if (m.reference === 'pooled') {
      what =
        `All ${n(reads)} reads of ${tmpl}${split}, one per row, against the consensus of all of them ` +
        'pooled: a split shows as blocks of reads that differ together';
    } else if (m.kind === 'node' || m.kind === 'merged') {
      what =
        `Cluster ${m.cluster_id} is node ${m.haplotype_id} of ${tmpl}${split}. ` +
        `Its ${n(reads)} reads, one per row, against node ${m.reference}'s consensus (${n(m.n_columns)} nt)`;
    } else {
      what = `Node ${m.reference} of ${tmpl}${split}: its ${n(reads)} reads, one per row, against its consensus`;
    }
    const arcs = data.arcs.n;
    return (
      `${what}. Grey: the read agrees with that consensus; colour: it differs. ` +
      `${n(candidates)} candidate column${candidates === 1 ? '' : 's'} (ticks under the minor-allele track)` +
      (arcs ? `, ${n(arcs)} covariance arc${arcs === 1 ? '' : 's'} between them (click one to sort the reads by it).` : '.')
    );
  }

  private legend(): HTMLDivElement {
    const box = document.createElement('div');
    box.className = 'tx-legend';
    const swatch = (colour: string, label: string, title = '') => {
      const s = document.createElement('span');
      s.className = 'tx-swatch';
      s.title = title;
      const c = document.createElement('i');
      c.style.background = colour;
      s.append(c, label);
      return s;
    };
    const style = DEFAULT_MATRIX_STYLE;
    box.append(
      swatch(style.span, 'agrees'),
      ...['A', 'C', 'G', 'T'].map((b, i) => swatch(style.allele[i], b, `the read has ${b} where the consensus differs`)),
      swatch(style.allele[4], 'gap', 'the read lacks a base the consensus has'),
      swatch(style.allele[5], 'N', 'an ambiguous base: no evidence'),
      swatch(style.insertion, 'insertion', 'bases the read has between two columns'),
      swatch(ARC_POSITIVE, 'arc: differ together', 'φ > 0: reads that differ at one column differ at the other'),
      swatch(ARC_NEGATIVE, 'arc: exclusive', 'φ < 0: reads differ at one column or the other, not both'),
    );
    const side = document.createElement('span');
    side.className = 'tx-legend-side';
    side.textContent =
      'per read: sample · identity to the template · Dorado quality · how surely the M-step placed it · departures over what 1.2% error expects (brighter = higher)';
    box.appendChild(side);
    const samples = Object.entries(this.opts.samples);
    if (samples.length) {
      const sl = document.createElement('span');
      sl.className = 'tx-legend-samples';
      samples.forEach(([id, name]) => sl.appendChild(swatch(SAMPLE_COLOURS[Math.abs(Number(id)) % SAMPLE_COLOURS.length], name)));
      box.appendChild(sl);
    }
    return box;
  }

  // ── drawing ───────────────────────────────────────────────────────

  private scheduleRender(): void {
    if (this.pendingRender !== null) return;
    this.pendingRender = setTimeout(() => {
      this.pendingRender = null;
      this.renderNow();
    }, 0);
  }

  private viewport(): MatrixViewport {
    return { cols: this.view.x, rows: this.view.y, widthPx: this.size.w, heightPx: this.size.h };
  }

  /** Draw everything for the current view, now. */
  renderNow(): void {
    if (this.pendingRender !== null) {
      clearTimeout(this.pendingRender);
      this.pendingRender = null;
    }
    const w = this.matrixHost.clientWidth || FALLBACK.w;
    const h = this.matrixHost.clientHeight || FALLBACK.h;
    if (w !== this.size.w || h !== this.size.h || this.layer.size().widthPx !== w) {
      this.size = { w, h };
      this.layer.resize(w, h);
    } else {
      this.layer.drawNow();
    }
    this.drawOverlay();
    this.drawTracks();
    this.drawSide();
    this.drawFoot();
  }

  private paint(ctx: CanvasRenderingContext2D, w: number, h: number): void {
    if (!this.shownIndex) return;
    drawMatrix(canvasSink(ctx), this.shownIndex, this.order, {
      cols: this.view.x,
      rows: this.view.y,
      widthPx: w,
      heightPx: h,
    });
  }

  private pxPerCol(): number {
    return this.size.w / Math.max(1e-9, this.view.x[1] - this.view.x[0]);
  }

  private pxPerRow(): number {
    return this.size.h / Math.max(1e-9, this.view.y[1] - this.view.y[0]);
  }

  /** Node boundaries and the sorted pair, over the matrix. */
  private drawOverlay(): void {
    const svg = this.layer.overlay;
    clear(svg);
    if (!this.data) return;
    const ppr = this.pxPerRow();
    const node = this.data.reads.node;
    const several = new Set(Array.from(node)).size > 1;
    for (let i = 0; i < this.order.length; i += 1) {
      if (i > 0 && node[this.order[i]] === node[this.order[i - 1]]) continue;
      const y = (i - this.view.y[0]) * ppr;
      if (y < -1 || y > this.size.h) continue;
      if (i > 0) {
        svg.appendChild(svgEl('line', { x1: 0, x2: this.size.w, y1: y, y2: y, stroke: '#edecee', 'stroke-opacity': 0.6, 'stroke-dasharray': '4 3' }));
      }
      if (!several) break;
      const label = svgEl('text', { x: 4, y: Math.max(0, y) + 11, fill: '#edecee', 'font-size': 10 });
      label.textContent = `node ${node[this.order[i]]}`;
      svg.appendChild(label);
    }
    // Where the shown reads are among all of them.
    const n = this.order.length;
    const [y0, y1] = this.view.y;
    if (n > 0 && (y0 > 0.5 || y1 < n - 0.5)) {
      const x = this.size.w - 6;
      svg.appendChild(svgEl('rect', { x, y: 0, width: 5, height: this.size.h, fill: '#edecee', 'fill-opacity': 0.12, rx: 2 }));
      svg.appendChild(svgEl('rect', { x, y: (y0 / n) * this.size.h, width: 5, height: Math.max(3, ((y1 - y0) / n) * this.size.h), fill: '#a277ff', 'fill-opacity': 0.8, rx: 2 }));
    }
    if (this.pair) {
      const ppc = this.pxPerCol();
      for (const c of this.pair) {
        const j = this.displayCol(c);
        if (j < 0) continue;
        const x = (j + 0.5 - this.view.x[0]) * ppc;
        svg.appendChild(svgEl('line', { x1: x, x2: x, y1: 0, y2: this.size.h, stroke: '#61ffca', 'stroke-opacity': 0.7 }));
      }
    }
  }

  private drawTracks(): void {
    const svg = this.tracks;
    clear(svg);
    clear(this.corner);
    svg.setAttribute('width', String(this.size.w));
    const data = this.data;
    if (!data) return;
    const labels: Array<[string, number]> = [
      ['', ROW.ruler[1]],
      ['consensus', ROW.consensus[1] - 3],
      [this.compare ? 'of each node' : 'coverage', ROW.coverage[1] - 3],
      ['minor / ε', ROW.minor[1] - 3],
      ['candidates', ROW.candidates[1]],
      ['covariance', ROW.arcsTop + 10],
    ];
    for (const [text, y] of labels) {
      if (!text) continue;
      const t = svgEl('text', { x: SIDE_W - 6, y, fill: '#adacae', 'font-size': 10, 'text-anchor': 'end' });
      t.textContent = text;
      this.corner.appendChild(t);
    }
    const x0 = this.view.x[0];
    const ppc = this.pxPerCol();
    const j0 = Math.max(0, Math.floor(x0));
    const j1 = Math.min(this.nCols(), Math.ceil(this.view.x[1]));
    const left = (j: number) => (j - x0) * ppc;
    const { columns, columnRow, meta } = data;

    // Ruler (column numbers, 1-based, in this space).
    const step = niceStep(this.view.x[1] - x0, this.size.w);
    for (let j = Math.ceil(j0 / step) * step; j < j1; j += step) {
      const x = left(j) + ppc / 2;
      svg.appendChild(svgEl('line', { x1: x, x2: x, y1: 10, y2: 16, stroke: '#6d6d6d' }));
      const t = svgEl('text', { x, y: 9, fill: '#adacae', 'font-size': 9, 'text-anchor': 'middle' });
      t.textContent = String(this.dataCol(j) + 1);
      svg.appendChild(t);
    }

    // Per-pixel maxima, so a zoomed-out track is one rect a pixel.
    const binned = (value: (row: number) => number): Map<number, [number, number]> => {
      const out = new Map<number, [number, number]>();
      for (let j = j0; j < j1; j += 1) {
        const row = columnRow[this.dataCol(j)] ?? -1;
        if (row < 0) continue;
        const px = ppc >= 1 ? j : Math.floor(left(j));
        const v = value(row);
        const prev = out.get(px);
        if (!prev || v > prev[0]) out.set(px, [v, row]);
      }
      return out;
    };
    const rectFor = (key: number): [number, number] => (ppc >= 1 ? [left(key), Math.max(1, ppc - (ppc >= 4 ? 1 : 0))] : [key, 1]);

    // Consensus.
    if (ppc >= 7 && !this.compare) {
      for (let j = j0; j < j1; j += 1) {
        const ch = meta.consensus[this.dataCol(j)] ?? '';
        const t = svgEl('text', { x: left(j) + ppc / 2, y: ROW.consensus[1] - 3, fill: '#edecee', 'font-size': Math.min(12, ppc), 'text-anchor': 'middle', 'font-family': 'ui-monospace, monospace' });
        t.textContent = ch;
        svg.appendChild(t);
      }
    } else if (this.compare) {
      this.drawNodeAlleles(svg, j0, j1, left, ppc);
    } else {
      svg.appendChild(svgEl('rect', { x: left(j0), y: ROW.consensus[0] + 6, width: left(j1) - left(j0), height: 2, fill: '#6d6d6d' }));
    }

    // Coverage: depth, and base coverage within it. (Compare mode gives
    // its band to the nodes' alleles.)
    const depth = this.compare ? new Map<number, [number, number]>() : binned((r) => columns.depth[r]);
    let maxDepth = 1;
    for (const [v] of depth.values()) maxDepth = Math.max(maxDepth, v);
    const [c0, c1] = ROW.coverage;
    for (const [key, [v, row]] of depth) {
      const [x, wd] = rectFor(key);
      const hd = ((c1 - c0) * v) / maxDepth;
      const hb = ((c1 - c0) * columns.baseCov[row]) / maxDepth;
      svg.appendChild(svgEl('rect', { x, y: c1 - hd, width: wd, height: hd, fill: '#3b334b' }));
      svg.appendChild(svgEl('rect', { x, y: c1 - hb, width: wd, height: hb, fill: '#6d6a7d' }));
    }
    if (!this.compare) {
      const dl = svgEl('text', { x: 2, y: c0 + 8, fill: '#6d6d6d', 'font-size': 8 });
      dl.textContent = `${Math.round(maxDepth).toLocaleString('en-US')}`;
      svg.appendChild(dl);
    }

    // Minor-allele fraction against the column's ε (0-0.5).
    const [m0, m1] = ROW.minor;
    const minor = binned((r) => columns.minorFrac[r]);
    for (const [key, [v, row]] of minor) {
      if (v <= 0) continue;
      const [x, wd] = rectFor(key);
      const eps = columns.eps[row];
      const above = Number.isFinite(eps) ? v > 3 * eps : v > 0.05;
      const hgt = (m1 - m0) * Math.min(1, v / 0.5);
      svg.appendChild(svgEl('rect', { x, y: m1 - hgt, width: wd, height: hgt, fill: above ? '#a277ff' : '#4d4a5c' }));
      if (Number.isFinite(eps) && ppc >= 3) {
        const ye = m1 - (m1 - m0) * Math.min(1, eps / 0.5);
        svg.appendChild(svgEl('line', { x1: x, x2: x + wd, y1: ye, y2: ye, stroke: '#ffca85', 'stroke-opacity': 0.8 }));
      }
    }

    // Candidate columns, by route.
    for (let j = j0; j < j1; j += 1) {
      const row = columnRow[this.dataCol(j)] ?? -1;
      if (row < 0 || !columns.candidate[row]) continue;
      svg.appendChild(svgEl('rect', { x: left(j), y: ROW.candidates[0], width: Math.max(1, ppc), height: ROW.candidates[1] - ROW.candidates[0], fill: ROUTE_COLOURS[columns.route[row]] ?? '#adacae' }));
    }

    // Covariance arcs.
    const g = svgEl('g', { class: 'tx-arcs' }) as SVGGElement;
    const displayArcs = {
      ...data.arcs,
      u: Int32Array.from(data.arcs.u, (c) => this.displayCol(c)),
      v: Int32Array.from(data.arcs.v, (c) => this.displayCol(c)),
    };
    const keep: number[] = [];
    for (let i = 0; i < data.arcs.n; i += 1) if (displayArcs.u[i] >= 0 && displayArcs.v[i] >= 0) keep.push(i);
    const filtered = {
      n: keep.length,
      u: Int32Array.from(keep, (i) => displayArcs.u[i]),
      v: Int32Array.from(keep, (i) => displayArcs.v[i]),
      selfEdge: Uint8Array.from(keep, (i) => data.arcs.selfEdge[i]),
      n11: Float64Array.from(keep, (i) => data.arcs.n11[i]),
      total: Float64Array.from(keep, (i) => data.arcs.total[i]),
      phi: Float64Array.from(keep, (i) => data.arcs.phi[i]),
      logOdds: Float64Array.from(keep, (i) => data.arcs.logOdds[i]),
      pValue: Float64Array.from(keep, (i) => data.arcs.pValue[i]),
    };
    const geoms = arcGeometry(filtered, (j) => left(j) + ppc / 2, this.size.w, ROW.arcsBase - ROW.arcsTop, Math.max(1, this.view.x[1] - x0));
    drawArcs(
      g,
      geoms,
      ROW.arcsBase,
      (k) => this.sortByPair(data.arcs.u[keep[k]], data.arcs.v[keep[k]]),
      (k) => this.arcTooltip(k === null ? null : keep[k]),
    );
    svg.appendChild(g);
  }

  /** Compare mode: each node's consensus at the shown columns. */
  private drawNodeAlleles(svg: SVGSVGElement, j0: number, j1: number, left: (j: number) => number, ppc: number): void {
    const data = this.data!;
    const letters = 'ACGT-N';
    const nodes = [...new Set(Array.from(data.differences.haplotypeId))].sort((a, b) => a - b);
    // The consensus and coverage bands, one row a node.
    const top = ROW.consensus[0];
    const rowH = Math.max(6, (ROW.coverage[1] - top) / Math.max(1, nodes.length));
    nodes.forEach((h, k) => {
      const label = svgEl('text', { x: 2, y: top + (k + 1) * rowH - 2, fill: '#adacae', 'font-size': Math.min(10, rowH) });
      label.textContent = `node ${h}`;
      svg.appendChild(label);
    });
    for (let i = 0; i < data.differences.n; i += 1) {
      const c = data.colOfFrame[data.differences.frameCol[i]];
      const j = this.displayCol(c);
      if (j < j0 || j >= j1) continue;
      const k = nodes.indexOf(data.differences.haplotypeId[i]);
      const t = svgEl('text', { x: left(j) + ppc / 2, y: top + (k + 1) * rowH - 2, fill: '#edecee', 'font-size': Math.min(11, rowH, Math.max(6, ppc)), 'text-anchor': 'middle', 'font-family': 'ui-monospace, monospace' });
      t.textContent = letters[data.differences.allele[i]] ?? '?';
      svg.appendChild(t);
    }
  }

  private sideValue(key: string, r: number): string | null {
    const reads = this.data!.reads;
    const seq = (v: number, lo: number, hi: number, to = '#61ffca') =>
      Number.isFinite(v) ? mix('#252231', to, (v - lo) / (hi - lo)) : null;
    switch (key) {
      case 'sample': {
        const s = reads.sampleId[r];
        return s === null ? null : SAMPLE_COLOURS[Math.abs(s) % SAMPLE_COLOURS.length];
      }
      case 'identity':
        return seq(reads.identity[r], 0.9, 1);
      case 'quality':
        return seq(reads.quality[r], 10, 40);
      case 'margin':
        return seq(reads.margin[r], 0, 20, '#a277ff');
      case 'excess': {
        const v = this.excess[r];
        if (!Number.isFinite(v)) return null;
        return v <= 1 ? mix('#252231', '#6d6a7d', v) : mix('#6d6a7d', '#ff6767', (v - 1) / 3);
      }
      default:
        return null;
    }
  }

  private drawSide(): void {
    const svg = this.side;
    clear(svg);
    svg.setAttribute('height', String(this.size.h));
    const corner = this.corner;
    SIDE_COLS.forEach((c, k) => {
      const t = svgEl('text', { x: 4 + k * (SIDE_COL_W + 2) + SIDE_COL_W / 2, y: TRACKS_H - 4, fill: '#adacae', 'font-size': 8, 'text-anchor': 'middle' });
      t.textContent = c.label;
      const title = svgEl('title');
      title.textContent = c.title;
      t.appendChild(title);
      corner.appendChild(t);
    });
    if (!this.data) return;
    const ppr = this.pxPerRow();
    const y0 = this.view.y[0];
    const n = this.order.length;
    // A row of at least a pixel is drawn; below, one read per pixel row
    // stands for its bin.
    const rows: Array<[number, number, number]> = [];
    if (ppr >= 1) {
      for (let i = Math.max(0, Math.floor(y0)); i < Math.min(n, Math.ceil(this.view.y[1])); i += 1) {
        rows.push([this.order[i], (i - y0) * ppr, Math.max(1, ppr - (ppr >= 4 ? 1 : 0))]);
      }
    } else {
      for (let p = 0; p < this.size.h; p += 1) {
        const i = Math.floor(y0 + (p + 0.5) / ppr);
        if (i >= 0 && i < n) rows.push([this.order[i], p, 1]);
      }
    }
    SIDE_COLS.forEach((c, k) => {
      const x = 4 + k * (SIDE_COL_W + 2);
      for (const [r, y, h] of rows) {
        const fill = this.sideValue(c.key, r);
        if (fill) svg.appendChild(svgEl('rect', { x, y, width: SIDE_COL_W, height: h, fill }));
      }
    });
  }

  private drawFoot(): void {
    const data = this.data;
    if (!data) {
      this.foot.textContent = '';
      return;
    }
    const m = data.meta;
    const lod = matrixLod(this.viewport());
    const parts = [
      m.kind === 'template' ? templateLabel(m.template_id ?? 0) : `${m.kind} cluster ${m.cluster_id} (round ${m.round})`,
      `reference ${m.reference === 'pooled' ? 'pooled consensus' : `node ${m.reference}`}`,
      `${data.reads.n.toLocaleString('en-US')} reads (${m.n_sampled.toLocaleString('en-US')} voted in the PWM${m.n_absorbed_reads ? `, ${m.n_absorbed_reads} merged in` : ''})`,
      `${this.nCols().toLocaleString('en-US')} columns${this.compare ? ' where the nodes differ' : ''}`,
      `detail ${lod.x} × ${lod.y}`,
    ];
    if (this.orderNote) parts.push(this.orderNote);
    this.foot.innerHTML = '';
    const verdict = document.createElement('span');
    verdict.className = m.verified ? 'tx-ok' : 'tx-bad';
    verdict.textContent = m.verified ? '✓ verified against the run' : `✗ not the run: ${m.mismatch.join('; ')}`;
    const text = document.createElement('span');
    text.textContent = ` · ${parts.join(' · ')}`;
    this.foot.append(verdict, text);
    if (m.warnings.length) {
      const warn = document.createElement('span');
      warn.className = 'tx-warn';
      warn.textContent = ` · ${m.warnings.length} warning${m.warnings.length === 1 ? '' : 's'}`;
      warn.title = m.warnings.join('\n');
      this.foot.appendChild(warn);
    }
  }

  // ── hover ─────────────────────────────────────────────────────────

  private placeTooltip(lines: string[]): void {
    this.tooltip.innerHTML = '';
    for (const l of lines.slice(0, 6)) {
      const d = document.createElement('div');
      d.textContent = l;
      this.tooltip.appendChild(d);
    }
    const r = this.body.getBoundingClientRect();
    this.tooltip.style.left = `${this.pointer.x - r.left + 14}px`;
    this.tooltip.style.top = `${this.pointer.y - r.top + 14}px`;
    this.tooltip.hidden = false;
  }

  private hover(at: { x: number; y: number } | null): void {
    const data = this.data;
    if (!at || !data) {
      this.tooltip.hidden = true;
      return;
    }
    const j = Math.floor(at.x);
    const c = this.dataCol(j);
    const row = c >= 0 ? (data.columnRow[c] ?? -1) : -1;
    if (at.y < this.view.y[0] || row < 0) {
      if (row < 0) {
        this.tooltip.hidden = true;
        return;
      }
      const cols = data.columns;
      this.placeTooltip([
        `column ${c + 1}${data.meta.space === 'consensus' ? '' : ` (template ${cols.templateCol[row] + 1 || '—'})`}`,
        `consensus ${'ACGT-N'[cols.refBase[row]] ?? '?'} · depth ${cols.depth[row].toFixed(0)}`,
        `minor ${(100 * cols.minorFrac[row]).toFixed(2)}% · ε ${Number.isFinite(cols.eps[row]) ? (100 * cols.eps[row]).toFixed(2) + '%' : '—'}`,
        cols.candidate[row] ? `candidate (${routeName(cols.route[row])})${cols.signature[row] >= 0 ? `, signature ${cols.signature[row]}` : ''}` : 'not a candidate',
      ]);
      return;
    }
    const i = Math.floor(at.y);
    if (i < 0 || i >= this.order.length) {
      this.tooltip.hidden = true;
      return;
    }
    const r = this.order[i];
    const reads = data.reads;
    const s = reads.sampleId[r];
    const name = s === null ? '—' : (this.opts.samples[String(s)] ?? `sample ${s}`);
    const fmt = (v: number, d = 3) => (Number.isFinite(v) ? v.toFixed(d) : '—');
    const allele = c >= 0 ? 'ACGT-N'[this.alleleAt(r, c)] ?? 'uncovered' : '';
    this.placeTooltip([
      reads.readId[r] ?? `read row ${reads.readRow[r]}`,
      `node ${reads.node[r]}${reads.absorbedFrom[r] ? ` (merged in from ${reads.absorbedFrom[r]})` : ''} · ${name}`,
      `identity ${fmt(reads.identity[r])} · Q ${fmt(reads.quality[r], 1)}`,
      `margin ${fmt(reads.margin[r], 2)} · Δlogl ${fmt(reads.loglDelta[r], 2)}`,
      reads.sampled[r] ? `column ${c + 1}: ${allele}` : 'not in the PWM sample',
      `departures ${fmt(this.excess[r], 2)}× expected`,
    ]);
  }

  private alleleAt(r: number, c: number): number {
    const data = this.data!;
    const m = data.matrix;
    if (m.spanLo[r] < 0 || c < m.spanLo[r] || c >= m.spanHi[r]) return -1;
    const a = data.index.depStart[r];
    const b = data.index.depStart[r + 1];
    for (let k = a; k < b; k += 1) if (data.index.depCol[k] === c) return data.index.depAllele[k];
    return m.refAllele ? m.refAllele[c] : -1;
  }

  private arcTooltip(i: number | null): void {
    const data = this.data;
    if (i === null || !data) {
      this.tooltip.hidden = true;
      return;
    }
    const a = data.arcs;
    const fmt = (v: number) => (Number.isFinite(v) ? v.toFixed(3) : '—');
    this.placeTooltip([
      `columns ${a.u[i] + 1} ↔ ${a.v[i] + 1}${a.selfEdge[i] ? ' (one collapsed run)' : ''}`,
      `both minor in ${Number.isFinite(a.n11[i]) ? a.n11[i].toFixed(0) : '—'} of ${Number.isFinite(a.total[i]) ? a.total[i].toFixed(0) : '—'} co-observed reads`,
      `φ ${fmt(a.phi[i])} · log odds ${fmt(a.logOdds[i])}`,
      `p ${a.pValue[i].toExponential(1)}`,
      'click: sort reads by their state at the pair',
    ]);
  }

  // ── export ────────────────────────────────────────────────────────

  /** The view as one SVG document: tracks, per-read columns, and the
   *  matrix as vectors under the glyph budget or as an image over it. */
  exportSvg(): string | null {
    if (!this.data || !this.shownIndex) return null;
    const vp = this.viewport();
    const index = this.shownIndex;
    const order = this.order;
    const matrix = layerSvg({
      widthPx: vp.widthPx,
      heightPx: vp.heightPx,
      glyphs: matrixGlyphCost(index, order, vp),
      vector: (g) => drawMatrix(svgSink(g), index, order, vp),
      raster: () => {
        try {
          return this.layer.canvas.toDataURL('image/png');
        } catch {
          return null;
        }
      },
    });
    for (const child of Array.from(this.layer.overlay.childNodes)) matrix.svg.appendChild(child.cloneNode(true));
    const cells: ExportCell[] = [
      { svg: this.corner, x: 0, y: 0, width: SIDE_W, height: TRACKS_H },
      { svg: this.tracks, x: SIDE_W, y: 0, width: vp.widthPx, height: TRACKS_H },
      { svg: this.side, x: 0, y: TRACKS_H, width: SIDE_W, height: vp.heightPx },
      { svg: matrix.svg, x: SIDE_W, y: TRACKS_H, width: vp.widthPx, height: vp.heightPx },
    ];
    return buildGridSvg(cells, {
      title: `Support of ${this.describeTarget()}`,
      widthPx: SIDE_W + vp.widthPx,
      heightPx: TRACKS_H + vp.heightPx,
    }).svg;
  }

  dispose(): void {
    this.generation += 1;
    if (this.ticker) clearInterval(this.ticker);
    if (this.pendingRender !== null) clearTimeout(this.pendingRender);
    this.detachPanZoom();
    this.resizeObs?.disconnect();
    this.layer.dispose();
    this.root.remove();
  }
}
