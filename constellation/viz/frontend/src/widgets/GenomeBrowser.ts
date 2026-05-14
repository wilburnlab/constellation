// GenomeBrowser — composes a locus picker + overview + ruler + a stack
// of tracks. The widget owns the track list, a ViewportBus, a
// ResizeObserver-driven re-render scheduler, and chrome-level controls
// (zoom buttons, feature search, label toggle).
//
// Each track's renderer module knows nothing about the widget — it just
// consumes a (table, mode, ctx) tuple. Per-feature state (showLabels)
// rides on the RenderContext.

import { axisBottom } from 'd3-axis';
import { select } from 'd3-selection';
import { fetchJson, fetchTrackData, TrackMode } from '../engine/arrow_client';
import { attachPanZoom, zoomLocus, ZOOM_STEP } from '../engine/interactions';
import { GenomicScale, makeAxis, xScale, formatGenomic } from '../engine/scales';
import { svgEl, ensureSvg } from '../engine/svg_layer';
import { Locus, ViewportBus } from '../engine/viewport_bus';
import { buildCompositeSvg, downloadSvg, estimateGlyphCount } from '../engine/export';
import { getRenderer } from '../track_renderers';
import { TrackMetadata } from '../track_renderers/base';
import './GenomeBrowser.css';

interface ContigInfo {
  contig_id: number;
  name: string;
  length: number;
}

interface TrackEntry {
  kind: string;
  binding_id: string;
  label: string;
}

interface MountedTrack {
  entry: TrackEntry;
  meta: TrackMetadata;
  panel: HTMLElement;
  statusEl: HTMLElement;
  cancel?: AbortController;
}

interface SearchHit {
  feature_id: number;
  name: string;
  type: string;
  strand: string;
  contig_name: string;
  start: number;
  end: number;
  source: string;
}

export interface GenomeBrowserOptions {
  host: HTMLElement;
  sessionId: string;
}

const LABELS_KEY = 'constellation.genome.labels';
const OVERVIEW_HEIGHT = 18;
const SEARCH_DEBOUNCE_MS = 200;

export class GenomeBrowser {
  readonly bus = new ViewportBus();
  private readonly host: HTMLElement;
  private readonly sessionId: string;
  private toolbar!: HTMLElement;
  private overviewHost!: HTMLElement;
  private rulerHost!: HTMLElement;
  private trackHost!: HTMLElement;
  private browser!: HTMLElement;
  private contigs: ContigInfo[] = [];
  private tracks: MountedTrack[] = [];
  private currentContig: ContigInfo | null = null;
  private rerenderTimer: number | null = null;
  private resizeObserver: ResizeObserver | null = null;
  private detachPanZoom: (() => void) | null = null;
  private showLabels = readLabelsPref();
  private searchTimer: number | null = null;
  private searchAbort: AbortController | null = null;

  constructor(opts: GenomeBrowserOptions) {
    this.host = opts.host;
    this.sessionId = opts.sessionId;
  }

  async mount(): Promise<void> {
    this.host.innerHTML = '';
    this.host.classList.add('genome-browser-root');

    this.toolbar = document.createElement('div');
    this.toolbar.className = 'toolbar';
    this.host.appendChild(this.toolbar);

    this.browser = document.createElement('div');
    this.browser.className = 'browser';
    this.host.appendChild(this.browser);

    this.overviewHost = document.createElement('div');
    this.overviewHost.className = 'overview';
    this.browser.appendChild(this.overviewHost);

    this.rulerHost = document.createElement('div');
    this.rulerHost.className = 'ruler';
    this.browser.appendChild(this.rulerHost);

    this.trackHost = document.createElement('div');
    this.browser.appendChild(this.trackHost);

    await this.loadContigs();
    await this.loadAvailableTracks();
    this.buildToolbar();

    if (this.contigs.length > 0) {
      this.currentContig = this.contigs[0];
      const span = Math.min(50_000, this.currentContig.length);
      this.bus.setLocus({
        contig: this.currentContig.name,
        start: 0,
        end: span,
      });
    }

    this.detachPanZoom = attachPanZoom({
      bus: this.bus,
      surface: this.browser,
      getWidthPx: () => this.browser.clientWidth || 1200,
      getContigLength: () => this.currentContig?.length ?? 0,
    });

    this.bus.on('locus:changed', () => this.scheduleRender());

    this.resizeObserver = new ResizeObserver(() => this.scheduleRender());
    this.resizeObserver.observe(this.host);

    this.attachOverviewInteractions();
    this.scheduleRender();
  }

  dispose(): void {
    if (this.rerenderTimer !== null) {
      window.clearTimeout(this.rerenderTimer);
      this.rerenderTimer = null;
    }
    if (this.searchTimer !== null) {
      window.clearTimeout(this.searchTimer);
      this.searchTimer = null;
    }
    this.searchAbort?.abort();
    this.searchAbort = null;
    this.resizeObserver?.disconnect();
    this.resizeObserver = null;
    this.detachPanZoom?.();
    this.detachPanZoom = null;
    for (const track of this.tracks) track.cancel?.abort();
    this.host.classList.remove('genome-browser-root');
  }

  private async loadContigs(): Promise<void> {
    this.contigs = await fetchJson<ContigInfo[]>(
      `/api/sessions/${encodeURIComponent(this.sessionId)}/contigs`,
    );
  }

  private async loadAvailableTracks(): Promise<void> {
    const entries = await fetchJson<TrackEntry[]>(
      `/api/tracks?session=${encodeURIComponent(this.sessionId)}`,
    );
    for (const entry of entries) {
      if (!getRenderer(entry.kind)) continue;
      try {
        const meta = await fetchJson<TrackMetadata>(
          `/api/tracks/${encodeURIComponent(entry.kind)}/metadata?session=${encodeURIComponent(this.sessionId)}&binding=${encodeURIComponent(entry.binding_id)}`,
        );
        const panel = document.createElement('div');
        panel.className = 'track';
        const header = document.createElement('div');
        header.className = 'track-header';
        const labelEl = document.createElement('span');
        labelEl.className = 'track-header-label';
        labelEl.textContent = entry.label;
        const statusEl = document.createElement('span');
        statusEl.className = 'track-header-status';
        statusEl.textContent = '—';
        header.appendChild(labelEl);
        header.appendChild(statusEl);
        panel.appendChild(header);
        this.trackHost.appendChild(panel);
        this.tracks.push({ entry, meta, panel, statusEl });
      } catch (err) {
        console.warn(`failed to load track ${entry.kind}/${entry.binding_id}`, err);
      }
    }
  }

  private buildToolbar(): void {
    // Contig selector + Go-to input
    const contigSelect = document.createElement('select');
    for (const c of this.contigs) {
      const opt = document.createElement('option');
      opt.value = c.name;
      opt.textContent = `${c.name} (${formatGenomic(c.length)})`;
      contigSelect.appendChild(opt);
    }
    contigSelect.addEventListener('change', () => {
      const name = contigSelect.value;
      const c = this.contigs.find((x) => x.name === name);
      if (!c) return;
      this.currentContig = c;
      const span = Math.min(50_000, c.length);
      this.bus.setLocus({ contig: c.name, start: 0, end: span });
    });
    this.toolbar.appendChild(labeled('Contig:', contigSelect));

    const locusInput = document.createElement('input');
    locusInput.type = 'text';
    locusInput.placeholder = 'chr:start-end';
    locusInput.size = 22;
    const applyLocus = () => {
      const parsed = parseLocus(locusInput.value);
      if (!parsed) return;
      const c = this.contigs.find((x) => x.name === parsed.contig);
      if (!c) return;
      this.currentContig = c;
      contigSelect.value = c.name;
      this.bus.setLocus(parsed);
    };
    locusInput.addEventListener('keydown', (e) => {
      if (e.key === 'Enter') applyLocus();
    });
    this.toolbar.appendChild(labeled('Go to:', locusInput));
    this.bus.on('locus:changed', (locus) => {
      locusInput.value = `${locus.contig}:${locus.start}-${locus.end}`;
    });

    // Zoom + Fit buttons
    const zoomGroup = document.createElement('div');
    zoomGroup.className = 'btn-group';
    const zoomOut = makeIconButton('−', 'Zoom out', () => this.zoomBy(ZOOM_STEP));
    const zoomIn = makeIconButton('+', 'Zoom in', () => this.zoomBy(1 / ZOOM_STEP));
    const fit = makeIconButton('Fit', 'Fit to contig', () => this.fitContig());
    zoomGroup.appendChild(zoomOut);
    zoomGroup.appendChild(zoomIn);
    zoomGroup.appendChild(fit);
    this.toolbar.appendChild(zoomGroup);

    // Labels toggle
    const labelsBtn = document.createElement('button');
    labelsBtn.type = 'button';
    labelsBtn.className = `toggle${this.showLabels ? ' on' : ''}`;
    labelsBtn.textContent = 'Labels';
    labelsBtn.title = 'Show feature names on annotation tracks';
    labelsBtn.addEventListener('click', () => {
      this.showLabels = !this.showLabels;
      labelsBtn.classList.toggle('on', this.showLabels);
      writeLabelsPref(this.showLabels);
      this.scheduleRender();
    });
    this.toolbar.appendChild(labelsBtn);

    // Feature search
    this.toolbar.appendChild(this.buildSearchControl());

    // Right-side: track count + Save SVG
    const exportBtn = document.createElement('button');
    exportBtn.type = 'button';
    exportBtn.textContent = 'Save SVG';
    exportBtn.addEventListener('click', () => this.exportSvg());

    const right = document.createElement('div');
    right.className = 'toolbar-right';
    const status = document.createElement('span');
    status.className = 'label-dim';
    status.textContent = `${this.tracks.length} tracks`;
    right.appendChild(status);
    right.appendChild(exportBtn);
    this.toolbar.appendChild(right);
  }

  private buildSearchControl(): HTMLElement {
    const wrap = document.createElement('div');
    wrap.className = 'search-wrap';
    const input = document.createElement('input');
    input.type = 'text';
    input.placeholder = 'Search features…';
    input.size = 18;
    input.className = 'search-input';
    const dropdown = document.createElement('div');
    dropdown.className = 'search-results';
    dropdown.hidden = true;
    wrap.appendChild(input);
    wrap.appendChild(dropdown);

    const hide = (): void => {
      dropdown.hidden = true;
      dropdown.replaceChildren();
    };
    const onQuery = (text: string): void => {
      if (this.searchTimer !== null) window.clearTimeout(this.searchTimer);
      this.searchAbort?.abort();
      const trimmed = text.trim();
      if (!trimmed) {
        hide();
        return;
      }
      this.searchTimer = window.setTimeout(() => {
        this.searchTimer = null;
        void this.runSearch(trimmed, dropdown);
      }, SEARCH_DEBOUNCE_MS);
    };
    input.addEventListener('input', () => onQuery(input.value));
    input.addEventListener('keydown', (e) => {
      if (e.key === 'Escape') {
        input.value = '';
        hide();
      }
    });
    document.addEventListener('mousedown', (e) => {
      if (!wrap.contains(e.target as Node)) hide();
    });

    return wrap;
  }

  private async runSearch(
    query: string,
    dropdown: HTMLElement,
  ): Promise<void> {
    this.searchAbort = new AbortController();
    try {
      const url =
        `/api/sessions/${encodeURIComponent(this.sessionId)}/search` +
        `?q=${encodeURIComponent(query)}&limit=20`;
      const hits = await fetchJson<SearchHit[]>(url, this.searchAbort.signal);
      this.renderSearchResults(dropdown, hits);
    } catch (err) {
      if ((err as Error)?.name === 'AbortError') return;
      console.warn('feature search failed', err);
    }
  }

  private renderSearchResults(dropdown: HTMLElement, hits: SearchHit[]): void {
    dropdown.replaceChildren();
    if (hits.length === 0) {
      const empty = document.createElement('div');
      empty.className = 'search-empty';
      empty.textContent = 'No matches';
      dropdown.appendChild(empty);
      dropdown.hidden = false;
      return;
    }
    for (const hit of hits) {
      const row = document.createElement('div');
      row.className = 'search-row';
      const name = document.createElement('span');
      name.className = 'search-name';
      name.textContent = hit.name ?? `#${hit.feature_id}`;
      const type = document.createElement('span');
      type.className = 'search-badge';
      type.textContent = hit.type;
      const source = document.createElement('span');
      source.className = `search-badge src-${hit.source}`;
      source.textContent = hit.source;
      const coord = document.createElement('span');
      coord.className = 'search-coord';
      coord.textContent = `${hit.contig_name}:${hit.start.toLocaleString()}-${hit.end.toLocaleString()}`;
      row.appendChild(name);
      row.appendChild(type);
      row.appendChild(source);
      row.appendChild(coord);
      row.addEventListener('click', () => {
        this.jumpToFeature(hit);
        dropdown.hidden = true;
        dropdown.replaceChildren();
      });
      dropdown.appendChild(row);
    }
    dropdown.hidden = false;
  }

  private jumpToFeature(hit: SearchHit): void {
    const c = this.contigs.find((x) => x.name === hit.contig_name);
    if (!c) return;
    this.currentContig = c;
    const span = Math.max(1, hit.end - hit.start);
    const flank = Math.max(200, Math.round(span * 0.25));
    const start = Math.max(0, hit.start - flank);
    const end = Math.min(c.length, hit.end + flank);
    this.bus.setLocus({ contig: c.name, start, end });
  }

  private zoomBy(factor: number): void {
    if (!this.currentContig) return;
    this.bus.setLocus(zoomLocus(this.bus.locus, this.currentContig.length, factor, 0.5));
  }

  private fitContig(): void {
    if (!this.currentContig) return;
    this.bus.setLocus({
      contig: this.currentContig.name,
      start: 0,
      end: this.currentContig.length,
    });
  }

  private attachOverviewInteractions(): void {
    let dragStart: { clientX: number; locus: Locus } | null = null;
    const centerAt = (clientX: number): void => {
      if (!this.currentContig) return;
      const rect = this.overviewHost.getBoundingClientRect();
      if (rect.width <= 0) return;
      const fraction = Math.min(
        1,
        Math.max(0, (clientX - rect.left) / rect.width),
      );
      const targetBp = Math.round(fraction * this.currentContig.length);
      const span = this.bus.locus.end - this.bus.locus.start;
      const half = Math.round(span / 2);
      let start = targetBp - half;
      let end = start + span;
      if (start < 0) {
        start = 0;
        end = span;
      }
      if (end > this.currentContig.length) {
        end = this.currentContig.length;
        start = Math.max(0, end - span);
      }
      this.bus.setLocus({ contig: this.currentContig.name, start, end });
    };
    this.overviewHost.addEventListener('mousedown', (e) => {
      if (e.button !== 0) return;
      dragStart = { clientX: e.clientX, locus: this.bus.locus };
      centerAt(e.clientX);
      e.preventDefault();
    });
    window.addEventListener('mousemove', (e) => {
      if (!dragStart) return;
      centerAt(e.clientX);
    });
    window.addEventListener('mouseup', () => {
      dragStart = null;
    });
  }

  private scheduleRender(): void {
    if (this.rerenderTimer !== null) {
      window.clearTimeout(this.rerenderTimer);
    }
    this.rerenderTimer = window.setTimeout(() => {
      this.rerenderTimer = null;
      void this.render();
    }, 60);
  }

  private async render(): Promise<void> {
    const locus = this.bus.locus;
    if (!locus.contig) return;
    const widthPx = Math.max(200, this.host.clientWidth - 40);
    this.renderOverview(widthPx, locus);
    this.renderRuler(widthPx, locus);

    for (const track of this.tracks) {
      track.cancel?.abort();
      track.cancel = new AbortController();
      const heightPx = Number(track.meta.default_height_px ?? 80);
      const trackHost = track.panel;
      const svg = ensureSvg(trackHost, widthPx, heightPx);
      const ctx = {
        svg,
        widthPx,
        heightPx,
        xScale: xScale([locus.start, locus.end], widthPx),
        meta: track.meta,
        showLabels: this.showLabels,
      };

      try {
        const { table, mode } = await fetchTrackData(
          track.entry.kind,
          {
            session: this.sessionId,
            binding: track.entry.binding_id,
            contig: locus.contig,
            start: locus.start,
            end: locus.end,
            viewport_px: widthPx,
          },
          track.cancel.signal,
        );
        const renderer = getRenderer(track.entry.kind);
        if (!renderer) continue;
        renderer.render(table, mode as TrackMode, ctx);
        track.statusEl.textContent =
          table.numRows === 0
            ? '— no data in window'
            : `showing ${table.numRows.toLocaleString()} ${pluralUnit(track.entry.kind, table.numRows)}`;
      } catch (err) {
        if ((err as Error).name === 'AbortError') continue;
        console.warn(`render failed for ${track.entry.kind}`, err);
        track.statusEl.textContent = 'render failed';
      }
    }
  }

  private renderRuler(widthPx: number, locus: Locus): void {
    const rulerHeight = 28;
    const svg = ensureSvg(this.rulerHost, widthPx, rulerHeight);
    while (svg.firstChild) svg.removeChild(svg.firstChild);
    const scale: GenomicScale = xScale([locus.start, locus.end], widthPx);
    const axis = makeAxis(scale);
    const g = svgEl('g', { transform: `translate(0 4)` });
    svg.appendChild(g);
    select(g as Element).call(axisBottom(scale).ticks(8) as any);
    void axis;
  }

  private renderOverview(widthPx: number, locus: Locus): void {
    if (!this.currentContig) return;
    const contigLen = this.currentContig.length;
    const svg = ensureSvg(this.overviewHost, widthPx, OVERVIEW_HEIGHT);
    while (svg.firstChild) svg.removeChild(svg.firstChild);
    svg.appendChild(
      svgEl('rect', {
        x: 0,
        y: 5,
        width: widthPx,
        height: OVERVIEW_HEIGHT - 10,
        fill: '#2a2a33',
      }),
    );
    if (contigLen <= 0) return;
    const x0 = (locus.start / contigLen) * widthPx;
    const x1 = (locus.end / contigLen) * widthPx;
    const xClamped = Math.max(0, Math.min(widthPx, x0));
    const wClamped = Math.max(2, Math.min(widthPx, x1) - xClamped);
    svg.appendChild(
      svgEl('rect', {
        x: xClamped,
        y: 2,
        width: wClamped,
        height: OVERVIEW_HEIGHT - 4,
        fill: '#4f9efb',
        opacity: '0.55',
        class: 'overview-viewport',
      }),
    );
    const contigLabel = svgEl('text', {
      x: 6,
      y: OVERVIEW_HEIGHT - 5,
      'font-size': '10',
      fill: '#e3e3e8',
      'pointer-events': 'none',
      'paint-order': 'stroke',
      stroke: '#0f0f12',
      'stroke-width': '2',
      'stroke-linejoin': 'round',
    });
    contigLabel.textContent = `${this.currentContig.name}  ${formatGenomic(contigLen)}`;
    svg.appendChild(contigLabel);
  }

  private exportSvg(): void {
    const totalHeight =
      OVERVIEW_HEIGHT +
      28 +
      this.tracks.reduce(
        (acc, t) => acc + Number(t.meta.default_height_px ?? 80),
        0,
      );
    const widthPx = Math.max(200, this.host.clientWidth - 40);
    const cost = estimateGlyphCount(this.tracks.map((t) => t.panel));
    if (cost > 50_000) {
      const ok = window.confirm(
        `This export contains ~${cost.toLocaleString()} glyphs. ` +
          `The resulting file may be large and slow to open in Illustrator. Continue?`,
      );
      if (!ok) return;
    }
    const ruler = this.rulerHost.querySelector('svg.track-canvas') as SVGSVGElement | null;
    const svgString = buildCompositeSvg({
      title: `${this.bus.locus.contig}:${this.bus.locus.start}-${this.bus.locus.end}`,
      trackPanels: this.tracks.map((t) => t.panel),
      rulerSvg: ruler,
      totalWidthPx: widthPx,
      totalHeightPx: totalHeight,
    });
    const filename = `${this.bus.locus.contig}_${this.bus.locus.start}_${this.bus.locus.end}.svg`;
    downloadSvg(filename, svgString);
  }
}

function labeled(labelText: string, control: HTMLElement): HTMLElement {
  const wrap = document.createElement('label');
  wrap.className = 'labeled-control';
  const span = document.createElement('span');
  span.className = 'label-dim';
  span.textContent = labelText;
  wrap.appendChild(span);
  wrap.appendChild(control);
  return wrap;
}

function makeIconButton(
  text: string,
  title: string,
  onClick: () => void,
): HTMLButtonElement {
  const b = document.createElement('button');
  b.type = 'button';
  b.className = 'icon-btn';
  b.textContent = text;
  b.title = title;
  b.addEventListener('click', onClick);
  return b;
}

function parseLocus(input: string): Locus | null {
  const m = /^\s*([\w.\-]+)\s*[:\s]\s*([\d,]+)\s*[-\s]\s*([\d,]+)\s*$/.exec(input);
  if (!m) return null;
  const start = parseInt(m[2].replace(/,/g, ''), 10);
  const end = parseInt(m[3].replace(/,/g, ''), 10);
  if (!Number.isFinite(start) || !Number.isFinite(end) || end <= start) return null;
  return { contig: m[1], start, end };
}

function pluralUnit(kind: string, n: number): string {
  const map: Record<string, [string, string]> = {
    gene_annotation: ['feature', 'features'],
    coverage_histogram: ['bin', 'bins'],
    read_pileup: ['read', 'reads'],
    cluster_pileup: ['cluster', 'clusters'],
    splice_junctions: ['junction', 'junctions'],
    reference_sequence: ['base', 'bases'],
  };
  const [singular, plural] = map[kind] ?? ['row', 'rows'];
  return n === 1 ? singular : plural;
}

function readLabelsPref(): boolean {
  try {
    const raw = window.localStorage.getItem(LABELS_KEY);
    if (raw === null) return true;
    return raw === 'true';
  } catch {
    return true;
  }
}

function writeLabelsPref(value: boolean): void {
  try {
    window.localStorage.setItem(LABELS_KEY, value ? 'true' : 'false');
  } catch {
    // ignored — storage may be disabled in private modes
  }
}
