// GenomeBrowser — composes a locus picker + ruler + a stack of tracks.
//
// The widget owns the track list, a ViewportBus, and a re-render
// scheduler that debounces fetches when the user pans/zooms quickly.
// Each track's renderer module knows nothing about the widget — it
// just consumes a (table, mode, ctx) tuple.

import { axisBottom } from 'd3-axis';
import { select } from 'd3-selection';
import { fetchJson, fetchTrackData, TrackMode } from '../engine/arrow_client';
import { attachPanZoom } from '../engine/interactions';
import { GenomicScale, makeAxis, xScale, formatGenomic } from '../engine/scales';
import { svgEl, ensureSvg } from '../engine/svg_layer';
import { Locus, ViewportBus } from '../engine/viewport_bus';
import { buildCompositeSvg, downloadSvg, estimateGlyphCount } from '../engine/export';
import { getRenderer } from '../track_renderers';
import { TrackMetadata } from '../track_renderers/base';

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
  cancel?: AbortController;
}

export interface GenomeBrowserOptions {
  host: HTMLElement;
  sessionId: string;
}

export class GenomeBrowser {
  readonly bus = new ViewportBus();
  private readonly host: HTMLElement;
  private readonly sessionId: string;
  private toolbar!: HTMLElement;
  private rulerHost!: HTMLElement;
  private trackHost!: HTMLElement;
  private contigs: ContigInfo[] = [];
  private tracks: MountedTrack[] = [];
  private currentContig: ContigInfo | null = null;
  private rerenderTimer: number | null = null;

  constructor(opts: GenomeBrowserOptions) {
    this.host = opts.host;
    this.sessionId = opts.sessionId;
  }

  async mount(): Promise<void> {
    this.host.innerHTML = '';
    this.host.style.display = 'flex';
    this.host.style.flexDirection = 'column';
    this.host.style.height = '100%';

    this.toolbar = document.createElement('div');
    this.toolbar.className = 'toolbar';
    this.host.appendChild(this.toolbar);

    const browser = document.createElement('div');
    browser.className = 'browser';
    this.host.appendChild(browser);

    this.rulerHost = document.createElement('div');
    this.rulerHost.className = 'ruler';
    browser.appendChild(this.rulerHost);

    this.trackHost = document.createElement('div');
    browser.appendChild(this.trackHost);

    await this.loadContigs();
    await this.loadAvailableTracks();
    this.buildToolbar();

    if (this.contigs.length > 0) {
      this.currentContig = this.contigs[0];
      // Default view: first 50kb of first contig (or whole contig if shorter).
      const span = Math.min(50_000, this.currentContig.length);
      this.bus.setLocus({
        contig: this.currentContig.name,
        start: 0,
        end: span,
      });
    }

    attachPanZoom({
      bus: this.bus,
      surface: browser,
      getWidthPx: () => browser.clientWidth || 1200,
      getContigLength: () => this.currentContig?.length ?? 0,
    });

    this.bus.on('locus:changed', () => this.scheduleRender());
    this.scheduleRender();
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
        header.textContent = entry.label;
        panel.appendChild(header);
        this.trackHost.appendChild(panel);
        this.tracks.push({ entry, meta, panel });
      } catch (err) {
        console.warn(`failed to load track ${entry.kind}/${entry.binding_id}`, err);
      }
    }
  }

  private buildToolbar(): void {
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
    locusInput.size = 24;
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

    const exportBtn = document.createElement('button');
    exportBtn.textContent = 'Save SVG';
    exportBtn.addEventListener('click', () => this.exportSvg());
    this.toolbar.appendChild(exportBtn);

    const status = document.createElement('span');
    status.className = 'label-dim';
    status.style.marginLeft = 'auto';
    status.textContent = `${this.tracks.length} tracks`;
    this.toolbar.appendChild(status);
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
      } catch (err) {
        if ((err as Error).name === 'AbortError') continue;
        console.warn(`render failed for ${track.entry.kind}`, err);
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

  private exportSvg(): void {
    const totalHeight =
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
  wrap.style.display = 'flex';
  wrap.style.alignItems = 'center';
  wrap.style.gap = '6px';
  const span = document.createElement('span');
  span.className = 'label-dim';
  span.textContent = labelText;
  wrap.appendChild(span);
  wrap.appendChild(control);
  return wrap;
}

function parseLocus(input: string): Locus | null {
  const m = /^\s*([\w.\-]+)\s*[:\s]\s*([\d,]+)\s*[-\s]\s*([\d,]+)\s*$/.exec(input);
  if (!m) return null;
  const start = parseInt(m[2].replace(/,/g, ''), 10);
  const end = parseInt(m[3].replace(/,/g, ''), 10);
  if (!Number.isFinite(start) || !Number.isFinite(end) || end <= start) return null;
  return { contig: m[1], start, end };
}
