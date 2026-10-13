// GenomeBrowser — a locus picker, overview bar and ruler over a stack of
// genome tracks.
//
// The stack itself — the per-track chrome, order, drag-to-reorder,
// resize, hide / collapse, the gear popover and layout persistence — is
// the generic `PanelStack`. What lives here is what makes it a genome
// browser: the session's reference and sources, the locus (a
// `ViewportBus`), the toolbar (contig / Go-to / zoom / Labels / feature
// search / Options / Datasets / Save SVG), the overview and ruler, and
// the two things the stack asks of its host — fetch a track's data for
// a locus, and draw it with that kind's renderer.
//
// Layout persists to localStorage keyed by sessionId and, when the
// session was saved with a slug, also to the saved-session TOML via
// PATCH /api/saved-sessions/{slug}/layout. Layout entries are keyed by
// (source_id, kind) so they survive runtime source add/remove (POST /
// DELETE /api/sessions/{id}/sources, which rebuild the session in place
// under the same session_id).
//
// A renderer knows nothing about the browser — it consumes a
// (table, mode, ctx) tuple. Host UI state it may follow (showLabels)
// rides on the RenderContext.

import { axisBottom } from 'd3-axis';
import { select } from 'd3-selection';
import {
  FetchedTable,
  fetchJson,
  fetchJsonMethod,
  fetchTrackData,
} from '../../engine/arrow_client';
import { attachPanZoom, zoomLocus, ZOOM_STEP } from './interactions';
import { GenomicScale, makeAxis, xScale, formatGenomic } from './scales';
import { svgEl, ensureSvg } from '../../engine/svg_layer';
import { Locus, ViewportBus } from './viewport_bus';
import { buildCompositeSvg, downloadSvg, estimateGlyphCount } from '../../engine/export';
import { encodePushdown } from '../../panels/kind';
import { Panel, PanelEntry } from '../../panels/Panel';
import { PanelStack, PanelView } from '../../panels/PanelStack';
import { getRenderer, kindRank } from './renderers';
import { FALLBACK_SETTINGS } from './renderers/settings_common';
import { TrackMetadata } from './renderers/base';
import {
  BindingRow,
  DatasetManagerPopover,
  SourceRow,
} from './DatasetManagerPopover';
import {
  LayoutEntry,
  LayoutStore,
  createLayoutStore,
} from '../../panels/layout';
import {
  BrowserOptions,
  DEFAULT_BROWSER_OPTIONS,
  OptionsPopover,
  parseBrowserOptions,
} from '../../panels/OptionsPopover';
import './genome.css';

interface ContigInfo {
  contig_id: number;
  name: string;
  length: number;
}

/** A track's data as fetched: the Arrow table plus the mode the server
 *  chose for it. */
type TrackPanel = Panel<FetchedTable>;

/** What every track is fetched and drawn for in one render pass. */
interface GenomeView extends PanelView {
  locus: Locus;
  showLabels: boolean;
}

interface SessionManifest {
  session_id: string;
  label: string;
  reference: {
    handle: string;
    path: string;
    genome: string;
    annotation: string | null;
    assembly_accession: string | null;
  };
  sources: ManifestSource[];
  warnings: string[];
  saved_as: string | null;
  stages_present: Record<string, boolean>;
}

interface ManifestSource {
  source_id: string;
  path: string;
  kind: 'align' | 'cluster';
  label: string;
  assembly_accession: string | null;
  reference_handle: string | null;
  samples: string[];
  slots: Record<string, string | null>;
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
  /** Optional initial layout (e.g. restored from a saved-session TOML).
   * Takes precedence over localStorage when present. */
  initialLayout?: LayoutEntry[];
}

const LABELS_KEY = 'constellation.genome.labels';
/** localStorage prefix for this browser's layout + options. */
const STORAGE_NAMESPACE = 'constellation.genome';
const OVERVIEW_HEIGHT = 18;
const RULER_HEIGHT = 28;
const SEARCH_DEBOUNCE_MS = 200;

export class GenomeBrowser {
  readonly bus = new ViewportBus();
  private readonly host: HTMLElement;
  private readonly sessionId: string;
  private readonly initialLayout: LayoutEntry[] | null;
  private toolbar!: HTMLElement;
  private overviewHost!: HTMLElement;
  private rulerHost!: HTMLElement;
  private trackHost!: HTMLElement;
  private emptyPlaceholder!: HTMLElement;
  private browser!: HTMLElement;
  private contigs: ContigInfo[] = [];
  private manifest: SessionManifest | null = null;
  private stack!: PanelStack<GenomeView, FetchedTable, BrowserOptions>;
  private currentContig: ContigInfo | null = null;
  private rerenderTimer: number | null = null;
  private resizeObserver: ResizeObserver | null = null;
  private detachPanZoom: (() => void) | null = null;
  /** Undo for everything registered outside the browser's own DOM —
   *  bus subscriptions and document / window listeners. Its own elements
   *  take their listeners with them. */
  private teardown: Array<() => void> = [];
  private showLabels = readLabelsPref();
  private searchTimer: number | null = null;
  private searchAbort: AbortController | null = null;
  private datasetBtn!: HTMLButtonElement;
  private optionsBtn!: HTMLButtonElement;
  private trackCountStatus!: HTMLElement;
  private popover: DatasetManagerPopover | null = null;
  private optionsPopover: OptionsPopover | null = null;
  private options: BrowserOptions = { ...DEFAULT_BROWSER_OPTIONS };
  private readonly layoutStore: LayoutStore<BrowserOptions>;

  constructor(opts: GenomeBrowserOptions) {
    this.host = opts.host;
    this.sessionId = opts.sessionId;
    this.initialLayout = opts.initialLayout ?? null;
    this.layoutStore = createLayoutStore<BrowserOptions>({
      namespace: STORAGE_NAMESPACE,
      sessionId: this.sessionId,
      parseOptions: parseBrowserOptions,
      savedSlug: () => this.manifest?.saved_as ?? null,
    });
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
    this.trackHost.className = 'track-stack';
    this.browser.appendChild(this.trackHost);

    this.emptyPlaceholder = document.createElement('div');
    this.emptyPlaceholder.className = 'track-stack-empty';
    this.emptyPlaceholder.textContent =
      'All tracks hidden — open the Datasets menu to enable some.';
    this.emptyPlaceholder.hidden = true;
    this.browser.appendChild(this.emptyPlaceholder);

    this.stack = new PanelStack<GenomeView, FetchedTable, BrowserOptions>({
      stackHost: this.trackHost,
      emptyPlaceholder: this.emptyPlaceholder,
      driver: {
        view: () => this.currentView(),
        fetch: (track, view, signal) => this.fetchTrack(track, view, signal),
        draw: (track, data, view, svg, size) =>
          this.drawTrack(track, data, view, svg, size),
      },
      kindOf: getRenderer,
      fallbackSettings: FALLBACK_SETTINGS,
      settingsHost: () => ({ showLabels: this.showLabels }),
      store: this.layoutStore,
      options: () => this.options,
      requestRender: () => this.scheduleRender(),
      onChanged: () => {
        this.updateTrackCountStatus();
        this.refreshPopoverIfOpen();
      },
    });

    await Promise.all([this.loadManifest(), this.loadContigs()]);
    await this.loadAvailableTracks();
    this.applyPersistedLayout();
    this.stack.refreshOrder();
    this.updateTrackCountStatus();
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

    this.teardown.push(this.bus.on('locus:changed', () => this.scheduleRender()));

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
    for (const undo of this.teardown.splice(0)) undo();
    this.popover?.dispose();
    this.popover = null;
    this.optionsPopover?.dispose();
    this.optionsPopover = null;
    this.stack?.dispose();
    this.host.classList.remove('genome-browser-root');
  }

  // --------------------------------------------------------------------
  // Initial load
  // --------------------------------------------------------------------

  private async loadManifest(): Promise<void> {
    this.manifest = await fetchJson<SessionManifest>(
      `/api/sessions/${encodeURIComponent(this.sessionId)}/manifest`,
    );
  }

  private async loadContigs(): Promise<void> {
    this.contigs = await fetchJson<ContigInfo[]>(
      `/api/sessions/${encodeURIComponent(this.sessionId)}/contigs`,
    );
  }

  private async loadAvailableTracks(): Promise<void> {
    const entries = await fetchJson<PanelEntry[]>(
      `/api/tracks?session=${encodeURIComponent(this.sessionId)}`,
    );
    // Stable canonical order: by each kind's declared rank, then per-kind insertion
    // order from the endpoint (which iterates session.sources in order).
    entries.sort((a, b) => kindRank(a.kind) - kindRank(b.kind));

    let order = 0;
    for (const entry of entries) {
      if (!getRenderer(entry.kind)) continue;
      try {
        const meta = await fetchJson<TrackMetadata>(
          `/api/tracks/${encodeURIComponent(entry.kind)}/metadata?session=${encodeURIComponent(this.sessionId)}&binding=${encodeURIComponent(entry.binding_id)}`,
        );
        const defaultHeight = Number(meta.default_height_px ?? 80);
        this.stack.add({
          entry,
          meta,
          visible: true,
          collapsed: false,
          heightPx: defaultHeight,
          displayOrder: order++,
          style: {},
          filter: {},
        });
      } catch (err) {
        console.warn(`failed to load track ${entry.kind}/${entry.binding_id}`, err);
      }
    }
  }

  // --------------------------------------------------------------------
  // Layout state — persistence + apply
  // --------------------------------------------------------------------

  private applyPersistedLayout(): void {
    // Browser-wide options are loaded independently of track layout.
    const storedOptions = this.layoutStore.loadOptions();
    if (storedOptions) this.options = { ...this.options, ...storedOptions };

    let entries: LayoutEntry[] | null = this.initialLayout;
    if (!entries || entries.length === 0) {
      entries = this.layoutStore.loadLayout();
    }
    if (!entries || entries.length === 0) return;
    this.stack.applyLayout(entries);
  }

  // --------------------------------------------------------------------
  // Toolbar + popover wiring
  // --------------------------------------------------------------------

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
    const applyLocus = (): void => {
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
    this.teardown.push(
      this.bus.on('locus:changed', (locus) => {
        locusInput.value = `${locus.contig}:${locus.start}-${locus.end}`;
      }),
    );

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

    // Right-side: Options, Datasets popovers, track count, Save SVG
    this.optionsBtn = document.createElement('button');
    this.optionsBtn.type = 'button';
    this.optionsBtn.className = 'options-btn';
    this.optionsBtn.textContent = 'Options ▾';
    this.optionsBtn.title = 'Browser-wide settings';
    this.optionsBtn.addEventListener('click', () => this.toggleOptionsPopover());

    this.datasetBtn = document.createElement('button');
    this.datasetBtn.type = 'button';
    this.datasetBtn.className = 'dataset-btn';
    this.datasetBtn.textContent = 'Datasets ▾';
    this.datasetBtn.title = 'Toggle tracks, add/remove datasets';
    this.datasetBtn.addEventListener('click', () => this.togglePopover());

    const exportBtn = document.createElement('button');
    exportBtn.type = 'button';
    exportBtn.textContent = 'Save SVG';
    exportBtn.addEventListener('click', () => this.exportSvg());

    const right = document.createElement('div');
    right.className = 'toolbar-right';
    this.trackCountStatus = document.createElement('span');
    this.trackCountStatus.className = 'label-dim';
    right.appendChild(this.optionsBtn);
    right.appendChild(this.datasetBtn);
    right.appendChild(this.trackCountStatus);
    right.appendChild(exportBtn);
    this.toolbar.appendChild(right);
    this.updateTrackCountStatus();
  }

  private updateTrackCountStatus(): void {
    if (!this.trackCountStatus) return;
    const total = this.stack.panels.length;
    const visible = this.stack.panels.filter((t) => t.visible).length;
    this.trackCountStatus.textContent =
      total === visible
        ? `${total} tracks`
        : `${visible} / ${total} tracks`;
  }

  private togglePopover(): void {
    if (this.popover) {
      this.popover.dispose();
      this.popover = null;
      return;
    }
    this.openPopover();
  }

  private refreshPopoverIfOpen(): void {
    if (!this.popover) return;
    this.popover.dispose();
    this.popover = null;
    this.openPopover();
  }

  private openPopover(): void {
    if (!this.manifest) return;
    const referenceLabel =
      this.manifest.reference.handle ||
      this.manifest.reference.path ||
      'reference';
    const referenceBindings: BindingRow[] = this.stack.panels
      .filter((t) => t.entry.source_id === null)
      .map((t) => this.toBindingRow(t));
    const sourceLookup = new Map<string, ManifestSource>();
    for (const s of this.manifest.sources) sourceLookup.set(s.source_id, s);
    const sources: SourceRow[] = this.manifest.sources.map((s) => ({
      source_id: s.source_id,
      label: s.label,
      kind: s.kind,
      path: s.path,
      warning: warningFor(s, this.manifest!),
    }));
    const bindingsBySource = new Map<string, BindingRow[]>();
    for (const t of this.stack.panels) {
      const sid = t.entry.source_id;
      if (sid === null) continue;
      if (!bindingsBySource.has(sid)) bindingsBySource.set(sid, []);
      bindingsBySource.get(sid)!.push(this.toBindingRow(t));
    }
    this.popover = new DatasetManagerPopover({
      anchor: this.datasetBtn,
      referenceLabel,
      referenceBindings,
      sources,
      bindingsBySource,
      handlers: {
        onToggleBinding: (binding_id, visible) => {
          const t = this.stack.panels.find((x) => x.entry.binding_id === binding_id);
          if (t) this.stack.setVisible(t, visible);
        },
        onRemoveSource: async (sid) => {
          await this.removeSource(sid);
        },
        onAddSource: async (path) => {
          return this.addSource(path);
        },
      },
      onClose: () => {
        this.popover?.dispose();
        this.popover = null;
      },
    });
    this.popover.mount(document.body);
  }

  private toBindingRow(t: TrackPanel): BindingRow {
    return {
      binding_id: t.entry.binding_id,
      kind: t.entry.kind,
      label: t.entry.label,
      source_id: t.entry.source_id,
      visible: t.visible,
    };
  }

  // --------------------------------------------------------------------
  // Options popover
  // --------------------------------------------------------------------

  private toggleOptionsPopover(): void {
    if (this.optionsPopover) {
      this.optionsPopover.dispose();
      this.optionsPopover = null;
      return;
    }
    this.optionsPopover = new OptionsPopover({
      anchor: this.optionsBtn,
      options: { ...this.options },
      handlers: {
        onChange: (key, value) => {
          this.options = { ...this.options, [key]: value };
          this.stack.schedulePersist();
        },
      },
      onClose: () => {
        this.optionsPopover?.dispose();
        this.optionsPopover = null;
      },
    });
    this.optionsPopover.mount(document.body);
  }

  // --------------------------------------------------------------------
  // Runtime source mutation
  // --------------------------------------------------------------------

  private async addSource(
    path: string,
  ): Promise<{ ok: true } | { ok: false; error: string }> {
    try {
      await fetchJsonMethod<SessionManifest>(
        `/api/sessions/${encodeURIComponent(this.sessionId)}/sources`,
        'POST',
        { path },
      );
    } catch (err) {
      return { ok: false, error: (err as Error).message };
    }
    await this.reloadTracksAfterSourceChange();
    return { ok: true };
  }

  private async removeSource(sourceId: string): Promise<void> {
    try {
      await fetchJsonMethod<SessionManifest>(
        `/api/sessions/${encodeURIComponent(this.sessionId)}/sources/${encodeURIComponent(sourceId)}`,
        'DELETE',
      );
    } catch (err) {
      console.warn('failed to remove source', err);
      return;
    }
    await this.reloadTracksAfterSourceChange();
  }

  private async reloadTracksAfterSourceChange(): Promise<void> {
    // Snapshot current layout BEFORE tearing down, then rebuild and
    // restore by (source_id, kind). New bindings inherit kind-grouped
    // defaults at the tail of their kind cluster.
    // Any open per-track settings popover anchors a torn-down DOM node
    // — close it before the rebuild.
    this.stack.closeSettings();
    const previous = this.stack.snapshot();
    this.stack.clear();
    await this.loadManifest();
    await this.loadAvailableTracks();
    this.stack.mergeAfterReload(previous);
    this.stack.refreshOrder();
    this.updateTrackCountStatus();
    this.refreshPopoverIfOpen();
    this.stack.schedulePersist();
    this.scheduleRender();
  }

  // --------------------------------------------------------------------
  // Search
  // --------------------------------------------------------------------

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
    const onOutsideMouseDown = (e: MouseEvent): void => {
      if (!wrap.contains(e.target as Node)) hide();
    };
    document.addEventListener('mousedown', onOutsideMouseDown);
    this.teardown.push(() =>
      document.removeEventListener('mousedown', onOutsideMouseDown),
    );

    return wrap;
  }

  private async runSearch(query: string, dropdown: HTMLElement): Promise<void> {
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

  // --------------------------------------------------------------------
  // Zoom + overview
  // --------------------------------------------------------------------

  private zoomBy(factor: number): void {
    if (!this.currentContig) return;
    this.bus.setLocus(
      zoomLocus(this.bus.locus, this.currentContig.length, factor, 0.5),
    );
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
    const onMouseMove = (e: MouseEvent): void => {
      if (!dragStart) return;
      centerAt(e.clientX);
    };
    const onMouseUp = (): void => {
      dragStart = null;
    };
    window.addEventListener('mousemove', onMouseMove);
    window.addEventListener('mouseup', onMouseUp);
    this.teardown.push(() => {
      window.removeEventListener('mousemove', onMouseMove);
      window.removeEventListener('mouseup', onMouseUp);
    });
  }

  // --------------------------------------------------------------------
  // Render loop
  // --------------------------------------------------------------------

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
    const view = this.currentView();
    if (!view) return;
    this.renderOverview(view.widthPx, view.locus);
    this.renderRuler(view.widthPx, view.locus);
    await this.stack.render();
  }

  // --------------------------------------------------------------------
  // What the track stack asks of its host
  // --------------------------------------------------------------------

  /** The locus and width every track is drawn for right now; null until
   *  a contig is selected. */
  private currentView(): GenomeView | null {
    const locus = this.bus.locus;
    if (!locus.contig) return null;
    const widthPx = Math.max(200, this.host.clientWidth - 40);
    return {
      key: `${locus.contig}|${locus.start}|${locus.end}|${widthPx}`,
      widthPx,
      locus,
      showLabels: this.showLabels,
    };
  }

  private fetchTrack(
    track: TrackPanel,
    view: GenomeView,
    signal: AbortSignal,
  ): Promise<FetchedTable> {
    return fetchTrackData(
      track.entry.kind,
      {
        session: this.sessionId,
        binding: track.entry.binding_id,
        contig: view.locus.contig,
        start: view.locus.start,
        end: view.locus.end,
        viewport_px: view.widthPx,
        // The kind's server-side filters, as it declares them.
        ...encodePushdown(getRenderer(track.entry.kind), track.filter),
      },
      signal,
    );
  }

  private drawTrack(
    track: TrackPanel,
    data: FetchedTable,
    view: GenomeView,
    svg: SVGSVGElement,
    size: { widthPx: number; heightPx: number },
  ): number {
    getRenderer(track.entry.kind)?.render(data.table, data.mode, {
      svg,
      widthPx: size.widthPx,
      heightPx: size.heightPx,
      xScale: xScale([view.locus.start, view.locus.end], size.widthPx),
      meta: track.meta as TrackMetadata,
      showLabels: view.showLabels,
      style: track.style,
      filter: track.filter,
    });
    return data.table.numRows;
  }

  private renderRuler(widthPx: number, locus: Locus): void {
    const svg = ensureSvg(this.rulerHost, widthPx, RULER_HEIGHT);
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
        fill: '#2d2b3a',
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
        fill: '#a277ff',
        opacity: '0.55',
        class: 'overview-viewport',
      }),
    );
    const contigLabel = svgEl('text', {
      x: 6,
      y: OVERVIEW_HEIGHT - 5,
      'font-size': '10',
      fill: '#edecee',
      'pointer-events': 'none',
      'paint-order': 'stroke',
      stroke: '#15141b',
      'stroke-width': '2',
      'stroke-linejoin': 'round',
    });
    contigLabel.textContent = `${this.currentContig.name}  ${formatGenomic(contigLen)}`;
    svg.appendChild(contigLabel);
  }

  // --------------------------------------------------------------------
  // SVG export
  // --------------------------------------------------------------------

  private exportSvg(): void {
    const visible = this.stack.visibleSorted().filter((t) => !t.collapsed);
    const widthPx = Math.max(200, this.host.clientWidth - 40);
    const panels = visible.map((t) => t.element);
    const cost = estimateGlyphCount(panels);
    if (cost > 50_000) {
      const ok = window.confirm(
        `This export contains ~${cost.toLocaleString()} glyphs. ` +
          `The resulting file may be large and slow to open in Illustrator. Continue?`,
      );
      if (!ok) return;
    }
    const ruler = this.rulerHost.querySelector(
      'svg.track-canvas',
    ) as SVGSVGElement | null;
    const { svg: svgString } = buildCompositeSvg({
      title: `${this.bus.locus.contig}:${this.bus.locus.start}-${this.bus.locus.end}`,
      trackPanels: panels,
      rulerSvg: ruler,
      totalWidthPx: widthPx,
      clip: this.options.clip_svg,
    });
    const filename = `${this.bus.locus.contig}_${this.bus.locus.start}_${this.bus.locus.end}.svg`;
    downloadSvg(filename, svgString);
  }
}

// ----------------------------------------------------------------------
// Helpers
// ----------------------------------------------------------------------

function warningFor(
  source: ManifestSource,
  manifest: SessionManifest,
): string | null {
  const refAssembly = manifest.reference.assembly_accession;
  if (
    source.assembly_accession &&
    refAssembly &&
    source.assembly_accession !== refAssembly
  ) {
    return `assembly ${source.assembly_accession} ≠ reference ${refAssembly}`;
  }
  return null;
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
