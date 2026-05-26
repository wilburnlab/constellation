// GenomeBrowser — composes a locus picker + overview + ruler + a
// dynamically ordered stack of tracks. The widget owns the track list,
// per-binding layout state (visible / display order / height / collapsed),
// a ViewportBus, a ResizeObserver-driven re-render scheduler, and the
// chrome-level controls (zoom buttons, feature search, label toggle,
// dataset manager popover, Save SVG).
//
// Per-track layout state persists to localStorage keyed by sessionId and,
// when the session was saved with a slug, also syncs to the saved-session
// TOML via PATCH /api/saved-sessions/{slug}/layout. Layout entries are
// keyed by (source_id, kind) so they survive runtime source add/remove
// (handled by POST/DELETE /api/sessions/{id}/sources, which rebuilds the
// Session in place under the same session_id).
//
// Each track's renderer module knows nothing about the widget — it just
// consumes a (table, mode, ctx) tuple. Per-feature host UI state
// (showLabels) rides on the RenderContext.

import { Table } from 'apache-arrow';
import { axisBottom } from 'd3-axis';
import { select } from 'd3-selection';
import {
  fetchJson,
  fetchJsonMethod,
  fetchTrackData,
  TrackMode,
} from '../engine/arrow_client';
import { attachPanZoom, zoomLocus, ZOOM_STEP } from '../engine/interactions';
import { GenomicScale, makeAxis, xScale, formatGenomic } from '../engine/scales';
import { svgEl, ensureSvg } from '../engine/svg_layer';
import { Locus, ViewportBus } from '../engine/viewport_bus';
import { buildCompositeSvg, downloadSvg, estimateGlyphCount } from '../engine/export';
import { getRenderer } from '../track_renderers';
import { TrackMetadata } from '../track_renderers/base';
import {
  BindingRow,
  DatasetManagerPopover,
  SourceRow,
} from './DatasetManagerPopover';
import {
  BrowserOptions,
  DEFAULT_BROWSER_OPTIONS,
  OptionsPopover,
} from './OptionsPopover';
import { TrackSettingsPanel } from './TrackSettingsPanel';
import './GenomeBrowser.css';

/** Filter keys whose values are pushed down to the kernel — changing
 *  any of these requires a server refetch, not a client-side restyle.
 *  Everything else (palette overrides, visible-strands allowlist, etc.)
 *  is satisfied off the cached Arrow table via restyleTrack().
 *
 *  Currently: `min_mapq` only. Future additions land here (e.g. PR 5
 *  will add `cluster_view`). */
const PUSHDOWN_FILTER_KEYS: ReadonlySet<string> = new Set(['min_mapq']);

interface ContigInfo {
  contig_id: number;
  name: string;
  length: number;
}

interface TrackEntry {
  kind: string;
  binding_id: string;
  label: string;
  source_id: string | null;
}

interface MountedTrack {
  entry: TrackEntry;
  meta: TrackMetadata;
  panel: HTMLElement;
  headerLabelEl: HTMLElement;
  statusEl: HTMLElement;
  bodyHost: HTMLElement;
  collapseBtn: HTMLButtonElement;
  settingsBtn: HTMLButtonElement;
  hideBtn: HTMLButtonElement;
  visible: boolean;
  collapsed: boolean;
  heightPx: number;
  displayOrder: number;
  style: Record<string, unknown>;
  filter: Record<string, unknown>;
  /** Last successfully fetched Arrow data + locus. Reused by
   *  restyleTrack() so style/filter changes don't trigger a refetch. */
  lastFetched?: { table: Table; mode: TrackMode; locusKey: string };
  cancel?: AbortController;
}

interface TrackLayoutEntry {
  source_id: string;        // "" for reference-only bindings
  kind: string;
  visible: boolean;
  display_order: number;
  height_px: number;
  collapsed: boolean;
  style?: Record<string, unknown>;
  filter?: Record<string, unknown>;
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
  initialLayout?: TrackLayoutEntry[];
}

const LABELS_KEY = 'constellation.genome.labels';
const layoutStorageKey = (sessionId: string): string =>
  `constellation.genome.layout.${sessionId}`;
const optionsStorageKey = (sessionId: string): string =>
  `constellation.genome.options.${sessionId}`;
const OVERVIEW_HEIGHT = 18;
const RULER_HEIGHT = 28;
const COLLAPSED_BODY_HEIGHT = 0;
const SEARCH_DEBOUNCE_MS = 200;
const LAYOUT_PERSIST_DEBOUNCE_MS = 200;
const MIN_TRACK_HEIGHT = 24;
const MAX_TRACK_HEIGHT = 800;

// Canonical kind order — initial render stacks tracks in this order;
// new bindings added at runtime get slotted into their kind's cluster.
const KIND_ORDER: readonly string[] = [
  'reference_sequence',
  'gene_annotation',
  'coverage_histogram',
  'read_pileup',
  'cluster_pileup',
  'splice_junctions',
];

const layoutKey = (sourceId: string | null | undefined, kind: string): string =>
  `${sourceId ?? ''}|${kind}`;

export class GenomeBrowser {
  readonly bus = new ViewportBus();
  private readonly host: HTMLElement;
  private readonly sessionId: string;
  private readonly initialLayout: TrackLayoutEntry[] | null;
  private toolbar!: HTMLElement;
  private overviewHost!: HTMLElement;
  private rulerHost!: HTMLElement;
  private trackHost!: HTMLElement;
  private emptyPlaceholder!: HTMLElement;
  private browser!: HTMLElement;
  private contigs: ContigInfo[] = [];
  private manifest: SessionManifest | null = null;
  private tracks: MountedTrack[] = [];
  private currentContig: ContigInfo | null = null;
  private rerenderTimer: number | null = null;
  private resizeObserver: ResizeObserver | null = null;
  private detachPanZoom: (() => void) | null = null;
  private showLabels = readLabelsPref();
  private searchTimer: number | null = null;
  private searchAbort: AbortController | null = null;
  private datasetBtn!: HTMLButtonElement;
  private optionsBtn!: HTMLButtonElement;
  private trackCountStatus!: HTMLElement;
  private popover: DatasetManagerPopover | null = null;
  private optionsPopover: OptionsPopover | null = null;
  private settingsPopover: TrackSettingsPanel | null = null;
  private settingsAnchor: MountedTrack | null = null;
  private persistTimer: number | null = null;
  private dragSourceTrack: MountedTrack | null = null;
  private options: BrowserOptions = { ...DEFAULT_BROWSER_OPTIONS };

  constructor(opts: GenomeBrowserOptions) {
    this.host = opts.host;
    this.sessionId = opts.sessionId;
    this.initialLayout = opts.initialLayout ?? null;
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

    await Promise.all([this.loadManifest(), this.loadContigs()]);
    await this.loadAvailableTracks();
    this.applyPersistedLayout();
    this.refreshTrackStackOrder();
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
    if (this.persistTimer !== null) {
      window.clearTimeout(this.persistTimer);
      this.persistTimer = null;
    }
    this.searchAbort?.abort();
    this.searchAbort = null;
    this.resizeObserver?.disconnect();
    this.resizeObserver = null;
    this.detachPanZoom?.();
    this.detachPanZoom = null;
    this.popover?.dispose();
    this.popover = null;
    this.optionsPopover?.dispose();
    this.optionsPopover = null;
    this.settingsPopover?.dispose();
    this.settingsPopover = null;
    this.settingsAnchor = null;
    for (const track of this.tracks) track.cancel?.abort();
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
    const entries = await fetchJson<TrackEntry[]>(
      `/api/tracks?session=${encodeURIComponent(this.sessionId)}`,
    );
    // Stable canonical order: KIND_ORDER groups, then per-kind insertion
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
        const mounted = this.mountTrackPanel({
          entry,
          meta,
          visible: true,
          collapsed: false,
          heightPx: defaultHeight,
          displayOrder: order++,
          style: {},
          filter: {},
        });
        this.tracks.push(mounted);
      } catch (err) {
        console.warn(`failed to load track ${entry.kind}/${entry.binding_id}`, err);
      }
    }
  }

  private mountTrackPanel(args: {
    entry: TrackEntry;
    meta: TrackMetadata;
    visible: boolean;
    collapsed: boolean;
    heightPx: number;
    displayOrder: number;
    style: Record<string, unknown>;
    filter: Record<string, unknown>;
  }): MountedTrack {
    const { entry, meta } = args;
    const panel = document.createElement('div');
    panel.className = 'track';
    panel.dataset.bindingId = entry.binding_id;
    panel.dataset.kind = entry.kind;
    panel.draggable = true;

    const header = document.createElement('div');
    header.className = 'track-header';

    const dragHandle = document.createElement('span');
    dragHandle.className = 'track-handle';
    dragHandle.textContent = '⋮⋮';
    dragHandle.title = 'Drag to reorder';

    const collapseBtn = document.createElement('button');
    collapseBtn.type = 'button';
    collapseBtn.className = 'track-collapse-btn';
    collapseBtn.title = args.collapsed ? 'Expand track' : 'Collapse track';
    collapseBtn.textContent = args.collapsed ? '▸' : '▾';

    const labelEl = document.createElement('span');
    labelEl.className = 'track-header-label';
    labelEl.textContent = entry.label;

    const statusEl = document.createElement('span');
    statusEl.className = 'track-header-status';
    statusEl.textContent = '—';

    const settingsBtn = document.createElement('button');
    settingsBtn.type = 'button';
    settingsBtn.className = 'track-settings-btn';
    settingsBtn.textContent = '⚙';
    settingsBtn.title = 'Style and filter controls';

    const hideBtn = document.createElement('button');
    hideBtn.type = 'button';
    hideBtn.className = 'track-hide-btn';
    hideBtn.textContent = '👁';
    hideBtn.title = 'Hide track (re-enable from the Datasets menu)';

    header.appendChild(dragHandle);
    header.appendChild(collapseBtn);
    header.appendChild(labelEl);
    header.appendChild(statusEl);
    header.appendChild(settingsBtn);
    header.appendChild(hideBtn);

    const body = document.createElement('div');
    body.className = 'track-body';

    const resizeHandle = document.createElement('div');
    resizeHandle.className = 'track-resize-handle';
    resizeHandle.title = 'Drag to resize';

    panel.appendChild(header);
    panel.appendChild(body);
    panel.appendChild(resizeHandle);

    const mounted: MountedTrack = {
      entry,
      meta,
      panel,
      headerLabelEl: labelEl,
      statusEl,
      bodyHost: body,
      collapseBtn,
      settingsBtn,
      hideBtn,
      visible: args.visible,
      collapsed: args.collapsed,
      heightPx: args.heightPx,
      displayOrder: args.displayOrder,
      style: { ...args.style },
      filter: { ...args.filter },
    };

    collapseBtn.addEventListener('click', () => {
      this.setCollapsed(mounted, !mounted.collapsed);
    });
    hideBtn.addEventListener('click', () => {
      this.setVisible(mounted, false);
    });
    settingsBtn.addEventListener('click', () => {
      this.toggleSettingsPanel(mounted);
    });
    this.attachReorderDrag(mounted);
    this.attachResize(mounted, resizeHandle);

    return mounted;
  }

  // --------------------------------------------------------------------
  // Layout state — persistence + apply
  // --------------------------------------------------------------------

  private snapshotLayout(): TrackLayoutEntry[] {
    return this.tracks.map((t) => {
      const entry: TrackLayoutEntry = {
        source_id: t.entry.source_id ?? '',
        kind: t.entry.kind,
        visible: t.visible,
        display_order: t.displayOrder,
        height_px: Math.round(t.heightPx),
        collapsed: t.collapsed,
      };
      if (Object.keys(t.style).length > 0) entry.style = { ...t.style };
      if (Object.keys(t.filter).length > 0) entry.filter = { ...t.filter };
      return entry;
    });
  }

  private applyPersistedLayout(): void {
    // Browser-wide options are loaded independently of track layout.
    const storedOptions = readOptionsFromStorage(this.sessionId);
    if (storedOptions) this.options = { ...this.options, ...storedOptions };

    let entries: TrackLayoutEntry[] | null = this.initialLayout;
    if (!entries || entries.length === 0) {
      entries = readLayoutFromStorage(this.sessionId);
    }
    if (!entries || entries.length === 0) return;
    const byKey = new Map<string, TrackLayoutEntry>();
    for (const e of entries) {
      byKey.set(layoutKey(e.source_id || null, e.kind), e);
    }
    for (const t of this.tracks) {
      const e = byKey.get(layoutKey(t.entry.source_id, t.entry.kind));
      if (!e) continue;
      t.visible = e.visible;
      t.collapsed = e.collapsed;
      t.displayOrder = e.display_order;
      if (
        Number.isFinite(e.height_px) &&
        e.height_px >= MIN_TRACK_HEIGHT
      ) {
        t.heightPx = Math.min(MAX_TRACK_HEIGHT, e.height_px);
      }
      if (e.style && typeof e.style === 'object') t.style = { ...e.style };
      if (e.filter && typeof e.filter === 'object') t.filter = { ...e.filter };
      t.collapseBtn.textContent = t.collapsed ? '▸' : '▾';
      t.collapseBtn.title = t.collapsed ? 'Expand track' : 'Collapse track';
    }
  }

  private schedulePersistLayout(): void {
    if (this.persistTimer !== null) {
      window.clearTimeout(this.persistTimer);
    }
    this.persistTimer = window.setTimeout(() => {
      this.persistTimer = null;
      this.persistLayoutNow();
    }, LAYOUT_PERSIST_DEBOUNCE_MS);
  }

  private persistLayoutNow(): void {
    const entries = this.snapshotLayout();
    writeLayoutToStorage(this.sessionId, entries);
    writeOptionsToStorage(this.sessionId, this.options);
    const slug = this.manifest?.saved_as ?? null;
    if (slug) {
      const payload: Record<string, unknown> = {
        track_layout: entries,
        options: { ...this.options },
      };
      fetchJsonMethod<unknown>(
        `/api/saved-sessions/${encodeURIComponent(slug)}/layout`,
        'PATCH',
        payload,
      ).catch((err) => {
        console.warn('failed to PATCH saved-session layout', err);
      });
    }
  }

  // --------------------------------------------------------------------
  // Per-track mutations
  // --------------------------------------------------------------------

  private setVisible(track: MountedTrack, visible: boolean): void {
    if (track.visible === visible) return;
    track.visible = visible;
    this.refreshTrackStackOrder();
    this.updateTrackCountStatus();
    this.refreshPopoverIfOpen();
    this.schedulePersistLayout();
    if (visible) this.scheduleRender();
  }

  private setCollapsed(track: MountedTrack, collapsed: boolean): void {
    if (track.collapsed === collapsed) return;
    track.collapsed = collapsed;
    track.collapseBtn.textContent = collapsed ? '▸' : '▾';
    track.collapseBtn.title = collapsed ? 'Expand track' : 'Collapse track';
    if (collapsed) {
      track.bodyHost.style.height = `${COLLAPSED_BODY_HEIGHT}px`;
      track.bodyHost.style.overflow = 'hidden';
    } else {
      track.bodyHost.style.height = '';
      track.bodyHost.style.overflow = '';
      this.scheduleRender();
    }
    this.schedulePersistLayout();
  }

  // --------------------------------------------------------------------
  // Drag-to-reorder
  // --------------------------------------------------------------------

  private attachReorderDrag(track: MountedTrack): void {
    const panel = track.panel;
    panel.addEventListener('dragstart', (e) => {
      this.dragSourceTrack = track;
      panel.classList.add('track-dragging');
      e.dataTransfer?.setData('text/plain', track.entry.binding_id);
      if (e.dataTransfer) e.dataTransfer.effectAllowed = 'move';
    });
    panel.addEventListener('dragend', () => {
      panel.classList.remove('track-dragging');
      this.trackHost
        .querySelectorAll('.track-drop-target')
        .forEach((el) => el.classList.remove('track-drop-target'));
      this.dragSourceTrack = null;
    });
    panel.addEventListener('dragover', (e) => {
      if (!this.dragSourceTrack || this.dragSourceTrack === track) return;
      e.preventDefault();
      if (e.dataTransfer) e.dataTransfer.dropEffect = 'move';
      panel.classList.add('track-drop-target');
    });
    panel.addEventListener('dragleave', () => {
      panel.classList.remove('track-drop-target');
    });
    panel.addEventListener('drop', (e) => {
      e.preventDefault();
      panel.classList.remove('track-drop-target');
      if (!this.dragSourceTrack || this.dragSourceTrack === track) return;
      this.reorderTrack(this.dragSourceTrack, track);
    });
  }

  private reorderTrack(moved: MountedTrack, target: MountedTrack): void {
    const visible = this.visibleSortedTracks();
    const movedIdx = visible.indexOf(moved);
    const targetIdx = visible.indexOf(target);
    if (movedIdx === -1 || targetIdx === -1) return;
    visible.splice(movedIdx, 1);
    visible.splice(targetIdx, 0, moved);
    visible.forEach((t, i) => {
      t.displayOrder = i;
    });
    // Hidden tracks retain their existing order values; they don't
    // participate in the visible sequence but persist for re-show.
    this.refreshTrackStackOrder();
    this.schedulePersistLayout();
  }

  // --------------------------------------------------------------------
  // Drag-to-resize
  // --------------------------------------------------------------------

  private attachResize(track: MountedTrack, handle: HTMLElement): void {
    let startY = 0;
    let startH = 0;
    let active = false;

    const onMove = (e: PointerEvent): void => {
      if (!active) return;
      const dy = e.clientY - startY;
      const next = Math.max(
        MIN_TRACK_HEIGHT,
        Math.min(MAX_TRACK_HEIGHT, startH + dy),
      );
      track.heightPx = next;
      // Live re-render is heavy; just resize the body and re-render on
      // pointerup. We resize the SVG via re-render to keep glyph scaling
      // consistent. Cheap path: just update body min-height so the
      // visual feedback is immediate.
      track.bodyHost.style.minHeight = `${next}px`;
    };
    const onUp = (e: PointerEvent): void => {
      if (!active) return;
      active = false;
      handle.releasePointerCapture(e.pointerId);
      window.removeEventListener('pointermove', onMove);
      window.removeEventListener('pointerup', onUp);
      track.bodyHost.style.minHeight = '';
      this.scheduleRender();
      this.schedulePersistLayout();
    };
    handle.addEventListener('pointerdown', (e) => {
      if (e.button !== 0) return;
      active = true;
      startY = e.clientY;
      startH = track.heightPx;
      handle.setPointerCapture(e.pointerId);
      window.addEventListener('pointermove', onMove);
      window.addEventListener('pointerup', onUp);
      e.preventDefault();
    });
  }

  // --------------------------------------------------------------------
  // Render-stack ordering
  // --------------------------------------------------------------------

  private visibleSortedTracks(): MountedTrack[] {
    return this.tracks
      .filter((t) => t.visible)
      .sort((a, b) => a.displayOrder - b.displayOrder);
  }

  private refreshTrackStackOrder(): void {
    const visible = this.visibleSortedTracks();
    // Detach all panels and re-append in current visible order; hidden
    // panels are removed from the DOM but kept alive in this.tracks.
    while (this.trackHost.firstChild) {
      this.trackHost.removeChild(this.trackHost.firstChild);
    }
    for (const t of visible) {
      this.trackHost.appendChild(t.panel);
    }
    this.emptyPlaceholder.hidden = visible.length > 0;
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
    const total = this.tracks.length;
    const visible = this.tracks.filter((t) => t.visible).length;
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
    const referenceBindings: BindingRow[] = this.tracks
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
    for (const t of this.tracks) {
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
          const t = this.tracks.find((x) => x.entry.binding_id === binding_id);
          if (t) this.setVisible(t, visible);
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

  private toBindingRow(t: MountedTrack): BindingRow {
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
          this.schedulePersistLayout();
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
  // Per-track gear popover
  // --------------------------------------------------------------------

  private toggleSettingsPanel(track: MountedTrack): void {
    if (this.settingsPopover && this.settingsAnchor === track) {
      this.closeSettingsPanel();
      return;
    }
    this.closeSettingsPanel();
    this.openSettingsPanel(track);
  }

  private openSettingsPanel(track: MountedTrack): void {
    this.settingsAnchor = track;
    this.settingsPopover = new TrackSettingsPanel({
      anchor: track.settingsBtn,
      kind: track.entry.kind,
      label: track.entry.label,
      meta: track.meta,
      style: { ...track.style },
      filter: { ...track.filter },
      onStyleChange: (style) => {
        track.style = { ...style };
        this.restyleTrack(track);
        this.schedulePersistLayout();
      },
      onFilterChange: (filter) => {
        const before = track.filter;
        track.filter = { ...filter };
        if (pushdownFiltersChanged(before, track.filter)) {
          // Server-side filter — invalidate the cache so the next
          // render fetches fresh data, then schedule. The 60ms render
          // debounce absorbs slider drag.
          track.lastFetched = undefined;
          this.scheduleRender();
        } else {
          this.restyleTrack(track);
        }
        this.schedulePersistLayout();
      },
      onReset: () => {
        const before = track.filter;
        track.style = {};
        track.filter = {};
        if (pushdownFiltersChanged(before, track.filter)) {
          track.lastFetched = undefined;
          this.scheduleRender();
        } else {
          this.restyleTrack(track);
        }
        this.schedulePersistLayout();
      },
      onClose: () => {
        this.closeSettingsPanel();
      },
    });
    this.settingsPopover.mount(document.body);
  }

  private closeSettingsPanel(): void {
    this.settingsPopover?.dispose();
    this.settingsPopover = null;
    this.settingsAnchor = null;
  }

  /** Re-run the renderer against `track.lastFetched` so style/filter
   *  changes apply without a server round-trip. Falls back to the
   *  scheduler when the cache is empty (e.g. before first render). */
  private restyleTrack(track: MountedTrack): void {
    if (!track.visible || track.collapsed) return;
    const cached = track.lastFetched;
    if (!cached) {
      this.scheduleRender();
      return;
    }
    const locus = this.bus.locus;
    const widthPx = Math.max(200, this.host.clientWidth - 40);
    const expectedKey = `${locus.contig}|${locus.start}|${locus.end}|${widthPx}`;
    if (cached.locusKey !== expectedKey) {
      // Viewport moved since the last fetch — schedule a full render.
      this.scheduleRender();
      return;
    }
    const heightPx = Math.max(MIN_TRACK_HEIGHT, Math.round(track.heightPx));
    const svg = ensureSvg(track.bodyHost, widthPx, heightPx);
    const ctx = {
      svg,
      widthPx,
      heightPx,
      xScale: xScale([locus.start, locus.end], widthPx),
      meta: track.meta,
      showLabels: this.showLabels,
      style: track.style,
      filter: track.filter,
    };
    const renderer = getRenderer(track.entry.kind);
    if (!renderer) return;
    try {
      renderer.render(cached.table, cached.mode, ctx);
    } catch (err) {
      console.warn(
        `restyle failed for ${track.entry.kind}/${track.entry.binding_id}`,
        err,
      );
    }
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
    this.closeSettingsPanel();
    const previous = this.snapshotLayout();
    for (const t of this.tracks) {
      t.cancel?.abort();
      t.panel.remove();
    }
    this.tracks = [];
    await this.loadManifest();
    await this.loadAvailableTracks();
    this.mergeLayoutAfterReload(previous);
    this.refreshTrackStackOrder();
    this.updateTrackCountStatus();
    this.refreshPopoverIfOpen();
    this.schedulePersistLayout();
    this.scheduleRender();
  }

  private mergeLayoutAfterReload(previous: TrackLayoutEntry[]): void {
    const byKey = new Map<string, TrackLayoutEntry>();
    for (const e of previous) {
      byKey.set(layoutKey(e.source_id || null, e.kind), e);
    }
    // Maximum displayOrder we've seen so we can extend it for fresh
    // bindings that weren't in the previous snapshot.
    let maxOrder = previous.reduce(
      (m, e) => (e.display_order > m ? e.display_order : m),
      -1,
    );
    for (const t of this.tracks) {
      const e = byKey.get(layoutKey(t.entry.source_id, t.entry.kind));
      if (e) {
        t.visible = e.visible;
        t.collapsed = e.collapsed;
        t.displayOrder = e.display_order;
        if (
          Number.isFinite(e.height_px) &&
          e.height_px >= MIN_TRACK_HEIGHT
        ) {
          t.heightPx = Math.min(MAX_TRACK_HEIGHT, e.height_px);
        }
        if (e.style && typeof e.style === 'object') t.style = { ...e.style };
        if (e.filter && typeof e.filter === 'object') t.filter = { ...e.filter };
        t.collapseBtn.textContent = t.collapsed ? '▸' : '▾';
        t.collapseBtn.title = t.collapsed
          ? 'Expand track'
          : 'Collapse track';
      } else {
        // New binding: slot it in at the end of its kind cluster.
        maxOrder += 1;
        t.displayOrder = this.computeInsertOrder(t, byKey, maxOrder);
      }
    }
  }

  private computeInsertOrder(
    track: MountedTrack,
    previousByKey: Map<string, TrackLayoutEntry>,
    fallback: number,
  ): number {
    // Find the largest display_order in `previousByKey` whose kindRank
    // is <= this track's kindRank. Place new binding right after that.
    const rank = kindRank(track.entry.kind);
    let bestOrder = -1;
    for (const e of previousByKey.values()) {
      if (kindRank(e.kind) <= rank && e.display_order > bestOrder) {
        bestOrder = e.display_order;
      }
    }
    // Push downstream entries by 1 to make room. This is small (≤10s of
    // tracks) so the O(N) shift is fine.
    if (bestOrder === -1) return fallback;
    const insertAt = bestOrder + 1;
    for (const t of this.tracks) {
      if (t === track) continue;
      if (t.displayOrder >= insertAt) {
        t.displayOrder += 1;
      }
    }
    return insertAt;
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
    document.addEventListener('mousedown', (e) => {
      if (!wrap.contains(e.target as Node)) hide();
    });

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
    window.addEventListener('mousemove', (e) => {
      if (!dragStart) return;
      centerAt(e.clientX);
    });
    window.addEventListener('mouseup', () => {
      dragStart = null;
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
    const locus = this.bus.locus;
    if (!locus.contig) return;
    const widthPx = Math.max(200, this.host.clientWidth - 40);
    this.renderOverview(widthPx, locus);
    this.renderRuler(widthPx, locus);

    const locusKey = `${locus.contig}|${locus.start}|${locus.end}|${widthPx}`;
    for (const track of this.visibleSortedTracks()) {
      if (track.collapsed) {
        // Collapsed tracks render the header only; clear any prior SVG
        // so the panel doesn't keep stale glyphs in DOM (which would
        // also surface in Save SVG).
        while (track.bodyHost.firstChild) {
          track.bodyHost.removeChild(track.bodyHost.firstChild);
        }
        track.statusEl.textContent = 'collapsed';
        continue;
      }
      track.cancel?.abort();
      track.cancel = new AbortController();
      const heightPx = Math.max(MIN_TRACK_HEIGHT, Math.round(track.heightPx));
      const svg = ensureSvg(track.bodyHost, widthPx, heightPx);
      const ctx = {
        svg,
        widthPx,
        heightPx,
        xScale: xScale([locus.start, locus.end], widthPx),
        meta: track.meta,
        showLabels: this.showLabels,
        style: track.style,
        filter: track.filter,
      };

      try {
        const minMapq = readNumberFilter(track.filter, 'min_mapq');
        const { table, mode } = await fetchTrackData(
          track.entry.kind,
          {
            session: this.sessionId,
            binding: track.entry.binding_id,
            contig: locus.contig,
            start: locus.start,
            end: locus.end,
            viewport_px: widthPx,
            min_mapq: minMapq > 0 ? minMapq : undefined,
          },
          track.cancel.signal,
        );
        const renderer = getRenderer(track.entry.kind);
        if (!renderer) continue;
        renderer.render(table, mode as TrackMode, ctx);
        track.lastFetched = { table, mode: mode as TrackMode, locusKey };
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
    const visible = this.visibleSortedTracks().filter((t) => !t.collapsed);
    const widthPx = Math.max(200, this.host.clientWidth - 40);
    const panels = visible.map((t) => t.panel);
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

function kindRank(kind: string): number {
  const idx = KIND_ORDER.indexOf(kind);
  return idx === -1 ? KIND_ORDER.length : idx;
}

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

function readLayoutFromStorage(sessionId: string): TrackLayoutEntry[] | null {
  try {
    const raw = window.localStorage.getItem(layoutStorageKey(sessionId));
    if (!raw) return null;
    const parsed = JSON.parse(raw);
    if (!Array.isArray(parsed)) return null;
    return parsed as TrackLayoutEntry[];
  } catch {
    return null;
  }
}

function writeLayoutToStorage(
  sessionId: string,
  entries: TrackLayoutEntry[],
): void {
  try {
    window.localStorage.setItem(
      layoutStorageKey(sessionId),
      JSON.stringify(entries),
    );
  } catch {
    // ignored — storage may be disabled
  }
}

function readOptionsFromStorage(sessionId: string): BrowserOptions | null {
  try {
    const raw = window.localStorage.getItem(optionsStorageKey(sessionId));
    if (!raw) return null;
    const parsed = JSON.parse(raw);
    if (!parsed || typeof parsed !== 'object') return null;
    const out: BrowserOptions = { ...DEFAULT_BROWSER_OPTIONS };
    if (typeof parsed.clip_svg === 'boolean') out.clip_svg = parsed.clip_svg;
    return out;
  } catch {
    return null;
  }
}

function writeOptionsToStorage(
  sessionId: string,
  options: BrowserOptions,
): void {
  try {
    window.localStorage.setItem(
      optionsStorageKey(sessionId),
      JSON.stringify(options),
    );
  } catch {
    // ignored — storage may be disabled
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


/** True when any pushdown-routed filter key differs between the two
 *  filter dicts (including a transition between present-and-absent).
 *  The route between restyle vs refetch on filter change depends on
 *  this signal — non-pushdown changes (palettes, allowlists already in
 *  the cached payload) get the cheap client-side restyle. */
function pushdownFiltersChanged(
  before: Record<string, unknown>,
  after: Record<string, unknown>,
): boolean {
  for (const key of PUSHDOWN_FILTER_KEYS) {
    if (!shallowEqual(before[key], after[key])) return true;
  }
  return false;
}


function shallowEqual(a: unknown, b: unknown): boolean {
  if (a === b) return true;
  if (a === undefined || b === undefined) return false;
  if (a === null || b === null) return false;
  return JSON.stringify(a) === JSON.stringify(b);
}


/** Coerce a filter dict value to a finite number, defaulting to 0. */
function readNumberFilter(
  filter: Record<string, unknown>,
  key: string,
): number {
  const v = filter[key];
  if (typeof v === 'number' && Number.isFinite(v)) return v;
  if (typeof v === 'string') {
    const parsed = Number(v);
    if (Number.isFinite(parsed)) return parsed;
  }
  return 0;
}
