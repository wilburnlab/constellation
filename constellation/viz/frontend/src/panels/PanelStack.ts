// PanelStack — a vertical, reorderable stack of panels.
//
// Owns everything about the stack that is the same whatever the panels
// show: their order, drag-to-reorder, drag-to-resize, hide and collapse,
// the gear popover, when a change needs a refetch rather than a redraw,
// the status line, and persisting the layout.
//
// It is a component its host drives, not a framework. The host decides
// which panels exist, owns its own toolbar and viewport chrome, and
// calls `render()`; the stack calls back only through `PanelDriver` (to
// fetch and draw one panel for the host's current view) and two
// notifications. It knows no panel kind: what a kind needs is asked of
// its `PanelKind` descriptor.

import { ensureSvg } from '../engine/svg_layer';
import { Panel, PanelInit } from './Panel';
import { SettingsPanel } from './SettingsPanel';
import { PanelKind, UNKNOWN_KIND_ORDER, pushdownChanged, unitFor } from './kind';
import {
  LayoutEntry,
  LayoutStore,
  MAX_PANEL_HEIGHT,
  MIN_PANEL_HEIGHT,
  applyLayout,
  mergeLayoutAfterReload,
  reorderVisible,
  snapshotLayout,
  visibleSorted,
} from './layout';
import { Bag, SettingsSchema } from './settings_schema';

const COLLAPSED_BODY_HEIGHT = 0;
const LAYOUT_PERSIST_DEBOUNCE_MS = 200;

/** A snapshot of the host's viewport, taken once per render so every
 *  panel is fetched and drawn for the same view. `key` identifies the
 *  view: cached data is reused only while it is unchanged. Hosts extend
 *  this with whatever their panels need (a locus, a time window, …). */
export interface PanelView {
  key: string;
  /** Width, in pixels, each panel is drawn at. */
  widthPx: number;
}

/** How the stack gets one panel onto the screen. `D` is whatever the
 *  host's fetch returns and its draw consumes. */
export interface PanelDriver<V extends PanelView, D> {
  /** The current view, or null when there is nothing to show yet. */
  view(): V | null;
  fetch(panel: Panel<D>, view: V, signal: AbortSignal): Promise<D>;
  /** Draw `data` into the panel's SVG. Returns the number of items drawn
   *  from, for the status line. */
  draw(
    panel: Panel<D>,
    data: D,
    view: V,
    svg: SVGSVGElement,
    size: { widthPx: number; heightPx: number },
  ): number;
}

export interface PanelStackConfig<V extends PanelView, D, O> {
  /** Element the visible panels are appended to, in order. */
  stackHost: HTMLElement;
  /** Shown instead when every panel is hidden. */
  emptyPlaceholder: HTMLElement;
  driver: PanelDriver<V, D>;
  /** The descriptor for a kind, or null when none is registered. */
  kindOf(kind: string): PanelKind | null;
  /** Settings shown for a kind that declares none. */
  fallbackSettings: SettingsSchema;
  /** Host state the settings schemas may consult. */
  settingsHost?(): Bag;
  store: LayoutStore<O>;
  /** The host's browser-wide options, saved alongside the layout. */
  options(): O;
  /** Ask the host to render soon. The host owns the debounce: it has
   *  chrome of its own to draw before the panels. */
  requestRender(): void;
  /** Which panels are visible changed. */
  onChanged(): void;
}

export class PanelStack<V extends PanelView, D, O> {
  private readonly config: PanelStackConfig<V, D, O>;
  private panelList: Panel<D>[] = [];
  private settingsPopover: SettingsPanel | null = null;
  private settingsAnchor: Panel<D> | null = null;
  private persistTimer: number | null = null;
  private dragSource: Panel<D> | null = null;

  constructor(config: PanelStackConfig<V, D, O>) {
    this.config = config;
  }

  /** Every panel, visible or not, in the order they were added. */
  get panels(): readonly Panel<D>[] {
    return this.panelList;
  }

  // --------------------------------------------------------------------
  // Membership
  // --------------------------------------------------------------------

  add(init: PanelInit): Panel<D> {
    const panel = new Panel<D>(init);
    panel.collapseBtn.addEventListener('click', () => {
      this.setCollapsed(panel, !panel.collapsed);
    });
    panel.hideBtn.addEventListener('click', () => {
      this.setVisible(panel, false);
    });
    panel.settingsBtn.addEventListener('click', () => {
      this.toggleSettings(panel);
    });
    this.attachReorderDrag(panel);
    this.attachResize(panel);
    this.panelList.push(panel);
    return panel;
  }

  /** Remove every panel (their in-flight fetches are aborted). */
  clear(): void {
    for (const t of this.panelList) {
      t.cancel?.abort();
      t.element.remove();
    }
    this.panelList = [];
  }

  dispose(): void {
    if (this.persistTimer !== null) {
      window.clearTimeout(this.persistTimer);
      this.persistTimer = null;
    }
    this.closeSettings();
    for (const t of this.panelList) t.cancel?.abort();
  }

  // --------------------------------------------------------------------
  // Layout state — persistence + apply
  // --------------------------------------------------------------------

  snapshot(): LayoutEntry[] {
    return snapshotLayout(this.panelList);
  }

  applyLayout(entries: readonly LayoutEntry[]): void {
    applyLayout(this.panelList, entries, (t) => t.syncCollapseButton());
  }

  /** After the panels were rebuilt: restore survivors from `previous` and
   *  slot new ones into their kind's group. */
  mergeAfterReload(previous: readonly LayoutEntry[]): void {
    mergeLayoutAfterReload(
      this.panelList,
      previous,
      (kind) => this.config.kindOf(kind)?.order ?? UNKNOWN_KIND_ORDER,
      (t) => t.syncCollapseButton(),
    );
  }

  schedulePersist(): void {
    if (this.persistTimer !== null) {
      window.clearTimeout(this.persistTimer);
    }
    this.persistTimer = window.setTimeout(() => {
      this.persistTimer = null;
      this.config.store.save(this.snapshot(), this.config.options());
    }, LAYOUT_PERSIST_DEBOUNCE_MS);
  }

  // --------------------------------------------------------------------
  // Per-panel mutations
  // --------------------------------------------------------------------

  setVisible(panel: Panel<D>, visible: boolean): void {
    if (panel.visible === visible) return;
    panel.visible = visible;
    this.refreshOrder();
    this.config.onChanged();
    this.schedulePersist();
    if (visible) this.config.requestRender();
  }

  setCollapsed(panel: Panel<D>, collapsed: boolean): void {
    if (panel.collapsed === collapsed) return;
    panel.collapsed = collapsed;
    panel.syncCollapseButton();
    if (collapsed) {
      panel.bodyHost.style.height = `${COLLAPSED_BODY_HEIGHT}px`;
      panel.bodyHost.style.overflow = 'hidden';
    } else {
      panel.bodyHost.style.height = '';
      panel.bodyHost.style.overflow = '';
      this.config.requestRender();
    }
    this.schedulePersist();
  }

  // --------------------------------------------------------------------
  // Drag-to-reorder
  // --------------------------------------------------------------------

  private attachReorderDrag(track: Panel<D>): void {
    const panel = track.element;
    panel.addEventListener('dragstart', (e) => {
      this.dragSource = track;
      panel.classList.add('track-dragging');
      e.dataTransfer?.setData('text/plain', track.entry.binding_id);
      if (e.dataTransfer) e.dataTransfer.effectAllowed = 'move';
    });
    panel.addEventListener('dragend', () => {
      panel.classList.remove('track-dragging');
      this.config.stackHost
        .querySelectorAll('.track-drop-target')
        .forEach((el) => el.classList.remove('track-drop-target'));
      this.dragSource = null;
    });
    panel.addEventListener('dragover', (e) => {
      if (!this.dragSource || this.dragSource === track) return;
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
      if (!this.dragSource || this.dragSource === track) return;
      this.reorder(this.dragSource, track);
    });
  }

  private reorder(moved: Panel<D>, target: Panel<D>): void {
    if (!reorderVisible(this.panelList, moved, target)) return;
    this.refreshOrder();
    this.schedulePersist();
  }

  // --------------------------------------------------------------------
  // Drag-to-resize
  // --------------------------------------------------------------------

  private attachResize(track: Panel<D>): void {
    const handle = track.resizeHandle;
    let startY = 0;
    let startH = 0;
    let active = false;

    const onMove = (e: PointerEvent): void => {
      if (!active) return;
      const dy = e.clientY - startY;
      const next = Math.max(
        MIN_PANEL_HEIGHT,
        Math.min(MAX_PANEL_HEIGHT, startH + dy),
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
      this.config.requestRender();
      this.schedulePersist();
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
  // Order
  // --------------------------------------------------------------------

  visibleSorted(): Panel<D>[] {
    return visibleSorted(this.panelList);
  }

  /** Re-append the visible panels in display order. Hidden panels leave
   *  the DOM but stay in the stack. */
  refreshOrder(): void {
    const visible = this.visibleSorted();
    const host = this.config.stackHost;
    while (host.firstChild) {
      host.removeChild(host.firstChild);
    }
    for (const t of visible) {
      host.appendChild(t.element);
    }
    this.config.emptyPlaceholder.hidden = visible.length > 0;
  }

  // --------------------------------------------------------------------
  // Gear popover
  // --------------------------------------------------------------------

  private toggleSettings(track: Panel<D>): void {
    if (this.settingsPopover && this.settingsAnchor === track) {
      this.closeSettings();
      return;
    }
    this.closeSettings();
    this.openSettings(track);
  }

  private openSettings(track: Panel<D>): void {
    const kind = this.config.kindOf(track.entry.kind);
    this.settingsAnchor = track;
    this.settingsPopover = new SettingsPanel({
      anchor: track.settingsBtn,
      kind: track.entry.kind,
      label: track.entry.label,
      schema: kind?.settings ?? this.config.fallbackSettings,
      meta: track.meta,
      style: { ...track.style },
      filter: { ...track.filter },
      host: this.config.settingsHost?.(),
      onStyleChange: (style) => {
        track.style = { ...style };
        this.restyle(track);
        this.schedulePersist();
      },
      onFilterChange: (filter) => {
        const before = track.filter;
        track.filter = { ...filter };
        if (pushdownChanged(kind, before, track.filter)) {
          // Server-side filter — invalidate the cache so the next
          // render fetches fresh data, then ask for one. The host's
          // render debounce absorbs rapid edits.
          track.lastFetched = undefined;
          this.config.requestRender();
        } else {
          this.restyle(track);
        }
        this.schedulePersist();
      },
      onReset: () => {
        const before = track.filter;
        track.style = {};
        track.filter = {};
        if (pushdownChanged(kind, before, track.filter)) {
          track.lastFetched = undefined;
          this.config.requestRender();
        } else {
          this.restyle(track);
        }
        this.schedulePersist();
      },
      onClose: () => {
        this.closeSettings();
      },
    });
    this.settingsPopover.mount(document.body);
  }

  closeSettings(): void {
    this.settingsPopover?.dispose();
    this.settingsPopover = null;
    this.settingsAnchor = null;
  }

  // --------------------------------------------------------------------
  // Drawing
  // --------------------------------------------------------------------

  /** Redraw one panel from its cached data, so a style or client-side
   *  filter change needs no request. Falls back to a full render when
   *  there is no cache or the view has moved since it was filled. */
  restyle(track: Panel<D>): void {
    if (!track.visible || track.collapsed) return;
    const cached = track.lastFetched;
    if (!cached) {
      this.config.requestRender();
      return;
    }
    const view = this.config.driver.view();
    if (!view || cached.viewKey !== view.key) {
      // Viewport moved since the last fetch — schedule a full render.
      this.config.requestRender();
      return;
    }
    const heightPx = Math.max(MIN_PANEL_HEIGHT, Math.round(track.heightPx));
    const svg = ensureSvg(track.bodyHost, view.widthPx, heightPx);
    try {
      this.config.driver.draw(track, cached.data, view, svg, {
        widthPx: view.widthPx,
        heightPx,
      });
    } catch (err) {
      console.warn(
        `restyle failed for ${track.entry.kind}/${track.entry.binding_id}`,
        err,
      );
    }
  }

  /** Fetch and draw every visible panel for the host's current view, one
   *  after another in display order. */
  async render(): Promise<void> {
    const view = this.config.driver.view();
    if (!view) return;
    const widthPx = view.widthPx;

    for (const track of this.visibleSorted()) {
      if (track.collapsed) {
        // Collapsed panels show the header only; clear any prior SVG so
        // the panel doesn't keep stale glyphs in the DOM (which would
        // also surface in an SVG export).
        while (track.bodyHost.firstChild) {
          track.bodyHost.removeChild(track.bodyHost.firstChild);
        }
        track.statusEl.textContent = 'collapsed';
        continue;
      }
      track.cancel?.abort();
      track.cancel = new AbortController();
      const heightPx = Math.max(MIN_PANEL_HEIGHT, Math.round(track.heightPx));
      const svg = ensureSvg(track.bodyHost, widthPx, heightPx);

      try {
        const data = await this.config.driver.fetch(track, view, track.cancel.signal);
        const count = this.config.driver.draw(track, data, view, svg, {
          widthPx,
          heightPx,
        });
        track.lastFetched = { data, viewKey: view.key };
        track.statusEl.textContent =
          count === 0
            ? '— no data in window'
            : `showing ${count.toLocaleString()} ${unitFor(this.config.kindOf(track.entry.kind), count)}`;
      } catch (err) {
        if ((err as Error).name === 'AbortError') continue;
        console.warn(`render failed for ${track.entry.kind}`, err);
        track.statusEl.textContent = 'render failed';
      }
    }
  }
}
