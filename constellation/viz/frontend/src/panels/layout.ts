// Panel layout state: what a browser remembers about each of its panels
// (shown or hidden, where in the order, how tall, collapsed, and the
// per-panel style / filter overrides), and how that is stored.
//
// Lifted out of the genome browser unchanged in behavior. The functions
// work on any record with the layout-bearing fields (`LayoutPanel`), so
// a host keeps its own richer panel objects and passes them straight in;
// anything that touches the DOM is left to the host through a callback.
//
// The entry shape and the storage keys are a persisted format — they are
// what saved-session TOMLs and each user's localStorage already hold.

import { fetchJsonMethod } from '../engine/arrow_client';

/** One panel's persisted layout record. */
export interface LayoutEntry {
  /** "" for a panel that belongs to no attached source. */
  source_id: string;
  kind: string;
  visible: boolean;
  display_order: number;
  height_px: number;
  collapsed: boolean;
  style?: Record<string, unknown>;
  filter?: Record<string, unknown>;
}

/** The layout-bearing part of a mounted panel. */
export interface LayoutPanel {
  entry: { source_id: string | null; kind: string };
  visible: boolean;
  collapsed: boolean;
  heightPx: number;
  displayOrder: number;
  style: Record<string, unknown>;
  filter: Record<string, unknown>;
}

export const MIN_PANEL_HEIGHT = 24;
export const MAX_PANEL_HEIGHT = 800;

/** Layout is keyed by (source, kind), not by binding id: binding ids are
 *  index-based and shift when a source is added or removed. */
export const layoutKey = (
  sourceId: string | null | undefined,
  kind: string,
): string => `${sourceId ?? ''}|${kind}`;

export function snapshotLayout(panels: readonly LayoutPanel[]): LayoutEntry[] {
  return panels.map((t) => {
    const entry: LayoutEntry = {
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

function indexByKey(entries: readonly LayoutEntry[]): Map<string, LayoutEntry> {
  const byKey = new Map<string, LayoutEntry>();
  for (const e of entries) {
    byKey.set(layoutKey(e.source_id || null, e.kind), e);
  }
  return byKey;
}

/** Copy a stored entry onto its panel. A height below the minimum is
 *  ignored; one above the maximum is capped. */
function restore<P extends LayoutPanel>(t: P, e: LayoutEntry): void {
  t.visible = e.visible;
  t.collapsed = e.collapsed;
  t.displayOrder = e.display_order;
  if (Number.isFinite(e.height_px) && e.height_px >= MIN_PANEL_HEIGHT) {
    t.heightPx = Math.min(MAX_PANEL_HEIGHT, e.height_px);
  }
  if (e.style && typeof e.style === 'object') t.style = { ...e.style };
  if (e.filter && typeof e.filter === 'object') t.filter = { ...e.filter };
}

/** Apply stored entries to the panels they match. `onRestored` runs for
 *  each panel that had an entry, so the host can sync its chrome. */
export function applyLayout<P extends LayoutPanel>(
  panels: readonly P[],
  entries: readonly LayoutEntry[],
  onRestored: (panel: P) => void = () => {},
): void {
  const byKey = indexByKey(entries);
  for (const t of panels) {
    const e = byKey.get(layoutKey(t.entry.source_id, t.entry.kind));
    if (!e) continue;
    restore(t, e);
    onRestored(t);
  }
}

/** After the panel list was rebuilt (a source was added or removed):
 *  restore each surviving panel from the snapshot taken beforehand, and
 *  slot each new one at the end of its kind's group. `rankOf` gives a
 *  kind's default position among kinds. */
export function mergeLayoutAfterReload<P extends LayoutPanel>(
  panels: readonly P[],
  previous: readonly LayoutEntry[],
  rankOf: (kind: string) => number,
  onRestored: (panel: P) => void = () => {},
): void {
  const byKey = indexByKey(previous);
  // Maximum displayOrder we've seen so we can extend it for fresh
  // bindings that weren't in the previous snapshot.
  let maxOrder = previous.reduce(
    (m, e) => (e.display_order > m ? e.display_order : m),
    -1,
  );
  for (const t of panels) {
    const e = byKey.get(layoutKey(t.entry.source_id, t.entry.kind));
    if (e) {
      restore(t, e);
      onRestored(t);
    } else {
      // New binding: slot it in at the end of its kind cluster.
      maxOrder += 1;
      t.displayOrder = computeInsertOrder(t, byKey, maxOrder, panels, rankOf);
    }
  }
}

/** Display order for a panel with no stored entry: right after the last
 *  stored entry whose kind ranks at or before its own.
 *
 *  Shifts every other panel at or past that slot down by one. Panels
 *  later in the list have not been restored yet when this runs, so their
 *  shifted value is overwritten by their stored one — which can leave
 *  two panels sharing a display order. The host's stable sort over the
 *  panel list still shows them in the right sequence. Kept as is. */
export function computeInsertOrder<P extends LayoutPanel>(
  panel: P,
  previousByKey: ReadonlyMap<string, LayoutEntry>,
  fallback: number,
  panels: readonly P[],
  rankOf: (kind: string) => number,
): number {
  // Find the largest display_order in `previousByKey` whose rank is <=
  // this panel's rank. Place the new panel right after that.
  const rank = rankOf(panel.entry.kind);
  let bestOrder = -1;
  for (const e of previousByKey.values()) {
    if (rankOf(e.kind) <= rank && e.display_order > bestOrder) {
      bestOrder = e.display_order;
    }
  }
  // Push downstream entries by 1 to make room. This is small (≤10s of
  // panels) so the O(N) shift is fine.
  if (bestOrder === -1) return fallback;
  const insertAt = bestOrder + 1;
  for (const t of panels) {
    if (t === panel) continue;
    if (t.displayOrder >= insertAt) {
      t.displayOrder += 1;
    }
  }
  return insertAt;
}

export function visibleSorted<P extends LayoutPanel>(panels: readonly P[]): P[] {
  return panels
    .filter((t) => t.visible)
    .sort((a, b) => a.displayOrder - b.displayOrder);
}

/** Move `moved` to `target`'s position among the visible panels and
 *  renumber them 0..n-1. Hidden panels keep their order values; they
 *  don't take part in the visible sequence but persist for re-show.
 *  Returns false when either panel is not visible. */
export function reorderVisible<P extends LayoutPanel>(
  panels: readonly P[],
  moved: P,
  target: P,
): boolean {
  const visible = visibleSorted(panels);
  const movedIdx = visible.indexOf(moved);
  const targetIdx = visible.indexOf(target);
  if (movedIdx === -1 || targetIdx === -1) return false;
  visible.splice(movedIdx, 1);
  visible.splice(targetIdx, 0, moved);
  visible.forEach((t, i) => {
    t.displayOrder = i;
  });
  return true;
}

// ----------------------------------------------------------------------
// Storage
// ----------------------------------------------------------------------

/** Where a browser's layout and browser-wide options are kept. */
export interface LayoutStore<O> {
  loadLayout(): LayoutEntry[] | null;
  loadOptions(): O | null;
  /** Persist both together. There is one writer on purpose: the remote
   *  copy is a whole-file rewrite on the server, so layout and options
   *  sent as separate requests could lose an update to each other. */
  save(layout: LayoutEntry[], options: O): void;
}

export interface LayoutStoreConfig<O> {
  /** localStorage key prefix, e.g. `constellation.genome`. */
  namespace: string;
  sessionId: string;
  /** Validate a stored options blob; null rejects it. */
  parseOptions(raw: unknown): O | null;
  /** Slug of the saved session this browser was opened from, if any. When
   *  set, each save is also PATCHed to the saved-session file. */
  savedSlug(): string | null;
}

/** localStorage-backed store, mirrored to the saved-session TOML when the
 *  session has one. Storage failures (disabled, full, private mode) are
 *  swallowed: layout persistence is a convenience. */
export function createLayoutStore<O extends object>(
  config: LayoutStoreConfig<O>,
): LayoutStore<O> {
  const layoutStorageKey = `${config.namespace}.layout.${config.sessionId}`;
  const optionsStorageKey = `${config.namespace}.options.${config.sessionId}`;

  return {
    loadLayout(): LayoutEntry[] | null {
      try {
        const raw = window.localStorage.getItem(layoutStorageKey);
        if (!raw) return null;
        const parsed = JSON.parse(raw);
        if (!Array.isArray(parsed)) return null;
        return parsed as LayoutEntry[];
      } catch {
        return null;
      }
    },

    loadOptions(): O | null {
      try {
        const raw = window.localStorage.getItem(optionsStorageKey);
        if (!raw) return null;
        const parsed = JSON.parse(raw);
        if (!parsed || typeof parsed !== 'object') return null;
        return config.parseOptions(parsed);
      } catch {
        return null;
      }
    },

    save(layout: LayoutEntry[], options: O): void {
      try {
        window.localStorage.setItem(layoutStorageKey, JSON.stringify(layout));
      } catch {
        // ignored — storage may be disabled
      }
      try {
        window.localStorage.setItem(optionsStorageKey, JSON.stringify(options));
      } catch {
        // ignored — storage may be disabled
      }
      const slug = config.savedSlug();
      if (slug) {
        const payload: Record<string, unknown> = {
          track_layout: layout,
          options: { ...options },
        };
        fetchJsonMethod<unknown>(
          `/api/saved-sessions/${encodeURIComponent(slug)}/layout`,
          'PATCH',
          payload,
        ).catch((err) => {
          console.warn('failed to PATCH saved-session layout', err);
        });
      }
    },
  };
}
