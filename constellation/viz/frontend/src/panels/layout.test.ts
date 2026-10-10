import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import {
  LayoutEntry,
  LayoutPanel,
  MAX_PANEL_HEIGHT,
  applyLayout,
  computeInsertOrder,
  createLayoutStore,
  layoutKey,
  mergeLayoutAfterReload,
  reorderVisible,
  snapshotLayout,
  visibleSorted,
} from './layout';

function panel(source_id: string | null, kind: string, displayOrder: number, extra: Partial<LayoutPanel> = {}): LayoutPanel {
  return {
    entry: { source_id, kind },
    visible: true,
    collapsed: false,
    heightPx: 80,
    displayOrder,
    style: {},
    filter: {},
    ...extra,
  };
}

function entry(source_id: string, kind: string, display_order: number, extra: Partial<LayoutEntry> = {}): LayoutEntry {
  return { source_id, kind, visible: true, display_order, height_px: 80, collapsed: false, ...extra };
}

const RANK: Record<string, number> = { ref: 0, cov: 1, reads: 2, junc: 3 };
const rankOf = (kind: string): number => RANK[kind] ?? 99;

describe('layoutKey', () => {
  it('treats a missing source and the empty string alike', () => {
    expect(layoutKey(null, 'cov')).toBe('|cov');
    expect(layoutKey(undefined, 'cov')).toBe('|cov');
    expect(layoutKey('', 'cov')).toBe('|cov');
    expect(layoutKey('src-1', 'cov')).toBe('src-1|cov');
  });
});

describe('snapshotLayout', () => {
  it('records each panel, rounding heights and omitting empty style/filter', () => {
    const panels = [
      panel(null, 'ref', 0, { heightPx: 24.4 }),
      panel('src-1', 'cov', 1, { visible: false, collapsed: true, style: { a: 1 }, filter: { b: [2] } }),
    ];
    expect(snapshotLayout(panels)).toEqual([
      { source_id: '', kind: 'ref', visible: true, display_order: 0, height_px: 24, collapsed: false },
      { source_id: 'src-1', kind: 'cov', visible: false, display_order: 1, height_px: 80, collapsed: true,
        style: { a: 1 }, filter: { b: [2] } },
    ]);
  });

  it('copies style and filter rather than aliasing them', () => {
    const p = panel('s', 'cov', 0, { style: { a: 1 } });
    const [snap] = snapshotLayout([p]);
    p.style.a = 2;
    expect(snap.style).toEqual({ a: 1 });
  });
});

describe('applyLayout', () => {
  it('restores matching panels and reports each one', () => {
    const a = panel(null, 'ref', 0);
    const b = panel('src-1', 'cov', 1);
    const untouched = panel('src-2', 'cov', 2);
    const restored: LayoutPanel[] = [];
    applyLayout(
      [a, b, untouched],
      [
        entry('', 'ref', 5, { visible: false }),
        entry('src-1', 'cov', 3, { collapsed: true, height_px: 120, style: { k: 'v' }, filter: { f: 1 } }),
        entry('src-9', 'cov', 0),
      ],
      (p) => restored.push(p),
    );
    expect(a).toMatchObject({ visible: false, displayOrder: 5 });
    expect(b).toMatchObject({ collapsed: true, displayOrder: 3, heightPx: 120, style: { k: 'v' }, filter: { f: 1 } });
    expect(untouched).toMatchObject({ displayOrder: 2, heightPx: 80 });
    expect(restored).toEqual([a, b]);
  });

  it('ignores a height below the minimum and caps one above the maximum', () => {
    const low = panel('s', 'cov', 0);
    const high = panel('s', 'reads', 1);
    const nan = panel('s', 'junc', 2);
    applyLayout([low, high, nan], [
      entry('s', 'cov', 0, { height_px: 5 }),
      entry('s', 'reads', 1, { height_px: 5000 }),
      entry('s', 'junc', 2, { height_px: Number.NaN }),
    ]);
    expect(low.heightPx).toBe(80);
    expect(high.heightPx).toBe(MAX_PANEL_HEIGHT);
    expect(nan.heightPx).toBe(80);
  });

  it('keeps existing style and filter when the entry carries none', () => {
    const p = panel('s', 'cov', 0, { style: { keep: true } });
    applyLayout([p], [entry('s', 'cov', 4)]);
    expect(p.style).toEqual({ keep: true });
    expect(p.displayOrder).toBe(4);
  });
});

describe('visibleSorted / reorderVisible', () => {
  it('sorts visible panels by display order, stably', () => {
    const a = panel(null, 'ref', 1);
    const b = panel('s', 'cov', 0);
    const hidden = panel('s', 'reads', -1, { visible: false });
    const tie1 = panel('s', 'junc', 1);
    expect(visibleSorted([a, b, hidden, tie1])).toEqual([b, a, tie1]);
  });

  it('moves a panel to the target slot and renumbers the visible ones', () => {
    const panels = [panel(null, 'ref', 0), panel('s', 'cov', 1), panel('s', 'reads', 2), panel('s', 'junc', 3)];
    const hidden = panel('s2', 'cov', 7, { visible: false });
    expect(reorderVisible([...panels, hidden], panels[3], panels[1])).toBe(true);
    expect(panels.map((p) => p.displayOrder)).toEqual([0, 2, 3, 1]);
    expect(hidden.displayOrder).toBe(7);
  });

  it('refuses when either panel is hidden', () => {
    const a = panel(null, 'ref', 0);
    const hidden = panel('s', 'cov', 1, { visible: false });
    expect(reorderVisible([a, hidden], a, hidden)).toBe(false);
    expect(reorderVisible([a, hidden], hidden, a)).toBe(false);
    expect(a.displayOrder).toBe(0);
  });
});

describe('mergeLayoutAfterReload', () => {
  it('restores survivors and appends a new panel of the last kind', () => {
    const previous = [entry('', 'ref', 0), entry('a', 'cov', 1, { visible: false })];
    const ref = panel(null, 'ref', 0);
    const cov = panel('a', 'cov', 1);
    const junc = panel('a', 'junc', 2);
    mergeLayoutAfterReload([ref, cov, junc], previous, rankOf);
    expect(cov.visible).toBe(false);
    expect([ref, cov, junc].map((p) => p.displayOrder)).toEqual([0, 1, 2]);
  });

  it('slots a new panel after the last stored panel of its kind group', () => {
    // Stored: ref, cov(a), junc(a). New: cov(b) belongs after cov(a).
    const previous = [entry('', 'ref', 0), entry('a', 'cov', 1), entry('a', 'junc', 2)];
    const ref = panel(null, 'ref', 0);
    const covA = panel('a', 'cov', 1);
    const covB = panel('b', 'cov', 2);
    const junc = panel('a', 'junc', 3);
    mergeLayoutAfterReload([ref, covA, covB, junc], previous, rankOf);
    expect(visibleSorted([ref, covA, covB, junc])).toEqual([ref, covA, covB, junc]);
    expect(covB.displayOrder).toBe(2);
  });

  it('can leave two panels sharing a display order (known quirk, sort is stable)', () => {
    // A later panel is shifted to make room, then overwritten by its own
    // stored order — which equals the slot the new panel just took.
    const previous = [entry('', 'ref', 0), entry('a', 'cov', 1), entry('a', 'junc', 2)];
    const ref = panel(null, 'ref', 0);
    const covA = panel('a', 'cov', 1);
    const covB = panel('b', 'cov', 2);
    const junc = panel('a', 'junc', 3);
    mergeLayoutAfterReload([ref, covA, covB, junc], previous, rankOf);
    expect([ref, covA, covB, junc].map((p) => p.displayOrder)).toEqual([0, 1, 2, 2]);
  });

  it('gives a new panel the next free order when nothing ranks before it', () => {
    const previous = [entry('a', 'reads', 0), entry('a', 'junc', 1)];
    const reads = panel('a', 'reads', 0);
    const junc = panel('a', 'junc', 1);
    const ref = panel(null, 'ref', 2);
    mergeLayoutAfterReload([reads, junc, ref], previous, rankOf);
    expect(ref.displayOrder).toBe(2);
  });

  it('reports only the restored panels', () => {
    const restored: LayoutPanel[] = [];
    const kept = panel('a', 'cov', 0);
    const fresh = panel('b', 'cov', 1);
    mergeLayoutAfterReload([kept, fresh], [entry('a', 'cov', 0)], rankOf, (p) => restored.push(p));
    expect(restored).toEqual([kept]);
  });
});

describe('computeInsertOrder', () => {
  it('returns the fallback when no stored entry ranks at or before the kind', () => {
    const p = panel('b', 'ref', 0);
    const byKey = new Map([['a|cov', entry('a', 'cov', 4)]]);
    expect(computeInsertOrder(p, byKey, 9, [p], rankOf)).toBe(9);
  });

  it('shifts the panels at or past the insertion slot', () => {
    const byKey = new Map([['|ref', entry('', 'ref', 0)], ['a|cov', entry('a', 'cov', 1)]]);
    const fresh = panel('b', 'cov', 5);
    const later = panel('a', 'junc', 2);
    const earlier = panel(null, 'ref', 0);
    expect(computeInsertOrder(fresh, byKey, 9, [earlier, fresh, later], rankOf)).toBe(2);
    expect(later.displayOrder).toBe(3);
    expect(earlier.displayOrder).toBe(0);
    expect(fresh.displayOrder).toBe(5); // the caller assigns the result
  });
});

describe('createLayoutStore', () => {
  interface Opts { clip: boolean }
  const parseOptions = (raw: unknown): Opts | null => {
    const o = raw as Record<string, unknown>;
    return typeof o.clip === 'boolean' ? { clip: o.clip } : null;
  };
  let slug: string | null;
  let fetchMock: ReturnType<typeof vi.fn>;
  const store = () =>
    createLayoutStore<Opts>({ namespace: 'ns.test', sessionId: 'sid', parseOptions, savedSlug: () => slug });

  beforeEach(() => {
    window.localStorage.clear();
    slug = null;
    fetchMock = vi.fn(async () => ({ ok: true, status: 200, json: async () => ({}) }));
    vi.stubGlobal('fetch', fetchMock);
  });
  afterEach(() => {
    vi.unstubAllGlobals();
    vi.restoreAllMocks();
  });

  it('returns null for missing, malformed or wrongly shaped stored values', () => {
    expect(store().loadLayout()).toBeNull();
    expect(store().loadOptions()).toBeNull();
    window.localStorage.setItem('ns.test.layout.sid', '{not json');
    window.localStorage.setItem('ns.test.options.sid', '[1,2]');
    expect(store().loadLayout()).toBeNull();
    expect(store().loadOptions()).toBeNull();
    window.localStorage.setItem('ns.test.layout.sid', '{"an":"object"}');
    window.localStorage.setItem('ns.test.options.sid', '"a string"');
    expect(store().loadLayout()).toBeNull();
    expect(store().loadOptions()).toBeNull();
  });

  it('round-trips layout and options under namespaced, per-session keys', () => {
    const layout = [entry('a', 'cov', 0, { style: { k: 1 } })];
    store().save(layout, { clip: true });
    expect(JSON.parse(window.localStorage.getItem('ns.test.layout.sid')!)).toEqual(layout);
    expect(JSON.parse(window.localStorage.getItem('ns.test.options.sid')!)).toEqual({ clip: true });
    expect(store().loadLayout()).toEqual(layout);
    expect(store().loadOptions()).toEqual({ clip: true });
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it('PATCHes layout and options in one request when the session is saved', () => {
    slug = 'my session/1';
    const layout = [entry('a', 'cov', 0)];
    store().save(layout, { clip: false });
    expect(fetchMock).toHaveBeenCalledTimes(1);
    const [url, init] = fetchMock.mock.calls[0];
    expect(url).toBe('/api/saved-sessions/my%20session%2F1/layout');
    expect(init.method).toBe('PATCH');
    expect(JSON.parse(init.body)).toEqual({ track_layout: layout, options: { clip: false } });
  });

  it('survives a failing remote save and a throwing localStorage', async () => {
    slug = 's';
    fetchMock.mockRejectedValueOnce(new Error('offline'));
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    vi.spyOn(Storage.prototype, 'setItem').mockImplementation(() => {
      throw new Error('quota');
    });
    expect(() => store().save([entry('a', 'cov', 0)], { clip: true })).not.toThrow();
    await Promise.resolve();
    await Promise.resolve();
    expect(warn).toHaveBeenCalled();
  });
});
