// PanelStack on its own, under a host that is not a genome browser: the
// view is a time window, the data is a list of numbers, and the two
// panel kinds belong to no modality. The genome browser's use of the
// stack is pinned separately by its black-box host test.

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { PanelInit } from './Panel';
import { PanelDriver, PanelStack, PanelView } from './PanelStack';
import { PanelKind } from './kind';
import { LayoutEntry, LayoutStore } from './layout';
import { SettingsSchema } from './settings_schema';

interface TimeView extends PanelView {
  t0: number;
  t1: number;
}

interface ToyOptions {
  compact: boolean;
}

const TRACE: PanelKind = {
  kind: 'trace',
  order: 10,
  unit: ['point', 'points'],
  pushdown: { smoothing: (v) => (typeof v === 'number' ? String(v) : undefined) },
  settings: {
    sections: [
      {
        title: 'Look',
        controls: [
          { type: 'number', target: 'style', key: 'radius', label: 'Radius', default: 2, min: 1, max: 9, step: 1 },
          { type: 'number', target: 'filter', key: 'floor', label: 'Floor', default: 0, min: 0, max: 9, step: 1 },
          { type: 'number', target: 'filter', key: 'smoothing', label: 'Smoothing', default: 0, min: 0, max: 9, step: 1 },
        ],
      },
    ],
  },
};

const MARKS: PanelKind = { kind: 'marks', order: 20, unit: ['mark', 'marks'] };

const KINDS: Record<string, PanelKind> = { trace: TRACE, marks: MARKS };

const FALLBACK: SettingsSchema = {
  sections: [{ title: 'Nothing', controls: [{ type: 'note', text: 'no controls' }] }],
};

interface Harness {
  stack: PanelStack<TimeView, number[], ToyOptions>;
  stackHost: HTMLElement;
  placeholder: HTMLElement;
  /** `kind/binding@key` per fetch, in call order. */
  fetched: string[];
  /** `binding:radius` per draw, in call order. */
  drawn: string[];
  saved: Array<{ layout: LayoutEntry[]; options: ToyOptions }>;
  requestRender: ReturnType<typeof vi.fn>;
  onChanged: ReturnType<typeof vi.fn>;
  view: TimeView | null;
  data: Record<string, number[] | Error>;
}

function harness(): Harness {
  const stackHost = document.createElement('div');
  const placeholder = document.createElement('div');
  placeholder.hidden = true;
  document.body.append(stackHost, placeholder);

  const h = {
    stackHost,
    placeholder,
    fetched: [] as string[],
    drawn: [] as string[],
    saved: [] as Array<{ layout: LayoutEntry[]; options: ToyOptions }>,
    requestRender: vi.fn(),
    onChanged: vi.fn(),
    view: { key: '0-10@400', widthPx: 400, t0: 0, t1: 10 } as TimeView | null,
    data: {} as Record<string, number[] | Error>,
  } as unknown as Harness;

  const driver: PanelDriver<TimeView, number[]> = {
    view: () => h.view,
    fetch: async (panel, view) => {
      h.fetched.push(`${panel.entry.kind}/${panel.entry.binding_id}@${view.key}`);
      const out = h.data[panel.entry.binding_id] ?? [];
      if (out instanceof Error) throw out;
      return out;
    },
    draw: (panel, data, _view, svg) => {
      const radius = Number(panel.style.radius ?? 2);
      const floor = Number(panel.filter.floor ?? 0);
      const kept = data.filter((v) => v >= floor);
      svg.replaceChildren(
        ...kept.map((v) => {
          const c = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
          c.setAttribute('cy', String(v));
          c.setAttribute('r', String(radius));
          return c;
        }),
      );
      h.drawn.push(`${panel.entry.binding_id}:${radius}`);
      return kept.length;
    },
  };

  const store: LayoutStore<ToyOptions> = {
    loadLayout: () => null,
    loadOptions: () => null,
    save: (layout, options) => h.saved.push({ layout, options }),
  };

  h.stack = new PanelStack<TimeView, number[], ToyOptions>({
    stackHost,
    emptyPlaceholder: placeholder,
    driver,
    kindOf: (kind) => KINDS[kind] ?? null,
    fallbackSettings: FALLBACK,
    store,
    options: () => ({ compact: true }),
    requestRender: h.requestRender,
    onChanged: h.onChanged,
  });
  return h;
}

function init(kind: string, binding: string, order: number, source: string | null = 'src-a'): PanelInit {
  return {
    entry: { kind, binding_id: binding, label: `${kind} ${binding}`, source_id: source },
    meta: {},
    visible: true,
    collapsed: false,
    heightPx: 60,
    displayOrder: order,
    style: {},
    filter: {},
  };
}

function order(h: Harness): string[] {
  return Array.from(h.stackHost.children, (el) => (el as HTMLElement).dataset.bindingId ?? '');
}

function status(h: Harness, binding: string): string {
  const panel = h.stack.panels.find((p) => p.entry.binding_id === binding)!;
  return panel.statusEl.textContent ?? '';
}

function setNumber(label: string, value: string): void {
  const row = Array.from(document.querySelectorAll<HTMLElement>('.track-settings-popover .settings-row')).find(
    (r) => r.querySelector('.settings-row-label')?.textContent === label,
  );
  if (!row) throw new Error(`no settings row ${label}`);
  const input = row.querySelector('input') as HTMLInputElement;
  input.value = value;
  input.dispatchEvent(new Event('input', { bubbles: true }));
  input.dispatchEvent(new Event('change', { bubbles: true }));
}

beforeEach(() => {
  vi.useFakeTimers();
});

afterEach(() => {
  vi.useRealTimers();
  document.body.replaceChildren();
});

describe('PanelStack membership and order', () => {
  it('stacks visible panels by display order and builds the standard chrome', () => {
    const h = harness();
    h.stack.add(init('marks', 'm0', 1));
    h.stack.add(init('trace', 't0', 0));
    h.stack.refreshOrder();

    expect(order(h)).toEqual(['t0', 'm0']);
    expect(h.placeholder.hidden).toBe(true);
    const panel = h.stackHost.firstElementChild as HTMLElement;
    expect(panel.className).toBe('track');
    expect(panel.dataset.kind).toBe('trace');
    expect(Array.from(panel.children, (c) => c.className)).toEqual([
      'track-header',
      'track-body',
      'track-resize-handle',
    ]);
    expect(Array.from(panel.querySelector('.track-header')!.children, (c) => c.className)).toEqual([
      'track-handle',
      'track-collapse-btn',
      'track-header-label',
      'track-header-status',
      'track-settings-btn',
      'track-hide-btn',
    ]);
    expect(panel.querySelector('.track-header-label')!.textContent).toBe('trace t0');
  });

  it('hides a panel from its header button and shows the placeholder when none are left', () => {
    const h = harness();
    const only = h.stack.add(init('trace', 't0', 0));
    h.stack.refreshOrder();

    only.hideBtn.click();

    expect(order(h)).toEqual([]);
    expect(h.placeholder.hidden).toBe(false);
    expect(h.stack.panels).toHaveLength(1);
    expect(h.onChanged).toHaveBeenCalledTimes(1);
    // Hiding needs no render; showing again does.
    expect(h.requestRender).not.toHaveBeenCalled();
    h.stack.setVisible(only, true);
    expect(order(h)).toEqual(['t0']);
    expect(h.requestRender).toHaveBeenCalledTimes(1);
  });

  it('saves the layout once, 200 ms after the last change, together with the host options', () => {
    const h = harness();
    const a = h.stack.add(init('trace', 't0', 0));
    const b = h.stack.add(init('marks', 'm0', 1, null));
    h.stack.refreshOrder();

    h.stack.setVisible(a, false);
    vi.advanceTimersByTime(150);
    h.stack.setCollapsed(b, true);
    vi.advanceTimersByTime(199);
    expect(h.saved).toEqual([]);
    vi.advanceTimersByTime(1);

    expect(h.saved).toEqual([
      {
        layout: [
          { source_id: 'src-a', kind: 'trace', visible: false, display_order: 0, height_px: 60, collapsed: false },
          { source_id: '', kind: 'marks', visible: true, display_order: 1, height_px: 60, collapsed: true },
        ],
        options: { compact: true },
      },
    ]);
  });

  it('drops a pending save on dispose', () => {
    const h = harness();
    const a = h.stack.add(init('trace', 't0', 0));
    h.stack.setCollapsed(a, true);
    h.stack.dispose();
    vi.advanceTimersByTime(1000);
    expect(h.saved).toEqual([]);
  });

  it('restores survivors and slots a new panel into its kind group after a rebuild', () => {
    const h = harness();
    h.stack.add(init('trace', 't0', 0));
    const marks = h.stack.add(init('marks', 'm0', 1));
    marks.heightPx = 140;
    marks.collapsed = true;
    const previous = h.stack.snapshot();

    h.stack.clear();
    expect(h.stack.panels).toHaveLength(0);
    // The server now lists a second source's trace; binding ids shifted.
    h.stack.add(init('trace', 't0', 0));
    h.stack.add(init('trace', 't1', 1, 'src-b'));
    const restored = h.stack.add(init('marks', 'm0', 2));
    h.stack.mergeAfterReload(previous);
    h.stack.refreshOrder();

    expect(order(h)).toEqual(['t0', 't1', 'm0']);
    expect(restored.heightPx).toBe(140);
    expect(restored.collapsed).toBe(true);
    expect(restored.collapseBtn.textContent).toBe('▸');
  });
});

describe('PanelStack rendering', () => {
  it('fetches and draws visible panels one after another in display order', async () => {
    const h = harness();
    h.stack.add(init('marks', 'm0', 1));
    h.stack.add(init('trace', 't0', 0));
    const hidden = h.stack.add(init('trace', 't1', 2));
    hidden.visible = false;
    h.stack.refreshOrder();
    h.data = { t0: [1, 2, 3], m0: [5] };

    await h.stack.render();

    expect(h.fetched).toEqual(['trace/t0@0-10@400', 'marks/m0@0-10@400']);
    expect(h.drawn).toEqual(['t0:2', 'm0:2']);
    expect(status(h, 't0')).toBe('showing 3 points');
    expect(status(h, 'm0')).toBe('showing 1 mark');
    const svg = h.stackHost.querySelector('[data-binding-id="t0"] svg.track-canvas')!;
    expect(svg.getAttribute('width')).toBe('400');
    expect(svg.getAttribute('height')).toBe('60');
    expect(svg.querySelectorAll('circle')).toHaveLength(3);
  });

  it('does nothing when the host has no view yet', async () => {
    const h = harness();
    h.stack.add(init('trace', 't0', 0));
    h.stack.refreshOrder();
    h.view = null;
    await h.stack.render();
    expect(h.fetched).toEqual([]);
    expect(status(h, 't0')).toBe('—');
  });

  it('reports an empty window, a failure, and a collapsed panel on the status line', async () => {
    const h = harness();
    h.stack.add(init('trace', 'empty', 0));
    h.stack.add(init('trace', 'broken', 1));
    const folded = h.stack.add(init('trace', 'folded', 2));
    h.stack.refreshOrder();
    h.data = { broken: new Error('boom'), folded: [1] };
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});

    await h.stack.render();
    folded.collapseBtn.click();
    await h.stack.render();

    expect(status(h, 'empty')).toBe('— no data in window');
    expect(status(h, 'broken')).toBe('render failed');
    expect(status(h, 'folded')).toBe('collapsed');
    expect(folded.bodyHost.children).toHaveLength(0);
    expect(folded.collapseBtn.textContent).toBe('▸');
    // The collapsed panel was fetched once, before it was collapsed.
    expect(h.fetched.filter((f) => f.includes('folded'))).toHaveLength(1);
    warn.mockRestore();
  });

  it('leaves the status alone when a fetch is aborted', async () => {
    const h = harness();
    h.stack.add(init('trace', 't0', 0));
    h.stack.refreshOrder();
    const aborted = new Error('aborted');
    aborted.name = 'AbortError';
    h.data = { t0: aborted };
    await h.stack.render();
    expect(status(h, 't0')).toBe('—');
  });
});

describe('PanelStack settings', () => {
  async function opened(): Promise<Harness> {
    const h = harness();
    h.stack.add(init('trace', 't0', 0));
    h.stack.refreshOrder();
    h.data = { t0: [1, 4, 7] };
    await h.stack.render();
    h.fetched.length = 0;
    h.drawn.length = 0;
    h.stack.panels[0].settingsBtn.click();
    return h;
  }

  it('opens the schema the kind declares, and toggles closed from the same gear', async () => {
    const h = await opened();
    const labels = Array.from(
      document.querySelectorAll('.track-settings-popover .settings-row-label'),
      (l) => l.textContent,
    );
    expect(labels).toEqual(['Radius', 'Floor', 'Smoothing']);
    h.stack.panels[0].settingsBtn.click();
    expect(document.querySelector('.track-settings-popover')).toBeNull();
  });

  it('falls back to the host schema for a kind that declares none', () => {
    const h = harness();
    const marks = h.stack.add(init('marks', 'm0', 0));
    marks.settingsBtn.click();
    expect(document.querySelector('.track-settings-popover .settings-section-title')!.textContent).toBe('Nothing');
    h.stack.closeSettings();
  });

  it('redraws from the fetched data on a style change, without a request', async () => {
    const h = await opened();
    setNumber('Radius', '5');
    expect(h.drawn).toEqual(['t0:5']);
    expect(h.fetched).toEqual([]);
    expect(h.requestRender).not.toHaveBeenCalled();
    expect(h.stack.panels[0].style).toEqual({ radius: 5 });
  });

  it('applies a client-side filter from the fetched data', async () => {
    const h = await opened();
    setNumber('Floor', '4');
    expect(h.fetched).toEqual([]);
    expect(h.stackHost.querySelectorAll('circle')).toHaveLength(2);
  });

  it('drops the cache and asks for a render when a pushdown filter changes', async () => {
    const h = await opened();
    setNumber('Smoothing', '3');
    expect(h.drawn).toEqual([]);
    expect(h.stack.panels[0].lastFetched).toBeUndefined();
    expect(h.requestRender).toHaveBeenCalledTimes(1);
  });

  it('asks for a render instead of redrawing when the view moved since the fetch', async () => {
    const h = await opened();
    h.view = { key: '5-15@400', widthPx: 400, t0: 5, t1: 15 };
    setNumber('Radius', '5');
    expect(h.drawn).toEqual([]);
    expect(h.requestRender).toHaveBeenCalledTimes(1);
  });

  it('persists a settings edit', async () => {
    const h = await opened();
    setNumber('Radius', '5');
    vi.advanceTimersByTime(200);
    expect(h.saved).toHaveLength(1);
    expect(h.saved[0].layout[0].style).toEqual({ radius: 5 });
  });
});
