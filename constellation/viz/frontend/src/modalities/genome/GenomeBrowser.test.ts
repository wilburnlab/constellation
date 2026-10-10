// GenomeBrowser — black-box host test.
//
// Mounts the real widget under jsdom against an in-memory fake of the viz
// server and pins what is observable from outside: the requests it makes
// and their order, the track stack it builds, header status text, what it
// persists (localStorage + the saved-session PATCH), and how edits route
// between a client-side restyle and a server refetch.
//
// jsdom has no layout engine, so host width is supplied explicitly, and
// drag-reorder / pointer-resize are driven by synthetic events. CSS and
// real pointer behaviour are outside what this can check.

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

vi.mock('../../engine/export', async (importOriginal) => {
  const actual = await importOriginal<typeof import('../../engine/export')>();
  return { ...actual, downloadSvg: vi.fn() };
});

import { downloadSvg } from '../../engine/export';
import {
  ADDED_SOURCE_ID,
  FakeServer,
  SESSION_ID,
} from './__fixtures__/fake_server';
import { GenomeBrowser } from './GenomeBrowser';

const LAYOUT_KEY = `constellation.genome.layout.${SESSION_ID}`;
const OPTIONS_KEY = `constellation.genome.options.${SESSION_ID}`;
const LABELS_KEY = 'constellation.genome.labels';

const SRC_A = 'src-aaaa0001';
const SRC_B = 'src-bbbb0002';

/** Binding ids in the default (kind-grouped) display order. */
const DEFAULT_ORDER = [
  'reference_sequence',
  'reference',
  'coverage-0',
  'read_pileup-0',
  'cluster_pileup-1',
  'splice_junctions-0',
];
const KIND_OF: Record<string, string> = {
  reference_sequence: 'reference_sequence',
  reference: 'gene_annotation',
  'coverage-0': 'coverage_histogram',
  'read_pileup-0': 'read_pileup',
  'cluster_pileup-1': 'cluster_pileup',
  'splice_junctions-0': 'splice_junctions',
};

interface LayoutEntry {
  source_id: string;
  kind: string;
  visible: boolean;
  display_order: number;
  height_px: number;
  collapsed: boolean;
  style?: Record<string, unknown>;
  filter?: Record<string, unknown>;
}

function entry(
  source_id: string,
  kind: string,
  display_order: number,
  height_px: number,
  extra: Partial<LayoutEntry> = {},
): LayoutEntry {
  return { source_id, kind, visible: true, display_order, height_px, collapsed: false, ...extra };
}

/** The layout a freshly opened fixture session persists. */
function defaultLayout(): LayoutEntry[] {
  return [
    entry('', 'reference_sequence', 0, 24),
    entry('', 'gene_annotation', 1, 60),
    entry(SRC_A, 'coverage_histogram', 2, 80),
    entry(SRC_A, 'read_pileup', 3, 240),
    entry(SRC_B, 'cluster_pileup', 4, 200),
    entry(SRC_A, 'splice_junctions', 5, 80),
  ];
}

// ----------------------------------------------------------------------
// Harness
// ----------------------------------------------------------------------

let server: FakeServer;
let hostWidth = 1240;
let resizeCallbacks: Array<() => void> = [];
const opened: GenomeBrowser[] = [];

class ResizeObserverStub {
  constructor(cb: () => void) {
    resizeCallbacks.push(cb);
  }
  observe(): void {}
  unobserve(): void {}
  disconnect(): void {}
}

/** Advance fake time, letting promise chains started by timers finish. */
async function settle(ms: number): Promise<void> {
  await vi.advanceTimersByTimeAsync(ms);
  for (let i = 0; i < 4; i++) await vi.advanceTimersByTimeAsync(0);
}

async function open(
  opts: { initialLayout?: LayoutEntry[] } = {},
): Promise<{ host: HTMLElement; browser: GenomeBrowser }> {
  const host = document.createElement('div');
  Object.defineProperty(host, 'clientWidth', { configurable: true, get: () => hostWidth });
  document.body.appendChild(host);
  const browser = new GenomeBrowser({
    host,
    sessionId: SESSION_ID,
    initialLayout: opts.initialLayout,
  });
  opened.push(browser);
  await browser.mount();
  await settle(60); // the mount-time render is debounced
  return { host, browser };
}

beforeEach(() => {
  vi.useFakeTimers();
  window.localStorage.clear();
  document.body.replaceChildren();
  hostWidth = 1240;
  resizeCallbacks = [];
  server = new FakeServer();
  vi.stubGlobal('fetch', server.fetch);
  vi.stubGlobal('ResizeObserver', ResizeObserverStub);
  // jsdom does not implement pointer capture.
  HTMLElement.prototype.setPointerCapture = () => {};
  HTMLElement.prototype.releasePointerCapture = () => {};
  vi.mocked(downloadSvg).mockClear();
});

afterEach(() => {
  for (const b of opened.splice(0)) b.dispose();
  vi.useRealTimers();
  vi.unstubAllGlobals();
});

function stack(host: HTMLElement): string[] {
  return Array.from(host.querySelectorAll<HTMLElement>('.track-stack > .track')).map(
    (el) => el.dataset.bindingId ?? '',
  );
}

function track(host: HTMLElement, bindingId: string): HTMLElement {
  const el = host.querySelector<HTMLElement>(`.track[data-binding-id="${bindingId}"]`);
  if (!el) throw new Error(`track ${bindingId} is not in the stack`);
  return el;
}

function status(host: HTMLElement, bindingId: string): string {
  return track(host, bindingId).querySelector('.track-header-status')?.textContent ?? '';
}

function svgOf(host: HTMLElement, bindingId: string): SVGSVGElement | null {
  return track(host, bindingId).querySelector<SVGSVGElement>('svg.track-canvas');
}

function click(el: Element | null): void {
  if (!el) throw new Error('nothing to click');
  (el as HTMLElement).click();
}

function setInput(el: HTMLInputElement | HTMLSelectElement, value: string, event = 'change'): void {
  el.value = value;
  el.dispatchEvent(new Event(event, { bubbles: true }));
}

/** The stored style of the reference annotation track, once the
 *  debounced save has run. */
function storedLayoutAfter(): Record<string, unknown> | undefined {
  vi.advanceTimersByTime(200);
  return storedLayout().find((e) => e.kind === 'gene_annotation' && e.source_id === '')?.style;
}

function storedLayout(): LayoutEntry[] {
  return JSON.parse(window.localStorage.getItem(LAYOUT_KEY) ?? 'null') as LayoutEntry[];
}

function dataUrl(bindingId: string, locus: string, extra = ''): string {
  const [contig, range] = locus.split(':');
  const [start, end] = range.split('-');
  return (
    `${KIND_OF[bindingId] ?? bindingId}?session=${SESSION_ID}&binding=${bindingId}` +
    `&contig=${contig}&start=${start}&end=${end}&viewport_px=${hostWidth - 40}${extra}`
  );
}

function locusInput(host: HTMLElement): HTMLInputElement {
  return host.querySelector<HTMLInputElement>('.toolbar input[placeholder="chr:start-end"]')!;
}

/** Find a row of the open gear popover by its label. */
function settingsRow(label: string): HTMLElement {
  const popover = document.body.querySelector('.track-settings-popover');
  if (!popover) throw new Error('settings popover is not open');
  const rows = Array.from(popover.querySelectorAll<HTMLElement>('.settings-row, .settings-checkbox-row'));
  const hit = rows.find(
    (r) => (r.querySelector('.settings-row-label, span')?.textContent ?? '') === label,
  );
  if (!hit) throw new Error(`no settings row labelled ${label}`);
  return hit;
}

// ----------------------------------------------------------------------
// Mount
// ----------------------------------------------------------------------

describe('GenomeBrowser mount', () => {
  it('loads the session, then metadata and data for every track in display order', async () => {
    await open();
    const meta = (kind: string, binding: string): string =>
      `GET /api/tracks/${kind}/metadata?session=${SESSION_ID}&binding=${binding}`;
    expect(server.calls.map((c) => `${c.method} ${c.url}`)).toEqual([
      `GET /api/sessions/${SESSION_ID}/manifest`,
      `GET /api/sessions/${SESSION_ID}/contigs`,
      `GET /api/tracks?session=${SESSION_ID}`,
      meta('reference_sequence', 'reference_sequence'),
      meta('gene_annotation', 'reference'),
      meta('coverage_histogram', 'coverage-0'),
      meta('read_pileup', 'read_pileup-0'),
      meta('cluster_pileup', 'cluster_pileup-1'),
      meta('splice_junctions', 'splice_junctions-0'),
      ...DEFAULT_ORDER.map((id) => `GET /api/tracks/${dataUrl(id, 'chr1:0-12000').replace('?', '/data?')}`),
    ]);
  });

  it('builds the kind-grouped track stack with labels, heights and status text', async () => {
    const { host } = await open();
    expect(host.classList.contains('genome-browser-root')).toBe(true);
    expect(stack(host)).toEqual(DEFAULT_ORDER);
    expect(
      DEFAULT_ORDER.map((id) => track(host, id).querySelector('.track-header-label')?.textContent),
    ).toEqual([
      'Reference sequence',
      'Annotation (reference)',
      'Coverage (run-a)',
      'Reads (run-a)',
      'Transcript clusters (run-a clusters)',
      'Junctions (run-a)',
    ]);
    expect(DEFAULT_ORDER.map((id) => svgOf(host, id)?.getAttribute('height'))).toEqual([
      '24', '60', '80', '240', '200', '80',
    ]);
    expect(DEFAULT_ORDER.map((id) => svgOf(host, id)?.getAttribute('width'))).toEqual(
      Array(6).fill('1200'),
    );
    expect(DEFAULT_ORDER.map((id) => status(host, id))).toEqual([
      'showing 48 bases',
      'showing 11 features',
      'showing 5 bins',
      'showing 4 reads',
      'showing 4 clusters',
      'showing 4 junctions',
    ]);
    expect(host.querySelector('.track-stack-empty')?.hasAttribute('hidden')).toBe(true);
  });

  it('fills the toolbar from the session', async () => {
    const { host } = await open();
    const contigSelect = host.querySelector<HTMLSelectElement>('.toolbar select')!;
    expect(Array.from(contigSelect.options).map((o) => o.textContent)).toEqual([
      'chr1 (12.00 kb)',
      'chr2 (80.00 kb)',
    ]);
    expect(locusInput(host).value).toBe('chr1:0-12000');
    expect(host.querySelector('.toolbar-right .label-dim')?.textContent).toBe('6 tracks');
    expect(
      Array.from(host.querySelectorAll('.btn-group .icon-btn')).map((b) => b.textContent),
    ).toEqual(['−', '+', 'Fit']);
  });

  it('reports an empty window and uses the singular unit for one row', async () => {
    server.rowLimit.splice_junctions = 0;
    server.rowLimit.read_pileup = 1;
    const { host } = await open();
    expect(status(host, 'splice_junctions-0')).toBe('— no data in window');
    expect(status(host, 'read_pileup-0')).toBe('showing 1 read');
  });

  it('persists nothing until the layout changes', async () => {
    await open();
    await settle(1000);
    expect(window.localStorage.getItem(LAYOUT_KEY)).toBeNull();
    expect(window.localStorage.getItem(OPTIONS_KEY)).toBeNull();
  });
});

// ----------------------------------------------------------------------
// Layout state and persistence
// ----------------------------------------------------------------------

describe('GenomeBrowser layout', () => {
  it('hides a track, updates the count and persists after the debounce', async () => {
    const { host } = await open();
    const mark = server.calls.length;
    click(track(host, 'splice_junctions-0').querySelector('.track-hide-btn'));

    expect(stack(host)).toEqual(DEFAULT_ORDER.slice(0, 5));
    expect(host.querySelector('.toolbar-right .label-dim')?.textContent).toBe('5 / 6 tracks');

    await settle(199);
    expect(window.localStorage.getItem(LAYOUT_KEY)).toBeNull();
    await settle(1);
    const expected = defaultLayout();
    expected[5].visible = false;
    expect(storedLayout()).toEqual(expected);
    expect(JSON.parse(window.localStorage.getItem(OPTIONS_KEY)!)).toEqual({ clip_svg: false });
    // Hiding neither refetches nor, for an unsaved session, PATCHes.
    expect(server.since(mark)).toEqual([]);
  });

  it('shows the empty-state placeholder when every track is hidden', async () => {
    const { host } = await open();
    for (const id of DEFAULT_ORDER) click(track(host, id).querySelector('.track-hide-btn'));
    expect(stack(host)).toEqual([]);
    expect(host.querySelector('.track-stack-empty')?.hasAttribute('hidden')).toBe(false);
    expect(host.querySelector('.toolbar-right .label-dim')?.textContent).toBe('0 / 6 tracks');
  });

  it('PATCHes layout and options together when the session is saved', async () => {
    server.savedAs = 'my-session-1a2b';
    const { host } = await open();
    const mark = server.calls.length;
    click(track(host, 'splice_junctions-0').querySelector('.track-hide-btn'));
    await settle(200);
    const expected = defaultLayout();
    expected[5].visible = false;
    expect(server.since(mark)).toEqual([
      {
        method: 'PATCH',
        url: '/api/saved-sessions/my-session-1a2b/layout',
        body: { track_layout: expected, options: { clip_svg: false } },
      },
    ]);
  });

  it('restores a layout from localStorage', async () => {
    const stored = defaultLayout();
    stored[1].visible = false; // gene_annotation
    stored[2].display_order = 5; // coverage to the bottom
    stored[2].style = { y_scale: 'log' };
    stored[5].display_order = 2; // junctions up
    stored[3].height_px = 120; // read_pileup
    stored[3].collapsed = true;
    window.localStorage.setItem(LAYOUT_KEY, JSON.stringify(stored));
    window.localStorage.setItem(OPTIONS_KEY, JSON.stringify({ clip_svg: true }));

    const { host } = await open();
    expect(stack(host)).toEqual([
      'reference_sequence',
      'splice_junctions-0',
      'read_pileup-0',
      'cluster_pileup-1',
      'coverage-0',
    ]);
    // Collapsed and hidden tracks are not fetched.
    expect(server.dataRequests().map((u) => u.split('?')[0])).toEqual([
      'reference_sequence',
      'splice_junctions',
      'cluster_pileup',
      'coverage_histogram',
    ]);
    expect(status(host, 'read_pileup-0')).toBe('collapsed');
    expect(svgOf(host, 'read_pileup-0')).toBeNull();
    expect(track(host, 'read_pileup-0').querySelector('.track-collapse-btn')?.textContent).toBe('▸');
    // The stored style reached the renderer.
    expect(svgOf(host, 'coverage-0')?.textContent).toContain('(log)');

    click(host.querySelector('.options-btn'));
    const clip = document.body.querySelector<HTMLInputElement>('.options-popover input[type=checkbox]')!;
    expect(clip.checked).toBe(true);
  });

  it('prefers an initial layout over localStorage, but still reads stored options', async () => {
    const stored = defaultLayout();
    stored[1].visible = false;
    window.localStorage.setItem(LAYOUT_KEY, JSON.stringify(stored));
    window.localStorage.setItem(OPTIONS_KEY, JSON.stringify({ clip_svg: true }));

    const initial = defaultLayout();
    initial[5].visible = false;
    const { host } = await open({ initialLayout: initial });

    expect(stack(host)).toEqual(DEFAULT_ORDER.slice(0, 5));
    click(host.querySelector('.options-btn'));
    expect(
      document.body.querySelector<HTMLInputElement>('.options-popover input[type=checkbox]')!.checked,
    ).toBe(true);
  });

  it('ignores a stored height below the minimum and caps one above the maximum', async () => {
    const stored = defaultLayout();
    stored[2].height_px = 5;
    stored[3].height_px = 5000;
    window.localStorage.setItem(LAYOUT_KEY, JSON.stringify(stored));
    const { host } = await open();
    expect(svgOf(host, 'coverage-0')?.getAttribute('height')).toBe('80');
    expect(svgOf(host, 'read_pileup-0')?.getAttribute('height')).toBe('800');
  });

  it('collapses without fetching and refetches on expand', async () => {
    const { host } = await open();
    const collapseBtn = track(host, 'coverage-0').querySelector<HTMLElement>('.track-collapse-btn')!;
    let mark = server.calls.length;
    click(collapseBtn);
    expect(collapseBtn.textContent).toBe('▸');
    expect(collapseBtn.title).toBe('Expand track');
    await settle(300);
    expect(server.dataRequests(mark)).toEqual([]);
    expect(storedLayout()[2].collapsed).toBe(true);

    // The next render skips it and clears its body.
    mark = server.calls.length;
    click(host.querySelector('.btn-group .icon-btn:nth-child(3)')); // Fit (no-op locus) …
    click(host.querySelector('.btn-group .icon-btn:nth-child(2)')); // … then zoom in
    await settle(60);
    expect(server.dataRequests(mark).map((u) => u.split('?')[0])).toEqual([
      'reference_sequence',
      'gene_annotation',
      'read_pileup',
      'cluster_pileup',
      'splice_junctions',
    ]);
    expect(status(host, 'coverage-0')).toBe('collapsed');
    expect(svgOf(host, 'coverage-0')).toBeNull();

    mark = server.calls.length;
    click(collapseBtn);
    expect(collapseBtn.textContent).toBe('▾');
    await settle(60);
    expect(server.dataRequests(mark)).toHaveLength(6);
    expect(status(host, 'coverage-0')).toBe('showing 5 bins');
  });

  it('reorders by drag and drop and renumbers the visible tracks', async () => {
    const { host } = await open();
    const moved = track(host, 'read_pileup-0');
    const target = track(host, 'reference');
    moved.dispatchEvent(new Event('dragstart', { bubbles: true }));
    expect(moved.classList.contains('track-dragging')).toBe(true);
    target.dispatchEvent(new Event('dragover', { bubbles: true, cancelable: true }));
    expect(target.classList.contains('track-drop-target')).toBe(true);
    target.dispatchEvent(new Event('drop', { bubbles: true, cancelable: true }));
    moved.dispatchEvent(new Event('dragend', { bubbles: true }));

    expect(stack(host)).toEqual([
      'reference_sequence',
      'read_pileup-0',
      'reference',
      'coverage-0',
      'cluster_pileup-1',
      'splice_junctions-0',
    ]);
    expect(moved.classList.contains('track-dragging')).toBe(false);
    expect(target.classList.contains('track-drop-target')).toBe(false);

    await settle(200);
    expect(storedLayout().map((e) => [e.kind, e.display_order])).toEqual([
      ['reference_sequence', 0],
      ['gene_annotation', 2],
      ['coverage_histogram', 3],
      ['read_pileup', 1],
      ['cluster_pileup', 4],
      ['splice_junctions', 5],
    ]);
  });

  it('resizes by dragging the bottom handle, within the height bounds', async () => {
    const { host } = await open();
    const handle = track(host, 'coverage-0').querySelector<HTMLElement>('.track-resize-handle')!;
    const pointer = (type: string, clientY: number): MouseEvent =>
      new MouseEvent(type, { bubbles: true, cancelable: true, button: 0, clientY });

    handle.dispatchEvent(pointer('pointerdown', 100));
    window.dispatchEvent(pointer('pointermove', 160));
    window.dispatchEvent(pointer('pointerup', 160));
    await settle(60);
    expect(svgOf(host, 'coverage-0')?.getAttribute('height')).toBe('140');
    await settle(200);
    expect(storedLayout()[2].height_px).toBe(140);

    handle.dispatchEvent(pointer('pointerdown', 100));
    window.dispatchEvent(pointer('pointermove', -5000));
    window.dispatchEvent(pointer('pointerup', -5000));
    await settle(260);
    expect(storedLayout()[2].height_px).toBe(24);

    handle.dispatchEvent(pointer('pointerdown', 100));
    window.dispatchEvent(pointer('pointermove', 9000));
    window.dispatchEvent(pointer('pointerup', 9000));
    await settle(260);
    expect(storedLayout()[2].height_px).toBe(800);
  });

  it('re-renders at the new width when the host is resized', async () => {
    const { host } = await open();
    const mark = server.calls.length;
    hostWidth = 1040;
    for (const cb of resizeCallbacks) cb();
    await settle(60);
    expect(server.dataRequests(mark)).toEqual(
      DEFAULT_ORDER.map((id) => dataUrl(id, 'chr1:0-12000')),
    );
    expect(svgOf(host, 'coverage-0')?.getAttribute('width')).toBe('1000');
  });

  it('persists the options popover toggle', async () => {
    const { host } = await open();
    click(host.querySelector('.options-btn'));
    const clip = document.body.querySelector<HTMLInputElement>('.options-popover input[type=checkbox]')!;
    expect(clip.checked).toBe(false);
    clip.checked = true;
    clip.dispatchEvent(new Event('change', { bubbles: true }));
    await settle(200);
    expect(JSON.parse(window.localStorage.getItem(OPTIONS_KEY)!)).toEqual({ clip_svg: true });
    // A second click on the button closes the popover.
    click(host.querySelector('.options-btn'));
    expect(document.body.querySelector('.options-popover')).toBeNull();
  });
});

// ----------------------------------------------------------------------
// Per-track settings: restyle vs refetch
// ----------------------------------------------------------------------

describe('GenomeBrowser track settings', () => {
  it('restyles from the cached table on a style edit, without a request', async () => {
    const { host } = await open();
    click(track(host, 'read_pileup-0').querySelector('.track-settings-btn'));
    const mark = server.calls.length;

    setInput(settingsRow('Read opacity').querySelector('input')!, '0.5');
    expect(svgOf(host, 'read_pileup-0')?.querySelector('rect')?.getAttribute('opacity')).toBe('0.5');
    await settle(260);
    expect(server.since(mark)).toEqual([]);
    expect(storedLayout()[3].style).toEqual({ read_opacity: 0.5 });
  });

  it('refetches every visible track when a pushdown filter changes', async () => {
    const { host } = await open();
    click(track(host, 'read_pileup-0').querySelector('.track-settings-btn'));
    const mark = server.calls.length;

    setInput(settingsRow('Min MAPQ').querySelector('input')!, '20');
    await settle(59);
    expect(server.dataRequests(mark)).toEqual([]);
    await settle(1);
    expect(server.dataRequests(mark)).toEqual(
      DEFAULT_ORDER.map((id) =>
        dataUrl(id, 'chr1:0-12000', id === 'read_pileup-0' ? '&min_mapq=20' : ''),
      ),
    );
    await settle(200);
    expect(storedLayout()[3].filter).toEqual({ min_mapq: 20 });
  });

  it('restyles on a client-side filter and refetches on reset of a pushdown filter', async () => {
    const { host } = await open();
    click(track(host, 'read_pileup-0').querySelector('.track-settings-btn'));
    setInput(settingsRow('Min MAPQ').querySelector('input')!, '20');
    await settle(60);

    // Unticking a strand is satisfied from the cached payload.
    let mark = server.calls.length;
    const minus = Array.from(
      document.body.querySelectorAll<HTMLElement>('.track-settings-popover label.settings-checkbox-row'),
    ).find((r) => r.querySelector('span')?.textContent === '-')!;
    const box = minus.querySelector<HTMLInputElement>('input')!;
    box.checked = false;
    box.dispatchEvent(new Event('change', { bubbles: true }));
    expect(svgOf(host, 'read_pileup-0')?.querySelector('[data-alignment-id="2"]')).toBeNull();
    await settle(260);
    expect(server.dataRequests(mark)).toEqual([]);
    expect(storedLayout()[3].filter).toEqual({ min_mapq: 20, visible_strands: ['+'] });

    // Reset drops the pushdown key, so it goes back to the server.
    mark = server.calls.length;
    click(document.body.querySelector('.settings-reset-btn'));
    await settle(60);
    expect(server.dataRequests(mark)).toEqual(
      DEFAULT_ORDER.map((id) => dataUrl(id, 'chr1:0-12000')),
    );
    expect(svgOf(host, 'read_pileup-0')?.querySelector('[data-alignment-id="2"]')).not.toBeNull();
    await settle(200);
    const read = storedLayout()[3];
    expect(read.style).toBeUndefined();
    expect(read.filter).toBeUndefined();
  });

  it('switches the cluster view through the server', async () => {
    const { host } = await open();
    click(track(host, 'cluster_pileup-1').querySelector('.track-settings-btn'));
    const mark = server.calls.length;
    setInput(settingsRow('Cluster view').querySelector('select')!, 'members');
    await settle(60);
    expect(server.dataRequests(mark)).toContain(
      dataUrl('cluster_pileup-1', 'chr1:0-12000', '&cluster_view=members'),
    );
    expect(svgOf(host, 'cluster_pileup-1')?.textContent).toContain('members · 4 reads');
    expect(status(host, 'cluster_pileup-1')).toBe('showing 4 clusters');
  });

  it('toggles the gear popover and moves it between tracks', async () => {
    const { host } = await open();
    const readGear = track(host, 'read_pileup-0').querySelector('.track-settings-btn');
    const covGear = track(host, 'coverage-0').querySelector('.track-settings-btn');
    click(readGear);
    expect(document.body.querySelector('.settings-kind')?.textContent).toBe('read_pileup');
    click(covGear);
    expect(document.body.querySelectorAll('.track-settings-popover')).toHaveLength(1);
    expect(document.body.querySelector('.settings-kind')?.textContent).toBe('coverage_histogram');
    click(covGear);
    expect(document.body.querySelector('.track-settings-popover')).toBeNull();
  });
});

// ----------------------------------------------------------------------
// Dataset manager
// ----------------------------------------------------------------------

function popoverTexts(selector: string): Array<string | null> {
  return Array.from(document.body.querySelectorAll(`.dataset-popover ${selector}`)).map(
    (el) => el.textContent,
  );
}

async function addDataset(path: string): Promise<void> {
  const input = document.body.querySelector<HTMLInputElement>('.dataset-popover .path-input-field')!;
  setInput(input, path, 'input');
  click(document.body.querySelector('.dataset-popover .dataset-add-btn'));
  await settle(0);
}

describe('GenomeBrowser dataset manager', () => {
  it('lists the reference and each source with its bindings and warnings', async () => {
    const { host } = await open();
    click(host.querySelector('.dataset-btn'));
    // The reference heads the list; it has no kind badge and no remove button.
    expect(popoverTexts('.dataset-source-label')).toEqual([
      'Reference: test_org@local_import-20260522',
      'run-a',
      'run-a clusters',
    ]);
    expect(popoverTexts('.dataset-readonly-tag')).toEqual(['read-only']);
    expect(document.body.querySelectorAll('.dataset-popover .dataset-remove-btn')).toHaveLength(2);
    expect(popoverTexts('.dataset-kind-badge')).toEqual(['align', 'cluster']);
    expect(popoverTexts('.dataset-warning')).toEqual([
      'assembly OtherAssembly.9 ≠ reference TestAssembly.1',
    ]);
    expect(popoverTexts('.dataset-binding-label')).toEqual([
      'Reference sequence',
      'Annotation (reference)',
      'Coverage (run-a)',
      'Reads (run-a)',
      'Junctions (run-a)',
      'Transcript clusters (run-a clusters)',
    ]);
  });

  it('hides and re-shows a track from its checkbox', async () => {
    const { host } = await open();
    click(host.querySelector('.dataset-btn'));
    const box = (): HTMLInputElement =>
      Array.from(document.body.querySelectorAll<HTMLElement>('.dataset-binding-row'))
        .find((r) => r.textContent === 'Junctions (run-a)')!
        .querySelector('input')!;

    let mark = server.calls.length;
    box().checked = false;
    box().dispatchEvent(new Event('change', { bubbles: true }));
    expect(stack(host)).toEqual(DEFAULT_ORDER.slice(0, 5));
    expect(box().checked).toBe(false); // the popover was rebuilt with the new state
    await settle(60);
    expect(server.dataRequests(mark)).toEqual([]);

    mark = server.calls.length;
    box().checked = true;
    box().dispatchEvent(new Event('change', { bubbles: true }));
    expect(stack(host)).toEqual(DEFAULT_ORDER);
    await settle(60);
    expect(server.dataRequests(mark)).toHaveLength(6);
  });

  it('adds a source and slots its tracks at the end of each kind group', async () => {
    const { host } = await open();
    click(track(host, 'splice_junctions-0').querySelector('.track-hide-btn'));
    click(host.querySelector('.dataset-btn'));
    const mark = server.calls.length;

    await addDataset('/data/run-b/align');

    const meta = (kind: string, binding: string): string =>
      `GET /api/tracks/${kind}/metadata?session=${SESSION_ID}&binding=${binding}`;
    expect(server.since(mark)[0]).toEqual({
      method: 'POST',
      url: `/api/sessions/${SESSION_ID}/sources`,
      body: { path: '/data/run-b/align' },
    });
    expect(server.since(mark).slice(1).map((c) => `${c.method} ${c.url}`)).toEqual([
      `GET /api/sessions/${SESSION_ID}/manifest`,
      `GET /api/tracks?session=${SESSION_ID}`,
      meta('reference_sequence', 'reference_sequence'),
      meta('gene_annotation', 'reference'),
      meta('coverage_histogram', 'coverage-0'),
      meta('coverage_histogram', 'coverage-2'),
      meta('read_pileup', 'read_pileup-0'),
      meta('read_pileup', 'read_pileup-2'),
      meta('cluster_pileup', 'cluster_pileup-1'),
      meta('splice_junctions', 'splice_junctions-0'),
    ]);

    // New tracks join their kind group; the hidden track stays hidden.
    expect(stack(host)).toEqual([
      'reference_sequence',
      'reference',
      'coverage-0',
      'coverage-2',
      'read_pileup-0',
      'read_pileup-2',
      'cluster_pileup-1',
    ]);
    expect(host.querySelector('.toolbar-right .label-dim')?.textContent).toBe('7 / 8 tracks');
    expect(popoverTexts('.dataset-source-label').slice(1)).toEqual(['run-a', 'run-a clusters', 'run-b']);

    await settle(200);
    expect(storedLayout().map((e) => [e.source_id, e.kind, e.display_order, e.visible])).toEqual([
      ['', 'reference_sequence', 0, true],
      ['', 'gene_annotation', 1, true],
      [SRC_A, 'coverage_histogram', 2, true],
      [ADDED_SOURCE_ID, 'coverage_histogram', 3, true],
      [SRC_A, 'read_pileup', 3, true],
      [ADDED_SOURCE_ID, 'read_pileup', 4, true],
      [SRC_B, 'cluster_pileup', 4, true],
      [SRC_A, 'splice_junctions', 5, false],
    ]);
  });

  it('shows the server error and reloads nothing when an add fails', async () => {
    const { host } = await open();
    click(host.querySelector('.dataset-btn'));
    server.addSourceError = 'not a directory: /nope';
    const mark = server.calls.length;

    await addDataset('/nope');

    expect(server.since(mark)).toHaveLength(1);
    const error = document.body.querySelector<HTMLElement>('.dataset-add-error')!;
    expect(error.hidden).toBe(false);
    expect(error.textContent).toBe(
      `POST /api/sessions/${SESSION_ID}/sources failed: 400 — not a directory: /nope`,
    );
    expect(stack(host)).toEqual(DEFAULT_ORDER);
  });

  it('removes a source and drops its tracks, keeping the rest of the layout', async () => {
    const { host } = await open();
    click(host.querySelector('.dataset-btn'));
    await addDataset('/data/run-b/align');
    await settle(60);

    const mark = server.calls.length;
    const removeButtons = document.body.querySelectorAll('.dataset-popover .dataset-remove-btn');
    click(removeButtons[2]);
    await settle(0);

    expect(server.since(mark)[0]).toEqual({
      method: 'DELETE',
      url: `/api/sessions/${SESSION_ID}/sources/${ADDED_SOURCE_ID}`,
    });
    expect(stack(host)).toEqual(DEFAULT_ORDER);
    expect(host.querySelector('.toolbar-right .label-dim')?.textContent).toBe('6 tracks');
    await settle(200);
    expect(storedLayout().map((e) => e.display_order)).toEqual([0, 1, 2, 3, 4, 5]);
  });
});

// ----------------------------------------------------------------------
// Navigation
// ----------------------------------------------------------------------

describe('GenomeBrowser navigation', () => {
  it('goes to a typed locus and zooms around its centre', async () => {
    const { host } = await open();
    const input = locusInput(host);
    let mark = server.calls.length;
    input.value = 'chr1:1,000-2,000';
    input.dispatchEvent(new KeyboardEvent('keydown', { key: 'Enter', bubbles: true }));
    expect(input.value).toBe('chr1:1000-2000');
    await settle(60);
    expect(server.dataRequests(mark)).toEqual(
      DEFAULT_ORDER.map((id) => dataUrl(id, 'chr1:1000-2000')),
    );

    const [zoomOut, zoomIn, fit] = Array.from(host.querySelectorAll('.btn-group .icon-btn'));
    click(zoomOut);
    expect(input.value).toBe('chr1:900-2100');
    click(zoomIn);
    expect(input.value).toBe('chr1:1000-2000');
    mark = server.calls.length;
    click(fit);
    expect(input.value).toBe('chr1:0-12000');
    // Three locus changes inside the debounce produce one render.
    await settle(60);
    expect(server.dataRequests(mark)).toHaveLength(6);
  });

  it('ignores an unparseable locus and an unknown contig', async () => {
    const { host } = await open();
    const input = locusInput(host);
    const mark = server.calls.length;
    for (const text of ['nonsense', 'chr1:500-100', 'chrZ:1-100']) {
      input.value = text;
      input.dispatchEvent(new KeyboardEvent('keydown', { key: 'Enter', bubbles: true }));
    }
    await settle(60);
    expect(server.dataRequests(mark)).toEqual([]);
  });

  it('switches contig to its first 50 kb', async () => {
    const { host } = await open();
    setInput(host.querySelector<HTMLSelectElement>('.toolbar select')!, 'chr2');
    expect(locusInput(host).value).toBe('chr2:0-50000');
  });

  it('zooms on vertical wheel, pans on horizontal wheel and on drag', async () => {
    const { host } = await open();
    setInput(host.querySelector<HTMLSelectElement>('.toolbar select')!, 'chr2');
    const surface = host.querySelector<HTMLElement>('.browser')!;
    const input = locusInput(host);

    surface.dispatchEvent(
      new WheelEvent('wheel', { deltaY: 100, clientX: 600, bubbles: true, cancelable: true }),
    );
    expect(input.value).toBe('chr2:0-60000');

    surface.dispatchEvent(
      new WheelEvent('wheel', { deltaX: 120, deltaY: 0, clientX: 600, bubbles: true, cancelable: true }),
    );
    expect(input.value).toBe('chr2:6000-66000');

    surface.dispatchEvent(new MouseEvent('mousedown', { button: 0, clientX: 600, bubbles: true }));
    window.dispatchEvent(new MouseEvent('mousemove', { clientX: 480, bubbles: true }));
    window.dispatchEvent(new MouseEvent('mouseup', { bubbles: true }));
    expect(input.value).toBe('chr2:12000-72000');
    // Released: further movement does nothing.
    window.dispatchEvent(new MouseEvent('mousemove', { clientX: 100, bubbles: true }));
    expect(input.value).toBe('chr2:12000-72000');
  });

  it('searches after a debounce and jumps to a hit with flanking context', async () => {
    server.searchHits = [
      {
        feature_id: 9,
        name: 'geneB',
        type: 'gene',
        strand: '-',
        contig_name: 'chr2',
        start: 100,
        end: 900,
        source: 'reference',
      },
    ];
    const { host } = await open();
    const search = host.querySelector<HTMLInputElement>('.search-input')!;
    const dropdown = host.querySelector<HTMLElement>('.search-results')!;
    let mark = server.calls.length;

    setInput(search, ' geneB ', 'input');
    await settle(199);
    expect(server.since(mark)).toEqual([]);
    await settle(1);
    expect(server.since(mark)).toEqual([
      { method: 'GET', url: `/api/sessions/${SESSION_ID}/search?q=geneB&limit=20` },
    ]);
    expect(dropdown.hidden).toBe(false);
    const row = dropdown.querySelector('.search-row')!;
    expect(Array.from(row.children).map((c) => c.textContent)).toEqual([
      'geneB',
      'gene',
      'reference',
      'chr2:100-900',
    ]);

    mark = server.calls.length;
    click(row);
    expect(dropdown.hidden).toBe(true);
    // 800 bp feature + max(200, 25%) flank each side, clamped at 0.
    expect(locusInput(host).value).toBe('chr2:0-1100');
    await settle(60);
    expect(server.dataRequests(mark)[0]).toBe(dataUrl('reference_sequence', 'chr2:0-1100'));
  });

  it('shows "No matches" for an empty result and clears on Escape', async () => {
    const { host } = await open();
    const search = host.querySelector<HTMLInputElement>('.search-input')!;
    const dropdown = host.querySelector<HTMLElement>('.search-results')!;
    setInput(search, 'zzz', 'input');
    await settle(200);
    expect(dropdown.querySelector('.search-empty')?.textContent).toBe('No matches');
    search.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape', bubbles: true }));
    expect(search.value).toBe('');
    expect(dropdown.hidden).toBe(true);
  });
});

// ----------------------------------------------------------------------
// Labels, export, dispose
// ----------------------------------------------------------------------

describe('GenomeBrowser toolbar actions', () => {
  it('toggles feature labels, remembers the choice and re-renders', async () => {
    const { host } = await open();
    const labels = host.querySelector<HTMLElement>('button.toggle')!;
    expect(labels.classList.contains('on')).toBe(true);
    const before = svgOf(host, 'reference')!.querySelectorAll('text').length;
    expect(before).toBeGreaterThan(0);

    const mark = server.calls.length;
    click(labels);
    expect(labels.classList.contains('on')).toBe(false);
    expect(window.localStorage.getItem(LABELS_KEY)).toBe('false');
    await settle(60);
    expect(server.dataRequests(mark)).toHaveLength(6);
    expect(svgOf(host, 'reference')!.querySelectorAll('text').length).toBeLessThan(before);
  });

  it('shows the Labels toggle state on a track that has no choice of its own', async () => {
    const { host } = await open();
    const gear = track(host, 'reference').querySelector('.track-settings-btn');
    const box = (): HTMLInputElement => settingsRow('Show labels').querySelector('input')!;

    click(gear);
    expect(box().checked).toBe(true);
    click(gear);

    click(host.querySelector<HTMLElement>('button.toggle'));
    await settle(60);
    click(gear);
    // The track is drawn without labels, and the box now says so.
    expect(svgOf(host, 'reference')!.querySelectorAll('text')).toHaveLength(0);
    expect(box().checked).toBe(false);

    // Ticking it pins this track on, whatever the toolbar says.
    box().click();
    expect(svgOf(host, 'reference')!.querySelectorAll('text').length).toBeGreaterThan(0);
    expect(storedLayoutAfter()).toMatchObject({ show_labels: true });
  });

  it('starts with labels off when that was the stored choice', async () => {
    window.localStorage.setItem(LABELS_KEY, 'false');
    const { host } = await open();
    expect(host.querySelector('button.toggle')?.classList.contains('on')).toBe(false);
  });

  it('exports the visible tracks as one SVG named after the locus', async () => {
    const { host } = await open();
    click(track(host, 'splice_junctions-0').querySelector('.track-hide-btn'));
    const save = Array.from(host.querySelectorAll('.toolbar-right button')).find(
      (b) => b.textContent === 'Save SVG',
    )!;
    click(save);

    expect(downloadSvg).toHaveBeenCalledTimes(1);
    const [name, svg] = vi.mocked(downloadSvg).mock.calls[0];
    expect(name).toBe('chr1_0_12000.svg');
    expect(svg).toContain('<title>chr1:0-12000</title>');
    expect(svg).toContain('width="1200"');
    expect(svg).not.toContain('<clipPath');
    // Ruler + five visible panels, each translated into place.
    expect(svg.split('<g transform="translate(0 ').length - 1).toBeGreaterThanOrEqual(6);
  });

  it('clips each exported panel when the option is on', async () => {
    window.localStorage.setItem(OPTIONS_KEY, JSON.stringify({ clip_svg: true }));
    const { host } = await open();
    const save = Array.from(host.querySelectorAll('.toolbar-right button')).find(
      (b) => b.textContent === 'Save SVG',
    )!;
    click(save);
    const svg = vi.mocked(downloadSvg).mock.calls[0][1];
    expect(svg.split('<clipPath').length - 1).toBe(6);
  });

  it('dispose releases every document and window listener and stops following the locus', async () => {
    // Count listeners per target as add/remove pairs, from before the
    // browser exists until after it is disposed.
    const live = new Map<EventTarget, Map<unknown, string>>([
      [document, new Map()],
      [window, new Map()],
    ]);
    const restore: Array<() => void> = [];
    for (const [target, held] of live) {
      const add = target.addEventListener;
      const remove = target.removeEventListener;
      target.addEventListener = function (this: EventTarget, type: string, fn: unknown, ...rest: unknown[]) {
        held.set(fn, type);
        return (add as (...a: unknown[]) => void).call(this, type, fn, ...rest);
      } as typeof target.addEventListener;
      target.removeEventListener = function (this: EventTarget, type: string, fn: unknown, ...rest: unknown[]) {
        held.delete(fn);
        return (remove as (...a: unknown[]) => void).call(this, type, fn, ...rest);
      } as typeof target.removeEventListener;
      restore.push(() => {
        target.addEventListener = add;
        target.removeEventListener = remove;
      });
    }
    const held = (target: EventTarget): string[] => Array.from(live.get(target)!.values()).sort();

    try {
      const { host, browser } = await open();
      // Open every popover and start a resize drag, so their listeners
      // are live too when the browser goes away.
      click(host.querySelector('.dataset-btn'));
      click(host.querySelector('.options-btn'));
      click(track(host, 'coverage-0').querySelector('.track-settings-btn'));
      track(host, 'coverage-0')
        .querySelector('.track-resize-handle')!
        .dispatchEvent(new MouseEvent('pointerdown', { bubbles: true, cancelable: true, button: 0, clientY: 100 }));
      expect(held(document).length).toBeGreaterThan(0);
      expect(held(window)).toEqual(
        ['mousemove', 'mousemove', 'mouseup', 'mouseup', 'pointermove', 'pointerup'].sort(),
      );

      browser.dispose();
      expect(held(document)).toEqual([]);
      expect(held(window)).toEqual([]);

      // The locus bus is the browser's own, but it must not keep the
      // disposed browser rendering.
      const mark = server.calls.length;
      browser.bus.setLocus({ contig: 'chr1', start: 100, end: 900 });
      await settle(200);
      expect(server.calls.length).toBe(mark);
      expect(locusInput(host).value).toBe('chr1:0-12000');
    } finally {
      for (const undo of restore) undo();
    }
  });

  it('dispose closes popovers, drops the root class and cancels a pending render', async () => {
    const { host, browser } = await open();
    click(host.querySelector('.dataset-btn'));
    click(host.querySelector('.options-btn'));
    click(track(host, 'coverage-0').querySelector('.track-settings-btn'));
    click(host.querySelector('.btn-group .icon-btn:nth-child(2)')); // schedules a render
    const mark = server.calls.length;

    browser.dispose();

    expect(document.body.querySelector('.dataset-popover')).toBeNull();
    expect(document.body.querySelector('.options-popover')).toBeNull();
    expect(document.body.querySelector('.track-settings-popover')).toBeNull();
    expect(host.classList.contains('genome-browser-root')).toBe(false);
    await settle(1000);
    expect(server.since(mark)).toEqual([]);
  });
});
