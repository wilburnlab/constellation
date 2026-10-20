// The mass-spec browser against an in-memory server: what it asks for,
// how picking a scan on one level carries to the others, and keyboard
// stepping. The wire shapes are those of viz/modalities/massspec.

import { Float64, Int32, Int8, Table, tableToIPC, vectorFromArray, Bool } from 'apache-arrow';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { MassSpecBrowser, rowId } from './MassSpecBrowser';
import type { AcquisitionInfo, IsolationWindow } from './types';

/** Four DIA cycles: an MS1 scan, then one MS2 scan in each of two windows.
 *  Scan n is at n seconds; the third cycle's MS1 scan is the most intense. */
const SCANS = Array.from({ length: 12 }, (_, i) => {
  const scan = i + 1;
  const position = i % 3;
  const cycleStart = scan - position;
  return {
    scan,
    level: position === 0 ? 1 : 2,
    rt: scan,
    window: position === 0 ? null : position - 1,
    parent_scan: position === 0 ? null : cycleStart,
    precursor_mz: position === 0 ? null : position === 1 ? 401 : 403,
    tic: position === 0 ? (scan === 7 ? 9e6 : 1e6) : 1e5,
  };
});

const ACQ: AcquisitionInfo = {
  key: 'src-run',
  label: 'run_a',
  bundle: '/data/run_a',
  source_id: 'src-run',
  bundle_kind: 'thermo_acquisition',
  source_file: '/data/run_a.raw',
  source_sha256: null,
  acquisition_type: 'DIA',
  staggered: false,
  profile_mode: false,
  levels: [1, 2],
  scans_per_level: { '1': 4, '2': 8 },
  analyzers: { '1': ['ITMS'], '2': ['ITMS'] },
  rt_range: [1, 12],
  has_index: false,
  default_tolerance: { '1': [0.35, 'Da'], '2': [0.35, 'Da'] },
};

const WINDOWS: IsolationWindow[] = [
  { level: 2, window: 0, isolation_lower: 400, isolation_upper: 402, center: 401, n_scans: 4 },
  { level: 2, window: 1, isolation_lower: 402, isolation_upper: 404, center: 403, n_scans: 4 },
];

function scanTable(): Table {
  return new Table({
    scan: vectorFromArray(SCANS.map((s) => s.scan), new Int32()),
    level: vectorFromArray(SCANS.map((s) => s.level), new Int8()),
    rt: vectorFromArray(SCANS.map((s) => s.rt), new Float64()),
    window: vectorFromArray(SCANS.map((s) => s.window), new Int32()),
    parent_scan: vectorFromArray(SCANS.map((s) => s.parent_scan), new Int32()),
    precursor_mz: vectorFromArray(SCANS.map((s) => s.precursor_mz), new Float64()),
    tic: vectorFromArray(SCANS.map((s) => s.tic), new Float64()),
  });
}

function trace(rows: typeof SCANS): Table {
  return new Table({
    scan: vectorFromArray(rows.map((s) => s.scan), new Int32()),
    rt: vectorFromArray(rows.map((s) => s.rt), new Float64()),
    intensity: vectorFromArray(rows.map((s) => s.tic), new Float64()),
    mz: vectorFromArray(rows.map(() => null), new Float64()),
    decimated: vectorFromArray(rows.map(() => false), new Bool()),
  });
}

function spectrum(scan: number): Table {
  const mz = [200 + scan, 300 + scan, 400 + scan];
  return new Table({
    mz: vectorFromArray(mz, new Float64()),
    intensity: vectorFromArray([10, 30, 20], new Float64()),
    decimated: vectorFromArray([false, false, false], new Bool()),
  });
}

interface Server {
  requests: string[];
}

function fakeServer(): Server {
  const server: Server = { requests: [] };
  const arrow = (t: Table) =>
    new Response(tableToIPC(t, 'stream') as unknown as BodyInit, { headers: { 'X-Track-Mode': 'vector' } });
  vi.stubGlobal('fetch', async (input: string) => {
    const url = new URL(input, 'http://localhost');
    server.requests.push(url.pathname + url.search);
    const p = url.pathname;
    if (p.endsWith('/acquisitions')) return Response.json([ACQ]);
    if (p.endsWith('/windows')) return Response.json(WINDOWS);
    if (p.endsWith('/scans')) return arrow(scanTable());
    if (p === '/api/tracks/ms_chromatogram/data') {
      const level = Number(url.searchParams.get('level'));
      const window = url.searchParams.get('window');
      return arrow(trace(SCANS.filter((s) => s.level === level && (window === null || s.window === Number(window)))));
    }
    if (p === '/api/tracks/ms_spectrum/data') return arrow(spectrum(Number(url.searchParams.get('scan'))));
    return new Response('not found', { status: 404 });
  });
  return server;
}

let host: HTMLElement;
let browser: MassSpecBrowser | null = null;
beforeEach(() => {
  window.localStorage.clear();
  host = document.createElement('div');
  document.body.appendChild(host);
});
afterEach(() => {
  browser?.dispose();
  browser = null;
  host.remove();
  vi.unstubAllGlobals();
});

async function open(): Promise<{ b: MassSpecBrowser; server: Server }> {
  const server = fakeServer();
  const b = (browser = new MassSpecBrowser({ host, sessionId: 'sess-1' }));
  await b.mount();
  // jsdom lays nothing out: give the grid the size a window would.
  b.panelGrid!.resize({ widthPx: 1200, heightPx: 600 });
  await b.renderNow();
  return { b, server };
}

describe('MassSpecBrowser', () => {
  it('lays out a row per MS level and opens on the most intense MS1 scan', async () => {
    const { b, server } = await open();
    const view = b.currentView!;
    expect(Object.keys(view.rows)).toEqual(['ms1', 'ms2']);
    expect(view.rows.ms1.scan).toBe(7);
    // The MS2 row shows the first scan after it in its (middle) window.
    expect(view.rows.ms2.window).toBe(1);
    expect(view.rows.ms2.scan).toBe(9);
    expect(server.requests).toContain('/api/massspec/sessions/sess-1/acquisitions/src-run/scans');
    const chrom = server.requests.filter((r) => r.startsWith('/api/tracks/ms_chromatogram/data'));
    expect(chrom.some((r) => r.includes('level=2') && r.includes('window=1') && r.includes('trace=tic'))).toBe(true);
    expect(chrom.some((r) => r.includes('level=1') && !r.includes('window='))).toBe(true);
    // The window picker sits in the MS2 row's header.
    const select = host.querySelector<HTMLSelectElement>('.panel-grid-row-header[data-row="ms2"] select');
    expect([...select!.options].map((o) => o.textContent)).toEqual(['400.0–402.0', '402.0–404.0']);
  });

  it('labels the spectra with their scans and shades the isolation window on MS1', async () => {
    const { b } = await open();
    const grid = b.panelGrid!;
    expect(grid.cell('ms2', 'spec')!.panel.headerLabelEl.textContent).toBe('MS2 scan 9 · 0.15 min');
    const ms1 = grid.cell('ms1', 'spec')!.panel.bodyHost;
    expect(ms1.querySelector('.ms-isolation')).not.toBeNull();
    expect(ms1.querySelector('.ms-target')).not.toBeNull();
    expect(ms1.querySelectorAll('.ms-peak-label').length).toBe(3);
  });

  it('a scan picked on MS2 brings its parent MS1 scan along', async () => {
    const { b } = await open();
    b.selectScan(rowId(2), 3);
    await b.renderNow();
    const view = b.currentView!;
    expect(view.rows.ms2.scan).toBe(3);
    expect(view.rows.ms1.scan).toBe(1);
  });

  it('arrow keys step the active row within its window', async () => {
    const { b } = await open();
    b.selectScan(rowId(2), 6);
    const root = host.querySelector<HTMLElement>('.ms-browser')!;
    root.dispatchEvent(new KeyboardEvent('keydown', { key: 'ArrowRight', bubbles: true }));
    expect(b.currentView!.rows.ms2.scan).toBe(9);
    root.dispatchEvent(new KeyboardEvent('keydown', { key: 'ArrowLeft', bubbles: true }));
    root.dispatchEvent(new KeyboardEvent('keydown', { key: 'ArrowLeft', bubbles: true }));
    expect(b.currentView!.rows.ms2.scan).toBe(3);
  });

  it('switching the DIA window keeps the time', async () => {
    const { b } = await open();
    b.selectScan(rowId(2), 9); // window 1 at 9 s
    b.setWindow(rowId(2), 0);
    expect(b.currentView!.rows.ms2.window).toBe(0);
    expect(b.currentView!.rows.ms2.scan).toBe(8);
  });

  it('dispose removes the widget', async () => {
    const { b } = await open();
    b.dispose();
    browser = null;
    expect(host.querySelector('.ms-browser')).toBeNull();
  });
});
