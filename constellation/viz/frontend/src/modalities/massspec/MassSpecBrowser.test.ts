// The mass-spec browser against an in-memory server: what it asks for,
// how picking a scan on one level carries to the others, keyboard
// stepping, and what a target (an identification, or a peptide typed by
// hand) does to every panel. The wire shapes are those of
// viz/modalities/massspec.

import { Bool, Float64, Int32, Int64, Int8, Int16, Table, Utf8, tableToIPC, vectorFromArray } from 'apache-arrow';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { MassSpecBrowser, rowId, toleranceDa } from './MassSpecBrowser';
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
  searches: [
    { experiment: '/data/exp', experiment_source_id: 'src-exp', experiment_label: 'exp', acquisition_id: 0, units: ['run_a-1234abcd'] },
  ],
};

/** Above 2^53, so a JavaScript number would round it. */
const BIG_ID = '9007199254740993';

/** Two identifications: one in window 1 (402–404), one in window 0. */
const IDS = [
  { precursor_id: BIG_ID, modified_sequence: 'PEPTIDEK', sequence: 'PEPTIDEK', charge: 2, precursor_mz: 403.2, rt: 7.5, qvalue: 0.001, proteins: 'P12345', intensity: 5e6 },
  { precursor_id: '17', modified_sequence: 'ELVISK', sequence: 'ELVISK', charge: 2, precursor_mz: 401.1, rt: 4.0, qvalue: 0.004, proteins: 'Q99999', intensity: 1e6 },
];

function identifications(): Table {
  return new Table({
    unit: vectorFromArray(IDS.map(() => 'run_a-1234abcd'), new Utf8()),
    precursor_id: vectorFromArray(IDS.map((r) => BigInt(r.precursor_id)), new Int64()),
    modified_sequence: vectorFromArray(IDS.map((r) => r.modified_sequence), new Utf8()),
    sequence: vectorFromArray(IDS.map((r) => r.sequence), new Utf8()),
    charge: vectorFromArray(IDS.map((r) => r.charge), new Int32()),
    precursor_mz: vectorFromArray(IDS.map((r) => r.precursor_mz), new Float64()),
    rt: vectorFromArray(IDS.map((r) => r.rt), new Float64()),
    intensity: vectorFromArray(IDS.map((r) => r.intensity), new Float64()),
    qvalue: vectorFromArray(IDS.map((r) => r.qvalue), new Float64()),
    proteins: vectorFromArray(IDS.map((r) => r.proteins), new Utf8()),
    scan: vectorFromArray(IDS.map(() => null), new Int32()),
    experiment: vectorFromArray(IDS.map(() => 'src-exp'), new Utf8()),
  });
}

const LADDER = [
  { ion_type: 'b', ion_number: 2, position: 1, fragment_charge: 1, loss: null, mz: 227.1, annotation: 'b2' },
  { ion_type: 'y', ion_number: 4, position: 3, fragment_charge: 1, loss: null, mz: 500.3, annotation: 'y4' },
];

function precursorJson(id: string, scan: number | null = null): Record<string, unknown> {
  const found = IDS.find((r) => r.precursor_id === id)!;
  // Asked for one match (spectrum-centric): that match's scan, time and score.
  const row = scan === null ? found : { ...found, rt: scan, qvalue: null, scan };
  const window = row.precursor_mz > 402 ? 1 : 0;
  const ms = scan ?? (window === 1 ? 9 : 5);
  return {
    identification: { ...row, precursor_id: Number(id), rt_start: row.rt - 1.5, rt_end: row.rt + 1.5 },
    experiment: 'src-exp',
    unit: 'run_a-1234abcd',
    located: { window, isolation: [400, 402], ms_scan: ms, parent_scan: ms - ((ms - 1) % 3) },
    ms1_targets: [
      { mz: row.precursor_mz, label: 'M' },
      { mz: row.precursor_mz + 0.5, label: 'M+1' },
    ],
    ms2_targets: [
      { mz: 500.3, label: 'y4', quantified: true },
      { mz: 600.3, label: 'y5', quantified: false },
    ],
    ladder: LADDER,
    quantified_ions: [],
    chromatogram: [
      { rt: 7, intensity: 1 },
      { rt: 8, intensity: 2 },
    ],
    fragment_tolerance: { value: 0.4, unit: 'Da' },
    precursor_tolerance: { value: 10, unit: 'ppm' },
    library_rt: 400,
    predicted_rt: 7.2,
  };
}

function xic(url: URL): Table {
  const mzs = url.searchParams.getAll('mz').map(Number);
  const labels = url.searchParams.getAll('label');
  const rows = mzs.flatMap((mz, k) => [5, 6, 7].map((rt) => ({ k, mz, rt })));
  return new Table({
    trace: vectorFromArray(rows.map((r) => r.k), new Int16()),
    label: vectorFromArray(rows.map((r) => labels[r.k] ?? null), new Utf8()),
    mz_target: vectorFromArray(rows.map((r) => r.mz), new Float64()),
    scan: vectorFromArray(rows.map((r) => r.rt), new Int32()),
    rt: vectorFromArray(rows.map((r) => r.rt), new Float64()),
    intensity: vectorFromArray(rows.map((r) => 100 * (r.k + 1)), new Float64()),
    mz_observed: vectorFromArray(rows.map((r) => r.mz), new Float64()),
  });
}

const WINDOWS: IsolationWindow[] = [
  { level: 2, window: 0, isolation_lower: 400, isolation_upper: 402, center: 401, n_scans: 4 },
  { level: 2, window: 1, isolation_lower: 402, isolation_upper: 404, center: 403, n_scans: 4 },
];

type ScanRow = (typeof SCANS)[number];

function scanTable(rows: ScanRow[]): Table {
  return new Table({
    scan: vectorFromArray(rows.map((s) => s.scan), new Int32()),
    level: vectorFromArray(rows.map((s) => s.level), new Int8()),
    rt: vectorFromArray(rows.map((s) => s.rt), new Float64()),
    window: vectorFromArray(rows.map((s) => s.window), new Int32()),
    parent_scan: vectorFromArray(rows.map((s) => s.parent_scan), new Int32()),
    precursor_mz: vectorFromArray(rows.map((s) => s.precursor_mz), new Float64()),
    tic: vectorFromArray(rows.map((s) => s.tic), new Float64()),
  });
}

/** A DDA run: four cycles of an MS1 scan and the two scans it triggered.
 *  Scans 2 and 11 isolated 401.2 (SAMPLER/2), scan 5 its M+1; scan 11 has
 *  the most signal. */
const DDA_PRECURSORS = [401.2, 500, 401.7, 600, 700, 800, 401.2, 900];
const DDA_SCANS: ScanRow[] = Array.from({ length: 12 }, (_, i) => {
  const scan = i + 1;
  const position = i % 3;
  const ms2 = Math.floor(i / 3) * 2 + position - 1;
  return {
    scan,
    level: position === 0 ? 1 : 2,
    rt: scan,
    window: null,
    parent_scan: position === 0 ? null : scan - position,
    precursor_mz: position === 0 ? null : DDA_PRECURSORS[ms2],
    tic: position === 0 ? 1e6 : scan === 11 ? 5e5 : 1e5,
  };
});
const DDA_ACQ: AcquisitionInfo = { ...ACQ, acquisition_type: 'DDA', has_index: true, searches: [] };

function trace(rows: ScanRow[]): Table {
  return new Table({
    scan: vectorFromArray(rows.map((s) => s.scan), new Int32()),
    rt: vectorFromArray(rows.map((s) => s.rt), new Float64()),
    intensity: vectorFromArray(rows.map((s) => s.tic), new Float64()),
    mz: vectorFromArray(rows.map((s) => s.precursor_mz), new Float64()),
    decimated: vectorFromArray(rows.map(() => false), new Bool()),
  });
}

function spectrum(scan: number, annotate: boolean): Table {
  const mz = [200 + scan, 300 + scan, 400 + scan];
  // Annotated: the last peak is y4, 2 ppm off.
  return new Table({
    mz: vectorFromArray(mz, new Float64()),
    intensity: vectorFromArray([10, 30, 20], new Float64()),
    annotation: vectorFromArray([null, null, annotate ? 'y4' : null], new Utf8()),
    ion_type: vectorFromArray([null, null, annotate ? 'y' : null], new Utf8()),
    error_ppm: vectorFromArray([null, null, annotate ? 2 : null], new Float64()),
    error_da: vectorFromArray([null, null, annotate ? 0.001 : null], new Float64()),
    decimated: vectorFromArray([false, false, false], new Bool()),
  });
}

interface Server {
  requests: string[];
  /** Bodies of POST requests, by path. */
  posted: Record<string, unknown>;
  /** What `GET /api/commands/{id}` answers. */
  job: { state: string; exit_code: number | null };
  acquisitions: AcquisitionInfo[];
  /** A scan table that answers only when this resolves, by acquisition key. */
  holdScans: Record<string, Promise<void>>;
  /** Scans a reduced chromatogram leaves out. */
  dropScans: number[];
  /** Spectra come back reduced. */
  reduced: boolean;
}

function fakeServer(dda = false): Server {
  const server: Server = {
    requests: [],
    posted: {},
    job: { state: 'running', exit_code: null },
    acquisitions: [dda ? DDA_ACQ : ACQ],
    holdScans: {},
    dropScans: [],
    reduced: false,
  };
  const scans = dda ? DDA_SCANS : SCANS;
  const arrow = (t: Table) =>
    new Response(tableToIPC(t, 'stream') as unknown as BodyInit, { headers: { 'X-Track-Mode': 'vector' } });
  vi.stubGlobal('fetch', async (input: string, init?: RequestInit) => {
    const url = new URL(input, 'http://localhost');
    server.requests.push(url.pathname + url.search);
    const p = url.pathname;
    if (init?.method === 'POST') {
      server.posted[p] = JSON.parse(String(init.body));
      return Response.json({ job_id: 'job-1', argv: [], started_at: '', state: 'running' });
    }
    if (p === '/api/commands/job-1') return Response.json({ job_id: 'job-1', ...server.job });
    if (p.endsWith('/acquisitions')) return Response.json(server.acquisitions);
    if (p.endsWith('/windows')) return Response.json(dda ? [] : WINDOWS);
    if (p.endsWith('/scans')) {
      await server.holdScans[p.split('/').at(-2)!];
      return arrow(scanTable(scans));
    }
    if (/\/scans\/\d+$/.test(p)) return Response.json({ scan: Number(p.split('/').pop()), filter_string: 'ITMS + c NSI d Full ms2' });
    if (p.endsWith('/identifications')) return arrow(identifications());
    if (p.endsWith('/precursor')) {
      const scan = url.searchParams.get('scan');
      return Response.json(precursorJson(url.searchParams.get('precursor_id')!, scan === null ? null : Number(scan)));
    }
    if (p.endsWith('/resolve')) {
      return Response.json({
        modified_sequence: 'SAMPLER',
        sequence: 'SAMPLER',
        charges: [{ charge: 2, precursor_mz: 401.2, precursor_id: 5 }],
        charge: 2,
        ms1_targets: [{ mz: 401.2, label: 'M' }],
        ms2_targets: [{ mz: 333.3, label: 'y3', quantified: null }],
        ladder: LADDER,
      });
    }
    if (p === '/api/tracks/ms_xic/data') return arrow(xic(url));
    if (p === '/api/tracks/ms_chromatogram/data') {
      const level = Number(url.searchParams.get('level'));
      const window = url.searchParams.get('window');
      const kept = scans.filter((s) => s.level === level && (window === null || s.window === Number(window)));
      return arrow(trace(kept.filter((s) => !server.dropScans.includes(s.scan))));
    }
    if (p === '/api/tracks/ms_spectrum/data') {
      const table = spectrum(Number(url.searchParams.get('scan')), url.searchParams.has('modified_sequence'));
      if (!server.reduced) return arrow(table);
      const flag = vectorFromArray(Array.from({ length: table.numRows }, () => true), new Bool());
      return arrow(new Table({ ...Object.fromEntries(table.schema.fields.map((f) => [f.name, table.getChild(f.name)!])), decimated: flag }));
    }
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

async function open(dda = false, setup?: (server: Server) => void): Promise<{ b: MassSpecBrowser; server: Server }> {
  const server = fakeServer(dda);
  setup?.(server);
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

  it('lists what the searches found in the drawer', async () => {
    const { b, server } = await open();
    await vi.waitFor(() => expect(b.identificationRows.length).toBe(2));
    expect(server.requests).toContain('/api/massspec/sessions/sess-1/acquisitions/src-run/identifications?max_q=0.01');
    // The 64-bit id survives as a string.
    expect(b.identificationRows[0].precursor_id).toBe(BIG_ID);
    const rows = host.querySelectorAll('.ms-id-table tbody tr');
    expect(rows.length).toBe(2);
    // Sorted by q-value: PEPTIDEK first.
    expect(rows[0].textContent).toContain('PEPTIDEK');
  });

  it('picking an identification shows its traces, window, apex and annotated spectrum', async () => {
    const { b, server } = await open();
    await vi.waitFor(() => expect(b.identificationRows.length).toBe(2));
    b.selectScan(rowId(2), 3); // somewhere else, in window 0
    server.requests.length = 0;
    await b.selectIdentification(b.identificationRows[0]);
    await b.renderNow();
    const precursorReq = server.requests.find((r) => r.includes('/precursor?'))!;
    expect(precursorReq).toContain(`precursor_id=${BIG_ID}`);
    expect(precursorReq).toContain('experiment=src-exp');
    const view = b.currentView!;
    expect(view.target!.label).toBe('PEPTIDEK/2');
    expect(view.rows.ms2.window).toBe(1);
    expect(view.rows.ms2.scan).toBe(9);
    expect(view.rows.ms1.scan).toBe(7);
    // Zoomed to the integration window with a margin, inside the run.
    expect(view.rt[0]).toBeGreaterThanOrEqual(1);
    expect(view.rt[1]).toBeLessThanOrEqual(12);
    // Traces: the isotopes in MS1 at the search's precursor tolerance, the
    // fragments in MS2 at its fragment tolerance; no index, so a time range.
    const xics = server.requests.filter((r) => r.startsWith('/api/tracks/ms_xic/data'));
    const ms1 = xics.find((r) => r.includes('level=1'))!;
    expect(ms1).toContain('mz=403.2');
    expect(ms1).toContain('label=M%2B1');
    expect(ms1).toContain('tolerance=10&tolerance_unit=ppm');
    expect(ms1).toContain('rt_start=');
    const ms2 = xics.find((r) => r.includes('level=2'))!;
    expect(ms2).toContain('window=1');
    expect(ms2).toContain('mz=500.3');
    expect(ms2).toContain('tolerance=0.4&tolerance_unit=Da');
    // The MS2 spectrum is annotated with the target's ions.
    const spec = server.requests.find((r) => r.startsWith('/api/tracks/ms_spectrum/data') && r.includes('scan=9'))!;
    expect(spec).toContain('modified_sequence=PEPTIDEK');
    expect(spec).toContain('charge=2');
    expect(spec).toContain('ion_types=b&ion_types=y');
    const grid = b.panelGrid!;
    const chrom2 = grid.cell('ms2', 'chrom')!.panel.bodyHost;
    expect(chrom2.querySelectorAll('.ms-xic').length).toBe(2);
    expect(chrom2.querySelectorAll('.ms-xic[stroke-dasharray]').length).toBe(1); // y5 not quantified
    expect(chrom2.querySelector('.ms-integration-window')).not.toBeNull();
    expect(chrom2.querySelector('.ms-median')).not.toBeNull();
    expect(grid.cell('ms1', 'chrom')!.panel.bodyHost.querySelectorAll('.ms-xic').length).toBe(2);
    const spec2 = grid.cell('ms2', 'spec')!.panel.bodyHost;
    expect(spec2.querySelector('.ms-ion-label')!.textContent).toBe('y4');
    expect(spec2.querySelector('.ms-error-strip')).not.toBeNull();
    expect(grid.cell('ms1', 'spec')!.panel.bodyHost.querySelectorAll('.ms-isotope').length).toBe(2);
    // The chip names it; the drawer lights the matched ion.
    expect(host.querySelector('.ms-chip-target')!.textContent).toContain('PEPTIDEK/2');
    expect(host.querySelector('.ms-ion-table .matched')!.textContent).toBe('500.30');
  });

  it('a peptide typed in the search box is resolved and traced', async () => {
    const { b, server } = await open();
    await vi.waitFor(() => expect(b.identificationRows.length).toBe(2));
    const suggestions = b.suggestionsFor('SAMPLER/2');
    expect(suggestions.map((s) => s.detail)).toContain('trace this peptide (not identified)');
    await b.tracePeptide('SAMPLER', 2);
    await b.renderNow();
    expect(server.requests.some((r) => r.startsWith('/api/massspec/sessions/sess-1/resolve?q=SAMPLER'))).toBe(true);
    const view = b.currentView!;
    expect(view.target!.source).toBeNull();
    // DIA: the window that isolates 401.2.
    expect(view.rows.ms2.window).toBe(0);
    expect(host.querySelector('.ms-chip-target')!.textContent).toContain('not identified');
  });

  it('search suggestions cover identifications, scans, times and m/z', async () => {
    const { b } = await open();
    await vi.waitFor(() => expect(b.identificationRows.length).toBe(2));
    expect(b.suggestionsFor('PEPT')[0].label).toBe('PEPTIDEK/2');
    expect(b.suggestionsFor('q99999')[0].label).toBe('ELVISK/2');
    expect(b.suggestionsFor('scan 4')[0].detail).toBe('go to it');
    expect(b.suggestionsFor('#400')[0].detail).toBe('not in this run');
    b.suggestionsFor('scan 4')[0].run();
    expect(b.currentView!.rows.ms1.scan).toBe(4);
    b.suggestionsFor('512.27')[0].run();
    expect(b.currentView!.adhoc.ms1).toEqual([512.27]);
  });

  it('a clicked peak becomes a trace of its row, at most two per row', async () => {
    const { b, server } = await open();
    server.requests.length = 0;
    b.addAdhoc(rowId(2), 309);
    await b.renderNow();
    const req = server.requests.find((r) => r.startsWith('/api/tracks/ms_xic/data'))!;
    expect(req).toContain('level=2');
    expect(req).toContain('window=1');
    expect(req).toContain('mz=309');
    expect(host.querySelectorAll('.ms-chip-adhoc').length).toBe(1);
    b.addAdhoc(rowId(2), 409);
    b.addAdhoc(rowId(2), 209);
    expect(b.currentView!.adhoc.ms2).toEqual([409, 209]);
    b.removeAdhoc(rowId(2), 409);
    expect(b.currentView!.adhoc.ms2).toEqual([209]);
  });

  it('steps through identifications and back through history', async () => {
    const { b } = await open();
    await vi.waitFor(() => expect(b.identificationRows.length).toBe(2));
    await b.stepIdentification(1);
    expect(b.currentView!.target!.label).toBe('PEPTIDEK/2');
    await b.stepIdentification(1);
    expect(b.currentView!.target!.label).toBe('ELVISK/2');
    await b.go(-1);
    expect(b.currentView!.target!.label).toBe('PEPTIDEK/2');
    await b.go(-1);
    expect(b.currentView!.target).toBeNull();
    await b.go(1);
    expect(b.currentView!.target!.label).toBe('PEPTIDEK/2');
    b.clearTarget();
    expect(b.currentView!.target).toBeNull();
    expect(host.querySelector('.ms-chip-target')).toBeNull();
  });

  it('exports a cell as CSV', async () => {
    const { b } = await open();
    const csv = b.cellCsv(b.panelGrid!.cell('ms2', 'spec')!)!;
    expect(csv.split('\n')[0]).toBe('mz,intensity,annotation,ion_type,error_ppm');
    expect(csv.split('\n')[1]).toBe('209,10,,,');
  });

  it('DDA: a typed peptide shows the scans that isolated it, and steps among them', async () => {
    const { b, server } = await open(true);
    server.requests.length = 0;
    await b.tracePeptide('SAMPLER', 2);
    await b.renderNow();
    // The scan of the three that isolated it (M or M+1) with the most signal.
    expect(b.currentView!.rows.ms2.scan).toBe(11);
    expect(server.requests.some((r) => r.includes('ms_spectrum') && r.includes('scan=11') && r.includes('modified_sequence=SAMPLER'))).toBe(true);
    // No DIA window: the MS2 row keeps its scan events, only the target's.
    expect(server.requests.some((r) => r.startsWith('/api/tracks/ms_xic/data') && r.includes('level=2'))).toBe(false);
    const sticks = b.panelGrid!.cell('ms2', 'chrom')!.panel.bodyHost.querySelector('.ms-trace')!.getAttribute('d')!;
    expect(sticks.match(/M/g)!.length).toBe(3);
    // With a peak index the traces cover the whole run.
    const xic = server.requests.find((r) => r.startsWith('/api/tracks/ms_xic/data'))!;
    expect(xic).not.toContain('rt_start');
    const root = host.querySelector<HTMLElement>('.ms-browser')!;
    root.dispatchEvent(new KeyboardEvent('keydown', { key: 'ArrowLeft', bubbles: true }));
    expect(b.currentView!.rows.ms2.scan).toBe(5);
    root.dispatchEvent(new KeyboardEvent('keydown', { key: 'ArrowLeft', bubbles: true }));
    expect(b.currentView!.rows.ms2.scan).toBe(2);
    root.dispatchEvent(new KeyboardEvent('keydown', { key: 'ArrowLeft', bubbles: true }));
    expect(b.currentView!.rows.ms2.scan).toBe(2);
    // A trace added on the DDA MS2 row covers the target's time, not the run.
    b.addAdhoc(rowId(2), 300);
    await b.renderNow();
    const added = server.requests.filter((r) => r.startsWith('/api/tracks/ms_xic/data') && r.includes('level=2'));
    expect(added.length).toBe(1);
    expect(added[0]).toContain('rt_start=');
    // Without a target the arrows step every scan again.
    b.clearTarget();
    root.dispatchEvent(new KeyboardEvent('keydown', { key: 'ArrowRight', bubbles: true }));
    expect(b.currentView!.rows.ms2.scan).toBe(3);
  });

  it('builds a missing peak index through the command runner', async () => {
    const { b, server } = await open();
    const button = [...host.querySelectorAll<HTMLButtonElement>('.ms-btn')].find((x) => x.textContent === 'Build index')!;
    expect(button.hidden).toBe(false);
    vi.stubGlobal('confirm', () => true);
    vi.useFakeTimers();
    try {
      await b.buildIndex();
      expect(server.posted['/api/commands']).toEqual({
        argv: ['massspec', 'chromatogram', 'build-index', '--peaks-dir', '/data/run_a', '-o', '/data/run_a/mz_index'],
      });
      expect(button.disabled).toBe(true);
      await vi.advanceTimersByTimeAsync(2000);
      expect(button.disabled).toBe(true); // still running: polls again
      server.job = { state: 'completed', exit_code: 0 };
      await vi.advanceTimersByTimeAsync(3000);
      expect(button.hidden).toBe(true);
      // Traces now cover the whole run.
      b.addAdhoc(rowId(1), 300);
      expect(b.currentView!.xicRt).toBeNull();
    } finally {
      vi.useRealTimers();
    }
  });

  it('a spectrum-centric match is shown at its own scan, time and score', async () => {
    const { b, server } = await open();
    await vi.waitFor(() => expect(b.identificationRows.length).toBe(2));
    const match = { ...b.identificationRows[0], scan: 12, key: `${b.identificationRows[0].key}|12` };
    await b.selectIdentification(match);
    expect(server.requests.find((r) => r.includes('/precursor?'))).toContain('scan=12');
    const t = b.currentView!.target!;
    expect([t.rt, t.qvalue, t.apexScans.ms2, t.key.endsWith('|12')]).toEqual([12, null, 12, true]);
    expect(b.currentView!.rows.ms2.scan).toBe(12);
  });

  it('a slow reply for a run already left does not replace the run chosen after it', async () => {
    let release!: () => void;
    const { b } = await open(false, (server) => {
      server.acquisitions = [ACQ, { ...ACQ, key: 'src-b', label: 'run_b' }, { ...ACQ, key: 'src-c', label: 'run_c' }];
      server.holdScans['src-b'] = new Promise<void>((resolve) => (release = resolve));
    });
    const slow = b.showAcquisition('src-b');
    await b.showAcquisition('src-c');
    release();
    await slow;
    expect(b.currentView!.acquisition).toBe('src-c');
    expect(host.querySelector<HTMLSelectElement>('.ms-toolbar select')!.value).toBe('src-c');
  });

  it("DDA: a target's scan events are drawn even when the server's reduced trace left them out", async () => {
    const { b } = await open(true, (server) => {
      server.dropScans = [11];
    });
    await b.tracePeptide('SAMPLER', 2);
    await b.renderNow();
    const sticks = b.panelGrid!.cell('ms2', 'chrom')!.panel.bodyHost.querySelector('.ms-trace')!.getAttribute('d')!;
    expect(sticks.match(/M/g)!.length).toBe(3);
  });

  it('a reduced spectrum is fetched again for the m/z window shown', async () => {
    const { b, server } = await open();
    // A spectrum sent whole zooms on what it has.
    server.requests.length = 0;
    b.currentView!.rows.ms1.mz = [250, 260];
    await b.renderNow();
    expect(server.requests.filter((r) => r.startsWith('/api/tracks/ms_spectrum/data'))).toEqual([]);
    // One the server reduced is asked for again, for the window shown.
    server.reduced = true;
    b.selectScan(rowId(2), 12);
    await b.renderNow();
    server.requests.length = 0;
    b.currentView!.rows.ms2.mz = [300, 320];
    await b.renderNow();
    const spec = server.requests.filter((r) => r.startsWith('/api/tracks/ms_spectrum/data'));
    // Each reduced spectrum, for its own window (MS1 kept the one set above).
    expect(spec.map((r) => new URL(r, 'http://x').searchParams.get('scan')).sort()).toEqual(['10', '12']);
    expect(spec.find((r) => r.includes('scan=12'))).toContain('mz_start=300&mz_end=320');
    expect(spec.find((r) => r.includes('scan=10'))).toContain('mz_start=250&mz_end=260');
  });

  it("back restores each row's m/z window with the target", async () => {
    const { b } = await open();
    await vi.waitFor(() => expect(b.identificationRows.length).toBe(2));
    await b.selectIdentification(b.identificationRows[0]); // PEPTIDEK, 403.2
    const first = [...b.currentView!.rows.ms1.mz!];
    await b.selectIdentification(b.identificationRows[1]); // ELVISK, 401.1
    expect(b.currentView!.rows.ms1.mz).not.toEqual(first);
    await b.go(-1);
    expect(b.currentView!.target!.label).toBe('PEPTIDEK/2');
    expect(b.currentView!.rows.ms1.mz).toEqual(first);
  });

  it('the ions are no longer lit once the spectrum shown did not isolate the target', async () => {
    const { b } = await open();
    await vi.waitFor(() => expect(b.identificationRows.length).toBe(2));
    await b.selectIdentification(b.identificationRows[0]);
    await b.renderNow();
    expect(host.querySelectorAll('.ms-ion-table .matched').length).toBe(1);
    b.setWindow(rowId(2), 0); // 400-402: not PEPTIDEK's 403.2
    await b.renderNow();
    expect(host.querySelectorAll('.ms-ion-table .matched').length).toBe(0);
    expect(host.querySelector('.ms-drawer-ions')!.textContent).toContain('did not isolate');
  });

  it('converts tolerances to m/z', () => {
    expect(toleranceDa(500, [10, 'ppm'])).toBeCloseTo(0.005);
    expect(toleranceDa(500, [0.35, 'Da'])).toBe(0.35);
  });

  it('dispose removes the widget', async () => {
    const { b } = await open();
    b.dispose();
    browser = null;
    expect(host.querySelector('.ms-browser')).toBeNull();
  });
});
