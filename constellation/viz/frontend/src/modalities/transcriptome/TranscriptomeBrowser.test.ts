import { Bool, Int32, Int64, Table, tableToIPC, vectorFromArray } from 'apache-arrow';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { supportMeta, supportTables } from './__fixtures__/support';
import { TranscriptomeBrowser, parseInitial } from './TranscriptomeBrowser';
import type { SupportView } from './SupportView';
import type { RunInfo, SupportMeta } from './types';

const SID = 's1';
const KEY = 'src-run';
const BASE = `/api/transcriptome/sessions/${SID}/runs/${KEY}`;

const RUN: RunInfo = {
  key: KEY,
  label: 'em_run',
  path: '/data/em_run',
  finished: true,
  rounds: [1],
  n_clusters: 3,
  n_reads: 110,
  has_cluster_nodes: true,
  has_mstep_record: true,
  has_compact_evidence: true,
  has_round_params: true,
  samples: { '0': 'testis_a', '1': 'testis_b', '2': 'testis_c' },
  demux_dir: '/data/demux',
};

function clusterPage(): Table {
  return new Table({
    cluster_id: vectorFromArray([42n, 7n, 3n], new Int64()),
    n_reads: vectorFromArray([6n, 4n, 2n], new Int64()),
    length: vectorFromArray([12, 900, 400], new Int32()),
    protein_length: vectorFromArray([0, 250, 80], new Int32()),
    rebuilt: vectorFromArray([false, true, false], new Bool()),
    n_absorbed: vectorFromArray([0n, 2n, 0n], new Int64()),
  });
}

const DETAIL = {
  cluster_id: 42,
  n_reads: 6,
  n_unique_sequences: 6,
  orf_start: 0,
  orf_end: 0,
  length: 12,
  protein_length: 0,
  round: 1,
  parent_template_id: 7,
  haplotype_id: 0,
  node_template_id: 1099511627776,
  rebuilt: false,
  n_absorbed: 0,
  consensus_content_id: '9007199254740993',
  protein_content_id: null,
  protein: null,
  samples: [
    { sample_id: 0, sample: 'testis_a', count: 2, tpm: 10 },
    { sample_id: 1, sample: 'testis_b', count: 3, tpm: 15 },
  ],
};

interface Server {
  urls: string[];
  meta: (url: URL) => SupportMeta;
}

function install(server: Server): void {
  vi.stubGlobal('fetch', async (input: string) => {
    server.urls.push(input);
    const url = new URL(input, 'http://x');
    const path = url.pathname;
    const json = (body: unknown) => new Response(JSON.stringify(body), { status: 200 });
    const arrow = (t: Table, headers: Record<string, string> = {}) =>
      new Response(tableToIPC(t, 'stream') as Uint8Array<ArrayBuffer>, { status: 200, headers });
    if (path === `/api/transcriptome/sessions/${SID}/runs`) return json([RUN]);
    if (path === `${BASE}/clusters`) return arrow(clusterPage(), { 'X-Total-Rows': '3' });
    const table = path.match(/\/support\/(\w+)$/);
    if (table) return arrow(supportTables()[table[1]]);
    if (path.endsWith('/support')) return json(server.meta(url));
    if (path === `${BASE}/clusters/42`) return json(DETAIL);
    return new Response('not found', { status: 404 });
  });
}

async function flush(): Promise<void> {
  for (let i = 0; i < 20; i += 1) await new Promise((r) => setTimeout(r, 0));
}

let host: HTMLElement;
let browser: TranscriptomeBrowser | null = null;
let server: Server;

beforeEach(() => {
  // jsdom has no canvas; say so quietly (the matrix is drawn on export).
  vi.spyOn(HTMLCanvasElement.prototype, 'getContext').mockReturnValue(null);
  host = document.createElement('div');
  document.body.appendChild(host);
  server = {
    urls: [],
    meta: (url) =>
      url.pathname.includes('/templates/')
        ? supportMeta({ space: (url.searchParams.get('space') as 'frame') ?? 'consensus' })
        : supportMeta({
            kind: 'node',
            cluster_id: 42,
            haplotype_id: 0,
            reference: url.searchParams.get('node') === 'pooled' ? 'pooled' : '0',
            space: (url.searchParams.get('space') as 'frame') ?? 'consensus',
          }),
  };
  install(server);
});

afterEach(() => {
  browser?.dispose();
  browser = null;
  host.remove();
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

/** Mount, and pick the first cluster. */
async function open(): Promise<SupportView> {
  browser = new TranscriptomeBrowser({ host, sessionId: SID });
  await browser.mount();
  await flush();
  (host.querySelector('.tx-picker-table tbody tr') as HTMLTableRowElement).click();
  await flush();
  return (browser as unknown as { support: SupportView }).support;
}

describe('TranscriptomeBrowser', () => {
  it('opens nothing until a cluster is picked (the largest can take a minute)', async () => {
    browser = new TranscriptomeBrowser({ host, sessionId: SID });
    await browser.mount();
    await flush();
    expect(server.urls.some((u) => u.includes('/support'))).toBe(false);
    expect(host.querySelector('.tx-loading')!.textContent).toContain('Pick a cluster');
  });

  it('opens what its URL names', async () => {
    expect(parseInitial('#cluster=42')).toEqual({ cluster: 42 });
    expect(parseInitial('#template=12:13194139969773')).toEqual({ template: { round: 12, templateId: 13194139969773 } });
    expect(parseInitial('#nothing')).toBeUndefined();
    browser = new TranscriptomeBrowser({ host, sessionId: SID, initial: { template: { round: 1, templateId: 7 } } });
    await browser.mount();
    await flush();
    expect(server.urls.filter((u) => u.includes('/rounds/1/templates/7/support'))).toHaveLength(7);
  });

  it('lists the run’s clusters and opens the first, its detail and its support', async () => {
    await open();
    const page = server.urls.find((u) => u.startsWith(`${BASE}/clusters?`))!;
    const q = new URL(page, 'http://x').searchParams;
    expect(Object.fromEntries(q)).toEqual({
      sort: 'n_reads',
      descending: 'true',
      min_reads: '2',
      offset: '0',
      limit: '200',
    });
    expect(host.querySelectorAll('.tx-picker-table tbody tr')).toHaveLength(3);
    expect(host.querySelector('.tx-picker-status')!.textContent).toBe('3 clusters');
    expect(host.querySelector('tr.selected')!.getAttribute('data-cluster')).toBe('42');
    // One summary and the six tables, all for this cluster's own node.
    const support = server.urls.filter((u) => u.includes('/clusters/42/support'));
    expect(support).toHaveLength(7);
    for (const u of support) expect(u).toContain('node=cluster&space=consensus');
    expect(host.querySelector('.tx-detail-title')!.textContent).toBe('cluster 42');
    expect(host.querySelector('.tx-detail')!.textContent).toContain('9007199254740993');
    expect(host.querySelectorAll('.tx-sample')).toHaveLength(2);
  });

  it('draws the tracks, the arcs and a verified status', async () => {
    await open();
    const foot = host.querySelector('.tx-foot')!.textContent!;
    expect(foot).toContain('✓ verified against the run');
    expect(foot).toContain('6 reads (5 voted in the PWM)');
    expect(host.querySelectorAll('.tx-tracks path.tx-arc')).toHaveLength(1);
    const options = Array.from(host.querySelectorAll('.tx-support select')[0].querySelectorAll('option')).map(
      (o) => (o as HTMLOptionElement).value,
    );
    expect(options).toEqual(['cluster', 'pooled', '0', '1']);
    // Node boundaries over the matrix.
    expect(host.querySelector('.canvas-layer-overlay')!.textContent).toContain('node 1');
  });

  it('clicking an arc sorts the reads by their state at its columns', async () => {
    await open();
    (host.querySelector('.tx-tracks path.tx-arc') as SVGPathElement).dispatchEvent(
      new MouseEvent('click', { bubbles: true }),
    );
    expect(host.querySelector('.tx-foot')!.textContent).toContain('rows by their state at columns 4 and 9');
    (Array.from(host.querySelectorAll('.tx-support .tx-btn')).find((b) => b.textContent === 'Default order') as HTMLButtonElement).click();
    expect(host.querySelector('.tx-foot')!.textContent).not.toContain('rows by their state');
  });

  it('compare re-reads the pooled template in frame space and keeps the differing columns', async () => {
    await open();
    const before = server.urls.length;
    (Array.from(host.querySelectorAll('.tx-support .tx-btn')).find((b) => b.textContent === 'Compare nodes') as HTMLButtonElement).click();
    await flush();
    const again = server.urls.slice(before);
    expect(again).toHaveLength(7);
    for (const u of again) expect(u).toContain('node=pooled&space=frame');
    const foot = host.querySelector('.tx-foot')!.textContent!;
    expect(foot).toContain('2 columns where the nodes differ');
    expect(foot).toContain('reference pooled consensus');
  });

  it('a replay that is not the run says so', async () => {
    const meta = server.meta;
    server.meta = (url) => ({ ...meta(url), verified: false, mismatch: ['columns: minor_frac differs'] });
    await open();
    expect(host.querySelector('.tx-foot')!.textContent).toContain('✗ not the run: columns: minor_frac differs');
  });

  it('opens a template of any round by id', async () => {
    await open();
    (host.querySelector('.tx-round, .tx-num') as HTMLInputElement).value = '1';
    (host.querySelector('.tx-tid') as HTMLInputElement).value = '7';
    (Array.from(host.querySelectorAll('.tx-top .tx-btn')).find((b) => b.textContent === 'Open template') as HTMLButtonElement).click();
    await flush();
    expect(server.urls.some((u) => u.includes('/rounds/1/templates/7/support?node=pooled&space=consensus'))).toBe(true);
    expect(host.querySelector('.tx-title')!.textContent).toBe('template 7 (round 1)');
  });

  it('exports the view as one SVG, the small matrix as vectors', async () => {
    const support = await open();
    const svg = support.exportSvg()!;
    expect(svg).toContain('<svg');
    expect(svg).toContain('tx-arc');
    expect(svg).not.toContain('Raster:');
    // The matrix cell holds rects (spans and ticks), not an image.
    expect((svg.match(/<rect/g) ?? []).length).toBeGreaterThan(10);
  });

  it('dispose removes the widget and drops late replies', async () => {
    browser = new TranscriptomeBrowser({ host, sessionId: SID });
    const mounting = browser.mount();
    browser.dispose();
    await mounting;
    await flush();
    expect(host.querySelector('.tx-browser')).toBeNull();
    browser = null;
  });
});
