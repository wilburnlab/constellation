// Test-only in-memory stand-in for the viz FastAPI server.
//
// Answers the endpoints the genome browser calls with payloads shaped
// like the real ones (track data comes from the kernel-generated Arrow
// fixtures) and records every request, so a test can assert on exactly
// what the widget asked for and in what order.

import { tableToIPC } from 'apache-arrow';
import { loadFixture, loadFixtureBytes, loadMetadata } from './load';

export const SESSION_ID = 'sess-1';
export const ORIGIN = 'http://localhost:3000';

export interface Call {
  method: string;
  /** Path + query string, origin stripped. */
  url: string;
  body?: unknown;
}

interface TrackEntry {
  kind: string;
  binding_id: string;
  label: string;
  source_id: string | null;
}

interface Source {
  source_id: string;
  path: string;
  kind: 'align' | 'cluster';
  label: string;
  assembly_accession: string | null;
}

const REFERENCE_ASSEMBLY = 'TestAssembly.1';

const SRC_ALIGN: Source = {
  source_id: 'src-aaaa0001',
  path: '/data/run-a/align',
  kind: 'align',
  label: 'run-a',
  assembly_accession: REFERENCE_ASSEMBLY,
};
const SRC_CLUSTER: Source = {
  source_id: 'src-bbbb0002',
  path: '/data/run-a/cluster',
  kind: 'cluster',
  label: 'run-a clusters',
  // Differs from the reference: the dataset manager shows a warning.
  assembly_accession: 'OtherAssembly.9',
};
export const ADDED_SOURCE_ID = 'src-cccc0003';

/** Fixture file serving each kind's data. */
const DATA_FIXTURE: Record<string, string> = {
  reference_sequence: 'reference_sequence.letters',
  gene_annotation: 'gene_annotation',
  coverage_histogram: 'coverage_histogram',
  read_pileup: 'read_pileup',
  cluster_pileup: 'cluster_pileup.clusters',
  splice_junctions: 'splice_junctions',
};

interface FakeResponse {
  ok: boolean;
  status: number;
  statusText: string;
  headers: Headers;
  json(): Promise<unknown>;
  text(): Promise<string>;
  arrayBuffer(): Promise<ArrayBuffer>;
}

function json(body: unknown, status = 200): FakeResponse {
  return {
    ok: status >= 200 && status < 300,
    status,
    statusText: status === 200 ? 'OK' : 'Error',
    headers: new Headers({ 'Content-Type': 'application/json' }),
    json: async () => body,
    text: async () => JSON.stringify(body),
    arrayBuffer: async () => new ArrayBuffer(0),
  };
}

function arrow(bytes: Uint8Array, mode: string): FakeResponse {
  const copy = bytes.slice();
  return {
    ok: true,
    status: 200,
    statusText: 'OK',
    headers: new Headers({ 'X-Track-Mode': mode }),
    json: async () => ({}),
    text: async () => '',
    arrayBuffer: async () => copy.buffer as ArrayBuffer,
  };
}

export class FakeServer {
  readonly calls: Call[] = [];
  /** `saved_as` reported by the manifest; a slug enables layout PATCHes. */
  savedAs: string | null = null;
  /** Cap the rows returned for a kind (0 = an empty table). */
  readonly rowLimit: Record<string, number> = {};
  /** Hits returned by the feature-search endpoint. */
  searchHits: unknown[] = [];
  /** When set, POST …/sources fails with this detail. */
  addSourceError: string | null = null;

  private sources: Source[] = [SRC_ALIGN, SRC_CLUSTER];

  /** Install as the global `fetch`. */
  readonly fetch = async (
    input: RequestInfo | URL,
    init?: RequestInit,
  ): Promise<Response> => {
    if (init?.signal?.aborted) {
      throw new DOMException('aborted', 'AbortError');
    }
    const url = new URL(String(input), ORIGIN);
    const method = (init?.method ?? 'GET').toUpperCase();
    const call: Call = { method, url: url.pathname + url.search };
    if (typeof init?.body === 'string') call.body = JSON.parse(init.body);
    this.calls.push(call);
    return this.route(method, url, call.body) as unknown as Response;
  };

  /** Requests recorded since `mark`, optionally narrowed by a substring. */
  since(mark: number, contains?: string): Call[] {
    const out = this.calls.slice(mark);
    return contains ? out.filter((c) => c.url.includes(contains)) : out;
  }

  /** Data-endpoint requests since `mark`, as `kind?query` strings. */
  dataRequests(mark = 0): string[] {
    return this.since(mark)
      .filter((c) => /^\/api\/tracks\/[^/]+\/data\?/.test(c.url))
      .map((c) => c.url.replace('/api/tracks/', '').replace('/data?', '?'));
  }

  // ------------------------------------------------------------------

  private route(method: string, url: URL, body: unknown): FakeResponse {
    const path = url.pathname;
    const sessionBase = `/api/sessions/${SESSION_ID}`;

    if (method === 'GET' && path === `${sessionBase}/manifest`) {
      return json(this.manifest());
    }
    if (method === 'GET' && path === `${sessionBase}/contigs`) {
      return json([
        { contig_id: 1, name: 'chr1', length: 12_000 },
        { contig_id: 2, name: 'chr2', length: 80_000 },
      ]);
    }
    if (method === 'GET' && path === `${sessionBase}/search`) {
      return json(this.searchHits);
    }
    if (method === 'POST' && path === `${sessionBase}/sources`) {
      if (this.addSourceError) return json({ detail: this.addSourceError }, 400);
      const payload = body as { path: string };
      this.sources = [
        ...this.sources,
        {
          source_id: ADDED_SOURCE_ID,
          path: payload.path,
          kind: 'align',
          label: 'run-b',
          assembly_accession: REFERENCE_ASSEMBLY,
        },
      ];
      return json(this.manifest(), 201);
    }
    if (method === 'DELETE' && path.startsWith(`${sessionBase}/sources/`)) {
      const sourceId = decodeURIComponent(path.slice(`${sessionBase}/sources/`.length));
      this.sources = this.sources.filter((s) => s.source_id !== sourceId);
      return json(this.manifest());
    }
    if (method === 'GET' && path === '/api/tracks') {
      return json(this.tracks());
    }
    if (method === 'PATCH' && path.startsWith('/api/saved-sessions/')) {
      return json({ slug: this.savedAs });
    }
    const track = /^\/api\/tracks\/([^/]+)\/(metadata|data)$/.exec(path);
    if (method === 'GET' && track) {
      const kind = decodeURIComponent(track[1]);
      if (track[2] === 'metadata') {
        return json(this.metadata(kind, url.searchParams.get('binding') ?? ''));
      }
      return this.data(kind, url);
    }
    return json({ detail: `unrouted ${method} ${path}` }, 404);
  }

  private manifest(): unknown {
    return {
      session_id: SESSION_ID,
      label: 'fixture session',
      reference: {
        handle: 'test_org@local_import-20260522',
        path: '/refs/test_org/local_import-20260522',
        genome: '/refs/test_org/local_import-20260522/genome',
        annotation: '/refs/test_org/local_import-20260522/annotation',
        assembly_accession: REFERENCE_ASSEMBLY,
      },
      sources: this.sources.map((s) => ({
        ...s,
        reference_handle: 'test_org@local_import-20260522',
        samples: [],
        slots: {},
      })),
      warnings: [],
      saved_as: this.savedAs,
      stages_present: {},
    };
  }

  /** Listing order is deliberately not the display order. */
  private tracks(): TrackEntry[] {
    const out: TrackEntry[] = [];
    this.sources.forEach((s, idx) => {
      if (s.kind === 'align') {
        out.push({
          kind: 'splice_junctions',
          binding_id: `splice_junctions-${idx}`,
          label: `Junctions (${s.label})`,
          source_id: s.source_id,
        });
      }
    });
    this.sources.forEach((s, idx) => {
      if (s.kind === 'cluster') {
        out.push({
          kind: 'cluster_pileup',
          binding_id: `cluster_pileup-${idx}`,
          label: `Transcript clusters (${s.label})`,
          source_id: s.source_id,
        });
      }
    });
    out.push({
      kind: 'reference_sequence',
      binding_id: 'reference_sequence',
      label: 'Reference sequence',
      source_id: null,
    });
    this.sources.forEach((s, idx) => {
      if (s.kind === 'align') {
        out.push({
          kind: 'coverage_histogram',
          binding_id: `coverage-${idx}`,
          label: `Coverage (${s.label})`,
          source_id: s.source_id,
        });
      }
    });
    out.push({
      kind: 'gene_annotation',
      binding_id: 'reference',
      label: 'Annotation (reference)',
      source_id: null,
    });
    this.sources.forEach((s, idx) => {
      if (s.kind === 'align') {
        out.push({
          kind: 'read_pileup',
          binding_id: `read_pileup-${idx}`,
          label: `Reads (${s.label})`,
          source_id: s.source_id,
        });
      }
    });
    // The added source ships no junctions, so its kind set differs.
    return out.filter(
      (t) => !(t.kind === 'splice_junctions' && t.source_id === ADDED_SOURCE_ID),
    );
  }

  private metadata(kind: string, bindingId: string): unknown {
    const all = loadMetadata();
    const key = Object.keys(all).find((k) => k.startsWith(`${kind}/`));
    const entry = this.tracks().find((t) => t.binding_id === bindingId);
    return { ...(key ? all[key] : {}), kind, binding_id: bindingId, label: entry?.label ?? kind };
  }

  private data(kind: string, url: URL): FakeResponse {
    let fixture = DATA_FIXTURE[kind];
    if (!fixture) return json({ detail: `unknown kind ${kind}` }, 404);
    if (kind === 'cluster_pileup' && url.searchParams.get('cluster_view') === 'members') {
      fixture = 'cluster_pileup.members';
    }
    const limit = this.rowLimit[kind];
    if (limit === undefined) return arrow(loadFixtureBytes(fixture), 'vector');
    return arrow(tableToIPC(loadFixture(fixture).slice(0, limit), 'stream'), 'vector');
  }
}
