// Apache Arrow IPC fetcher — decodes /api/tracks/{kind}/data into Tables.
//
// The server speaks `application/vnd.apache.arrow.stream`; the JS package
// `apache-arrow` decodes the stream zero-copy into typed columnar buffers.
// We also surface the `X-Track-Mode` response header so the renderer
// can branch on vector vs hybrid without inspecting the schema.

import { tableFromIPC, Table } from 'apache-arrow';

export type TrackMode = 'vector' | 'hybrid';

export interface FetchedTable {
  table: Table;
  mode: TrackMode;
}

export interface TrackQueryParams {
  session: string;
  binding: string;
  contig: string;
  start: number;
  end: number;
  samples?: string[];
  viewport_px?: number;
  max_glyphs?: number;
  /** Kernel-pushdown filter (read_pileup only at present): drop
   *  alignments with mapq below this threshold at scan time. 0 admits
   *  every primary alignment (the default). */
  min_mapq?: number;
  /** Cluster_pileup view selector (PR 5): 'clusters' renders the
   *  default rectangle-per-cluster view; 'members' expands clusters
   *  into their member-read alignments with the read_pileup visual
   *  vocabulary. Undefined defaults to 'clusters' server-side. */
  cluster_view?: 'clusters' | 'members';
  force?: TrackMode;
}

export async function fetchTrackData(
  kind: string,
  params: TrackQueryParams,
  signal?: AbortSignal,
): Promise<FetchedTable> {
  const url = new URL(
    `/api/tracks/${encodeURIComponent(kind)}/data`,
    window.location.origin,
  );
  url.searchParams.set('session', params.session);
  url.searchParams.set('binding', params.binding);
  url.searchParams.set('contig', params.contig);
  url.searchParams.set('start', String(params.start));
  url.searchParams.set('end', String(params.end));
  if (params.viewport_px !== undefined) {
    url.searchParams.set('viewport_px', String(params.viewport_px));
  }
  if (params.max_glyphs !== undefined) {
    url.searchParams.set('max_glyphs', String(params.max_glyphs));
  }
  if (params.min_mapq !== undefined && params.min_mapq > 0) {
    url.searchParams.set('min_mapq', String(params.min_mapq));
  }
  if (params.cluster_view) {
    url.searchParams.set('cluster_view', params.cluster_view);
  }
  if (params.force) {
    url.searchParams.set('force', params.force);
  }
  for (const sample of params.samples ?? []) {
    url.searchParams.append('samples', sample);
  }

  const response = await fetch(url.toString(), { signal });
  if (!response.ok) {
    const body = await response.text().catch(() => '');
    throw new Error(
      `track fetch failed (${response.status}): ${body || response.statusText}`,
    );
  }

  const mode = (response.headers.get('X-Track-Mode') ?? 'vector') as TrackMode;
  const table = tableFromIPC(await response.arrayBuffer());
  return { table, mode };
}

export async function fetchJson<T>(path: string, signal?: AbortSignal): Promise<T> {
  const response = await fetch(path, { signal });
  if (!response.ok) {
    throw new Error(`fetch ${path} failed: ${response.status}`);
  }
  return (await response.json()) as T;
}

export async function fetchJsonMethod<T>(
  path: string,
  method: 'POST' | 'PUT' | 'PATCH' | 'DELETE',
  body?: unknown,
  signal?: AbortSignal,
): Promise<T> {
  const init: RequestInit = { method, signal };
  if (body !== undefined) {
    init.headers = { 'Content-Type': 'application/json' };
    init.body = JSON.stringify(body);
  }
  const response = await fetch(path, init);
  if (!response.ok) {
    let detail = '';
    try {
      const data = await response.json();
      if (data && typeof data === 'object' && 'detail' in data) {
        detail = ` — ${(data as { detail: unknown }).detail}`;
      }
    } catch {
      /* response wasn't JSON */
    }
    throw new Error(`${method} ${path} failed: ${response.status}${detail}`);
  }
  if (response.status === 204) {
    return undefined as T;
  }
  return (await response.json()) as T;
}
