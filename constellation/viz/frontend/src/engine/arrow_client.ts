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
