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

/** One query-string value: a scalar, or a list sent as a repeated
 *  parameter. `undefined` leaves the parameter out. */
export type TrackQueryValue = string | number | readonly string[] | undefined;

/** Query parameters for a track's data request. `session` and `binding`
 *  address the track; everything else is whatever that track's kernel
 *  declares in its query model (a locus for the genome kernels) — this
 *  layer does not know or name any of it. */
export interface TrackQueryParams {
  session: string;
  binding: string;
  [field: string]: TrackQueryValue;
}

/** Build the absolute `/api/tracks/{kind}/data` URL for a query. Pure,
 *  so the exact wire form is unit-testable. Parameters are emitted in
 *  the order they appear in `params`. */
export function buildTrackDataUrl(
  kind: string,
  params: TrackQueryParams,
  origin: string,
): string {
  const url = new URL(
    `/api/tracks/${encodeURIComponent(kind)}/data`,
    origin,
  );
  for (const [name, value] of Object.entries(params)) {
    if (value === undefined) continue;
    if (typeof value === 'string' || typeof value === 'number') {
      url.searchParams.set(name, String(value));
    } else {
      for (const item of value) url.searchParams.append(name, item);
    }
  }
  return url.toString();
}

export async function fetchTrackData(
  kind: string,
  params: TrackQueryParams,
  signal?: AbortSignal,
): Promise<FetchedTable> {
  const response = await fetch(
    buildTrackDataUrl(kind, params, window.location.origin),
    { signal },
  );
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
