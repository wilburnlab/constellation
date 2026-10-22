// ScanIndex — the scan table of one acquisition, held by the client so
// that moving between scans needs no request.
//
// Built from `…/acquisitions/{key}/scans` (one row per scan, in scan
// order: scan, level, rt, window, parent_scan, precursor_mz, …). Answers
// the questions a browser asks on every click and key press: which scan of
// a level and window is nearest a retention time, which is the next one,
// which scan another was taken from, which scans it led to.

import type { Table } from 'apache-arrow';

const NONE = -1;

export class ScanIndex {
  readonly size: number;
  readonly scans: Int32Array;
  readonly levels: Int8Array;
  readonly rts: Float64Array;
  /** DIA window id; -1 for none. */
  readonly windows: Int32Array;
  /** Parent scan number; -1 for none. */
  readonly parents: Int32Array;
  /** Precursor m/z; NaN for none. */
  readonly precursorMz: Float64Array;
  /** Isolation window bounds; NaN for none. */
  readonly isolationLower: Float64Array;
  readonly isolationUpper: Float64Array;
  /** Total ion current; 0 when not recorded. */
  readonly tics: Float64Array;
  private readonly rowOfScan = new Map<number, number>();
  private readonly groups = new Map<string, number[]>();

  constructor(table: Table) {
    const n = table.numRows;
    this.size = n;
    this.scans = new Int32Array(n);
    this.levels = new Int8Array(n);
    this.rts = new Float64Array(n);
    this.windows = new Int32Array(n).fill(NONE);
    this.parents = new Int32Array(n).fill(NONE);
    this.precursorMz = new Float64Array(n).fill(Number.NaN);
    this.isolationLower = new Float64Array(n).fill(Number.NaN);
    this.isolationUpper = new Float64Array(n).fill(Number.NaN);
    this.tics = new Float64Array(n);
    const col = (name: string) => table.getChild(name);
    const scan = col('scan');
    const level = col('level');
    const rt = col('rt');
    const window = col('window');
    const parent = col('parent_scan');
    const precursor = col('precursor_mz');
    const tic = col('tic');
    const lower = col('isolation_lower');
    const upper = col('isolation_upper');
    for (let i = 0; i < n; i++) {
      this.scans[i] = Number(scan?.get(i));
      this.levels[i] = Number(level?.get(i));
      this.rts[i] = Number(rt?.get(i));
      const w = window?.get(i);
      if (w !== null && w !== undefined) this.windows[i] = Number(w);
      const p = parent?.get(i);
      if (p !== null && p !== undefined) this.parents[i] = Number(p);
      const m = precursor?.get(i);
      if (m !== null && m !== undefined) this.precursorMz[i] = Number(m);
      const lo = lower?.get(i);
      if (lo !== null && lo !== undefined) this.isolationLower[i] = Number(lo);
      const hi = upper?.get(i);
      if (hi !== null && hi !== undefined) this.isolationUpper[i] = Number(hi);
      const t = tic?.get(i);
      if (t !== null && t !== undefined) this.tics[i] = Number(t);
      this.rowOfScan.set(this.scans[i], i);
      for (const key of [`${this.levels[i]}|*`, `${this.levels[i]}|${this.windows[i]}`]) {
        let rows = this.groups.get(key);
        if (!rows) this.groups.set(key, (rows = []));
        rows.push(i);
      }
    }
  }

  /** Rows of a level (and window; null = any), in scan order. */
  rows(level: number, window: number | null): readonly number[] {
    return this.groups.get(`${level}|${window ?? '*'}`) ?? [];
  }

  has(scan: number): boolean {
    return this.rowOfScan.has(scan);
  }

  row(scan: number): number | undefined {
    return this.rowOfScan.get(scan);
  }

  rtOf(scan: number): number | null {
    const r = this.rowOfScan.get(scan);
    return r === undefined ? null : this.rts[r];
  }

  levelOf(scan: number): number | null {
    const r = this.rowOfScan.get(scan);
    return r === undefined ? null : this.levels[r];
  }

  windowOf(scan: number): number | null {
    const r = this.rowOfScan.get(scan);
    return r === undefined || this.windows[r] === NONE ? null : this.windows[r];
  }

  parentOf(scan: number): number | null {
    const r = this.rowOfScan.get(scan);
    return r === undefined || this.parents[r] === NONE ? null : this.parents[r];
  }

  /** Whether `scan` isolated `mz`: within its isolation window when the
   *  scan records one, else within `slop` of its precursor m/z. Null when
   *  the scan records neither. */
  isolates(scan: number, mz: number, slop = 1.5): boolean | null {
    const r = this.rowOfScan.get(scan);
    if (r === undefined) return null;
    const lo = this.isolationLower[r];
    const hi = this.isolationUpper[r];
    if (Number.isFinite(lo) && Number.isFinite(hi) && hi > lo) return mz >= lo && mz <= hi;
    const p = this.precursorMz[r];
    return Number.isFinite(p) ? Math.abs(p - mz) <= slop : null;
  }

  /** The scan of `level` (and `window`) nearest `rt`. */
  nearest(level: number, window: number | null, rt: number): number | null {
    const rows = this.rows(level, window);
    if (rows.length === 0) return null;
    let lo = 0;
    let hi = rows.length - 1;
    while (lo < hi) {
      const mid = (lo + hi) >> 1;
      if (this.rts[rows[mid]] < rt) lo = mid + 1;
      else hi = mid;
    }
    let best = rows[lo];
    if (lo > 0 && Math.abs(this.rts[rows[lo - 1]] - rt) <= Math.abs(this.rts[best] - rt)) best = rows[lo - 1];
    return this.scans[best];
  }

  /** The scan of `level` with the highest total ion current. */
  mostIntense(level: number): number | null {
    let best = -1;
    for (const r of this.rows(level, null)) if (best < 0 || this.tics[r] > this.tics[best]) best = r;
    return best < 0 ? null : this.scans[best];
  }

  /** The scan `delta` places away among its level (and window). */
  step(scan: number, delta: number, window: number | null): number | null {
    const r = this.rowOfScan.get(scan);
    if (r === undefined) return null;
    const rows = this.rows(this.levels[r], window);
    const at = rows.indexOf(r);
    if (at < 0) return null;
    const next = Math.min(rows.length - 1, Math.max(0, at + delta));
    return this.scans[rows[next]];
  }

  /** The first scan of `level` (and `window`) after `scan`. */
  firstAfter(scan: number, level: number, window: number | null): number | null {
    const rows = this.rows(level, window);
    let lo = 0;
    let hi = rows.length;
    while (lo < hi) {
      const mid = (lo + hi) >> 1;
      if (this.scans[rows[mid]] <= scan) lo = mid + 1;
      else hi = mid;
    }
    return lo < rows.length ? this.scans[rows[lo]] : null;
  }

  /** The scans taken from `scan` (DDA: what it triggered), in scan order. */
  children(scan: number): number[] {
    const r = this.rowOfScan.get(scan);
    if (r === undefined) return [];
    const out: number[] = [];
    for (let i = r + 1; i < this.size && this.levels[i] > this.levels[r]; i++) {
      if (this.parents[i] === scan) out.push(this.scans[i]);
    }
    return out;
  }
}
