// The support view's tables as arrays the drawing reads, and the row
// orders it shows them in. Pure: built from Arrow tables, tested apart from
// any DOM.
//
// Rows are reads, grouped by node, and within a node ordered by their
// *pattern* — their alleles at the template's candidate columns — then by
// where they start and how well they matched. Ordering by pattern is what
// makes covariance visible once rows are aggregated: reads that depart
// together sit together, so a split is a block.

import type { Table } from 'apache-arrow';
import { ALLELE_N, MatrixData, MatrixIndex, alleleAt, indexMatrix } from '../../engine/matrix/model';
import type { SupportMeta } from './types';

export interface ReadInfo {
  n: number;
  readRow: Int32Array;
  readId: (string | null)[];
  sampleId: (number | null)[];
  node: Int32Array;
  /** The node of the final merge's group the read came from, as a string
   *  (an int64 id), or null. */
  absorbedFrom: (string | null)[];
  sampled: Uint8Array;
  folded: (boolean | null)[];
  margin: Float64Array;
  identity: Float64Array;
  quality: Float64Array;
  loglDelta: Float64Array;
  runnerUp: (string | null)[];
  runnerUpGap: Float64Array;
  clip5: Float64Array;
  clip3: Float64Array;
}

export interface ColumnInfo {
  n: number;
  col: Int32Array;
  frameCol: Int32Array;
  templateCol: Int32Array;
  refBase: Int8Array;
  depth: Float64Array;
  baseCov: Float64Array;
  major: Int8Array;
  minorFrac: Float64Array;
  eps: Float64Array;
  hpRun: Float64Array;
  candidate: Uint8Array;
  route: Int8Array;
  signature: Int32Array;
}

export interface ArcInfo {
  n: number;
  u: Int32Array;
  v: Int32Array;
  selfEdge: Uint8Array;
  n11: Float64Array;
  total: Float64Array;
  phi: Float64Array;
  logOdds: Float64Array;
  pValue: Float64Array;
}

export interface DifferenceInfo {
  n: number;
  frameCol: Int32Array;
  haplotypeId: Int32Array;
  allele: Int8Array;
}

export interface SupportData {
  meta: SupportMeta;
  reads: ReadInfo;
  columns: ColumnInfo;
  arcs: ArcInfo;
  differences: DifferenceInfo;
  matrix: MatrixData;
  index: MatrixIndex;
  /** Column (this space) of each frame column, -1 where it has none. */
  colOfFrame: Int32Array;
  /** Row of `columns` describing each column (this space), -1 for none. */
  columnRow: Int32Array;
}

// ── decoding ──────────────────────────────────────────────────────────

function nums(t: Table, name: string, fill = NaN): Float64Array {
  const v = t.getChild(name);
  const out = new Float64Array(t.numRows).fill(fill);
  if (!v) return out;
  for (let i = 0; i < t.numRows; i += 1) {
    const x = v.get(i);
    if (x !== null && x !== undefined) out[i] = Number(x);
  }
  return out;
}

function ints(t: Table, name: string, fill = -1): Int32Array {
  return Int32Array.from(nums(t, name, fill));
}

function int8s(t: Table, name: string, fill = -1): Int8Array {
  return Int8Array.from(nums(t, name, fill));
}

function bools(t: Table, name: string): Uint8Array {
  const v = t.getChild(name);
  const out = new Uint8Array(t.numRows);
  if (!v) return out;
  for (let i = 0; i < t.numRows; i += 1) out[i] = v.get(i) ? 1 : 0;
  return out;
}

function nullableBools(t: Table, name: string): (boolean | null)[] {
  const v = t.getChild(name);
  return Array.from({ length: t.numRows }, (_, i) => {
    const x = v?.get(i);
    return x === null || x === undefined ? null : Boolean(x);
  });
}

function strings(t: Table, name: string): (string | null)[] {
  const v = t.getChild(name);
  return Array.from({ length: t.numRows }, (_, i) => {
    const x = v?.get(i);
    return x === null || x === undefined ? null : String(x);
  });
}

function maybeNumbers(t: Table, name: string): (number | null)[] {
  const v = t.getChild(name);
  return Array.from({ length: t.numRows }, (_, i) => {
    const x = v?.get(i);
    return x === null || x === undefined ? null : Number(x);
  });
}

export function supportFromTables(meta: SupportMeta, tables: Record<string, Table>): SupportData {
  const r = tables.reads;
  const reads: ReadInfo = {
    n: r.numRows,
    readRow: ints(r, 'read_row'),
    readId: strings(r, 'read_id'),
    sampleId: maybeNumbers(r, 'sample_id'),
    node: ints(r, 'node', 0),
    absorbedFrom: strings(r, 'absorbed_from'),
    sampled: bools(r, 'sampled'),
    folded: nullableBools(r, 'folded'),
    margin: nums(r, 'margin'),
    identity: nums(r, 'identity'),
    quality: nums(r, 'dorado_quality'),
    loglDelta: nums(r, 'logl_delta'),
    runnerUp: strings(r, 'runner_up_template_id'),
    runnerUpGap: nums(r, 'runner_up_logl_gap'),
    clip5: nums(r, 'clip_5p'),
    clip3: nums(r, 'clip_3p'),
  };
  const c = tables.columns;
  const columns: ColumnInfo = {
    n: c.numRows,
    col: ints(c, 'col'),
    frameCol: ints(c, 'frame_col'),
    templateCol: ints(c, 'template_col'),
    refBase: int8s(c, 'ref_base'),
    depth: nums(c, 'depth', 0),
    baseCov: nums(c, 'base_cov', 0),
    major: int8s(c, 'major'),
    minorFrac: nums(c, 'minor_frac', 0),
    eps: nums(c, 'eps'),
    hpRun: nums(c, 'hp_run'),
    candidate: bools(c, 'candidate'),
    route: int8s(c, 'route'),
    signature: ints(c, 'signature'),
  };
  const a = tables.arcs;
  const arcs: ArcInfo = {
    n: a.numRows,
    u: ints(a, 'u_col'),
    v: ints(a, 'v_col'),
    selfEdge: bools(a, 'self_edge'),
    n11: nums(a, 'n11'),
    total: nums(a, 'n'),
    phi: nums(a, 'phi'),
    logOdds: nums(a, 'log_odds'),
    pValue: nums(a, 'p_value', 1),
  };
  const d = tables.differences;
  const differences: DifferenceInfo = {
    n: d.numRows,
    frameCol: ints(d, 'frame_col'),
    haplotypeId: ints(d, 'haplotype_id'),
    allele: int8s(d, 'allele'),
  };
  const nCols = meta.n_columns;
  const refAllele = new Int8Array(nCols).fill(-1);
  for (let i = 0; i < columns.n; i += 1) {
    const col = columns.col[i];
    if (col >= 0 && col < nCols) refAllele[col] = columns.refBase[i];
  }
  const columnRow = new Int32Array(nCols).fill(-1);
  for (let i = 0; i < columns.n; i += 1) {
    const col = columns.col[i];
    if (col >= 0 && col < nCols) columnRow[col] = i;
  }
  // A loop, not `Math.max(...cols)`: a spread passes every column as an
  // argument, and a long alignment exceeds the engine's argument limit.
  let maxFrame = 0;
  for (let i = 0; i < columns.n; i += 1) maxFrame = Math.max(maxFrame, columns.frameCol[i] + 1);
  const colOfFrame = new Int32Array(Math.max(maxFrame, 0)).fill(-1);
  for (let i = 0; i < columns.n; i += 1) colOfFrame[columns.frameCol[i]] = columns.col[i];

  const dep = tables.departures;
  const ins = tables.insertions;
  const matrix: MatrixData = {
    nRows: reads.n,
    nCols,
    spanLo: ints(r, 'span_lo'),
    spanHi: ints(r, 'span_hi'),
    depRow: ints(dep, 'read_idx'),
    depCol: ints(dep, 'col'),
    depAllele: int8s(dep, 'allele'),
    insRow: ints(ins, 'read_idx'),
    insBefore: ints(ins, 'before_col'),
    insLength: ints(ins, 'length', 0),
    refAllele,
  };
  return {
    meta,
    reads,
    columns,
    arcs,
    differences,
    matrix,
    index: indexMatrix(matrix),
    colOfFrame,
    columnRow,
  };
}

// ── row order ─────────────────────────────────────────────────────────

/** A row's allele at a column for sorting: its allele, with uncovered
 *  (and unsampled) sorting after every allele. */
function sortAllele(index: MatrixIndex, row: number, col: number): number {
  const a = alleleAt(index, row, col);
  return a < 0 ? 99 : a;
}

function compareDefault(data: SupportData, pattern: number[][], a: number, b: number): number {
  const { reads, matrix } = data;
  if (reads.node[a] !== reads.node[b]) return reads.node[a] - reads.node[b];
  const sa = matrix.spanLo[a] < 0;
  const sb = matrix.spanLo[b] < 0;
  if (sa !== sb) return sa ? 1 : -1; // unsampled after sampled
  const pa = pattern[a];
  const pb = pattern[b];
  for (let k = 0; k < pa.length; k += 1) if (pa[k] !== pb[k]) return pa[k] - pb[k];
  if (matrix.spanLo[a] !== matrix.spanLo[b]) return matrix.spanLo[a] - matrix.spanLo[b];
  const ia = Number.isFinite(reads.identity[a]) ? reads.identity[a] : -1;
  const ib = Number.isFinite(reads.identity[b]) ? reads.identity[b] : -1;
  if (ia !== ib) return ib - ia;
  return a - b;
}

/** The candidate columns that carry a signature (else every candidate),
 *  in this space — what a row's pattern is read at. */
export function patternColumns(data: SupportData): number[] {
  const { columns } = data;
  const signed: number[] = [];
  const all: number[] = [];
  for (let i = 0; i < columns.n; i += 1) {
    if (!columns.candidate[i]) continue;
    all.push(columns.col[i]);
    if (columns.signature[i] >= 0) signed.push(columns.col[i]);
  }
  return signed.length ? signed : all;
}

/** Rows grouped by node, then by pattern, then start, then identity. */
export function defaultOrder(data: SupportData): Int32Array {
  const cols = patternColumns(data);
  const pattern = Array.from({ length: data.reads.n }, (_, r) =>
    cols.map((c) => sortAllele(data.index, r, c)),
  );
  const order = Array.from({ length: data.reads.n }, (_, i) => i);
  order.sort((a, b) => compareDefault(data, pattern, a, b));
  return Int32Array.from(order);
}

/** Rows sorted by their joint state at two columns (an arc's ends), then
 *  as {@link defaultOrder} — so the 2 x 2 an arc stands for is four blocks. */
export function orderByPair(data: SupportData, u: number, v: number): Int32Array {
  const cols = patternColumns(data);
  const pattern = Array.from({ length: data.reads.n }, (_, r) =>
    cols.map((c) => sortAllele(data.index, r, c)),
  );
  const joint = Array.from({ length: data.reads.n }, (_, r) => [
    sortAllele(data.index, r, u),
    sortAllele(data.index, r, v),
  ]);
  const order = Array.from({ length: data.reads.n }, (_, i) => i);
  order.sort((a, b) => {
    for (let k = 0; k < 2; k += 1) if (joint[a][k] !== joint[b][k]) return joint[a][k] - joint[b][k];
    return compareDefault(data, pattern, a, b);
  });
  return Int32Array.from(order);
}

/** The rows of one node, in `order`'s order. */
export function rowsOfNode(data: SupportData, order: Int32Array, node: number | null): Int32Array {
  if (node === null) return order;
  return order.filter((r) => data.reads.node[r] === node);
}

// ── comparison: only the columns where nodes differ ───────────────────

/** The columns (this space) where the template's nodes' consensus differ. */
export function differingColumns(data: SupportData): Int32Array {
  const out = new Set<number>();
  for (let i = 0; i < data.differences.n; i += 1) {
    const f = data.differences.frameCol[i];
    const c = f < data.colOfFrame.length ? data.colOfFrame[f] : -1;
    if (c >= 0) out.add(c);
  }
  return Int32Array.from([...out].sort((a, b) => a - b));
}

/** The matrix restricted to `cols` (ascending), renumbered 0..cols.length:
 *  a row's span is the selected columns inside it, its departures and the
 *  reference those at them. Insertions are dropped (no column between two
 *  selected ones is shown). */
export function restrictColumns(m: MatrixData, cols: Int32Array): MatrixData {
  const pos = new Map<number, number>();
  cols.forEach((c, i) => pos.set(c, i));
  const below = (v: number): number => {
    // number of selected columns < v
    let lo = 0;
    let hi = cols.length;
    while (lo < hi) {
      const mid = (lo + hi) >>> 1;
      if (cols[mid] < v) lo = mid + 1;
      else hi = mid;
    }
    return lo;
  };
  const spanLo = new Int32Array(m.nRows);
  const spanHi = new Int32Array(m.nRows);
  for (let r = 0; r < m.nRows; r += 1) {
    if (m.spanLo[r] < 0) {
      spanLo[r] = -1;
      spanHi[r] = -1;
      continue;
    }
    const a = below(m.spanLo[r]);
    const b = below(m.spanHi[r]);
    spanLo[r] = a < b ? a : -1;
    spanHi[r] = a < b ? b : -1;
  }
  const depRow: number[] = [];
  const depCol: number[] = [];
  const depAllele: number[] = [];
  for (let i = 0; i < m.depRow.length; i += 1) {
    const p = pos.get(m.depCol[i]);
    if (p === undefined) continue;
    depRow.push(m.depRow[i]);
    depCol.push(p);
    depAllele.push(m.depAllele[i]);
  }
  return {
    nRows: m.nRows,
    nCols: cols.length,
    spanLo,
    spanHi,
    depRow: Int32Array.from(depRow),
    depCol: Int32Array.from(depCol),
    depAllele: Int8Array.from(depAllele),
    insRow: new Int32Array(0),
    insBefore: new Int32Array(0),
    insLength: new Int32Array(0),
    refAllele: m.refAllele ? Int8Array.from(cols, (c) => m.refAllele![c]) : undefined,
  };
}

/** Each read's departure count against the expected for its length at a
 *  per-base error rate — the noise-conformity check a node of one
 *  transcript should pass: reads hotter than thermal are candidate
 *  mis-assignments or unresolved variants.
 *
 *  An ambiguous base (allele 5, N) is no evidence either way: it is not
 *  counted as a departure, and its column leaves the span the expectation
 *  is taken over. A read with no comparable column is NaN (no indicator). */
export function departureExcess(data: SupportData, errorRate = 0.012): Float64Array {
  const { matrix, index } = data;
  const out = new Float64Array(matrix.nRows).fill(NaN);
  for (let r = 0; r < matrix.nRows; r += 1) {
    if (matrix.spanLo[r] < 0) continue;
    let ambiguous = 0;
    for (let i = index.depStart[r]; i < index.depStart[r + 1]; i += 1) {
      if (index.depAllele[i] === ALLELE_N) ambiguous += 1;
    }
    const comparable = matrix.spanHi[r] - matrix.spanLo[r] - ambiguous;
    if (comparable <= 0) continue;
    const observed = index.depStart[r + 1] - index.depStart[r] - ambiguous;
    out[r] = observed / Math.max(1e-9, comparable * errorRate);
  }
  return out;
}
