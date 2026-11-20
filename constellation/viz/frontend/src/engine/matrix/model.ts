// An alignment matrix: rows (reads) across columns (alignment positions),
// held as each row's covered span plus its sparse *departures* from a
// reference — `(column, allele)` where it differs — and insertion markers.
//
// A read agrees with its reference at ~98-99% of columns, so this is the
// whole matrix at a few MB for 20,000 reads, and every level of detail is
// drawn from it on the client. Nothing here knows what a read or a
// transcript is: the genome pile-up could hand its rows over unchanged.

/** Allele codes: 0-3 = ACGT, 4 = gap (the row lacks the reference's
 *  base), 5 = N (covered, but no evidence). */
export const ALLELE_GAP = 4;
export const ALLELE_N = 5;
export const N_ALLELES = 6;

export interface MatrixData {
  nRows: number;
  nCols: number;
  /** Per row, its covered columns `[lo, hi)`; -1 for a row with no
   *  alignment in this space (listed, not drawn). */
  spanLo: Int32Array;
  spanHi: Int32Array;
  depRow: Int32Array;
  depCol: Int32Array;
  depAllele: Int8Array;
  insRow: Int32Array;
  /** The inserted bases sit just before this column. */
  insBefore: Int32Array;
  insLength: Int32Array;
  /** Per column, the reference's allele (for letters at matches); omit
   *  and matches draw as a plain span. */
  refAllele?: Int8Array;
}

/** {@link MatrixData} with each row's departures and insertions found in
 *  O(1): CSR offsets over entries sorted by (row, column). */
export interface MatrixIndex {
  data: MatrixData;
  depStart: Int32Array;
  depCol: Int32Array;
  depAllele: Int8Array;
  insStart: Int32Array;
  insBefore: Int32Array;
  insLength: Int32Array;
}

function csr(
  nRows: number,
  row: Int32Array,
  key: Int32Array,
): { start: Int32Array; order: Int32Array } {
  const n = row.length;
  const order = new Int32Array(n);
  for (let i = 0; i < n; i += 1) order[i] = i;
  order.sort((a, b) => row[a] - row[b] || key[a] - key[b]);
  const start = new Int32Array(nRows + 1);
  for (let i = 0; i < n; i += 1) {
    const r = row[i];
    if (r >= 0 && r < nRows) start[r + 1] += 1;
  }
  for (let r = 0; r < nRows; r += 1) start[r + 1] += start[r];
  return { start, order };
}

export function indexMatrix(data: MatrixData): MatrixIndex {
  const dep = csr(data.nRows, data.depRow, data.depCol);
  const depCol = new Int32Array(dep.order.length);
  const depAllele = new Int8Array(dep.order.length);
  dep.order.forEach((i, k) => {
    depCol[k] = data.depCol[i];
    depAllele[k] = data.depAllele[i];
  });
  const ins = csr(data.nRows, data.insRow, data.insBefore);
  const insBefore = new Int32Array(ins.order.length);
  const insLength = new Int32Array(ins.order.length);
  ins.order.forEach((i, k) => {
    insBefore[k] = data.insBefore[i];
    insLength[k] = data.insLength[i];
  });
  return {
    data,
    depStart: dep.start,
    depCol,
    depAllele,
    insStart: ins.start,
    insBefore,
    insLength,
  };
}

/** First index in `arr[lo:hi)` (ascending) whose value is >= `v`. */
export function lowerBound(arr: Int32Array, lo: number, hi: number, v: number): number {
  while (lo < hi) {
    const mid = (lo + hi) >>> 1;
    if (arr[mid] < v) lo = mid + 1;
    else hi = mid;
  }
  return lo;
}

/** The allele a row carries at a column: its departure, the reference's
 *  allele inside its span, or -1 outside it. The inverse of the encoding,
 *  and what a test rebuilds a row from. */
export function alleleAt(index: MatrixIndex, row: number, col: number): number {
  const { data } = index;
  const lo = data.spanLo[row];
  if (lo < 0 || col < lo || col >= data.spanHi[row]) return -1;
  const a = index.depStart[row];
  const b = index.depStart[row + 1];
  const k = lowerBound(index.depCol, a, b, col);
  if (k < b && index.depCol[k] === col) return index.depAllele[k];
  return data.refAllele ? data.refAllele[col] : -2;
}
