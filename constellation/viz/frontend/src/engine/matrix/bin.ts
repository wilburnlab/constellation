// Binning the matrix for the image levels of detail: per bin of rows x
// columns, how many cells are covered, how many depart from the reference,
// and by which allele. Pure, so the arithmetic is tested apart from any
// drawing.
//
// Bins tile an extent of the data: `[e0, e1)` columns into `binsX` bins
// and `[f0, f1)` display rows into `binsY`, a value going to bin
// `floor((v + 0.5 - e0) / (e1 - e0) * bins)` by its centre. With an extent
// of whole columns and one bin each, a bin IS a column.

import { ALLELE_N, MatrixIndex, N_ALLELES, lowerBound } from './model';

export interface MatrixBins {
  binsX: number;
  binsY: number;
  /** Cells per bin, row-major `[by * binsX + bx]`. */
  cells: Float32Array;
  covered: Float32Array;
  /** Departures that are evidence (an N is covered but says nothing). */
  departed: Float32Array;
  /** Departures by allele, `[(by * binsX + bx) * N_ALLELES + allele]`. */
  alleles: Float32Array;
}

export interface BinExtent {
  cols: readonly [number, number];
  rows: readonly [number, number];
}

function binOf(v: number, e0: number, e1: number, n: number): number {
  const b = Math.floor(((v + 0.5 - e0) / (e1 - e0)) * n);
  return b < 0 ? 0 : b >= n ? n - 1 : b;
}

export function binMatrix(
  index: MatrixIndex,
  order: Int32Array,
  extent: BinExtent,
  binsX: number,
  binsY: number,
): MatrixBins {
  const { data } = index;
  const [e0, e1] = extent.cols;
  const [f0, f1] = extent.rows;
  const nb = binsX * binsY;
  const out: MatrixBins = {
    binsX,
    binsY,
    cells: new Float32Array(nb),
    covered: new Float32Array(nb),
    departed: new Float32Array(nb),
    alleles: new Float32Array(nb * N_ALLELES),
  };
  if (binsX <= 0 || binsY <= 0 || e1 <= e0 || f1 <= f0) return out;
  const c0 = Math.max(0, Math.ceil(e0 - 0.5));
  const c1 = Math.min(data.nCols, Math.ceil(e1 - 0.5));
  const r0 = Math.max(0, Math.ceil(f0 - 0.5));
  const r1 = Math.min(order.length, Math.ceil(f1 - 0.5));
  const nc = Math.max(0, c1 - c0);
  if (nc === 0 || r1 <= r0) return out;

  const colBin = new Int32Array(nc);
  const colsIn = new Float32Array(binsX);
  for (let c = 0; c < nc; c += 1) {
    colBin[c] = binOf(c0 + c, e0, e1, binsX);
    colsIn[colBin[c]] += 1;
  }
  const rowsIn = new Float32Array(binsY);
  for (let r = r0; r < r1; r += 1) rowsIn[binOf(r, f0, f1, binsY)] += 1;
  for (let by = 0; by < binsY; by += 1) {
    for (let bx = 0; bx < binsX; bx += 1) out.cells[by * binsX + bx] = rowsIn[by] * colsIn[bx];
  }

  // Coverage: a difference array over the visible columns per row bin,
  // one O(1) update a row, then one pass a row bin.
  const diff = new Int32Array(binsY * (nc + 1));
  for (let r = r0; r < r1; r += 1) {
    const d = order[r];
    const lo = data.spanLo[d];
    if (lo < 0) continue;
    const a = Math.max(lo, c0);
    const b = Math.min(data.spanHi[d], c1);
    if (a >= b) continue;
    const base = binOf(r, f0, f1, binsY) * (nc + 1);
    diff[base + a - c0] += 1;
    diff[base + b - c0] -= 1;
  }
  for (let by = 0; by < binsY; by += 1) {
    const base = by * (nc + 1);
    let run = 0;
    for (let c = 0; c < nc; c += 1) {
      run += diff[base + c];
      if (run) out.covered[by * binsX + colBin[c]] += run;
    }
  }

  for (let r = r0; r < r1; r += 1) {
    const d = order[r];
    const a = index.depStart[d];
    const b = index.depStart[d + 1];
    if (a === b) continue;
    const by = binOf(r, f0, f1, binsY);
    for (let k = lowerBound(index.depCol, a, b, c0); k < b; k += 1) {
      const col = index.depCol[k];
      if (col >= c1) break;
      const cell = by * binsX + colBin[col - c0];
      const allele = index.depAllele[k];
      out.alleles[cell * N_ALLELES + allele] += 1;
      if (allele !== ALLELE_N) out.departed[cell] += 1;
    }
  }
  return out;
}
