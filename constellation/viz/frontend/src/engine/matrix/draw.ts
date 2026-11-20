// Drawing an alignment matrix at the level of detail its zoom calls for
// (lod.ts). Drawn through a small sink, so the same pass paints a canvas
// on screen and writes vectors for an export (sinks.ts).
//
// Cell by cell when every column and row is at least a pixel: a grey span
// per row, its departures as coloured ticks, or as letters (with the
// reference's letters faint at matches) once a column is 8 px and a row
// 10. Otherwise one binned image: each bin grey by how much of it is
// covered, tinted toward its commonest departing allele by how often the
// covered cells depart.

import { Interval } from '../interval';
import { MatrixBins, binMatrix } from './bin';
import { Lod, chooseLod, isCellwise } from './lod';
import { ALLELE_GAP, ALLELE_N, MatrixIndex, N_ALLELES, lowerBound } from './model';

export interface MatrixSink {
  rect(x: number, y: number, w: number, h: number, fill: string, alpha: number): void;
  /** A centred glyph. */
  text(x: number, y: number, s: string, fill: string, alpha: number, fontPx: number): void;
  /** `w x h` RGBA pixels drawn into the rectangle `(x, y, dw, dh)`,
   *  unsmoothed. */
  image(rgba: Uint8ClampedArray, w: number, h: number, x: number, y: number, dw: number, dh: number): void;
}

export interface MatrixStyle {
  /** By allele code: A, C, G, T, gap, N. */
  allele: readonly string[];
  span: string;
  insertion: string;
  /** Letters at matches. */
  match: string;
  matchAlpha: number;
  /** A departing fraction at which an image bin is fully its allele's
   *  colour (a 25% minor allele should read as strongly as a fixed one). */
  densityGain: number;
}

// IGV's base colours, brightened for a dark surface.
export const DEFAULT_MATRIX_STYLE: MatrixStyle = {
  allele: ['#3fbf5f', '#4f8fff', '#f0a030', '#ff4f5f', '#e8e6ef', '#8a8893'],
  span: '#3b3a46',
  insertion: '#b77dff',
  match: '#c9c7d1',
  matchAlpha: 0.35,
  densityGain: 4,
};

const LETTERS = ['A', 'C', 'G', 'T', '-', 'N'];

export interface MatrixViewport {
  /** Columns shown, `[c0, c1)`, fractional. */
  cols: Interval;
  /** Display rows shown, `[r0, r1)`, fractional. */
  rows: Interval;
  widthPx: number;
  heightPx: number;
}

export interface MatrixDrawResult {
  lod: Lod;
  mode: 'cells' | 'image';
  /** Rects and glyphs drawn (cells), or pixels (image). */
  glyphs: number;
}

function scales(vp: MatrixViewport) {
  const pxPerCol = vp.widthPx / Math.max(1e-9, vp.cols[1] - vp.cols[0]);
  const pxPerRow = vp.heightPx / Math.max(1e-9, vp.rows[1] - vp.rows[0]);
  return { pxPerCol, pxPerRow, lod: chooseLod(pxPerCol, pxPerRow) };
}

export function matrixLod(vp: MatrixViewport): Lod {
  return scales(vp).lod;
}

function visible(index: MatrixIndex, order: Int32Array, vp: MatrixViewport) {
  return {
    c0: Math.max(0, Math.floor(vp.cols[0])),
    c1: Math.min(index.data.nCols, Math.ceil(vp.cols[1])),
    r0: Math.max(0, Math.floor(vp.rows[0])),
    r1: Math.min(order.length, Math.ceil(vp.rows[1])),
  };
}

/** What drawing this view would emit: rects and glyphs cell by cell, or
 *  the image's pixels — the figure an export weighs against its budget. */
export function matrixGlyphCost(index: MatrixIndex, order: Int32Array, vp: MatrixViewport): number {
  const { lod } = scales(vp);
  if (!isCellwise(lod)) return Math.ceil(vp.widthPx) * Math.ceil(vp.heightPx);
  const { c0, c1, r0, r1 } = visible(index, order, vp);
  const letters = lod.x === 'letters' && lod.y === 'legible' && !!index.data.refAllele;
  let n = 0;
  for (let r = r0; r < r1; r += 1) {
    const d = order[r];
    const lo = index.data.spanLo[d];
    if (lo < 0) continue;
    const a = Math.max(lo, c0);
    const b = Math.min(index.data.spanHi[d], c1);
    if (a >= b) continue;
    if (letters) n += b - a;
    else {
      n += 1;
      const s = index.depStart[d];
      const e = index.depStart[d + 1];
      const k0 = lowerBound(index.depCol, s, e, a);
      n += lowerBound(index.depCol, s, e, b) - k0;
    }
    const is = index.insStart[d];
    const ie = index.insStart[d + 1];
    n += lowerBound(index.insBefore, is, ie, b + 1) - lowerBound(index.insBefore, is, ie, a);
  }
  return n;
}

export function drawMatrix(
  sink: MatrixSink,
  index: MatrixIndex,
  order: Int32Array,
  vp: MatrixViewport,
  style: MatrixStyle = DEFAULT_MATRIX_STYLE,
): MatrixDrawResult {
  const { pxPerCol, pxPerRow, lod } = scales(vp);
  if (isCellwise(lod)) {
    return { lod, mode: 'cells', glyphs: drawCells(sink, index, order, vp, style, pxPerCol, pxPerRow, lod) };
  }
  return { lod, mode: 'image', glyphs: drawImage(sink, index, order, vp, style, pxPerCol, pxPerRow) };
}

function drawCells(
  sink: MatrixSink,
  index: MatrixIndex,
  order: Int32Array,
  vp: MatrixViewport,
  style: MatrixStyle,
  pxPerCol: number,
  pxPerRow: number,
  lod: Lod,
): number {
  const { data } = index;
  const { c0, c1, r0, r1 } = visible(index, order, vp);
  const letters = lod.x === 'letters' && lod.y === 'legible';
  const matchLetters = letters && !!data.refAllele;
  const gapY = pxPerRow >= 4 ? 1 : 0;
  const rowH = Math.max(1, pxPerRow - gapY);
  const tickW = Math.max(1, pxPerCol - (pxPerCol >= 4 ? 1 : 0));
  const fontPx = Math.min(14, Math.max(7, Math.min(pxPerCol, rowH) * 0.9));
  const x = (c: number) => (c - vp.cols[0]) * pxPerCol;
  let n = 0;
  for (let r = r0; r < r1; r += 1) {
    const d = order[r];
    const lo = data.spanLo[d];
    if (lo < 0) continue;
    const a = Math.max(lo, c0);
    const b = Math.min(data.spanHi[d], c1);
    if (a >= b) continue;
    const y = (r - vp.rows[0]) * pxPerRow;
    const s = index.depStart[d];
    const e = index.depStart[d + 1];
    let k = lowerBound(index.depCol, s, e, a);
    if (matchLetters) {
      const ref = data.refAllele!;
      for (let c = a; c < b; c += 1) {
        const departs = k < e && index.depCol[k] === c;
        const allele = departs ? index.depAllele[k] : ref[c];
        if (departs) k += 1;
        if (allele < 0 || allele >= N_ALLELES) continue;
        sink.text(
          x(c) + pxPerCol / 2,
          y + rowH / 2,
          LETTERS[allele],
          departs ? style.allele[allele] : style.match,
          departs ? 1 : style.matchAlpha,
          fontPx,
        );
        n += 1;
      }
    } else {
      sink.rect(x(a), y, (b - a) * pxPerCol, rowH, style.span, 1);
      n += 1;
      for (; k < e && index.depCol[k] < b; k += 1) {
        const allele = index.depAllele[k];
        const c = index.depCol[k];
        if (letters) {
          sink.text(x(c) + pxPerCol / 2, y + rowH / 2, LETTERS[allele], style.allele[allele], 1, fontPx);
        } else {
          sink.rect(x(c), y, tickW, rowH, style.allele[allele], allele === ALLELE_N ? 0.5 : 1);
        }
        n += 1;
      }
    }
    // Insertions: a mark at the column boundary they sit before.
    const is = index.insStart[d];
    const ie = index.insStart[d + 1];
    const markW = Math.max(1, Math.min(2, pxPerCol / 4));
    for (let j = lowerBound(index.insBefore, is, ie, a); j < ie; j += 1) {
      const before = index.insBefore[j];
      if (before > b) break;
      sink.rect(x(before) - markW / 2, y, markW, rowH, style.insertion, 1);
      n += 1;
    }
  }
  return n;
}

function rgb(hex: string): [number, number, number] {
  const h = hex.replace('#', '');
  const v = h.length === 3 ? h.split('').map((c) => c + c).join('') : h;
  return [parseInt(v.slice(0, 2), 16), parseInt(v.slice(2, 4), 16), parseInt(v.slice(4, 6), 16)];
}

/** One bin's colour: grey by covered fraction, toward its commonest
 *  departing allele by departing fraction x gain. Exported for the tests. */
export function binColour(
  bins: MatrixBins,
  cell: number,
  style: MatrixStyle,
): [number, number, number, number] {
  const covered = bins.covered[cell];
  if (covered <= 0) return [0, 0, 0, 0];
  const cover = Math.min(1, covered / Math.max(1, bins.cells[cell]));
  const span = rgb(style.span);
  const evidence = covered - bins.alleles[cell * N_ALLELES + ALLELE_N];
  const t = evidence > 0 ? Math.min(1, (bins.departed[cell] / evidence) * style.densityGain) : 0;
  let best = -1;
  let most = 0;
  for (let a = 0; a <= ALLELE_GAP; a += 1) {
    const v = bins.alleles[cell * N_ALLELES + a];
    if (v > most) {
      most = v;
      best = a;
    }
  }
  const tint = best >= 0 ? rgb(style.allele[best]) : span;
  const alpha = Math.max(0.35 + 0.65 * cover, t);
  return [
    Math.round(span[0] + (tint[0] - span[0]) * t),
    Math.round(span[1] + (tint[1] - span[1]) * t),
    Math.round(span[2] + (tint[2] - span[2]) * t),
    Math.round(alpha * 255),
  ];
}

function drawImage(
  sink: MatrixSink,
  index: MatrixIndex,
  order: Int32Array,
  vp: MatrixViewport,
  style: MatrixStyle,
  pxPerCol: number,
  pxPerRow: number,
): number {
  const { c0, c1, r0, r1 } = visible(index, order, vp);
  if (c1 <= c0 || r1 <= r0) return 0;
  // A column (row) of at least a pixel is its own bin, drawn over its
  // whole extent; below a pixel, the bins are the pixels.
  const xs =
    pxPerCol >= 1
      ? { e: [c0, c1] as const, n: c1 - c0, px: (c0 - vp.cols[0]) * pxPerCol, w: (c1 - c0) * pxPerCol }
      : { e: vp.cols, n: Math.max(1, Math.ceil(vp.widthPx)), px: 0, w: vp.widthPx };
  const ys =
    pxPerRow >= 1
      ? { e: [r0, r1] as const, n: r1 - r0, px: (r0 - vp.rows[0]) * pxPerRow, h: (r1 - r0) * pxPerRow }
      : { e: vp.rows, n: Math.max(1, Math.ceil(vp.heightPx)), px: 0, h: vp.heightPx };
  const bins = binMatrix(index, order, { cols: xs.e, rows: ys.e }, xs.n, ys.n);
  const rgba = new Uint8ClampedArray(xs.n * ys.n * 4);
  for (let cell = 0; cell < xs.n * ys.n; cell += 1) {
    const c = binColour(bins, cell, style);
    rgba[cell * 4] = c[0];
    rgba[cell * 4 + 1] = c[1];
    rgba[cell * 4 + 2] = c[2];
    rgba[cell * 4 + 3] = c[3];
  }
  sink.image(rgba, xs.n, ys.n, xs.px, ys.px, xs.w, ys.h);
  return xs.n * ys.n;
}
