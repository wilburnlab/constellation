import { describe, expect, it } from 'vitest';
import { binMatrix } from './bin';
import {
  DEFAULT_MATRIX_STYLE,
  MatrixSink,
  MatrixViewport,
  binColour,
  drawMatrix,
  matrixGlyphCost,
  matrixLod,
} from './draw';
import { chooseLod, isCellwise } from './lod';
import { ALLELE_N, MatrixData, alleleAt, indexMatrix } from './model';
import { svgSink } from './sinks';

/** Four rows over ten columns:
 *  row 0  [0,10)  departs at 2 (T) and 7 (gap)
 *  row 1  [2, 8)  departs at 2 (T), an N at 5
 *  row 2  no span (unsampled)
 *  row 3  [5,10)  departs at 9 (A); an insertion before column 6 */
function data(): MatrixData {
  return {
    nRows: 4,
    nCols: 10,
    spanLo: Int32Array.from([0, 2, -1, 5]),
    spanHi: Int32Array.from([10, 8, -1, 10]),
    // Deliberately out of (row, column) order.
    depRow: Int32Array.from([0, 3, 1, 0, 1]),
    depCol: Int32Array.from([7, 9, 5, 2, 2]),
    depAllele: Int8Array.from([4, 0, ALLELE_N, 3, 3]),
    insRow: Int32Array.from([3]),
    insBefore: Int32Array.from([6]),
    insLength: Int32Array.from([2]),
    refAllele: Int8Array.from([0, 1, 2, 3, 0, 1, 2, 3, 0, 1]),
  };
}

const ORDER = Int32Array.from([0, 1, 2, 3]);

interface Op {
  op: 'rect' | 'text' | 'image';
  args: unknown[];
}

function recorder(): MatrixSink & { ops: Op[] } {
  const ops: Op[] = [];
  return {
    ops,
    rect: (...args) => ops.push({ op: 'rect', args }),
    text: (...args) => ops.push({ op: 'text', args }),
    image: (...args) => ops.push({ op: 'image', args }),
  };
}

function vp(widthPx: number, heightPx: number, cols: [number, number] = [0, 10], rows: [number, number] = [0, 4]): MatrixViewport {
  return { cols, rows, widthPx, heightPx };
}

describe('the model', () => {
  it('rebuilds every row from its span, departures and the reference', () => {
    const index = indexMatrix(data());
    const ref = data().refAllele!;
    const want: number[][] = [
      [0, 1, 3, 3, 0, 1, 2, 4, 0, 1],
      [-1, -1, 3, 3, 0, ALLELE_N, 2, 3, -1, -1],
      Array(10).fill(-1),
      [-1, -1, -1, -1, -1, 1, 2, 3, 0, 0],
    ];
    for (let r = 0; r < 4; r += 1) {
      const got = Array.from({ length: 10 }, (_, c) => alleleAt(index, r, c));
      expect(got, `row ${r}`).toEqual(want[r]);
    }
    expect(ref.length).toBe(10);
  });
});

describe('level of detail', () => {
  it('follows the 3 x 3 table on each axis independently', () => {
    expect(chooseLod(8, 10)).toEqual({ x: 'letters', y: 'legible' });
    expect(chooseLod(7.9, 9.9)).toEqual({ x: 'ticks', y: 'thin' });
    expect(chooseLod(1, 1)).toEqual({ x: 'ticks', y: 'thin' });
    expect(chooseLod(0.99, 0.99)).toEqual({ x: 'density', y: 'aggregated' });
    expect(chooseLod(20, 0.5)).toEqual({ x: 'letters', y: 'aggregated' });
    expect(chooseLod(0.5, 20)).toEqual({ x: 'density', y: 'legible' });
    expect(isCellwise(chooseLod(2, 2))).toBe(true);
    expect(isCellwise(chooseLod(20, 0.5))).toBe(false);
    expect(isCellwise(chooseLod(0.5, 20))).toBe(false);
  });
});

describe('binning', () => {
  it('one bin per column and row is the matrix itself', () => {
    const b = binMatrix(indexMatrix(data()), ORDER, { cols: [0, 10], rows: [0, 4] }, 10, 4);
    const covered = Array.from(b.covered);
    expect(covered.slice(0, 10)).toEqual(Array(10).fill(1));
    expect(covered.slice(10, 20)).toEqual([0, 0, 1, 1, 1, 1, 1, 1, 0, 0]);
    expect(covered.slice(20, 30)).toEqual(Array(10).fill(0));
    expect(covered.slice(30, 40)).toEqual([0, 0, 0, 0, 0, 1, 1, 1, 1, 1]);
    // Departures that are evidence: the N at (1, 5) is covered, not departed.
    expect(b.departed[0 * 10 + 2]).toBe(1);
    expect(b.departed[1 * 10 + 5]).toBe(0);
    expect(b.alleles[(1 * 10 + 5) * 6 + ALLELE_N]).toBe(1);
    expect(b.departed[3 * 10 + 9]).toBe(1);
  });

  it('coarser bins add their cells up', () => {
    // Two column bins of five, one row bin of all four rows.
    const b = binMatrix(indexMatrix(data()), ORDER, { cols: [0, 10], rows: [0, 4] }, 2, 1);
    expect(Array.from(b.cells)).toEqual([20, 20]);
    expect(Array.from(b.covered)).toEqual([5 + 3 + 0 + 0, 5 + 3 + 5]);
    expect(Array.from(b.departed)).toEqual([2, 2]); // (0,2),(1,2) | (0,7),(3,9)
  });

  it('a bin colour is grey where nothing departs and its allele where much does', () => {
    const index = indexMatrix(data());
    const b = binMatrix(index, ORDER, { cols: [0, 10], rows: [0, 4] }, 10, 4);
    const grey = binColour(b, 0, DEFAULT_MATRIX_STYLE);
    expect(grey.slice(0, 3)).toEqual([0x3b, 0x3a, 0x46]);
    const t = binColour(b, 2, DEFAULT_MATRIX_STYLE); // row 0, column 2: a T
    expect(t.slice(0, 3)).toEqual([0xff, 0x4f, 0x5f]);
    expect(binColour(b, 20, DEFAULT_MATRIX_STYLE)[3]).toBe(0); // uncovered
  });
});

describe('drawing', () => {
  it('letters, with the reference faint at matches, when columns and rows are legible', () => {
    const sink = recorder();
    const res = drawMatrix(sink, indexMatrix(data()), ORDER, vp(100, 40));
    expect(res.lod).toEqual({ x: 'letters', y: 'legible' });
    expect(res.mode).toBe('cells');
    const texts = sink.ops.filter((o) => o.op === 'text');
    // 10 + 6 + 0 + 5 covered cells, plus one insertion mark (a rect).
    expect(texts).toHaveLength(21);
    expect(sink.ops.filter((o) => o.op === 'rect')).toHaveLength(1);
    const strong = texts.filter((o) => o.args[4] === 1).map((o) => o.args[2]);
    expect(strong.sort()).toEqual(['-', 'A', 'N', 'T', 'T']);
    expect(res.glyphs).toBe(matrixGlyphCost(indexMatrix(data()), ORDER, vp(100, 40)));
  });

  it('ticks on a grey span when columns are narrower than a letter', () => {
    const sink = recorder();
    const res = drawMatrix(sink, indexMatrix(data()), ORDER, vp(40, 20));
    expect(res.lod).toEqual({ x: 'ticks', y: 'thin' });
    const rects = sink.ops.filter((o) => o.op === 'rect');
    // 3 spans, 5 departures, 1 insertion mark.
    expect(rects).toHaveLength(9);
    expect(sink.ops.some((o) => o.op === 'text')).toBe(false);
    expect(res.glyphs).toBe(9);
    expect(matrixGlyphCost(indexMatrix(data()), ORDER, vp(40, 20))).toBe(9);
  });

  it('one image when rows aggregate, bins of whole columns', () => {
    const sink = recorder();
    // 10 columns over 100 px (10 px each), 4 rows over 2 px.
    const res = drawMatrix(sink, indexMatrix(data()), ORDER, vp(100, 2));
    expect(res.lod).toEqual({ x: 'letters', y: 'aggregated' });
    expect(res.mode).toBe('image');
    const img = sink.ops.filter((o) => o.op === 'image');
    expect(img).toHaveLength(1);
    const [rgba, w, h, x, y, dw, dh] = img[0].args as [Uint8ClampedArray, number, number, number, number, number, number];
    expect([w, h, x, y, dw, dh]).toEqual([10, 2, 0, 0, 100, 2]);
    expect(rgba.length).toBe(10 * 2 * 4);
  });

  it('one image of pixels when columns are under a pixel', () => {
    const sink = recorder();
    const res = drawMatrix(sink, indexMatrix(data()), ORDER, vp(5, 40));
    expect(res.lod).toEqual({ x: 'density', y: 'legible' });
    const [, w, h] = sink.ops[0].args as [Uint8ClampedArray, number, number];
    expect([w, h]).toEqual([5, 4]);
    expect(matrixLod(vp(5, 40))).toEqual(res.lod);
  });

  it('draws only what is in view', () => {
    const sink = recorder();
    // Columns [5, 7) of rows [3, 4), at 5 px a column (ticks): row 3's span
    // from 5 and the insertion before 6; its departure at 9 is out of view.
    drawMatrix(sink, indexMatrix(data()), ORDER, vp(10, 10, [5, 7], [3, 4]));
    const rects = sink.ops.filter((o) => o.op === 'rect');
    expect(rects).toHaveLength(2);
    expect(rects[0].args.slice(0, 4)).toEqual([0, 0, 10, 9]);
  });

  it('writes the same cells into SVG for a vector export', () => {
    const g = document.createElementNS('http://www.w3.org/2000/svg', 'g') as SVGGElement;
    const sink = svgSink(g);
    drawMatrix(sink, indexMatrix(data()), ORDER, vp(40, 20));
    expect(g.querySelectorAll('rect')).toHaveLength(9);
    const texts = svgSink(g);
    drawMatrix(texts, indexMatrix(data()), ORDER, vp(100, 40));
    expect(Array.from(g.querySelectorAll('text')).filter((t) => !t.hasAttribute('fill-opacity'))).toHaveLength(5);
  });
});
