import { Int32, Table, vectorFromArray } from 'apache-arrow';
import { describe, expect, it } from 'vitest';
import { ALLELE_N, alleleAt, indexMatrix } from '../../engine/matrix/model';
import { supportMeta, supportTables } from './__fixtures__/support';
import { arcEffect, arcGeometry, arcOpacity, arcPath, arcWidth, ARC_NEGATIVE, ARC_POSITIVE, MIN_ARC_HEIGHT } from './arcs';
import {
  defaultOrder,
  departureExcess,
  differingColumns,
  orderByPair,
  patternColumns,
  restrictColumns,
  supportFromTables,
} from './support_data';

const data = () => supportFromTables(supportMeta(), supportTables());

describe('decoding the support tables', () => {
  it('builds the matrix: spans, departures, insertions and the reference', () => {
    const d = data();
    expect(d.reads.n).toBe(6);
    expect(Array.from(d.matrix.spanLo)).toEqual([0, 0, 2, 0, 1, -1]);
    expect(Array.from(d.matrix.refAllele!)).toEqual([0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3]);
    expect(alleleAt(d.index, 3, 3)).toBe(2); // node 1's G
    expect(alleleAt(d.index, 0, 3)).toBe(3); // the reference's T
    expect(alleleAt(d.index, 5, 3)).toBe(-1); // unsampled: no span
    expect(d.reads.sampleId).toEqual([0, 1, 0, 1, 2, null]);
    expect(d.reads.runnerUp[2]).toBe('9');
    expect(Number.isNaN(d.reads.margin[5])).toBe(true);
    expect(Array.from(d.columnRow)).toEqual(Array.from({ length: 12 }, (_, i) => i));
  });
});

describe('row order', () => {
  it('groups by node, then pattern at the signature columns, then start; unsampled last', () => {
    const d = data();
    expect(patternColumns(d)).toEqual([3, 8]);
    // Reads 0 and 1 tie on pattern and start; the better identity (read 1,
    // 0.995) goes first.
    expect(Array.from(defaultOrder(d))).toEqual([1, 0, 2, 5, 3, 4]);
  });

  it('an arc sorts the reads by their joint state at its two columns', () => {
    expect(Array.from(orderByPair(data(), 3, 8))).toEqual([3, 4, 1, 0, 2, 5]);
  });
});

describe('comparison', () => {
  it('restricts to the columns where the nodes differ', () => {
    const d = data();
    const cols = differingColumns(d);
    expect(Array.from(cols)).toEqual([3, 8]);
    const m = restrictColumns(d.matrix, cols);
    expect(m.nCols).toBe(2);
    expect(Array.from(m.spanLo)).toEqual([0, 0, 0, 0, 0, -1]);
    expect(Array.from(m.spanHi)).toEqual([2, 2, 2, 2, 2, -1]);
    // Read 2's error at column 5 is not a shown column.
    expect(Array.from(m.depRow)).toEqual([3, 3, 4, 4]);
    expect(Array.from(m.depCol)).toEqual([0, 1, 0, 1]);
    expect(Array.from(m.refAllele!)).toEqual([3, 0]);
    expect(m.insRow.length).toBe(0);
  });

  it('a read hotter than its expected error says so', () => {
    const x = departureExcess(data(), 0.012);
    expect(x[0]).toBe(0);
    expect(x[3]).toBeCloseTo(2 / (12 * 0.012));
    expect(Number.isNaN(x[5])).toBe(true);
  });

  it('an ambiguous base is no evidence: not a departure, and off the span it is judged over', () => {
    const d = data();
    // Read 3 (span 12) departs at two columns; make one of them an N.
    const i = Array.from(d.matrix.depRow).findIndex((r) => r === 3);
    d.matrix.depAllele[i] = ALLELE_N;
    d.index = indexMatrix(d.matrix);
    let x = departureExcess(d, 0.012);
    expect(x[3]).toBeCloseTo(1 / (11 * 0.012));
    // Only Ns: no excess at all (it was 2 / 0.144 = 13.9x before).
    d.matrix.depAllele.fill(ALLELE_N);
    d.index = indexMatrix(d.matrix);
    x = departureExcess(d, 0.012);
    expect(x[3]).toBe(0);
    expect(x[4]).toBe(0);
  });

  it('a read with no comparable column gets no indicator', () => {
    const d = data();
    // Read 3 covers 12 columns; make every one an N.
    const n = d.matrix.spanHi[3] - d.matrix.spanLo[3];
    d.matrix.depRow = Int32Array.from({ length: n }, () => 3);
    d.matrix.depCol = Int32Array.from({ length: n }, (_, c) => d.matrix.spanLo[3] + c);
    d.matrix.depAllele = new Int8Array(n).fill(ALLELE_N);
    d.index = indexMatrix(d.matrix);
    expect(Number.isNaN(departureExcess(d)[3])).toBe(true);
  });
});

describe('long alignments', () => {
  it('decodes 150,000 columns (no argument spread over the columns)', () => {
    const n = 150_000;
    const tables = supportTables();
    tables.columns = new Table({
      col: vectorFromArray(Array.from({ length: n }, (_, i) => i), new Int32()),
      frame_col: vectorFromArray(Array.from({ length: n }, (_, i) => i), new Int32()),
    });
    const d = supportFromTables(supportMeta({ n_columns: n, consensus: 'A'.repeat(n) }), tables);
    expect(d.colOfFrame.length).toBe(n);
    expect(d.colOfFrame[n - 1]).toBe(n - 1);
  });
});

describe('arcs', () => {
  it('colour by effect, opacity by significance, width by reads', () => {
    const d = data();
    expect(arcEffect(d.arcs, 0)).toBeCloseTo(0.9);
    expect(arcOpacity(1e-10)).toBe(1);
    expect(arcOpacity(0.5)).toBe(0.2);
    expect(arcWidth(1)).toBe(0.75);
    expect(arcWidth(1000)).toBe(3.75);
    const geoms = arcGeometry(d.arcs, (c) => c * 10 + 5, 120, 80, 12);
    expect(geoms).toHaveLength(1);
    expect(geoms[0]).toMatchObject({ x1: 35, x2: 85, colour: ARC_POSITIVE });
    // Half the distance on screen, at least MIN_ARC_HEIGHT, at most the cap.
    expect(geoms[0].height).toBe(25);
    expect(arcGeometry(d.arcs, (c) => c * 1, 120, 80, 12)[0].height).toBe(MIN_ARC_HEIGHT);
    expect(arcGeometry(d.arcs, (c) => c * 100, 2000, 80, 12)[0].height).toBe(80);
    expect(arcPath(geoms[0], 100)).toBe(`M 35 100 Q 60 ${100 - 2 * geoms[0].height} 85 100`);
  });

  it('drops arcs out of view, and draws the tallest first', () => {
    const d = data();
    const two = {
      ...d.arcs,
      n: 2,
      u: Int32Array.from([3, 0]),
      v: Int32Array.from([8, 11]),
      phi: Float64Array.from([0.9, -0.5]),
      pValue: Float64Array.from([1e-6, 1e-3]),
      total: Float64Array.from([5, 5]),
      logOdds: Float64Array.from([4, -2]),
      n11: Float64Array.from([2, 0]),
      selfEdge: Uint8Array.from([0, 0]),
    };
    const g = arcGeometry(two, (c) => c * 10, 200, 80, 12);
    expect(g.map((a) => a.index)).toEqual([1, 0]);
    expect(g[0].colour).toBe(ARC_NEGATIVE);
    expect(arcGeometry(two, (c) => c * 10 - 500, 200, 80, 12)).toEqual([]);
  });
});
