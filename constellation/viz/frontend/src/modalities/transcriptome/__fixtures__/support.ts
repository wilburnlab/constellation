// Test-only: a template of two nodes as the support routes send it, with
// the server's column names and types (em/support.py's schemas).
//
// Twelve columns, reference ACGTACGTACGT. Node 0: reads 0, 1 and 2 (read 2
// with one error, a T at column 5) and read 5, not in the PWM sample. Node
// 1: reads 3 and 4, which carry the haplotype — G at column 3 and C at
// column 8, the two candidate columns, joined by one arc.

import { Bool, Float32, Float64, Int32, Int64, Int8, Table, Utf8, vectorFromArray } from 'apache-arrow';
import type { SupportMeta } from '../types';

export const N_COLS = 12;

export function supportMeta(overrides: Partial<SupportMeta> = {}): SupportMeta {
  return {
    kind: 'template',
    round: 1,
    verified: true,
    mismatch: [],
    warnings: [],
    template_id: 7,
    params_source: 'round',
    status: 'ok',
    n_reads: 6,
    n_sampled: 5,
    nodes: [
      { haplotype_id: 0, n_reads: 4 },
      { haplotype_id: 1, n_reads: 2 },
    ],
    n_absorbed_reads: 0,
    reference: 'pooled',
    space: 'consensus',
    consensus: 'ACGTACGTACGT',
    n_columns: N_COLS,
    rows: { reads: 6, departures: 5, insertions: 1, columns: 12, arcs: 1, differences: 4 },
    ...overrides,
  };
}

export function supportTables(): Record<string, Table> {
  const reads = new Table({
    idx: vectorFromArray([0, 1, 2, 3, 4, 5], new Int32()),
    read_row: vectorFromArray([10, 11, 12, 13, 14, 15], new Int32()),
    read_id: vectorFromArray(['r0', 'r1', 'r2', 'r3', 'r4', 'r5'], new Utf8()),
    sample_id: vectorFromArray([0n, 1n, 0n, 1n, 2n, null], new Int64()),
    node: vectorFromArray([0, 0, 0, 1, 1, 0], new Int32()),
    absorbed_from: vectorFromArray([null, null, null, null, null, null], new Int64()),
    sampled: vectorFromArray([true, true, true, true, true, false], new Bool()),
    folded: vectorFromArray([false, false, false, false, false, null], new Bool()),
    margin: vectorFromArray([9.5, 9.5, 8.1, 9.5, 9.5, null], new Float32()),
    span_lo: vectorFromArray([0, 0, 2, 0, 1, null], new Int32()),
    span_hi: vectorFromArray([12, 12, 10, 12, 12, null], new Int32()),
    n_departures: vectorFromArray([0, 0, 1, 2, 2, 0], new Int32()),
    clip_5p: vectorFromArray([0, 0, 0, 0, 0, null], new Int32()),
    clip_3p: vectorFromArray([0, 0, 0, 0, 0, null], new Int32()),
    dorado_quality: vectorFromArray([30, 28, 22, 31, 29, 18], new Float32()),
    identity: vectorFromArray([0.99, 0.995, 0.98, 0.99, 0.99, 0.97], new Float32()),
    logl_delta: vectorFromArray([0, 0, 0, 0, 0, null], new Float32()),
    runner_up_template_id: vectorFromArray([null, null, 9n, null, null, null], new Int64()),
    runner_up_logl_gap: vectorFromArray([null, null, 3.5, null, null, null], new Float32()),
  });
  const departures = new Table({
    read_idx: vectorFromArray([2, 3, 3, 4, 4], new Int32()),
    col: vectorFromArray([5, 3, 8, 3, 8], new Int32()),
    allele: vectorFromArray([3, 2, 1, 2, 1], new Int8()),
  });
  const insertions = new Table({
    read_idx: vectorFromArray([1], new Int32()),
    before_col: vectorFromArray([6], new Int32()),
    length: vectorFromArray([2], new Int32()),
    placed: vectorFromArray([true], new Bool()),
  });
  const cols = Array.from({ length: N_COLS }, (_, i) => i);
  const candidate = cols.map((c) => c === 3 || c === 8);
  const columns = new Table({
    col: vectorFromArray(cols, new Int32()),
    frame_col: vectorFromArray(cols, new Int32()),
    template_col: vectorFromArray(cols, new Int32()),
    ref_base: vectorFromArray(cols.map((c) => c % 4), new Int8()),
    depth: vectorFromArray(cols.map((c) => (c < 2 ? 4 : 5)), new Float64()),
    base_cov: vectorFromArray(cols.map((c) => (c < 2 ? 4 : 5)), new Float64()),
    major: vectorFromArray(cols.map((c) => c % 4), new Int8()),
    minor_frac: vectorFromArray(cols.map((c) => (candidate[c] ? 0.4 : c === 5 ? 0.2 : 0)), new Float64()),
    eps: vectorFromArray(cols.map(() => 0.01), new Float64()),
    hp_run: vectorFromArray(cols.map(() => 1), new Int32()),
    candidate: vectorFromArray(candidate, new Bool()),
    route: vectorFromArray(cols.map((c) => (candidate[c] ? 1 : null)), new Int8()),
    signature: vectorFromArray(cols.map((c) => (candidate[c] ? 0 : null)), new Int32()),
  });
  const arcs = new Table({
    u_col: vectorFromArray([3], new Int32()),
    v_col: vectorFromArray([8], new Int32()),
    self_edge: vectorFromArray([false], new Bool()),
    n11: vectorFromArray([2], new Float64()),
    n: vectorFromArray([5], new Float64()),
    phi: vectorFromArray([0.9], new Float64()),
    log_odds: vectorFromArray([4.2], new Float64()),
    p_value: vectorFromArray([1e-6], new Float64()),
  });
  const differences = new Table({
    frame_col: vectorFromArray([3, 3, 8, 8], new Int32()),
    haplotype_id: vectorFromArray([0, 1, 0, 1], new Int32()),
    allele: vectorFromArray([3, 2, 0, 1], new Int8()),
  });
  return { reads, departures, insertions, columns, arcs, differences };
}
