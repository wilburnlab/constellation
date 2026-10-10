// What each genome track kind declares about itself to the host.

import { describe, expect, it } from 'vitest';
import { UNKNOWN_KIND_ORDER, encodePushdown } from '../../../panels/kind';
import { getRenderer, kindRank, registeredKinds } from './index';
import { clusterView, minMapq } from './pushdown';

describe('genome track descriptors', () => {
  it('stack in the canonical order', () => {
    const byOrder = registeredKinds().sort((a, b) => kindRank(a) - kindRank(b));
    expect(byOrder).toEqual([
      'reference_sequence',
      'gene_annotation',
      'coverage_histogram',
      'read_pileup',
      'cluster_pileup',
      'splice_junctions',
    ]);
    expect(new Set(registeredKinds().map(kindRank)).size).toBe(6);
  });

  it('rank an unknown kind after every known one', () => {
    expect(kindRank('not_a_kind')).toBe(UNKNOWN_KIND_ORDER);
    expect(Math.max(...registeredKinds().map(kindRank))).toBeLessThan(UNKNOWN_KIND_ORDER);
  });

  it('name their unit', () => {
    expect(Object.fromEntries(registeredKinds().map((k) => [k, getRenderer(k)!.unit]))).toEqual({
      cluster_pileup: ['cluster', 'clusters'],
      coverage_histogram: ['bin', 'bins'],
      gene_annotation: ['feature', 'features'],
      read_pileup: ['read', 'reads'],
      reference_sequence: ['base', 'bases'],
      splice_junctions: ['junction', 'junctions'],
    });
  });

  it('declare server-side filters only on the pile-up kinds', () => {
    const keys = Object.fromEntries(
      registeredKinds().map((k) => [k, Object.keys(getRenderer(k)!.pushdown ?? {})]),
    );
    expect(keys).toEqual({
      cluster_pileup: ['min_mapq', 'cluster_view'],
      coverage_histogram: [],
      gene_annotation: [],
      read_pileup: ['min_mapq'],
      reference_sequence: [],
      splice_junctions: [],
    });
  });

  it('send min_mapq before cluster_view, as the wire has always had them', () => {
    const params = encodePushdown(getRenderer('cluster_pileup'), {
      cluster_view: 'members',
      min_mapq: 20,
    });
    expect(Object.entries(params)).toEqual([
      ['min_mapq', '20'],
      ['cluster_view', 'members'],
    ]);
  });
});

describe('pushdown encoders', () => {
  it('minMapq sends a positive threshold and nothing otherwise', () => {
    expect(minMapq(20)).toBe('20');
    expect(minMapq('30')).toBe('30');
    expect(minMapq(0)).toBeUndefined();
    expect(minMapq(-5)).toBeUndefined();
    expect(minMapq(undefined)).toBeUndefined();
    expect(minMapq('abc')).toBeUndefined();
    expect(minMapq(Number.NaN)).toBeUndefined();
  });

  it('clusterView sends only a known view', () => {
    expect(clusterView('members')).toBe('members');
    expect(clusterView('clusters')).toBe('clusters');
    expect(clusterView('other')).toBeUndefined();
    expect(clusterView(undefined)).toBeUndefined();
    expect(clusterView(1)).toBeUndefined();
  });
});
