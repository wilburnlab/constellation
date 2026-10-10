// What each genome track kind declares about itself to the host.

import { describe, expect, it } from 'vitest';
import { ensureSvg } from '../../../engine/svg_layer';
import { UNKNOWN_KIND_ORDER, encodePushdown } from '../../../panels/kind';
import { SettingsEnv, resolve } from '../../../panels/settings_schema';
import { loadFixture, metadataFor } from '../__fixtures__/load';
import { xScale } from '../scales';
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

// A control is live when drawing consults it. Each case draws a kind
// from its kernel fixture through `style` / `filter` dicts that record
// every key looked up, then checks that everything the popover offers
// for that state was among them — or is a filter the server applies.
describe('genome track settings offer nothing the track ignores', () => {
  interface Case {
    kind: string;
    fixture: string;
    domain: [number, number];
    /** Stored filter state the popover is opened with. */
    filter?: Record<string, unknown>;
    /** Palette entries for categories this fixture's data does not hold;
     *  a colour is only looked up for a category that is drawn. */
    absentFromFixture?: string[];
  }

  const CASES: Case[] = [
    { kind: 'reference_sequence', fixture: 'reference_sequence.letters', domain: [0, 48],
      absentFromFixture: ['palette.U'] },
    { kind: 'gene_annotation', fixture: 'gene_annotation', domain: [0, 12_000] },
    { kind: 'coverage_histogram', fixture: 'coverage_histogram', domain: [0, 12_000] },
    { kind: 'read_pileup', fixture: 'read_pileup', domain: [0, 12_000] },
    { kind: 'cluster_pileup', fixture: 'cluster_pileup.clusters', domain: [0, 12_000] },
    { kind: 'splice_junctions', fixture: 'splice_junctions', domain: [0, 12_000] },
  ];

  /** A dict that reports nothing set and remembers what was asked for. */
  function recording(seen: Set<string>): Record<string, unknown> {
    return new Proxy({}, {
      get: (_target, key) => {
        if (typeof key === 'string') seen.add(key);
        return undefined;
      },
    });
  }

  function keysRead(c: Case): { style: Set<string>; filter: Set<string> } {
    const read = { style: new Set<string>(), filter: new Set<string>() };
    const svg = ensureSvg(document.createElement('div'), 1000, 120);
    getRenderer(c.kind)!.render(loadFixture(c.fixture), 'vector', {
      svg,
      widthPx: 1000,
      heightPx: 120,
      xScale: xScale(c.domain, 1000),
      meta: metadataFor(c.kind),
      showLabels: true,
      style: recording(read.style),
      filter: recording(read.filter),
    });
    return read;
  }

  /** `[target, key]` for every control the popover shows in this state. */
  function keysOffered(c: Case): Array<['style' | 'filter', string]> {
    const env: SettingsEnv = { meta: metadataFor(c.kind), style: {}, filter: c.filter ?? {}, host: {} };
    const out: Array<['style' | 'filter', string]> = [];
    for (const section of getRenderer(c.kind)!.settings!.sections) {
      if (section.when && !section.when(env)) continue;
      for (const control of section.controls) {
        if (control.type === 'note') continue;
        if (control.when && !control.when(env)) continue;
        if (control.type === 'palette') {
          for (const entry of resolve(control.entries, env)) out.push(['style', `palette.${entry.key}`]);
        } else {
          out.push([control.target, control.key]);
        }
      }
    }
    return out;
  }

  it.each(CASES)('$fixture', (c) => {
    const read = keysRead(c);
    const pushdown = new Set(Object.keys(getRenderer(c.kind)!.pushdown ?? {}));
    const ignored = keysOffered(c)
      .filter(([target, key]) => !read[target].has(key))
      .filter(([target, key]) => !(target === 'filter' && pushdown.has(key)))
      .map(([, key]) => key);
    expect(ignored.sort()).toEqual((c.absentFromFixture ?? []).sort());
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
