import { describe, expect, it } from 'vitest';
import { buildTrackDataUrl } from './arrow_client';

const ORIGIN = 'http://127.0.0.1:8765';
const BASE = {
  session: 'run-1a2b3c4d',
  binding: 'read_pileup-0',
  contig: 'chr1',
  start: 100,
  end: 2000,
};

describe('buildTrackDataUrl', () => {
  it('emits the required parameters in a fixed order', () => {
    expect(buildTrackDataUrl('read_pileup', BASE, ORIGIN)).toBe(
      `${ORIGIN}/api/tracks/read_pileup/data` +
        '?session=run-1a2b3c4d&binding=read_pileup-0&contig=chr1&start=100&end=2000',
    );
  });

  it('appends optional parameters after the required ones', () => {
    const url = buildTrackDataUrl(
      'cluster_pileup',
      {
        ...BASE,
        viewport_px: 1180,
        max_glyphs: 5000,
        min_mapq: 20,
        cluster_view: 'members',
        force: 'hybrid',
        samples: ['a', 'b'],
      },
      ORIGIN,
    );
    expect(url).toBe(
      `${ORIGIN}/api/tracks/cluster_pileup/data` +
        '?session=run-1a2b3c4d&binding=read_pileup-0&contig=chr1&start=100&end=2000' +
        '&viewport_px=1180&max_glyphs=5000&min_mapq=20&cluster_view=members' +
        '&force=hybrid&samples=a&samples=b',
    );
  });

  it('leaves out undefined parameters and keeps falsy ones', () => {
    const url = buildTrackDataUrl(
      'read_pileup',
      { ...BASE, viewport_px: 900, min_mapq: undefined, depth: 0, tag: '' },
      ORIGIN,
    );
    expect(url).not.toContain('min_mapq');
    expect(url.endsWith('&viewport_px=900&depth=0&tag=')).toBe(true);
  });

  it('emits parameters in the order given and repeats a list', () => {
    const url = buildTrackDataUrl(
      'toy_trace',
      { session: 's', binding: 'b', x1: 2.5, x0: 0.5, tags: ['a', 'b'], empty: [] },
      ORIGIN,
    );
    expect(url).toBe(
      `${ORIGIN}/api/tracks/toy_trace/data?session=s&binding=b&x1=2.5&x0=0.5&tags=a&tags=b`,
    );
  });

  it('percent-encodes the kind and parameter values', () => {
    const url = buildTrackDataUrl(
      'odd kind',
      { ...BASE, contig: 'chr 1/alt', session: 'a&b' },
      ORIGIN,
    );
    expect(url).toContain('/api/tracks/odd%20kind/data?');
    expect(url).toContain('session=a%26b');
    expect(url).toContain('contig=chr+1%2Falt');
  });
});
