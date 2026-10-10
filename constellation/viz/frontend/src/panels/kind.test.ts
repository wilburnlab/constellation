import { describe, expect, it } from 'vitest';
import { PanelKind, encodePushdown, pushdownChanged, unitFor } from './kind';

const reads: PanelKind = {
  kind: 'reads',
  order: 3,
  unit: ['read', 'reads'],
  pushdown: {
    min_q: (v) => (typeof v === 'number' && v > 0 ? String(v) : undefined),
    view: (v) => (v === 'a' || v === 'b' ? v : undefined),
  },
};
const plain: PanelKind = { kind: 'cov', order: 2, unit: ['bin', 'bins'] };

describe('unitFor', () => {
  it('uses the singular for exactly one item', () => {
    expect(unitFor(reads, 1)).toBe('read');
    expect(unitFor(reads, 0)).toBe('reads');
    expect(unitFor(reads, 2)).toBe('reads');
  });
  it('falls back to a generic noun for an unknown kind', () => {
    expect(unitFor(null, 1)).toBe('row');
    expect(unitFor(undefined, 5)).toBe('rows');
  });
});

describe('encodePushdown', () => {
  it('encodes declared keys in declared order and drops undefined results', () => {
    expect(encodePushdown(reads, { view: 'b', min_q: 20, visible: ['+'] })).toEqual({
      min_q: '20',
      view: 'b',
    });
    expect(Object.keys(encodePushdown(reads, { view: 'b', min_q: 20 }))).toEqual(['min_q', 'view']);
    expect(encodePushdown(reads, { min_q: 0, view: 'zzz' })).toEqual({});
  });
  it('is empty for a kind with no server-side filters', () => {
    expect(encodePushdown(plain, { min_q: 20 })).toEqual({});
    expect(encodePushdown(null, { min_q: 20 })).toEqual({});
  });
});

describe('pushdownChanged', () => {
  it('reports a change only in a declared pushdown key', () => {
    expect(pushdownChanged(reads, {}, { min_q: 20 })).toBe(true);
    expect(pushdownChanged(reads, { min_q: 20 }, { min_q: 20 })).toBe(false);
    expect(pushdownChanged(reads, { min_q: 20 }, {})).toBe(true);
    expect(pushdownChanged(reads, { view: 'a' }, { view: 'b' })).toBe(true);
    expect(pushdownChanged(reads, { visible: ['+'] }, { visible: [] })).toBe(false);
  });
  it('compares structurally and treats null as distinct', () => {
    expect(pushdownChanged(reads, { min_q: [1, 2] }, { min_q: [1, 2] })).toBe(false);
    expect(pushdownChanged(reads, { min_q: null }, { min_q: 0 })).toBe(true);
  });
  it('never fires for a kind without pushdown filters', () => {
    expect(pushdownChanged(plain, {}, { min_q: 20 })).toBe(false);
  });
});
