import { describe, expect, it } from 'vitest';
import {
  pickAllowList,
  pickBool,
  pickCycledColor,
  pickNumber,
  pickPaletteColor,
  pickString,
} from './style';

describe('pickString', () => {
  it('returns the stored string', () => {
    expect(pickString({ k: 'log' }, 'k', 'linear')).toBe('log');
  });
  it('falls back for missing, empty and non-string values', () => {
    expect(pickString(undefined, 'k', 'd')).toBe('d');
    expect(pickString({}, 'k', 'd')).toBe('d');
    expect(pickString({ k: '' }, 'k', 'd')).toBe('d');
    expect(pickString({ k: 3 }, 'k', 'd')).toBe('d');
  });
});

describe('pickNumber', () => {
  it('returns finite numbers, including zero', () => {
    expect(pickNumber({ k: 0 }, 'k', 9)).toBe(0);
    expect(pickNumber({ k: 0.4 }, 'k', 9)).toBe(0.4);
  });
  it('coerces numeric strings', () => {
    expect(pickNumber({ k: '12.5' }, 'k', 9)).toBe(12.5);
    // Number('') is 0, so an empty string reads as zero, not the fallback.
    expect(pickNumber({ k: '' }, 'k', 9)).toBe(0);
  });
  it('falls back for non-numeric and non-finite values', () => {
    expect(pickNumber(undefined, 'k', 9)).toBe(9);
    expect(pickNumber({ k: 'abc' }, 'k', 9)).toBe(9);
    expect(pickNumber({ k: NaN }, 'k', 9)).toBe(9);
    expect(pickNumber({ k: Infinity }, 'k', 9)).toBe(9);
    expect(pickNumber({ k: true }, 'k', 9)).toBe(9);
    expect(pickNumber({ k: null }, 'k', 9)).toBe(9);
  });
});

describe('pickBool', () => {
  it('returns real booleans only', () => {
    expect(pickBool({ k: false }, 'k', true)).toBe(false);
    expect(pickBool({ k: true }, 'k', false)).toBe(true);
    expect(pickBool({ k: 'false' }, 'k', true)).toBe(true);
    expect(pickBool({ k: 0 }, 'k', true)).toBe(true);
    expect(pickBool(undefined, 'k', false)).toBe(false);
  });
});

describe('pickPaletteColor', () => {
  it('reads the flat dotted form', () => {
    expect(pickPaletteColor({ 'palette.gene': '#111111' }, 'gene', '#fff')).toBe('#111111');
  });
  it('reads the nested table form', () => {
    expect(pickPaletteColor({ palette: { gene: '#222222' } }, 'gene', '#fff')).toBe('#222222');
  });
  it('prefers the dotted form when both are present', () => {
    const style = { 'palette.gene': '#111111', palette: { gene: '#222222' } };
    expect(pickPaletteColor(style, 'gene', '#fff')).toBe('#111111');
  });
  it('falls back for missing or empty entries', () => {
    expect(pickPaletteColor(undefined, 'gene', '#fff')).toBe('#fff');
    expect(pickPaletteColor({ 'palette.gene': '' }, 'gene', '#fff')).toBe('#fff');
    expect(pickPaletteColor({ palette: 'nope' }, 'gene', '#fff')).toBe('#fff');
    expect(pickPaletteColor({ palette: { mRNA: '#333' } }, 'gene', '#fff')).toBe('#fff');
  });
});

describe('pickCycledColor', () => {
  const cycle = ['#a', '#b', '#c'];
  it('cycles by index when there is no override', () => {
    expect(pickCycledColor(undefined, 7, cycle, 0)).toBe('#a');
    expect(pickCycledColor({}, 7, cycle, 4)).toBe('#b');
  });
  it('clamps a negative index to the first colour', () => {
    expect(pickCycledColor({}, 7, cycle, -2)).toBe('#a');
  });
  it('honours an override keyed by the stringified id', () => {
    expect(pickCycledColor({ 'palette.7': '#zzz' }, 7, cycle, 1)).toBe('#zzz');
  });
});

describe('pickAllowList', () => {
  it('returns null (no constraint) when unset, null or "all"', () => {
    expect(pickAllowList(undefined, 'k')).toBeNull();
    expect(pickAllowList({}, 'k')).toBeNull();
    expect(pickAllowList({ k: null }, 'k')).toBeNull();
    expect(pickAllowList({ k: 'all' }, 'k')).toBeNull();
    expect(pickAllowList({ k: 'chr1' }, 'k')).toBeNull();
  });
  it('stringifies members so numeric and string ids compare equal', () => {
    const set = pickAllowList({ k: [1, '2', null, undefined] }, 'k');
    expect(set).toEqual(new Set(['1', '2']));
  });
  it('returns an empty set for an empty list (nothing allowed)', () => {
    expect(pickAllowList({ k: [] }, 'k')).toEqual(new Set());
  });
});
