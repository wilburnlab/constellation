import { describe, expect, it } from 'vitest';
import { formatGenomic, xScale } from './scales';

describe('formatGenomic', () => {
  it('uses bp below 1 kb, with thousands separators unused at that size', () => {
    expect(formatGenomic(0)).toBe('0 bp');
    expect(formatGenomic(999)).toBe('999 bp');
  });

  it('switches to kb at 1,000 and Mb at 1,000,000', () => {
    expect(formatGenomic(1000)).toBe('1.00 kb');
    expect(formatGenomic(12_345)).toBe('12.35 kb');
    expect(formatGenomic(999_999)).toBe('1000.00 kb');
    expect(formatGenomic(1_000_000)).toBe('1.00 Mb');
    expect(formatGenomic(248_956_422)).toBe('248.96 Mb');
  });

  it('chooses the unit from the magnitude for negative values', () => {
    expect(formatGenomic(-2500)).toBe('-2.50 kb');
  });
});

describe('xScale', () => {
  it('maps the domain linearly onto [0, width]', () => {
    const scale = xScale([1000, 2000], 500);
    expect(scale(1000)).toBe(0);
    expect(scale(1500)).toBe(250);
    expect(scale(2000)).toBe(500);
    expect(scale.domain()).toEqual([1000, 2000]);
  });

  it('extrapolates outside the domain rather than clamping', () => {
    const scale = xScale([0, 100], 200);
    expect(scale(-50)).toBe(-100);
    expect(scale(150)).toBe(300);
  });
});
