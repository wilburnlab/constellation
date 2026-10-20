import { describe, expect, it } from 'vitest';
import { IntervalLimits, clampInterval, panBy, sameInterval, zoomAbout } from './interval';

const RT: IntervalLimits = { min: 0, max: 6000, minSpan: 1 };
const BP: IntervalLimits = { min: 0, max: 1000, minSpan: 20, round: Math.round };

describe('clampInterval', () => {
  it('keeps a window that fits', () => {
    expect(clampInterval([100, 200], RT)).toEqual([100, 200]);
  });
  it('slides a window back inside, keeping its span', () => {
    expect(clampInterval([-50, 50], RT)).toEqual([0, 100]);
    expect(clampInterval([5950, 6050], RT)).toEqual([5900, 6000]);
  });
  it('gives the whole range for a window wider than it', () => {
    expect(clampInterval([-1, 7000], RT)).toEqual([0, 6000]);
  });
  it('rounds both ends when the axis asks', () => {
    expect(clampInterval([10.4, 50.6], BP)).toEqual([10, 51]);
  });
});

describe('zoomAbout', () => {
  it('keeps the anchor fixed', () => {
    const out = zoomAbout([1000, 2000], 0.5, 0.25, RT);
    expect(out).toEqual([1125, 1625]);
    // The point a quarter of the way in stays a quarter of the way in.
    expect(out[0] + 0.25 * (out[1] - out[0])).toBe(1250);
  });
  it('stops at the minimum span', () => {
    const out = zoomAbout([100, 130], 0.01, 0.5, BP);
    expect(out[1] - out[0]).toBe(20);
  });
  it('zooms out to the limits and no further', () => {
    expect(zoomAbout([0, 6000], 2, 0.5, RT)).toEqual([0, 6000]);
  });
});

describe('panBy', () => {
  it('moves by a fraction of the span and stops at the edge', () => {
    expect(panBy([1000, 2000], 0.1, RT)).toEqual([1100, 2100]);
    expect(panBy([100, 200], -5, RT)).toEqual([0, 100]);
  });
});

describe('sameInterval', () => {
  it('compares with a tolerance and handles null', () => {
    expect(sameInterval([0, 1], [0, 1 + 1e-12])).toBe(true);
    expect(sameInterval([0, 1], [0, 1.1])).toBe(false);
    expect(sameInterval(null, null)).toBe(true);
    expect(sameInterval(null, [0, 1])).toBe(false);
  });
});
