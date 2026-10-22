import { describe, expect, it } from 'vitest';
import {
  AXIS_FORMATS,
  autoYDomain,
  decimalsFor,
  drawAxes,
  formatIntensity,
  plotFrame,
  xAxis,
  yAxis,
} from './axis';
import { ensureSvg } from './svg_layer';

const MARGIN = { top: 4, right: 8, bottom: 20, left: 50 };

describe('plotFrame', () => {
  it('puts the data area inside the margins', () => {
    const f = plotFrame({ widthPx: 400, heightPx: 200 }, MARGIN);
    expect(f.inner).toEqual({ x: 50, y: 4, w: 342, h: 176 });
  });
});

describe('scales', () => {
  it('map the domain onto the data area and back', () => {
    const f = plotFrame({ widthPx: 400, heightPx: 200 }, MARGIN);
    const x = xAxis(f, [600, 1200], AXIS_FORMATS.rtMinutes);
    expect(x.scale(600)).toBe(50);
    expect(x.scale(1200)).toBe(392);
    expect(x.invert(221)).toBeCloseTo(900);
    const y = yAxis(f, [0, 100], AXIS_FORMATS.intensity);
    expect(y.scale(0)).toBe(180); // bottom of the data area
    expect(y.scale(100)).toBe(4);
  });

  it('a log axis is defined at zero', () => {
    const f = plotFrame({ widthPx: 400, heightPx: 200 }, MARGIN);
    const y = yAxis(f, [0, 1e6], AXIS_FORMATS.intensity, { log: true });
    expect(y.scale(0)).toBe(180);
    expect(y.scale(1e3)).toBeLessThan(y.scale(10));
  });
});

describe('tick formats', () => {
  it('retention time is shown in minutes with decimals that follow the zoom', () => {
    expect(AXIS_FORMATS.rtMinutes.tick(1800, 600)).toBe('30');
    expect(AXIS_FORMATS.rtMinutes.tick(1803, 3)).toBe('30.05');
    expect(AXIS_FORMATS.rtMinutes.tick(1806, 6)).toBe('30.1');
  });
  it('m/z decimals follow the tick step', () => {
    expect(AXIS_FORMATS.mz.tick(500, 100)).toBe('500');
    expect(AXIS_FORMATS.mz.tick(500.25, 0.05)).toBe('500.25');
    expect(decimalsFor(0.001)).toBe(3);
  });
  it('intensities stay short', () => {
    expect(formatIntensity(0)).toBe('0');
    expect(formatIntensity(2.5)).toBe('2.5');
    expect(formatIntensity(3)).toBe('3');
    expect(formatIntensity(1234)).toBe('1234');
    expect(formatIntensity(2_400_000)).toBe('2.4e6');
    expect(formatIntensity(10_000)).toBe('1e4');
  });
});

describe('autoYDomain', () => {
  it('fits the points in view, with headroom', () => {
    const xs = [0, 1, 2, 3];
    const ys = [5, 50, 10, 500];
    expect(autoYDomain(ys)).toEqual([0, 540]);
    expect(autoYDomain(ys, xs, [0, 2], 1)).toEqual([0, 50]);
    expect(autoYDomain([0, 0])).toEqual([0, 1]);
  });
});

describe('drawAxes', () => {
  it('draws ticks with labels and returns a clipped plot group', () => {
    const host = document.createElement('div');
    const svg = ensureSvg(host, 400, 200);
    const f = plotFrame({ widthPx: 400, heightPx: 200 }, MARGIN);
    const x = xAxis(f, [0, 600], AXIS_FORMATS.rtMinutes, { ticks: 5 });
    const y = yAxis(f, [0, 100], AXIS_FORMATS.intensity, { ticks: 3 });
    const plot = drawAxes(svg, f, { x, y, xLabel: 'RT', yLabel: 'I' }, 'cell-1');
    expect(plot.getAttribute('clip-path')).toBe('url(#cell-1-clip)');
    expect(svg.querySelector('#cell-1-clip rect')?.getAttribute('width')).toBe('342');
    const labels = [...svg.querySelectorAll('.axis-tick-label')].map((t) => t.textContent);
    expect(labels).toContain('4'); // minutes: [0, 600] s ticks every 2 min
    expect(labels).toContain('100');
    expect([...svg.querySelectorAll('.axis-label')].map((t) => t.textContent)).toEqual(['RT', 'I']);
  });
});

describe('ticks in displayed units', () => {
  it('a seconds axis shown in minutes ticks on round minutes', () => {
    const f = plotFrame({ widthPx: 800, heightPx: 200 }, MARGIN);
    const x = xAxis(f, [0, 6425], AXIS_FORMATS.rtMinutes, { ticks: 6 });
    expect(x.ticks.map((t) => t / 60)).toEqual([0, 20, 40, 60, 80, 100]);
    expect(x.ticks.map((t) => x.format.tick(t, x.step))).toEqual(['0', '20', '40', '60', '80', '100']);
  });
});
