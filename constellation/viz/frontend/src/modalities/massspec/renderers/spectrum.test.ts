import { Bool, Float64, Table, Utf8, vectorFromArray } from 'apache-arrow';
import { describe, expect, it } from 'vitest';
import { areaPerHeight, drawSpectrum, SPECTRUM_MARGIN, type SpectrumOptions } from './spectrum';

/** A table as the kernel sends it. */
function points(mz: number[], intensity: number[], profile: boolean, labels?: Array<string | null>): Table {
  return new Table({
    mz: vectorFromArray(mz, new Float64()),
    intensity: vectorFromArray(intensity, new Float64()),
    annotation: vectorFromArray(labels ?? mz.map(() => null), new Utf8()),
    ion_type: vectorFromArray(labels?.map((l) => (l ? l[0] : null)) ?? mz.map(() => null), new Utf8()),
    profile: vectorFromArray(mz.map(() => profile), new Bool()),
  });
}

/** Two peaks on a 1/15 Th grid: heights 100 and 40. */
function aProfile(): Table {
  const mz = Array.from({ length: 61 }, (_, i) => 500 + i / 15);
  const peak = (m: number, c: number, h: number): number => h * Math.exp(-((m - c) ** 2) / (2 * 0.13 ** 2));
  return points(mz, mz.map((m) => peak(m, 501, 100) + peak(m, 503, 40)), true);
}

function draw(table: Table, opts: Partial<SpectrumOptions> = {}): SVGSVGElement {
  const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
  drawSpectrum(svg, { widthPx: 800, heightPx: 300 }, table, {
    mz: null,
    intensity: 'abs',
    isolation: null,
    target: null,
    style: {},
    uid: 'test',
    ...opts,
  });
  return svg;
}

const sticks = (): { name: string; table: Table; color: string } => ({
  name: 'thermo',
  // Areas are about six times the heights, as Thermo's are.
  table: points([501, 503], [650, 260], false),
  color: '#d95926',
});

describe('drawSpectrum', () => {
  it('draws profile points as a line, peaks as sticks', () => {
    const line = draw(aProfile());
    expect(line.querySelector('.ms-profile')).not.toBeNull();
    expect(line.querySelector('.ms-peaks')).toBeNull();
    const peaks = draw(points([501, 503], [650, 260], false));
    expect(peaks.querySelector('.ms-profile')).toBeNull();
    expect(peaks.querySelector('.ms-peaks')).not.toBeNull();
  });

  it('draws sticks over a profile, with a legend saying which is which', () => {
    const svg = draw(aProfile(), { overlays: [sticks()], baseName: 'profile', perHeight: 6.5 });
    expect(svg.querySelector('.ms-profile')).not.toBeNull();
    expect(svg.querySelector('.ms-overlay-peaks')?.getAttribute('stroke')).toBe('#d95926');
    expect([...svg.querySelectorAll('.ms-legend-label')].map((t) => t.textContent)).toEqual(['profile', 'thermo']);
  });

  it('puts peak areas on the right-hand axis, in step with the profile', () => {
    const svg = draw(aProfile(), { overlays: [sticks()], perHeight: 6.5 });
    const right = svg.querySelectorAll('.axis-tick-label-right');
    expect(right.length).toBeGreaterThan(1);
    // A stick of area 650 at 6.5 area per height stands as tall as the
    // profile's 100: the line's top and the stick's end meet.
    const d = svg.querySelector('.ms-overlay-peaks')!.getAttribute('d')!;
    const stickTop = Number(/M[\d.]+ [\d.]+V([\d.]+)/.exec(d)![1]);
    const line = svg.querySelector('.ms-profile')!.getAttribute('d')!;
    const ys = [...line.matchAll(/[ML][\d.]+ ([\d.]+)/g)].map((m) => Number(m[1]));
    expect(Math.abs(stickTop - Math.min(...ys))).toBeLessThan(1.5);
  });

  it('anchors sticks to the profile on request, with no area axis', () => {
    const svg = draw(aProfile(), { overlays: [sticks()], perHeight: 1, anchored: true });
    expect(svg.querySelector('.axis-tick-label-right')).toBeNull();
    // perHeight 1 would put a 650 stick far above the profile; anchored it
    // stays on it, so the axis still fits the profile.
    const top = Number([...svg.querySelectorAll('.axis-tick-label')].at(-1)!.textContent);
    expect(top).toBeLessThanOrEqual(120);
  });

  it('has no area axis on a log plot, or between lists of peaks', () => {
    expect(draw(aProfile(), { overlays: [sticks()], intensity: 'log' }).querySelector('.axis-tick-label-right')).toBeNull();
    const peaks = draw(points([501, 503], [650, 260], false), { overlays: [sticks()] });
    expect(peaks.querySelector('.axis-tick-label-right')).toBeNull();
  });

  it('labels and annotates the top list, not the points under it', () => {
    const top = { ...sticks(), table: points([501, 503], [650, 260], false, ['y7', null]) };
    const svg = draw(aProfile(), {
      overlays: [top],
      perHeight: 6.5,
      errorStrip: null,
    });
    expect([...svg.querySelectorAll('.ms-ion-label')].map((t) => t.textContent)).toEqual(['y7']);
    expect(svg.querySelector('.ms-peaks-annotated')).not.toBeNull();
  });

  it('keeps the right margin it reserves for the area axis', () => {
    expect(SPECTRUM_MARGIN.right).toBeGreaterThanOrEqual(40);
  });
});

describe('areaPerHeight', () => {
  it('is the median of area over the height under each strong stick', () => {
    const ratio = areaPerHeight(aProfile(), sticks().table);
    expect(ratio).toBeGreaterThan(6);
    expect(ratio).toBeLessThan(7);
  });

  it('is not moved by a stick with nothing under it, or by having no profile', () => {
    const ghost = points([501, 503, 900], [650, 260, 300], false);
    expect(areaPerHeight(aProfile(), ghost)).toBeGreaterThan(6);
    expect(areaPerHeight(points([], [], true), sticks().table)).toBe(1);
  });
});
