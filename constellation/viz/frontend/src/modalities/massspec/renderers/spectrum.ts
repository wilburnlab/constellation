// ms_spectrum — the peaks of one scan as sticks.
//
// Peaks the kernel annotated with a peptide's ions are coloured by ion
// series and labelled first (y7, b5^2, p^2); the most intense of the rest
// get their m/z, skipping any label that would overlap one already placed.
// On an MS1 spectrum the isolation window of the MS2 scan being shown is
// shaded, its target marked, and a target precursor's isotopes ticked.
// Below an annotated spectrum an optional strip plots each matched peak's
// mass error, within the tolerance it was matched at.

import type { Table } from 'apache-arrow';
import {
  AXIS_FORMATS,
  type AxisFormat,
  type Margins,
  drawAxes,
  plotFrame,
  xAxis,
  yAxis,
} from '../../../engine/axis';
import type { Interval } from '../../../engine/interval';
import { svgEl } from '../../../engine/svg_layer';
import type { PanelKind } from '../../../panels/kind';
import { pickBool, pickNumber, pickPaletteColor, pickString } from '../../../panels/style';
import { ionColor } from '../palette';
import { linePath, type IntensityMode } from './chromatogram';

// The right margin holds the area axis of sticks drawn over a profile.
export const SPECTRUM_MARGIN: Margins = { top: 14, right: 46, bottom: 32, left: 58 };
/** Height of the mass-error strip, when drawn. */
export const ERROR_STRIP_PX = 46;

export const SPECTRUM_DEFAULTS = {
  stick: '#6d6d6d',
  label: '#edecee',
  isolation: '#ffca85',
  labels: 8,
};

/** A list of peaks drawn as sticks over the spectrum. */
export interface SpectrumOverlay {
  name: string;
  table: Table;
  color: string;
}

export interface SpectrumOptions {
  /** More lists of peaks to draw over this one; the last is on top, and is
   *  the one labelled and annotated. */
  overlays?: SpectrumOverlay[];
  /** The name of the base list, for the legend when there are overlays. */
  baseName?: string;
  /** Peak area per unit of profile height, for sticks over a profile. Fixed
   *  per scan by the caller (see `areaPerHeight`), so zooming does not
   *  rescale them. */
  perHeight?: number;
  /** Draw sticks over a profile at the height of the profile under them,
   *  rather than by their area on the right-hand axis. */
  anchored?: boolean;
  /** m/z window; the data's own extent when null. */
  mz: Interval | null;
  intensity: IntensityMode;
  /** Isolation window to shade (an MS1 spectrum showing what an MS2 scan isolated). */
  isolation: Interval | null;
  /** Precursor m/z to mark. */
  target: number | null;
  /** A target precursor's isotopes, ticked on an MS1 spectrum. */
  isotopes?: Array<{ mz: number; label: string }>;
  /** Draw the mass-error strip, at this tolerance. */
  errorStrip?: { tolerance: number; unit: 'ppm' | 'Da' } | null;
  style: Record<string, unknown>;
  uid: string;
}

/** The fragment ions annotated, as the spectrum gear offers them. */
export const ION_TYPE_CHOICES: Record<string, string[]> = {
  by: ['b', 'y'],
  aby: ['a', 'b', 'y'],
  cz: ['c', 'z'],
  bcyz: ['b', 'c', 'y', 'z'],
  all: ['a', 'b', 'c', 'x', 'y', 'z'],
};
export const LOSS_CHOICES: Record<string, string[]> = {
  none: [],
  water_ammonia: ['H2O', 'NH3'],
  phospho: ['H3PO4'],
  all: ['H2O', 'NH3', 'H3PO4'],
};

export const SPECTRUM_KIND: PanelKind = {
  kind: 'ms_spectrum',
  order: 1,
  unit: ['peak', 'peaks'],
  // Annotation settings change what the server matches: refetch on change.
  pushdown: {
    ion_types: (v) => (typeof v === 'string' ? v : undefined),
    max_fragment_charge: (v) => (typeof v === 'string' ? v : undefined),
    neutral_losses: (v) => (typeof v === 'string' ? v : undefined),
  },
  settings: {
    sections: [
      {
        title: 'Style',
        controls: [
          {
            type: 'palette',
            entries: [
              { key: 'stick', label: 'Peaks', default: SPECTRUM_DEFAULTS.stick },
              { key: 'label', label: 'Labels', default: SPECTRUM_DEFAULTS.label },
              { key: 'isolation', label: 'Isolation window', default: SPECTRUM_DEFAULTS.isolation },
            ],
          },
          {
            type: 'number',
            target: 'style',
            key: 'labels',
            label: 'Unannotated peaks labelled',
            default: SPECTRUM_DEFAULTS.labels,
            min: 0,
            max: 50,
            step: 1,
          },
          { type: 'toggle', target: 'style', key: 'error_strip', label: 'Mass-error strip', default: true },
        ],
      },
      {
        title: 'Annotation',
        controls: [
          {
            type: 'select',
            target: 'filter',
            key: 'ion_types',
            label: 'Ion series',
            default: 'by',
            options: [
              { value: 'by', label: 'b, y' },
              { value: 'aby', label: 'a, b, y' },
              { value: 'cz', label: 'c, z (ETD)' },
              { value: 'bcyz', label: 'b, c, y, z' },
              { value: 'all', label: 'all' },
            ],
          },
          {
            type: 'select',
            target: 'filter',
            key: 'max_fragment_charge',
            label: 'Fragment charges',
            default: 'auto',
            options: [
              { value: 'auto', label: 'up to one below the precursor' },
              { value: '1', label: '1' },
              { value: '2', label: 'up to 2' },
              { value: '3', label: 'up to 3' },
            ],
          },
          {
            type: 'select',
            target: 'filter',
            key: 'neutral_losses',
            label: 'Neutral losses',
            default: 'none',
            options: [
              { value: 'none', label: 'none' },
              { value: 'water_ammonia', label: 'H₂O, NH₃' },
              { value: 'phospho', label: 'H₃PO₄' },
              { value: 'all', label: 'H₂O, NH₃, H₃PO₄' },
            ],
          },
          { type: 'note', text: 'Changing these annotates the spectrum again on the server.' },
        ],
      },
    ],
  },
};

/** The annotation settings of a spectrum panel, as query values. */
export function annotationSettings(filter: Record<string, unknown>): {
  ion_types: string[];
  max_fragment_charge: number | null;
  neutral_losses: string[];
} {
  const types = ION_TYPE_CHOICES[pickString(filter, 'ion_types', 'by')] ?? ION_TYPE_CHOICES.by;
  const charge = pickString(filter, 'max_fragment_charge', 'auto');
  const losses = LOSS_CHOICES[pickString(filter, 'neutral_losses', 'none')] ?? [];
  return { ion_types: types, max_fragment_charge: charge === 'auto' ? null : Number(charge), neutral_losses: losses };
}

/** The m/z extent of a spectrum with a little room either side. */
export function spectrumExtent(mzs: ArrayLike<number>): Interval {
  if (mzs.length === 0) return [0, 1];
  let lo = Infinity;
  let hi = -Infinity;
  for (let i = 0; i < mzs.length; i++) {
    if (mzs[i] < lo) lo = mzs[i];
    if (mzs[i] > hi) hi = mzs[i];
  }
  const pad = Math.max(1, (hi - lo) * 0.03);
  return [Math.max(0, lo - pad), hi + pad];
}

/** Index of the peak nearest `mz` within `within` (m/z units), or null.
 *  `mzs` is in ascending order, as the kernel sends it. */
export function nearestPeak(mzs: ArrayLike<number>, mz: number, within: number): number | null {
  let lo = 0;
  let hi = mzs.length - 1;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (mzs[mid] < mz) lo = mid + 1;
    else hi = mid;
  }
  let best: number | null = null;
  for (const i of [lo - 1, lo]) {
    if (i < 0 || i >= mzs.length) continue;
    const d = Math.abs(mzs[i] - mz);
    if (d <= within && (best === null || d < Math.abs(mzs[best] - mz))) best = i;
  }
  return best;
}

/** Indices of the peaks to label, in order of priority, skipping any whose
 *  label would overlap one already placed (by pixel extent). */
export function chooseLabels(
  order: readonly number[],
  xs: ArrayLike<number>,
  limit: number,
  widthOf: (i: number) => number,
): number[] {
  const placed: Array<[number, number]> = [];
  const out: number[] = [];
  for (const i of order) {
    if (out.length >= limit) break;
    const half = widthOf(i) / 2 + 2;
    const span: [number, number] = [xs[i] - half, xs[i] + half];
    if (placed.some(([a, b]) => span[0] < b && span[1] > a)) continue;
    placed.push(span);
    out.push(i);
  }
  return out;
}

function strings(table: Table, name: string): Array<string | null> {
  const col = table.getChild(name);
  const out: Array<string | null> = [];
  for (let i = 0; i < table.numRows; i++) out.push((col?.get(i) as string | null) ?? null);
  return out;
}

/** The height of a profile near `mz`: its highest of the nearest points. */
function heightNear(mzs: Float64Array, ys: Float64Array, mz: number): number {
  let lo = 0;
  let hi = mzs.length;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (mzs[mid] < mz) lo = mid + 1;
    else hi = mid;
  }
  let best = 0;
  for (let i = Math.max(0, lo - 2); i <= Math.min(mzs.length - 1, lo + 1); i++) best = Math.max(best, ys[i]);
  return best;
}

/** How much peak area one unit of profile height stands for in this scan:
 *  the median over its strongest sticks of area over the profile's height
 *  there. Peak intensities are areas (they sum a peak's points) and profile
 *  intensities heights, so on one axis sticks stand about six times too tall.
 *  1 when nothing can be measured. */
export function areaPerHeight(profile: Table, peaks: Table, strongest = 20): number {
  const mzs = profile.getChild('mz')?.toArray() as Float64Array | undefined;
  const ys = profile.getChild('intensity')?.toArray() as Float64Array | undefined;
  const peakMz = peaks.getChild('mz')?.toArray() as Float64Array | undefined;
  const area = peaks.getChild('intensity')?.toArray() as Float64Array | undefined;
  if (!mzs || !ys || !peakMz || !area || !mzs.length) return 1;
  const order = Array.from(area.keys()).sort((a, b) => area[b] - area[a]).slice(0, strongest);
  const ratios = order.flatMap((i) => {
    const height = heightNear(mzs, ys, peakMz[i]);
    return height > 0 && area[i] > 0 ? [area[i] / height] : [];
  });
  if (!ratios.length) return 1;
  ratios.sort((a, b) => a - b);
  return ratios[Math.floor(ratios.length / 2)];
}

interface SpectrumPoints {
  table: Table;
  mzs: Float64Array;
  raw: Float64Array;
  labels: (string | null)[];
  ionTypes: (string | null)[];
}

function pointsOf(table: Table): SpectrumPoints {
  return {
    table,
    mzs: (table.getChild('mz')?.toArray() as Float64Array | undefined) ?? new Float64Array(0),
    raw: (table.getChild('intensity')?.toArray() as Float64Array | undefined) ?? new Float64Array(0),
    labels: strings(table, 'annotation'),
    ionTypes: strings(table, 'ion_type'),
  };
}

function indicesIn(points: SpectrumPoints, domain: Interval): number[] {
  const inView: number[] = [];
  for (let i = 0; i < points.mzs.length; i++) {
    if (points.mzs[i] >= domain[0] && points.mzs[i] <= domain[1]) inView.push(i);
  }
  return inView;
}

/** Draw the spectrum: profile points as a line, peaks as sticks, and any
 *  overlays as sticks over them. Returns how many peaks of the top list
 *  were in view. */
export function drawSpectrum(
  svg: SVGSVGElement,
  size: { widthPx: number; heightPx: number },
  table: Table,
  opts: SpectrumOptions,
): number {
  const base = pointsOf(table);
  const overlays = (opts.overlays ?? []).map((o) => ({ ...o, points: pointsOf(o.table) }));
  // What is labelled and annotated: the top list.
  const top = overlays.length ? overlays[overlays.length - 1].points : base;
  const profile = table.numRows > 0 && Boolean(table.getChild('profile')?.get(0));
  const overProfile = profile && overlays.length > 0;
  const annotated = top.labels.some((l) => l !== null);
  const strip = annotated && opts.errorStrip && pickBool(opts.style, 'error_strip', true) ? opts.errorStrip : null;
  const domain = opts.mz ?? spectrumExtent(base.mzs);
  const mainSize = { widthPx: size.widthPx, heightPx: strip ? size.heightPx - ERROR_STRIP_PX : size.heightPx };
  const frame = plotFrame(mainSize, strip ? { ...SPECTRUM_MARGIN, bottom: 6 } : SPECTRUM_MARGIN);

  // A stick's height in the base list's units.
  const perHeight = overProfile ? (opts.perHeight ?? 1) : 1;
  const stickHeight = (mz: number, area: number): number =>
    overProfile && opts.anchored ? heightNear(base.mzs, base.raw, mz) : area / perHeight;

  const inView = indicesIn(base, domain);
  let max = 0;
  for (const i of inView) if (base.raw[i] > max) max = base.raw[i];
  const overlayViews = overlays.map((o) => indicesIn(o.points, domain));
  overlays.forEach((o, k) => {
    for (const i of overlayViews[k]) {
      const h = stickHeight(o.points.mzs[i], o.points.raw[i]);
      if (h > max) max = h;
    }
  });
  const rel = opts.intensity === 'rel';
  const scaleBy = rel && max > 0 ? 100 / max : 1;
  const topValue = rel ? 108 : max > 0 ? max * 1.12 : 1;
  const x = xAxis(frame, domain, AXIS_FORMATS.mz);
  const y = yAxis(frame, [0, topValue], rel ? AXIS_FORMATS.percent : AXIS_FORMATS.intensity, {
    log: opts.intensity === 'log',
  });
  // Areas on the right: linear, so not on a log plot, and nothing to say when
  // the sticks are anchored to the profile.
  const yRight =
    overProfile && !opts.anchored && opts.intensity !== 'log'
      ? yAxis(frame, [0, (topValue / scaleBy) * perHeight], { label: 'Area', tick: AXIS_FORMATS.intensity.tick })
      : undefined;
  const plot = drawAxes(svg, frame, strip ? { y, yRight } : { x, y, yRight, xLabel: AXIS_FORMATS.mz.label }, opts.uid);
  const isolationColor = pickPaletteColor(opts.style, 'isolation', SPECTRUM_DEFAULTS.isolation);

  if (opts.isolation) {
    const a = x.scale(Math.max(domain[0], opts.isolation[0]));
    const b = x.scale(Math.min(domain[1], opts.isolation[1]));
    if (b > a) {
      plot.appendChild(svgEl('rect', {
        x: a, y: frame.inner.y, width: b - a, height: frame.inner.h, fill: isolationColor, 'fill-opacity': 0.15,
        class: 'ms-isolation',
      }));
    }
  }
  if (opts.target !== null && opts.target >= domain[0] && opts.target <= domain[1]) {
    const px = x.scale(opts.target);
    plot.appendChild(svgEl('line', {
      x1: px, x2: px, y1: frame.inner.y, y2: frame.inner.y + frame.inner.h, stroke: isolationColor,
      'stroke-dasharray': '3 3', class: 'ms-target',
    }));
  }
  for (const iso of opts.isotopes ?? []) {
    if (iso.mz < domain[0] || iso.mz > domain[1]) continue;
    const px = x.scale(iso.mz);
    plot.appendChild(svgEl('path', {
      d: `M${px - 4} ${frame.inner.y} L${px + 4} ${frame.inner.y} L${px} ${frame.inner.y + 6} Z`,
      fill: isolationColor, class: 'ms-isotope',
    }));
  }

  const neutral = pickPaletteColor(opts.style, 'stick', SPECTRUM_DEFAULTS.stick);
  const zero = y.scale(0);
  const px: number[] = new Array(top.mzs.length);
  const stickTop: number[] = new Array(top.mzs.length);

  /** One list as sticks: unannotated in `color`, annotated by ion series,
   *  those on top. `only` leaves out what is not annotated. */
  const drawSticks = (
    points: SpectrumPoints,
    view: number[],
    color: string,
    height: (i: number) => number,
    cls: string,
    only?: 'annotated',
  ): void => {
    const byColor = new Map<string, string>();
    for (const i of view) {
      const isAnnotated = points.labels[i] !== null;
      if (only && !isAnnotated) continue;
      const c = isAnnotated ? (ionColor(points.ionTypes[i]) ?? color) : color;
      const sx = x.scale(points.mzs[i]);
      const sy = y.scale(height(i) * scaleBy);
      if (points === top) {
        px[i] = sx;
        stickTop[i] = sy;
      }
      byColor.set(c, (byColor.get(c) ?? '') + `M${sx.toFixed(1)} ${zero.toFixed(1)}V${sy.toFixed(1)}`);
    }
    for (const [c, d] of [...byColor.entries()].sort(([a], [b]) => (a === color ? -1 : b === color ? 1 : 0))) {
      plot.appendChild(svgEl('path', {
        d, stroke: c, 'stroke-width': c === color && !overlays.length ? 1 : 1.5, fill: 'none',
        class: c === color ? cls : 'ms-peaks-annotated',
      }));
    }
  };

  if (profile) {
    plot.appendChild(svgEl('path', {
      d: linePath(base.mzs, base.raw, domain, (v) => x.scale(v), (v) => y.scale(v * scaleBy)),
      stroke: neutral, 'stroke-width': 1, fill: 'none', 'stroke-linejoin': 'round', class: 'ms-profile',
    }));
    // A peptide's ions on the points they fell on.
    drawSticks(base, inView, neutral, (i) => base.raw[i], 'ms-peaks', 'annotated');
    if (top === base) {
      for (const i of inView) {
        px[i] = x.scale(base.mzs[i]);
        stickTop[i] = y.scale(base.raw[i] * scaleBy);
      }
    }
  } else {
    drawSticks(base, inView, neutral, (i) => base.raw[i], 'ms-peaks');
  }
  overlays.forEach((o, k) => {
    drawSticks(o.points, overlayViews[k], o.color, (i) => stickHeight(o.points.mzs[i], o.points.raw[i]), 'ms-overlay-peaks');
  });

  // Labels: annotations (most intense first), then m/z of the rest.
  const topView = overlays.length ? overlayViews[overlayViews.length - 1] : inView;
  const decimals = Math.max(2, Math.min(4, Math.ceil(-Math.log10(Math.max(1e-6, (domain[1] - domain[0]) / 2000)))));
  const textOf = (i: number): string => top.labels[i] ?? top.mzs[i].toFixed(decimals);
  const byIntensity = [...topView].sort((a, b) => top.raw[b] - top.raw[a]);
  const annotatedOrder = byIntensity.filter((i) => top.labels[i] !== null);
  const restOrder = byIntensity.filter((i) => top.labels[i] === null);
  const limit = annotatedOrder.length + pickNumber(opts.style, 'labels', SPECTRUM_DEFAULTS.labels);
  const labelColor = pickPaletteColor(opts.style, 'label', SPECTRUM_DEFAULTS.label);
  for (const i of chooseLabels([...annotatedOrder, ...restOrder], px, limit, (k) => textOf(k).length * 5.6)) {
    const text = svgEl('text', {
      x: px[i], y: Math.max(frame.inner.y + 9, stickTop[i] - 3), 'text-anchor': 'middle',
      fill: labelColor, 'font-size': 9, class: top.labels[i] !== null ? 'ms-ion-label' : 'ms-peak-label',
    });
    text.textContent = textOf(i);
    plot.appendChild(text);
  }

  if (overlays.length) {
    drawListLegend(svg, frame.inner, [
      { label: opts.baseName ?? (profile ? 'profile' : 'peaks'), color: neutral },
      ...overlays.map((o) => ({ label: o.name, color: o.color })),
    ], labelColor);
  }
  if (strip) drawErrorStrip(svg, size, frame, domain, top.table, topView, strip, opts.uid);
  svg.dataset.naturalHeight = String(size.heightPx);
  return topView.length;
}

/** Which list is which, in the plot's top-right corner. */
function drawListLegend(
  svg: SVGSVGElement,
  inner: { x: number; y: number; w: number; h: number },
  entries: Array<{ label: string; color: string }>,
  textColor: string,
): void {
  const g = svgEl('g', { class: 'ms-legend' });
  entries.forEach((entry, i) => {
    const yy = inner.y + 8 + i * 12;
    const xx = inner.x + inner.w - 110;
    g.appendChild(svgEl('line', { x1: xx, x2: xx + 14, y1: yy - 3, y2: yy - 3, stroke: entry.color, 'stroke-width': 2 }));
    const text = svgEl('text', { x: xx + 18, y: yy, class: 'ms-legend-label', 'font-size': 10, fill: textColor });
    text.textContent = entry.label;
    g.appendChild(text);
  });
  svg.appendChild(g);
}

/** The matched peaks' mass errors against m/z, in a strip under the plot
 *  sharing its x axis, its y axis the matching tolerance either way. */
function drawErrorStrip(
  svg: SVGSVGElement,
  size: { widthPx: number; heightPx: number },
  main: { inner: { x: number; w: number }; height: number },
  domain: Interval,
  table: Table,
  inView: number[],
  strip: { tolerance: number; unit: 'ppm' | 'Da' },
  uid: string,
): void {
  const nested = svgEl('svg', { x: 0, y: main.height, width: size.widthPx, height: ERROR_STRIP_PX, class: 'ms-error-strip' });
  svg.appendChild(nested);
  const frame = plotFrame({ widthPx: size.widthPx, heightPx: ERROR_STRIP_PX }, { top: 4, right: SPECTRUM_MARGIN.right, bottom: 30, left: SPECTRUM_MARGIN.left });
  const format: AxisFormat = { label: strip.unit, tick: (v) => (v === 0 ? '0' : v.toFixed(strip.unit === 'ppm' ? 0 : 2)) };
  const x = xAxis(frame, domain, AXIS_FORMATS.mz);
  const y = yAxis(frame, [-strip.tolerance, strip.tolerance], format, { ticks: 2 });
  const plot = drawAxes(nested as SVGSVGElement, frame, { x, y, xLabel: `${AXIS_FORMATS.mz.label} · error (${strip.unit})` }, `${uid}-err`);
  plot.appendChild(svgEl('line', { x1: frame.inner.x, x2: frame.inner.x + frame.inner.w, y1: y.scale(0), y2: y.scale(0), stroke: '#3b334b' }));
  const errors = table.getChild(strip.unit === 'ppm' ? 'error_ppm' : 'error_da');
  const ionTypes = table.getChild('ion_type');
  for (const i of inView) {
    const e = errors?.get(i);
    if (e === null || e === undefined) continue;
    plot.appendChild(svgEl('circle', {
      cx: x.scale(Number(table.getChild('mz')?.get(i))), cy: y.scale(Math.max(-strip.tolerance, Math.min(strip.tolerance, Number(e)))),
      r: 2.5, fill: ionColor(ionTypes?.get(i) as string | null) ?? '#adacae', class: 'ms-error-point',
    }));
  }
}
