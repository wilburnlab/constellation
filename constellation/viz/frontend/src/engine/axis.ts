// Continuous axes for plots: a plot frame inside a panel's SVG, x and y
// scales over float domains, tick formatting, and drawing the axes.
//
// A panel's SVG is a frame: margins hold the axes and their labels, the
// inner rectangle holds the data. Panels that share an x axis (the
// chromatograms of a mass-spec browser, stacked by MS level) use the
// same margins so their plot areas line up whatever their tick labels.
//
// Axes are drawn as plain SVG (lines and text), not through d3-axis, so
// they render identically under jsdom and in an export.

import { scaleLinear, scaleSymlog, type ScaleContinuousNumeric } from 'd3-scale';
import type { Interval } from './interval';
import { svgEl } from './svg_layer';
import './plot.css';

export interface Margins {
  top: number;
  right: number;
  bottom: number;
  left: number;
}

export interface PlotFrame {
  width: number;
  height: number;
  margin: Margins;
  /** The data area, in the SVG's pixel coordinates. */
  inner: { x: number; y: number; w: number; h: number };
}

export function plotFrame(size: { widthPx: number; heightPx: number }, margin: Margins): PlotFrame {
  return {
    width: size.widthPx,
    height: size.heightPx,
    margin,
    inner: {
      x: margin.left,
      y: margin.top,
      w: Math.max(1, size.widthPx - margin.left - margin.right),
      h: Math.max(1, size.heightPx - margin.top - margin.bottom),
    },
  };
}

/** How an axis writes its ticks, and its title. `step` is the spacing
 *  between ticks, so decimals can follow the zoom. */
export interface AxisFormat {
  tick(value: number, step: number): string;
  label: string;
  /** Data units per displayed unit (60 for seconds shown as minutes), so
   *  ticks fall on round displayed values. */
  per?: number;
}

/** Decimal places that tell ticks `step` apart. */
export function decimalsFor(step: number): number {
  if (!(step > 0) || !Number.isFinite(step)) return 0;
  return Math.min(6, Math.max(0, -Math.floor(Math.log10(step) + 1e-9)));
}

/** An intensity in the fewest characters that keep its size: plain below
 *  ten thousand, exponent notation above (`2.4e6`). */
export function formatIntensity(value: number): string {
  if (value === 0) return '0';
  const abs = Math.abs(value);
  if (abs < 1e4) return abs < 10 ? value.toPrecision(2).replace(/\.?0+$/, '') : String(Math.round(value));
  const [mantissa, exponent] = value.toExponential(1).split('e');
  return `${mantissa.replace(/\.0$/, '')}e${Number(exponent)}`;
}

export const AXIS_FORMATS = {
  /** Seconds on the wire, minutes on the axis. */
  rtMinutes: {
    label: 'Retention time (min)',
    per: 60,
    tick: (v: number, step: number) => (v / 60).toFixed(decimalsFor(step / 60)),
  },
  mz: {
    label: 'm/z',
    tick: (v: number, step: number) => v.toFixed(decimalsFor(step)),
  },
  intensity: {
    label: 'Intensity',
    tick: (v: number) => formatIntensity(v),
  },
  percent: {
    label: 'Relative intensity (%)',
    tick: (v: number) => String(Math.round(v)),
  },
} satisfies Record<string, AxisFormat>;

export interface ContinuousAxis {
  domain: Interval;
  /** Data value → SVG pixel. */
  scale(value: number): number;
  /** SVG pixel → data value. */
  invert(px: number): number;
  ticks: number[];
  /** Spacing of the ticks (for formatting). */
  step: number;
  format: AxisFormat;
}

function build(
  scale: ScaleContinuousNumeric<number, number>,
  domain: Interval,
  format: AxisFormat,
  count: number,
): ContinuousAxis {
  const per = format.per ?? 1;
  const ticks =
    per === 1
      ? scale.ticks(count)
      : scaleLinear().domain([domain[0] / per, domain[1] / per]).ticks(count).map((t) => t * per);
  const step = ticks.length > 1 ? ticks[1] - ticks[0] : domain[1] - domain[0];
  return {
    domain,
    scale: (v) => scale(v),
    invert: (px) => scale.invert(px),
    ticks,
    step,
    format,
  };
}

export function xAxis(
  frame: PlotFrame,
  domain: Interval,
  format: AxisFormat,
  opts: { ticks?: number } = {},
): ContinuousAxis {
  const { x, w } = frame.inner;
  const count = opts.ticks ?? Math.max(2, Math.floor(w / 90));
  return build(scaleLinear().domain(domain).range([x, x + w]), domain, format, count);
}

/** A vertical axis. `log` uses a symmetric log scale, which is defined at
 *  zero (intensities often are). */
export function yAxis(
  frame: PlotFrame,
  domain: Interval,
  format: AxisFormat,
  opts: { ticks?: number; log?: boolean } = {},
): ContinuousAxis {
  const { y, h } = frame.inner;
  const count = opts.ticks ?? Math.max(2, Math.floor(h / 40));
  const scale = (opts.log ? scaleSymlog().constant(1) : scaleLinear())
    .domain(domain)
    .range([y + h, y]);
  return build(scale, domain, format, count);
}

/** A y domain that fits the data in view: from 0 to the largest value
 *  among points whose x lies in `visible` (all points when omitted), with
 *  headroom above it. Never empty. */
export function autoYDomain(
  ys: ArrayLike<number>,
  xs?: ArrayLike<number>,
  visible?: Interval,
  headroom = 1.08,
): Interval {
  let max = 0;
  for (let i = 0; i < ys.length; i++) {
    const v = ys[i];
    if (!Number.isFinite(v)) continue;
    if (xs && visible && (xs[i] < visible[0] || xs[i] > visible[1])) continue;
    if (v > max) max = v;
  }
  return [0, max > 0 ? max * headroom : 1];
}

const TICK = 4;

/** Draw the axes into `svg` and return a group clipped to the data area,
 *  for the caller to draw into. `uid` makes the clip-path id unique
 *  among the SVGs on the page. */
export function drawAxes(
  svg: SVGSVGElement,
  frame: PlotFrame,
  axes: { x?: ContinuousAxis; y?: ContinuousAxis; xLabel?: string; yLabel?: string },
  uid: string,
): SVGGElement {
  const { x: ix, y: iy, w, h } = frame.inner;
  const defs = svgEl('defs');
  const clip = svgEl('clipPath', { id: `${uid}-clip` });
  clip.appendChild(svgEl('rect', { x: ix, y: iy, width: w, height: h }));
  defs.appendChild(clip);
  svg.appendChild(defs);

  const axisGroup = svgEl('g', { class: 'plot-axes' });
  if (axes.x) {
    const base = iy + h;
    axisGroup.appendChild(svgEl('line', { class: 'axis-line', x1: ix, x2: ix + w, y1: base, y2: base }));
    for (const t of axes.x.ticks) {
      const px = axes.x.scale(t);
      axisGroup.appendChild(svgEl('line', { class: 'axis-tick', x1: px, x2: px, y1: base, y2: base + TICK }));
      const text = svgEl('text', { class: 'axis-tick-label', x: px, y: base + TICK + 10, 'text-anchor': 'middle' });
      text.textContent = axes.x.format.tick(t, axes.x.step);
      axisGroup.appendChild(text);
    }
    if (axes.xLabel) {
      // Its own line under the tick labels (the margin must leave room).
      const label = svgEl('text', { class: 'axis-label', x: ix + w / 2, y: base + TICK + 22, 'text-anchor': 'middle' });
      label.textContent = axes.xLabel;
      axisGroup.appendChild(label);
    }
  }
  if (axes.y) {
    axisGroup.appendChild(svgEl('line', { class: 'axis-line', x1: ix, x2: ix, y1: iy, y2: iy + h }));
    for (const t of axes.y.ticks) {
      const py = axes.y.scale(t);
      axisGroup.appendChild(svgEl('line', { class: 'axis-tick', x1: ix - TICK, x2: ix, y1: py, y2: py }));
      const text = svgEl('text', { class: 'axis-tick-label', x: ix - TICK - 2, y: py + 3, 'text-anchor': 'end' });
      text.textContent = axes.y.format.tick(t, axes.y.step);
      axisGroup.appendChild(text);
    }
    if (axes.yLabel) {
      const label = svgEl('text', {
        class: 'axis-label',
        x: 0,
        y: 0,
        'text-anchor': 'middle',
        transform: `translate(10 ${iy + h / 2}) rotate(-90)`,
      });
      label.textContent = axes.yLabel;
      axisGroup.appendChild(label);
    }
  }
  svg.appendChild(axisGroup);

  const plot = svgEl('g', { class: 'plot-area', 'clip-path': `url(#${uid}-clip)` });
  svg.appendChild(plot);
  return plot;
}
