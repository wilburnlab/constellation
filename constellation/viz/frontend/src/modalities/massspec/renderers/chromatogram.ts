// ms_chromatogram — intensity over retention time for one MS level.
//
// Draws the trace the kernel sends (TIC, base peak, or one stick per scan
// for a DDA level's scan events) over the shared retention-time window,
// with the row's selected scan marked. With a target, the row's traced
// ions (ms_xic) take over the plot: one line each in the fixed series
// order, a legend, the engine's integration window shaded, its apex and
// the library's predicted time marked, and its median chromatogram
// dotted. The y axis fits what is in view.

import type { Table } from 'apache-arrow';
import {
  AXIS_FORMATS,
  type Margins,
  autoYDomain,
  drawAxes,
  plotFrame,
  xAxis,
  yAxis,
} from '../../../engine/axis';
import type { Interval } from '../../../engine/interval';
import { svgEl } from '../../../engine/svg_layer';
import type { PanelKind } from '../../../panels/kind';
import { pickBool, pickNumber, pickPaletteColor } from '../../../panels/style';

/** The same for every chromatogram, so their plot areas line up. */
export const CHROM_MARGIN: Margins = { top: 8, right: 12, bottom: 32, left: 58 };

export const CHROM_DEFAULTS = {
  line: '#a277ff',
  marker: '#61ffca',
  lineWidth: 1.25,
  boundary: '#a277ff',
  apex: '#adacae',
  predicted: '#ffca85',
  median: '#edecee',
  /** Legend text: the text colour, never a series colour. */
  text: '#edecee',
};

export type IntensityMode = 'abs' | 'rel' | 'log';

/** A traced ion drawn over the chromatogram. */
export interface Overlay {
  label: string;
  color: string;
  xs: Float64Array;
  ys: Float64Array;
  /** Drawn dashed: an ion the engine did not quantify from. */
  dashed?: boolean;
}

export interface ChromatogramOptions {
  rt: Interval;
  intensity: IntensityMode;
  /** Retention time of the row's selected scan. */
  selectedRt: number | null;
  /** Draw sticks rather than a line (scan events). */
  sticks: boolean;
  /** For sticks: draw only scans whose precursor m/z this accepts. */
  stickFilter?: ((mz: number) => boolean) | null;
  overlays?: Overlay[];
  /** The integration window, apex and predicted time (seconds). */
  marks?: { start: number | null; end: number | null; apex: number | null; predicted: number | null } | null;
  /** The engine's median chromatogram, scaled to the overlays. */
  median?: { xs: Float64Array; ys: Float64Array } | null;
  style: Record<string, unknown>;
  uid: string;
}

export const CHROMATOGRAM_KIND: PanelKind = {
  kind: 'ms_chromatogram',
  order: 0,
  unit: ['point', 'points'],
  settings: {
    sections: [
      {
        title: 'Style',
        controls: [
          {
            type: 'palette',
            entries: [
              { key: 'line', label: 'Trace', default: CHROM_DEFAULTS.line },
              { key: 'marker', label: 'Selected scan', default: CHROM_DEFAULTS.marker },
              { key: 'boundary', label: 'Integration window', default: CHROM_DEFAULTS.boundary },
            ],
          },
          {
            type: 'number',
            target: 'style',
            key: 'line_width',
            label: 'Line width',
            default: CHROM_DEFAULTS.lineWidth,
            min: 0.5,
            max: 4,
            step: 0.25,
          },
        ],
      },
      {
        title: 'Target',
        controls: [
          { type: 'toggle', target: 'style', key: 'show_window', label: 'Shade the integration window', default: true },
          { type: 'toggle', target: 'style', key: 'show_predicted', label: 'Mark the predicted time', default: true },
          { type: 'toggle', target: 'style', key: 'show_median', label: "Engine's median chromatogram", default: true },
          { type: 'toggle', target: 'style', key: 'show_legend', label: 'Legend', default: true },
        ],
      },
    ],
  },
};

function column(table: Table, name: string): Float64Array {
  const col = table.getChild(name);
  const out = new Float64Array(table.numRows);
  for (let i = 0; i < table.numRows; i++) {
    const v = col?.get(i);
    out[i] = v === null || v === undefined ? NaN : Number(v);
  }
  return out;
}

/** Indices of `xs` (ascending) in `[lo, hi]`, plus one on each side. */
function span(xs: Float64Array, iv: Interval): [number, number] {
  let first = 0;
  while (first < xs.length && xs[first] < iv[0]) first++;
  let last = xs.length - 1;
  while (last >= 0 && xs[last] > iv[1]) last--;
  return [Math.max(0, first - 1), Math.min(xs.length - 1, last + 1)];
}

function linePath(xs: Float64Array, ys: Float64Array, iv: Interval, sx: (v: number) => number, sy: (v: number) => number): string {
  const [lo, hi] = span(xs, iv);
  let d = '';
  for (let i = lo; i <= hi; i++) {
    if (!Number.isFinite(ys[i])) continue;
    d += `${d ? 'L' : 'M'}${sx(xs[i]).toFixed(1)} ${sy(ys[i]).toFixed(1)}`;
  }
  return d;
}

/** Draw the chromatogram; returns how many points were in view. */
export function drawChromatogram(
  svg: SVGSVGElement,
  size: { widthPx: number; heightPx: number },
  table: Table,
  opts: ChromatogramOptions,
): number {
  const overlays = opts.overlays ?? [];
  const traced = overlays.length > 0;
  const xs = column(table, 'rt');
  const values = column(table, 'intensity').map((v) => (Number.isFinite(v) ? v : 0));
  const mzs = opts.stickFilter ? column(table, 'mz') : null;
  const keep = (i: number): boolean => !mzs || !opts.stickFilter || opts.stickFilter(mzs[i]);

  // The y axis fits what is drawn: the traces when there are any.
  let max = 0;
  if (traced) {
    for (const o of overlays) max = Math.max(max, autoYDomain(o.ys, o.xs, opts.rt, 1)[1]);
  } else {
    for (let i = 0; i < xs.length; i++) {
      if (xs[i] >= opts.rt[0] && xs[i] <= opts.rt[1] && keep(i) && values[i] > max) max = values[i];
    }
  }
  const rel = opts.intensity === 'rel';
  const scaleBy = rel && max > 0 ? 100 / max : 1;
  const top = rel ? 108 : max > 0 ? max * 1.08 : 1;
  const frame = plotFrame(size, CHROM_MARGIN);
  const x = xAxis(frame, opts.rt, AXIS_FORMATS.rtMinutes);
  const y = yAxis(frame, [0, top], rel ? AXIS_FORMATS.percent : AXIS_FORMATS.intensity, {
    log: opts.intensity === 'log',
  });
  const plot = drawAxes(svg, frame, { x, y, xLabel: AXIS_FORMATS.rtMinutes.label }, opts.uid);
  const sy = (v: number): number => y.scale(v * scaleBy);
  const vline = (at: number, attrs: Record<string, string | number>): void => {
    if (at < opts.rt[0] || at > opts.rt[1]) return;
    const px = x.scale(at);
    plot.appendChild(svgEl('line', { x1: px, x2: px, y1: frame.inner.y, y2: frame.inner.y + frame.inner.h, ...attrs }));
  };

  const marks = opts.marks;
  if (marks && marks.start !== null && marks.end !== null && pickBool(opts.style, 'show_window', true)) {
    const a = x.scale(Math.max(opts.rt[0], marks.start));
    const b = x.scale(Math.min(opts.rt[1], marks.end));
    if (b > a) {
      plot.appendChild(svgEl('rect', {
        x: a, y: frame.inner.y, width: b - a, height: frame.inner.h,
        fill: pickPaletteColor(opts.style, 'boundary', CHROM_DEFAULTS.boundary), 'fill-opacity': 0.12,
        class: 'ms-integration-window',
      }));
    }
  }

  const color = pickPaletteColor(opts.style, 'line', CHROM_DEFAULTS.line);
  const width = pickNumber(opts.style, 'line_width', CHROM_DEFAULTS.lineWidth);
  let shown = 0;
  if (traced) {
    for (const o of overlays) {
      const [lo, hi] = span(o.xs, opts.rt);
      shown += Math.max(0, hi - lo + 1);
      const d = linePath(o.xs, o.ys, opts.rt, (v) => x.scale(v), sy);
      if (!d) continue;
      plot.appendChild(svgEl('path', {
        d, stroke: o.color, 'stroke-width': width, fill: 'none', 'stroke-linejoin': 'round',
        ...(o.dashed ? { 'stroke-dasharray': '4 3' } : {}), class: 'ms-xic',
      }));
    }
    const median = opts.median;
    if (median && median.xs.length && pickBool(opts.style, 'show_median', true)) {
      let mmax = 0;
      for (let i = 0; i < median.ys.length; i++) mmax = Math.max(mmax, median.ys[i]);
      if (mmax > 0) {
        const scaled = median.ys.map((v) => (v / mmax) * max);
        const d = linePath(median.xs, scaled, opts.rt, (v) => x.scale(v), sy);
        if (d) {
          plot.appendChild(svgEl('path', {
            d, stroke: CHROM_DEFAULTS.median, 'stroke-width': 1, 'stroke-dasharray': '1 3', fill: 'none', class: 'ms-median',
          }));
        }
      }
    }
  } else if (opts.sticks) {
    const base = y.scale(0);
    let d = '';
    const [lo, hi] = span(xs, opts.rt);
    for (let i = lo; i <= hi; i++) {
      if (!keep(i)) continue;
      shown++;
      d += `M${x.scale(xs[i]).toFixed(1)} ${base.toFixed(1)}V${sy(values[i]).toFixed(1)}`;
    }
    if (d) plot.appendChild(svgEl('path', { d, stroke: color, 'stroke-width': width, fill: 'none', class: 'ms-trace' }));
  } else if (xs.length) {
    const [lo, hi] = span(xs, opts.rt);
    shown = Math.max(0, hi - lo + 1);
    const d = linePath(xs, values, opts.rt, (v) => x.scale(v), sy);
    if (d) {
      plot.appendChild(svgEl('path', {
        d, stroke: color, 'stroke-width': width, fill: 'none', 'stroke-linejoin': 'round', class: 'ms-trace',
      }));
    }
  }

  if (marks?.apex !== null && marks?.apex !== undefined) {
    vline(marks.apex, { stroke: CHROM_DEFAULTS.apex, 'stroke-width': 1, class: 'ms-apex' });
  }
  if (marks?.predicted !== null && marks?.predicted !== undefined && pickBool(opts.style, 'show_predicted', true)) {
    vline(marks.predicted, { stroke: CHROM_DEFAULTS.predicted, 'stroke-width': 1, 'stroke-dasharray': '2 3', class: 'ms-predicted' });
  }
  if (opts.selectedRt !== null) {
    vline(opts.selectedRt, {
      stroke: pickPaletteColor(opts.style, 'marker', CHROM_DEFAULTS.marker), 'stroke-width': 1, class: 'ms-selected-scan',
    });
  }
  if (traced && pickBool(opts.style, 'show_legend', true)) drawLegend(svg, frame.inner, overlays);
  svg.dataset.naturalHeight = String(size.heightPx);
  return shown;
}

/** A legend in the plot's top-right corner: a coloured swatch per trace,
 *  its label in the text colour (identity is never colour alone). */
function drawLegend(svg: SVGSVGElement, inner: { x: number; y: number; w: number; h: number }, overlays: Overlay[]): void {
  const g = svgEl('g', { class: 'ms-legend' });
  const rowH = 12;
  const rows = Math.min(overlays.length, Math.max(1, Math.floor((inner.h - 8) / rowH)));
  overlays.slice(0, rows).forEach((o, i) => {
    const yy = inner.y + 8 + i * rowH;
    const xx = inner.x + inner.w - 70;
    g.appendChild(svgEl('line', {
      x1: xx, x2: xx + 14, y1: yy - 3, y2: yy - 3, stroke: o.color, 'stroke-width': 2,
      ...(o.dashed ? { 'stroke-dasharray': '4 3' } : {}),
    }));
    const text = svgEl('text', { x: xx + 18, y: yy, class: 'ms-legend-label', 'font-size': 10, fill: CHROM_DEFAULTS.text });
    text.textContent = o.label;
    g.appendChild(text);
  });
  svg.appendChild(g);
}
