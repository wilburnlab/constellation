// ms_chromatogram — intensity over retention time for one MS level.
//
// Draws the trace the kernel sends (TIC, base peak, or one stick per scan
// for a DDA level's scan events) over the shared retention-time window,
// with the row's selected scan marked. The y axis fits what is in view.

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
import { pickNumber, pickPaletteColor } from '../../../panels/style';

/** The same for every chromatogram, so their plot areas line up. */
export const CHROM_MARGIN: Margins = { top: 8, right: 12, bottom: 32, left: 58 };

export const CHROM_DEFAULTS = {
  line: '#a277ff',
  marker: '#61ffca',
  lineWidth: 1.25,
};

export type IntensityMode = 'abs' | 'rel' | 'log';

export interface ChromatogramOptions {
  rt: Interval;
  intensity: IntensityMode;
  /** Retention time of the row's selected scan. */
  selectedRt: number | null;
  /** Draw sticks rather than a line (scan events). */
  sticks: boolean;
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
    ],
  },
};

/** Draw the trace; returns how many points were in view. */
export function drawChromatogram(
  svg: SVGSVGElement,
  size: { widthPx: number; heightPx: number },
  table: Table,
  opts: ChromatogramOptions,
): number {
  const rts = table.getChild('rt')?.toArray() as Float64Array | undefined;
  const raw = table.getChild('intensity');
  const n = table.numRows;
  const values = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    const v = raw?.get(i);
    values[i] = v === null || v === undefined ? 0 : Number(v);
  }
  const xs = rts ?? new Float64Array(0);

  const frame = plotFrame(size, CHROM_MARGIN);
  const [, max] = autoYDomain(values, xs, opts.rt, 1);
  const rel = opts.intensity === 'rel';
  const scaleBy = rel && max > 0 ? 100 / max : 1;
  const top = rel ? 108 : max > 0 ? max * 1.08 : 1;
  const x = xAxis(frame, opts.rt, AXIS_FORMATS.rtMinutes);
  const y = yAxis(frame, [0, top], rel ? AXIS_FORMATS.percent : AXIS_FORMATS.intensity, {
    log: opts.intensity === 'log',
  });
  const plot = drawAxes(svg, frame, { x, y, xLabel: AXIS_FORMATS.rtMinutes.label }, opts.uid);

  const color = pickPaletteColor(opts.style, 'line', CHROM_DEFAULTS.line);
  const width = pickNumber(opts.style, 'line_width', CHROM_DEFAULTS.lineWidth);
  // Points in view, plus one on each side so the line reaches the edges.
  let first = 0;
  while (first < n && xs[first] < opts.rt[0]) first++;
  let last = n - 1;
  while (last >= 0 && xs[last] > opts.rt[1]) last--;
  const lo = Math.max(0, first - 1);
  const hi = Math.min(n - 1, last + 1);
  const shown = Math.max(0, last - first + 1);

  if (opts.sticks) {
    const base = y.scale(0);
    let d = '';
    for (let i = lo; i <= hi; i++) {
      const px = x.scale(xs[i]).toFixed(1);
      d += `M${px} ${base.toFixed(1)}V${y.scale(values[i] * scaleBy).toFixed(1)}`;
    }
    if (d) plot.appendChild(svgEl('path', { d, stroke: color, 'stroke-width': width, fill: 'none', class: 'ms-trace' }));
  } else if (hi >= lo) {
    let d = '';
    for (let i = lo; i <= hi; i++) {
      d += `${i === lo ? 'M' : 'L'}${x.scale(xs[i]).toFixed(1)} ${y.scale(values[i] * scaleBy).toFixed(1)}`;
    }
    plot.appendChild(
      svgEl('path', {
        d,
        stroke: color,
        'stroke-width': width,
        fill: 'none',
        'stroke-linejoin': 'round',
        class: 'ms-trace',
      }),
    );
  }

  if (opts.selectedRt !== null && opts.selectedRt >= opts.rt[0] && opts.selectedRt <= opts.rt[1]) {
    const px = x.scale(opts.selectedRt);
    plot.appendChild(
      svgEl('line', {
        x1: px,
        x2: px,
        y1: frame.inner.y,
        y2: frame.inner.y + frame.inner.h,
        stroke: pickPaletteColor(opts.style, 'marker', CHROM_DEFAULTS.marker),
        'stroke-width': 1,
        class: 'ms-selected-scan',
      }),
    );
  }
  svg.dataset.naturalHeight = String(size.heightPx);
  return shown;
}
