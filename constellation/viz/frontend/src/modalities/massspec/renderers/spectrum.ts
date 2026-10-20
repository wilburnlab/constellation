// ms_spectrum — the peaks of one scan as sticks.
//
// The most intense peaks in view are labelled with their m/z, skipping
// any label that would overlap one already placed. On an MS1 spectrum the
// isolation window of the MS2 scan being shown is shaded and its target
// marked, so it is clear which precursors that scan fragmented.

import type { Table } from 'apache-arrow';
import {
  AXIS_FORMATS,
  type Margins,
  drawAxes,
  plotFrame,
  xAxis,
  yAxis,
} from '../../../engine/axis';
import type { Interval } from '../../../engine/interval';
import { svgEl } from '../../../engine/svg_layer';
import type { PanelKind } from '../../../panels/kind';
import { pickNumber, pickPaletteColor } from '../../../panels/style';
import type { IntensityMode } from './chromatogram';

export const SPECTRUM_MARGIN: Margins = { top: 14, right: 12, bottom: 32, left: 58 };

export const SPECTRUM_DEFAULTS = {
  stick: '#adacae',
  label: '#edecee',
  isolation: '#ffca85',
  labels: 8,
};

export interface SpectrumOptions {
  /** m/z window; the data's own extent when null. */
  mz: Interval | null;
  intensity: IntensityMode;
  /** Isolation window to shade (an MS1 spectrum showing what an MS2 scan isolated). */
  isolation: Interval | null;
  /** Precursor m/z to mark. */
  target: number | null;
  style: Record<string, unknown>;
  uid: string;
}

export const SPECTRUM_KIND: PanelKind = {
  kind: 'ms_spectrum',
  order: 1,
  unit: ['peak', 'peaks'],
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
            label: 'Peaks labelled',
            default: SPECTRUM_DEFAULTS.labels,
            min: 0,
            max: 50,
            step: 1,
          },
        ],
      },
    ],
  },
};

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

/** Indices of the peaks to label: most intense first, skipping any whose
 *  label would overlap one already placed (by pixel extent). */
export function chooseLabels(
  xs: ArrayLike<number>,
  heights: ArrayLike<number>,
  limit: number,
  widthOf: (i: number) => number,
): number[] {
  const order = Array.from({ length: xs.length }, (_, i) => i).sort((a, b) => heights[b] - heights[a]);
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

/** Draw the spectrum; returns how many peaks were in view. */
export function drawSpectrum(
  svg: SVGSVGElement,
  size: { widthPx: number; heightPx: number },
  table: Table,
  opts: SpectrumOptions,
): number {
  const mzs = (table.getChild('mz')?.toArray() as Float64Array | undefined) ?? new Float64Array(0);
  const raw = (table.getChild('intensity')?.toArray() as Float64Array | undefined) ?? new Float64Array(0);
  const domain = opts.mz ?? spectrumExtent(mzs);
  const frame = plotFrame(size, SPECTRUM_MARGIN);

  const inView: number[] = [];
  let max = 0;
  for (let i = 0; i < mzs.length; i++) {
    if (mzs[i] < domain[0] || mzs[i] > domain[1]) continue;
    inView.push(i);
    if (raw[i] > max) max = raw[i];
  }
  const rel = opts.intensity === 'rel';
  const scaleBy = rel && max > 0 ? 100 / max : 1;
  const top = rel ? 108 : max > 0 ? max * 1.12 : 1;
  const x = xAxis(frame, domain, AXIS_FORMATS.mz);
  const y = yAxis(frame, [0, top], rel ? AXIS_FORMATS.percent : AXIS_FORMATS.intensity, {
    log: opts.intensity === 'log',
  });
  const plot = drawAxes(svg, frame, { x, y, xLabel: AXIS_FORMATS.mz.label }, opts.uid);

  if (opts.isolation) {
    const a = x.scale(Math.max(domain[0], opts.isolation[0]));
    const b = x.scale(Math.min(domain[1], opts.isolation[1]));
    if (b > a) {
      plot.appendChild(
        svgEl('rect', {
          x: a,
          y: frame.inner.y,
          width: b - a,
          height: frame.inner.h,
          fill: pickPaletteColor(opts.style, 'isolation', SPECTRUM_DEFAULTS.isolation),
          'fill-opacity': 0.15,
          class: 'ms-isolation',
        }),
      );
    }
  }
  if (opts.target !== null && opts.target >= domain[0] && opts.target <= domain[1]) {
    const px = x.scale(opts.target);
    plot.appendChild(
      svgEl('line', {
        x1: px,
        x2: px,
        y1: frame.inner.y,
        y2: frame.inner.y + frame.inner.h,
        stroke: pickPaletteColor(opts.style, 'isolation', SPECTRUM_DEFAULTS.isolation),
        'stroke-dasharray': '3 3',
        class: 'ms-target',
      }),
    );
  }

  const base = y.scale(0);
  let d = '';
  const px: number[] = [];
  const py: number[] = [];
  for (const i of inView) {
    const sx = x.scale(mzs[i]);
    const sy = y.scale(raw[i] * scaleBy);
    px.push(sx);
    py.push(raw[i]);
    d += `M${sx.toFixed(1)} ${base.toFixed(1)}V${sy.toFixed(1)}`;
  }
  if (d) {
    plot.appendChild(
      svgEl('path', {
        d,
        stroke: pickPaletteColor(opts.style, 'stick', SPECTRUM_DEFAULTS.stick),
        'stroke-width': 1,
        fill: 'none',
        class: 'ms-peaks',
      }),
    );
  }

  const limit = pickNumber(opts.style, 'labels', SPECTRUM_DEFAULTS.labels);
  const decimals = Math.max(2, Math.min(4, Math.ceil(-Math.log10(Math.max(1e-6, (domain[1] - domain[0]) / 2000)))));
  const text = (k: number): string => mzs[inView[k]].toFixed(decimals);
  const labelColor = pickPaletteColor(opts.style, 'label', SPECTRUM_DEFAULTS.label);
  for (const k of chooseLabels(px, py, limit, (k) => text(k).length * 5.6)) {
    const label = svgEl('text', {
      x: px[k],
      y: Math.max(frame.inner.y + 9, y.scale(raw[inView[k]] * scaleBy) - 3),
      'text-anchor': 'middle',
      fill: labelColor,
      'font-size': 9,
      class: 'ms-peak-label',
    });
    label.textContent = text(k);
    plot.appendChild(label);
  }
  svg.dataset.naturalHeight = String(size.heightPx);
  return inView.length;
}
