// coverage_histogram renderer — one <path> per sample.
//
// Wire schema (from constellation/viz/tracks/coverage_histogram.py):
//   start: int64, end: int64, depth: float64, sample_id: int64
//
// We draw a step path per `sample_id`. With one sample only (the
// `-1` unstratified case in the COVERAGE_TABLE pipeline default), the
// track shows a single area below the depth curve.

import { Table } from 'apache-arrow';
import { svgEl, clear } from '../../../engine/svg_layer';
import { TrackMode } from '../../../engine/arrow_client';
import { TrackRenderer, RenderContext } from './base';
import {
  pickAllowList,
  pickBool,
  pickCycledColor,
  pickNumber,
  pickString,
} from '../../../panels/style';
import { SettingsSchema } from '../../../panels/settings_schema';
import {
  SAMPLE_PALETTE_CYCLE,
  generalSection,
  num,
  opacity,
  sampleOptions,
  samplePalette,
  toggle,
} from './settings_common';

/** Style and filter defaults, shared by the drawing code and the
 *  settings popover. */
const DEFAULTS = {
  fill_opacity: 0.4,
  stroke_width_px: 1,
  y_scale: 'linear',
  show_sample_labels: true,
  show_max_depth: true,
  min_depth: 0,
} as const;

const NO_SAMPLES = 'no samples in window yet';

const SETTINGS: SettingsSchema = {
  sections: [
    generalSection(0.4),
    {
      title: 'Style',
      controls: [
        // An unstratified coverage table carries one pseudo-sample, -1.
        { type: 'palette', entries: samplePalette({ unstratified: 'all (unstratified)' }), emptyHint: NO_SAMPLES },
        opacity('fill_opacity', 'Fill opacity', DEFAULTS.fill_opacity),
        num('style', 'stroke_width_px', 'Stroke width (px)', DEFAULTS.stroke_width_px, 0, 6, 0.5),
        {
          type: 'select',
          target: 'style',
          key: 'y_scale',
          label: 'Y scale',
          default: DEFAULTS.y_scale,
          options: [
            { value: 'linear', label: 'linear' },
            { value: 'log', label: 'log' },
          ],
        },
        toggle('style', 'show_sample_labels', 'Show sample labels', DEFAULTS.show_sample_labels),
        toggle('style', 'show_max_depth', 'Show max depth', DEFAULTS.show_max_depth),
      ],
    },
    {
      title: 'Filter',
      controls: [
        {
          type: 'allowlist',
          target: 'filter',
          key: 'visible_samples',
          label: 'Visible samples',
          options: sampleOptions({ unstratified: 'all' }),
          emptyHint: NO_SAMPLES,
        },
        num('filter', 'min_depth', 'Min depth', DEFAULTS.min_depth, 0, 1_000_000, 1),
      ],
    },
  ],
};

const renderer: TrackRenderer = {
  kind: 'coverage_histogram',
  order: 2,
  unit: ['bin', 'bins'],
  settings: SETTINGS,
  render(table: Table, _mode: TrackMode, ctx: RenderContext): void {
    clear(ctx.svg);

    const fillOpacity = pickNumber(ctx.style, 'fill_opacity', DEFAULTS.fill_opacity);
    const strokeWidth = pickNumber(ctx.style, 'stroke_width_px', DEFAULTS.stroke_width_px);
    const yScaleMode = pickString(ctx.style, 'y_scale', DEFAULTS.y_scale);
    const showSampleLabels = pickBool(ctx.style, 'show_sample_labels', DEFAULTS.show_sample_labels);
    const showMaxDepth = pickBool(ctx.style, 'show_max_depth', DEFAULTS.show_max_depth);

    const allowedSamples = pickAllowList(ctx.filter, 'visible_samples');
    const minDepth = pickNumber(ctx.filter, 'min_depth', DEFAULTS.min_depth);

    const startCol = table.getChild('start');
    const endCol = table.getChild('end');
    const depthCol = table.getChild('depth');
    const sampleCol = table.getChild('sample_id');
    if (!startCol || !endCol || !depthCol || !sampleCol) {
      ctx.svg.dataset.naturalHeight = String(ctx.heightPx);
      return;
    }

    const n = table.numRows;
    if (n === 0) {
      drawEmpty(ctx, 'no coverage in window');
      ctx.svg.dataset.naturalHeight = String(ctx.heightPx);
      return;
    }

    // Bucket rows by sample_id; depth values within each sample stay
    // RLE-encoded so we draw them as horizontal segments.
    const bySample = new Map<number, Array<{ s: number; e: number; d: number }>>();
    let maxDepth = 0;
    for (let i = 0; i < n; i++) {
      const sample = Number(sampleCol.get(i));
      if (allowedSamples && !allowedSamples.has(String(sample))) continue;
      const d = Number(depthCol.get(i));
      if (d < minDepth) continue;
      const s = Number(startCol.get(i));
      const e = Number(endCol.get(i));
      maxDepth = Math.max(maxDepth, d);
      let arr = bySample.get(sample);
      if (!arr) {
        arr = [];
        bySample.set(sample, arr);
      }
      arr.push({ s, e, d });
    }

    if (bySample.size === 0) {
      drawEmpty(ctx, 'no coverage in window');
      ctx.svg.dataset.naturalHeight = String(ctx.heightPx);
      return;
    }

    const headerH = 14;
    const drawableH = Math.max(10, ctx.heightPx - headerH - 4);
    const useLog = yScaleMode === 'log';
    const logMax = useLog ? Math.log10(1 + maxDepth) : 0;
    const yScale = (depth: number): number => {
      if (maxDepth <= 0) return headerH + drawableH;
      const fraction = useLog
        ? Math.log10(1 + Math.max(0, depth)) / Math.max(logMax, 1e-9)
        : depth / maxDepth;
      return headerH + drawableH - fraction * drawableH;
    };

    let paletteIdx = 0;
    for (const [sample, intervals] of bySample) {
      const color = pickCycledColor(ctx.style, sample, SAMPLE_PALETTE_CYCLE, paletteIdx);
      paletteIdx++;
      intervals.sort((a, b) => a.s - b.s);
      // Build a stepped <path>: one move + alternating H/V segments.
      const baseY = headerH + drawableH;
      const cmds: string[] = [];
      let cursorX = ctx.xScale(intervals[0].s);
      cmds.push(`M ${cursorX.toFixed(2)} ${baseY.toFixed(2)}`);
      for (const { s, e, d } of intervals) {
        const x0 = ctx.xScale(s);
        const x1 = ctx.xScale(e);
        const y = yScale(d);
        cmds.push(`L ${x0.toFixed(2)} ${y.toFixed(2)}`);
        cmds.push(`L ${x1.toFixed(2)} ${y.toFixed(2)}`);
        cursorX = x1;
      }
      cmds.push(`L ${cursorX.toFixed(2)} ${baseY.toFixed(2)} Z`);
      const path = svgEl('path', {
        d: cmds.join(' '),
        fill: color,
        'fill-opacity': String(fillOpacity),
        stroke: color,
        'stroke-width': String(strokeWidth),
      });
      ctx.svg.appendChild(path);
      // Sample label
      if (showSampleLabels) {
        const label = svgEl('text', {
          x: 4,
          y: 10 + 12 * (paletteIdx - 1),
          'font-size': '10',
          fill: color,
        });
        label.textContent = sample === -1 ? `all` : `sample ${sample}`;
        ctx.svg.appendChild(label);
      }
    }

    // Max-depth annotation
    if (showMaxDepth) {
      const maxLabel = svgEl('text', {
        x: ctx.widthPx - 4,
        y: 10,
        'font-size': '10',
        'text-anchor': 'end',
        fill: '#8a8a93',
      });
      maxLabel.textContent = `max depth ${maxDepth.toFixed(0)}${useLog ? ' (log)' : ''}`;
      ctx.svg.appendChild(maxLabel);
    }
    ctx.svg.dataset.naturalHeight = String(ctx.heightPx);
  },
};

function drawEmpty(ctx: RenderContext, message: string): void {
  const text = svgEl('text', {
    x: ctx.widthPx / 2,
    y: ctx.heightPx / 2,
    'font-size': '11',
    fill: '#5a5a63',
    'text-anchor': 'middle',
  });
  text.textContent = message;
  ctx.svg.appendChild(text);
}

export default renderer;
