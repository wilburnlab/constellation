// coverage_histogram renderer — one <path> per sample.
//
// Wire schema (from constellation/viz/tracks/coverage_histogram.py):
//   start: int64, end: int64, depth: float64, sample_id: int64
//
// We draw a step path per `sample_id`. With one sample only (the
// `-1` unstratified case in the COVERAGE_TABLE pipeline default), the
// track shows a single area below the depth curve.

import { Table } from 'apache-arrow';
import { svgEl, clear } from '../engine/svg_layer';
import { TrackMode } from '../engine/arrow_client';
import { TrackRenderer, RenderContext } from './base';

const PALETTE = ['#4f9efb', '#fb7c4f', '#a4d65e', '#d65eb6', '#5ed6cf'];

const renderer: TrackRenderer = {
  kind: 'coverage_histogram',
  render(table: Table, _mode: TrackMode, ctx: RenderContext): void {
    clear(ctx.svg);

    const startCol = table.getChild('start');
    const endCol = table.getChild('end');
    const depthCol = table.getChild('depth');
    const sampleCol = table.getChild('sample_id');
    if (!startCol || !endCol || !depthCol || !sampleCol) return;

    const n = table.numRows;
    if (n === 0) {
      drawEmpty(ctx, 'no coverage in window');
      return;
    }

    // Bucket rows by sample_id; depth values within each sample stay
    // RLE-encoded so we draw them as horizontal segments.
    const bySample = new Map<number, Array<{ s: number; e: number; d: number }>>();
    let maxDepth = 0;
    for (let i = 0; i < n; i++) {
      const sample = Number(sampleCol.get(i));
      const s = Number(startCol.get(i));
      const e = Number(endCol.get(i));
      const d = Number(depthCol.get(i));
      maxDepth = Math.max(maxDepth, d);
      let arr = bySample.get(sample);
      if (!arr) {
        arr = [];
        bySample.set(sample, arr);
      }
      arr.push({ s, e, d });
    }

    const headerH = 14;
    const drawableH = Math.max(10, ctx.heightPx - headerH - 4);
    const yScale = (depth: number) =>
      headerH +
      drawableH -
      (maxDepth > 0 ? (depth / maxDepth) * drawableH : 0);

    let paletteIdx = 0;
    for (const [sample, intervals] of bySample) {
      const color = PALETTE[paletteIdx++ % PALETTE.length];
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
        'fill-opacity': '0.4',
        stroke: color,
        'stroke-width': '1',
      });
      ctx.svg.appendChild(path);
      // Sample label
      const label = svgEl('text', {
        x: 4,
        y: 10 + 12 * (paletteIdx - 1),
        'font-size': '10',
        fill: color,
      });
      label.textContent = sample === -1 ? `all` : `sample ${sample}`;
      ctx.svg.appendChild(label);
    }

    // Max-depth annotation
    const maxLabel = svgEl('text', {
      x: ctx.widthPx - 4,
      y: 10,
      'font-size': '10',
      'text-anchor': 'end',
      fill: '#8a8a93',
    });
    maxLabel.textContent = `max depth ${maxDepth.toFixed(0)}`;
    ctx.svg.appendChild(maxLabel);
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
