// reference_sequence renderer — per-base letters at deep zoom; colored
// blocks at moderate zoom; horizontal line when fully decimated.

import { Table } from 'apache-arrow';
import { svgEl, clear } from '../engine/svg_layer';
import { TrackMode } from '../engine/arrow_client';
import { TrackRenderer, RenderContext } from './base';

const BASE_COLOR: Record<string, string> = {
  A: '#5cd66e',
  C: '#5e9cd6',
  G: '#d6c95e',
  T: '#d65e5e',
  U: '#d65e5e',
  N: '#666',
};

const renderer: TrackRenderer = {
  kind: 'reference_sequence',
  render(table: Table, _mode: TrackMode, ctx: RenderContext): void {
    clear(ctx.svg);
    if (table.numRows === 0) {
      // Empty payload = no sequence (e.g. unknown contig)
      const text = svgEl('text', {
        x: ctx.widthPx / 2,
        y: ctx.heightPx / 2,
        'font-size': '11',
        fill: '#5a5a63',
        'text-anchor': 'middle',
      });
      text.textContent = 'reference unavailable';
      ctx.svg.appendChild(text);
      return;
    }

    const positionCol = table.getChild('position');
    const baseCol = table.getChild('base');
    const stepCol = table.getChild('step');
    if (!positionCol || !baseCol || !stepCol) return;

    const firstStep = Number(stepCol.get(0));
    if (firstStep === 1 && pixelsPerBp(ctx) > 6) {
      // Per-base letters
      for (let i = 0; i < table.numRows; i++) {
        const pos = Number(positionCol.get(i));
        const base = String(baseCol.get(i)).toUpperCase();
        const x = ctx.xScale(pos + 0.5);
        const fill = BASE_COLOR[base] ?? BASE_COLOR.N;
        const text = svgEl('text', {
          x,
          y: ctx.heightPx / 2 + 4,
          'font-family': 'ui-monospace, SF Mono, Menlo, monospace',
          'font-size': '11',
          'text-anchor': 'middle',
          fill,
        });
        text.textContent = base;
        ctx.svg.appendChild(text);
      }
    } else if (firstStep === 1) {
      // Per-base, but pixels-per-bp too small for letters → tiny color blocks
      for (let i = 0; i < table.numRows; i++) {
        const pos = Number(positionCol.get(i));
        const base = String(baseCol.get(i)).toUpperCase();
        const x = ctx.xScale(pos);
        const w = Math.max(1, ctx.xScale(pos + 1) - x);
        const rect = svgEl('rect', {
          x,
          y: 4,
          width: w,
          height: ctx.heightPx - 8,
          fill: BASE_COLOR[base] ?? BASE_COLOR.N,
        });
        ctx.svg.appendChild(rect);
      }
    } else {
      // Decimated — single horizontal line with periodic ticks.
      const line = svgEl('line', {
        x1: 0,
        x2: ctx.widthPx,
        y1: ctx.heightPx / 2,
        y2: ctx.heightPx / 2,
        stroke: '#666',
        'stroke-width': 1,
      });
      ctx.svg.appendChild(line);
    }
  },
};

function pixelsPerBp(ctx: RenderContext): number {
  const [s, e] = ctx.xScale.domain();
  return ctx.widthPx / Math.max(1, e - s);
}

export default renderer;
