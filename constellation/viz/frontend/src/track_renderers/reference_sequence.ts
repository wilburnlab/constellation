// reference_sequence renderer — per-base letters at deep zoom; colored
// blocks at moderate zoom; horizontal line when fully decimated.

import { Table } from 'apache-arrow';
import { svgEl, clear } from '../engine/svg_layer';
import { TrackMode } from '../engine/arrow_client';
import { TrackRenderer, RenderContext } from './base';
import { pickNumber, pickPaletteColor, pickString } from './style';

const BASE_COLOR_DEFAULTS: Record<string, string> = {
  A: '#5cd66e',
  C: '#5e9cd6',
  G: '#d6c95e',
  T: '#d65e5e',
  U: '#d65e5e',
  N: '#666',
};

const DEFAULT_LETTER_FONT = 'ui-monospace, SF Mono, Menlo, monospace';

const renderer: TrackRenderer = {
  kind: 'reference_sequence',
  render(table: Table, _mode: TrackMode, ctx: RenderContext): void {
    clear(ctx.svg);
    const fontFamily = pickString(
      ctx.style,
      'letter_font_family',
      DEFAULT_LETTER_FONT,
    );
    const fontSize = pickNumber(ctx.style, 'letter_font_size_px', 11);
    const letterThreshold = pickNumber(
      ctx.style,
      'letter_threshold_px_per_bp',
      6,
    );

    if (table.numRows === 0) {
      const text = svgEl('text', {
        x: ctx.widthPx / 2,
        y: ctx.heightPx / 2,
        'font-size': '11',
        fill: '#5a5a63',
        'text-anchor': 'middle',
      });
      text.textContent = 'reference unavailable';
      ctx.svg.appendChild(text);
      ctx.svg.dataset.naturalHeight = String(ctx.heightPx);
      return;
    }

    const positionCol = table.getChild('position');
    const baseCol = table.getChild('base');
    const stepCol = table.getChild('step');
    if (!positionCol || !baseCol || !stepCol) {
      ctx.svg.dataset.naturalHeight = String(ctx.heightPx);
      return;
    }

    const colorFor = (base: string): string =>
      pickPaletteColor(ctx.style, base, BASE_COLOR_DEFAULTS[base] ?? BASE_COLOR_DEFAULTS.N);

    const firstStep = Number(stepCol.get(0));
    if (firstStep === 1 && pixelsPerBp(ctx) > letterThreshold) {
      // Per-base letters
      for (let i = 0; i < table.numRows; i++) {
        const pos = Number(positionCol.get(i));
        const base = String(baseCol.get(i)).toUpperCase();
        const x = ctx.xScale(pos + 0.5);
        const text = svgEl('text', {
          x,
          y: ctx.heightPx / 2 + 4,
          'font-family': fontFamily,
          'font-size': String(fontSize),
          'text-anchor': 'middle',
          fill: colorFor(base),
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
          fill: colorFor(base),
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
    ctx.svg.dataset.naturalHeight = String(ctx.heightPx);
  },
};

function pixelsPerBp(ctx: RenderContext): number {
  const [s, e] = ctx.xScale.domain();
  return ctx.widthPx / Math.max(1, e - s);
}

export default renderer;
