// read_pileup renderer — vector mode draws per-glyph rectangles;
// hybrid mode paints the server-rendered datashader PNG.

import { Table } from 'apache-arrow';
import { svgEl, clear } from '../engine/svg_layer';
import { TrackMode } from '../engine/arrow_client';
import { decodeHybrid, appendHybridImage } from '../engine/hybrid_layer';
import { TrackRenderer, RenderContext } from './base';

const STRAND_COLOR: Record<string, string> = {
  '+': '#5e9cd6',
  '-': '#d6755e',
  default: '#888',
};

const renderer: TrackRenderer = {
  kind: 'read_pileup',
  render(table: Table, mode: TrackMode, ctx: RenderContext): void {
    clear(ctx.svg);

    if (mode === 'hybrid') {
      const frame = decodeHybrid(table);
      if (!frame) return;
      appendHybridImage(ctx.svg, frame);
      const label = svgEl('text', {
        x: ctx.widthPx - 4,
        y: 12,
        'font-size': '10',
        'text-anchor': 'end',
        fill: '#8a8a93',
      });
      label.textContent = `hybrid · ${frame.nItems.toLocaleString()} reads`;
      ctx.svg.appendChild(label);
      return;
    }

    if (table.numRows === 0) {
      const text = svgEl('text', {
        x: ctx.widthPx / 2,
        y: ctx.heightPx / 2,
        'font-size': '11',
        fill: '#5a5a63',
        'text-anchor': 'middle',
      });
      text.textContent = 'no reads in window';
      ctx.svg.appendChild(text);
      return;
    }

    const startCol = table.getChild('ref_start');
    const endCol = table.getChild('ref_end');
    const strandCol = table.getChild('strand');
    const rowCol = table.getChild('row');
    const idCol = table.getChild('alignment_id');
    if (!startCol || !endCol || !strandCol || !rowCol) return;

    let maxRow = 0;
    for (let i = 0; i < table.numRows; i++) {
      const r = Number(rowCol.get(i));
      if (r > maxRow) maxRow = r;
    }
    const stackH = maxRow + 1;
    const rowH = Math.max(2, Math.min(8, (ctx.heightPx - 4) / stackH));

    for (let i = 0; i < table.numRows; i++) {
      const start = Number(startCol.get(i));
      const end = Number(endCol.get(i));
      const strand = String(strandCol.get(i));
      const row = Number(rowCol.get(i));
      const x0 = ctx.xScale(start);
      const x1 = ctx.xScale(end);
      const rect = svgEl('rect', {
        x: x0,
        y: 4 + row * rowH,
        width: Math.max(1, x1 - x0),
        height: Math.max(1, rowH - 1),
        fill: STRAND_COLOR[strand] ?? STRAND_COLOR.default,
      });
      if (idCol) {
        rect.setAttribute('data-alignment-id', String(idCol.get(i)));
      }
      ctx.svg.appendChild(rect);
    }

    const label = svgEl('text', {
      x: ctx.widthPx - 4,
      y: 12,
      'font-size': '10',
      'text-anchor': 'end',
      fill: '#8a8a93',
    });
    label.textContent = `vector · ${table.numRows.toLocaleString()} reads`;
    ctx.svg.appendChild(label);
  },
};

export default renderer;
