// read_pileup renderer — vector mode draws per-glyph rectangles;
// hybrid mode paints the server-rendered datashader PNG.

import { Table } from 'apache-arrow';
import { svgEl, clear } from '../engine/svg_layer';
import { TrackMode } from '../engine/arrow_client';
import { decodeHybrid, appendHybridImage } from '../engine/hybrid_layer';
import { TrackRenderer, RenderContext } from './base';
import {
  pickAllowList,
  pickNumber,
  pickPaletteColor,
} from './style';

const STRAND_COLOR_DEFAULTS: Record<string, string> = {
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
      if (!frame) {
        ctx.svg.dataset.naturalHeight = String(ctx.heightPx);
        return;
      }
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
      ctx.svg.dataset.naturalHeight = String(ctx.heightPx);
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
      ctx.svg.dataset.naturalHeight = String(ctx.heightPx);
      return;
    }

    const startCol = table.getChild('ref_start');
    const endCol = table.getChild('ref_end');
    const strandCol = table.getChild('strand');
    const rowCol = table.getChild('row');
    const idCol = table.getChild('alignment_id');
    if (!startCol || !endCol || !strandCol || !rowCol) {
      ctx.svg.dataset.naturalHeight = String(ctx.heightPx);
      return;
    }

    const minRowH = pickNumber(ctx.style, 'min_row_height_px', 2);
    const maxRowH = pickNumber(ctx.style, 'max_row_height_px', 8);
    const readOpacity = pickNumber(ctx.style, 'read_opacity', 1.0);
    const allowedStrands = pickAllowList(ctx.filter, 'visible_strands');

    const admit: boolean[] = new Array(table.numRows);
    let maxRow = -1;
    let admittedRows = 0;
    for (let i = 0; i < table.numRows; i++) {
      const strand = String(strandCol.get(i));
      if (allowedStrands && !allowedStrands.has(strand)) {
        admit[i] = false;
        continue;
      }
      admit[i] = true;
      admittedRows++;
      const r = Number(rowCol.get(i));
      if (r > maxRow) maxRow = r;
    }

    if (admittedRows === 0) {
      const text = svgEl('text', {
        x: ctx.widthPx / 2,
        y: ctx.heightPx / 2,
        'font-size': '11',
        fill: '#5a5a63',
        'text-anchor': 'middle',
      });
      text.textContent = 'no reads in window';
      ctx.svg.appendChild(text);
      ctx.svg.dataset.naturalHeight = String(ctx.heightPx);
      return;
    }

    const stackH = maxRow + 1;
    const rowH = Math.max(
      minRowH,
      Math.min(maxRowH, (ctx.heightPx - 4) / Math.max(1, stackH)),
    );

    for (let i = 0; i < table.numRows; i++) {
      if (!admit[i]) continue;
      const start = Number(startCol.get(i));
      const end = Number(endCol.get(i));
      const strand = String(strandCol.get(i));
      const row = Number(rowCol.get(i));
      const x0 = ctx.xScale(start);
      const x1 = ctx.xScale(end);
      const fill = pickPaletteColor(
        ctx.style,
        strand,
        STRAND_COLOR_DEFAULTS[strand] ?? STRAND_COLOR_DEFAULTS.default,
      );
      const rect = svgEl('rect', {
        x: x0,
        y: 4 + row * rowH,
        width: Math.max(1, x1 - x0),
        height: Math.max(1, rowH - 1),
        fill,
        opacity: String(readOpacity),
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
    label.textContent = `vector · ${admittedRows.toLocaleString()} reads`;
    ctx.svg.appendChild(label);

    const naturalHeight = stackH > 0 ? 4 + stackH * rowH : ctx.heightPx;
    ctx.svg.dataset.naturalHeight = String(naturalHeight);
  },
};

export default renderer;
