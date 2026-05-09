// cluster_pileup renderer — same shape as read_pileup but glyphs are
// transcript clusters (one row per cluster, n_reads in label).

import { Table } from 'apache-arrow';
import { svgEl, clear } from '../engine/svg_layer';
import { TrackMode } from '../engine/arrow_client';
import { decodeHybrid, appendHybridImage } from '../engine/hybrid_layer';
import { TrackRenderer, RenderContext } from './base';

const MODE_COLOR: Record<string, string> = {
  'genome-guided': '#5ed6cf',
  'de-novo': '#a8d65e',
  default: '#888',
};

const renderer: TrackRenderer = {
  kind: 'cluster_pileup',
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
      label.textContent = `hybrid · ${frame.nItems.toLocaleString()} clusters`;
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
      text.textContent = 'no clusters in window';
      ctx.svg.appendChild(text);
      return;
    }

    const startCol = table.getChild('span_start');
    const endCol = table.getChild('span_end');
    const rowCol = table.getChild('row');
    const modeCol = table.getChild('mode');
    const nReadsCol = table.getChild('n_reads');
    const idCol = table.getChild('cluster_id');
    if (!startCol || !endCol || !rowCol || !modeCol) return;

    let maxRow = 0;
    let maxN = 1;
    for (let i = 0; i < table.numRows; i++) {
      const r = Number(rowCol.get(i));
      if (r > maxRow) maxRow = r;
      if (nReadsCol) {
        const n = Number(nReadsCol.get(i));
        if (n > maxN) maxN = n;
      }
    }
    const stackH = maxRow + 1;
    const rowH = Math.max(4, Math.min(10, (ctx.heightPx - 4) / stackH));

    for (let i = 0; i < table.numRows; i++) {
      const start = Number(startCol.get(i));
      const end = Number(endCol.get(i));
      const row = Number(rowCol.get(i));
      const m = String(modeCol.get(i));
      const n = nReadsCol ? Number(nReadsCol.get(i)) : 1;
      const x0 = ctx.xScale(start);
      const x1 = ctx.xScale(end);
      const opacity = 0.4 + 0.6 * (Math.log2(1 + n) / Math.log2(1 + maxN));
      const rect = svgEl('rect', {
        x: x0,
        y: 4 + row * rowH,
        width: Math.max(1, x1 - x0),
        height: Math.max(1, rowH - 2),
        fill: MODE_COLOR[m] ?? MODE_COLOR.default,
        opacity: opacity.toFixed(2),
      });
      if (idCol) {
        rect.setAttribute('data-cluster-id', String(idCol.get(i)));
      }
      const title = svgEl('title');
      title.textContent = `cluster (${m}) · ${n} reads`;
      rect.appendChild(title);
      ctx.svg.appendChild(rect);
    }
  },
};

export default renderer;
