// splice_junctions renderer — arc per intron cluster, stroke-width
// scaled by support, color by motif.

import { Table } from 'apache-arrow';
import { svgEl, clear } from '../engine/svg_layer';
import { TrackMode } from '../engine/arrow_client';
import { TrackRenderer, RenderContext } from './base';

const MOTIF_COLOR: Record<string, string> = {
  'GT-AG': '#5e9cd6',
  'GC-AG': '#9b5ed6',
  'AT-AC': '#d6a05e',
  default: '#888',
};

const renderer: TrackRenderer = {
  kind: 'splice_junctions',
  render(table: Table, _mode: TrackMode, ctx: RenderContext): void {
    clear(ctx.svg);
    if (table.numRows === 0) {
      const text = svgEl('text', {
        x: ctx.widthPx / 2,
        y: ctx.heightPx / 2,
        'font-size': '11',
        fill: '#5a5a63',
        'text-anchor': 'middle',
      });
      text.textContent = 'no junctions in window';
      ctx.svg.appendChild(text);
      return;
    }

    let maxSupport = 1;
    const supportCol = table.getChild('support');
    if (supportCol) {
      for (let i = 0; i < table.numRows; i++) {
        const s = Number(supportCol.get(i));
        if (s > maxSupport) maxSupport = s;
      }
    }

    const baseY = ctx.heightPx - 8;
    const arcMaxH = ctx.heightPx - 16;

    for (let i = 0; i < table.numRows; i++) {
      const row = table.get(i);
      if (!row) continue;
      const donor = Number(row.donor_pos);
      const acceptor = Number(row.acceptor_pos);
      const support = Number(row.support);
      const motif = String(row.motif ?? '');
      const color = MOTIF_COLOR[motif] ?? MOTIF_COLOR.default;

      const x0 = ctx.xScale(Math.min(donor, acceptor));
      const x1 = ctx.xScale(Math.max(donor, acceptor));
      const arcH = Math.max(8, (support / maxSupport) * arcMaxH);
      const cx = (x0 + x1) / 2;
      const cy = baseY - arcH;
      const path = svgEl('path', {
        d: `M ${x0} ${baseY} Q ${cx} ${cy} ${x1} ${baseY}`,
        stroke: color,
        'stroke-width': Math.max(1, Math.min(4, Math.log2(1 + support))),
        fill: 'none',
        opacity: '0.85',
      });
      const title = svgEl('title');
      title.textContent = `${motif || '?'}  support=${support}`;
      path.appendChild(title);
      ctx.svg.appendChild(path);
    }
  },
};

export default renderer;
