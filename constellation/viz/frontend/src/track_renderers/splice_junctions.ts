// splice_junctions renderer — arc per intron cluster, stroke-width
// scaled by support, color by motif.

import { Table } from 'apache-arrow';
import { svgEl, clear } from '../engine/svg_layer';
import { TrackMode } from '../engine/arrow_client';
import { TrackRenderer, RenderContext } from './base';
import {
  pickAllowList,
  pickBool,
  pickNumber,
  pickPaletteColor,
} from './style';

const MOTIF_COLOR_DEFAULTS: Record<string, string> = {
  'GT-AG': '#5e9cd6',
  'GC-AG': '#9b5ed6',
  'AT-AC': '#d6a05e',
  default: '#888',
};

const renderer: TrackRenderer = {
  kind: 'splice_junctions',
  render(table: Table, _mode: TrackMode, ctx: RenderContext): void {
    clear(ctx.svg);

    const strokeMin = pickNumber(ctx.style, 'arc_stroke_min_px', 1);
    const strokeMax = pickNumber(ctx.style, 'arc_stroke_max_px', 4);
    const arcOpacity = pickNumber(ctx.style, 'arc_opacity', 0.85);
    const allowedMotifs = pickAllowList(ctx.filter, 'visible_motifs');
    const minSupport = pickNumber(ctx.filter, 'min_support', 1);
    const annotatedOnly = pickBool(ctx.filter, 'annotated_only', false);

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
      ctx.svg.dataset.naturalHeight = String(ctx.heightPx);
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
      const motifRaw = row.motif;
      const motif = motifRaw === null || motifRaw === undefined ? '' : String(motifRaw);
      if (allowedMotifs && motif && !allowedMotifs.has(motif)) continue;
      const support = Number(row.support);
      if (support < minSupport) continue;
      if (annotatedOnly && row.annotated !== true) continue;

      const donor = Number(row.donor_pos);
      const acceptor = Number(row.acceptor_pos);
      const color = pickPaletteColor(
        ctx.style,
        motif || 'default',
        MOTIF_COLOR_DEFAULTS[motif] ?? MOTIF_COLOR_DEFAULTS.default,
      );

      const x0 = ctx.xScale(Math.min(donor, acceptor));
      const x1 = ctx.xScale(Math.max(donor, acceptor));
      const arcH = Math.max(8, (support / maxSupport) * arcMaxH);
      const cx = (x0 + x1) / 2;
      const cy = baseY - arcH;
      const strokeBp = Math.max(
        strokeMin,
        Math.min(strokeMax, Math.log2(1 + support)),
      );
      const path = svgEl('path', {
        d: `M ${x0} ${baseY} Q ${cx} ${cy} ${x1} ${baseY}`,
        stroke: color,
        'stroke-width': strokeBp,
        fill: 'none',
        opacity: String(arcOpacity),
      });
      const title = svgEl('title');
      title.textContent = `${motif || '?'}  support=${support}`;
      path.appendChild(title);
      ctx.svg.appendChild(path);
    }
    ctx.svg.dataset.naturalHeight = String(ctx.heightPx);
  },
};

export default renderer;
