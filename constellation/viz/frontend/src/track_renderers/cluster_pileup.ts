// cluster_pileup renderer — two views:
//
//   clusters (default) — one rectangle per cluster from span_start to
//     span_end, colored by `mode` (genome-guided vs de-novo), with
//     log-scaled opacity by n_reads. Hybrid mode paints a datashader
//     PNG.
//
//   members — one rectangle per member alignment (with CIGAR-aware
//     exon blocks + intron connectors + per-base X glyphs), colored
//     by `cluster_id` so the user can see which reads got grouped
//     together. Delegates to the shared _alignment_view helper.
//
// The active view is signaled by the response's X-Track-View header
// (kernel-side mode_extra.cluster_view) AND by which columns are
// present on the wire — the renderer branches on column presence as
// a defensive check.

import { Table } from 'apache-arrow';
import { svgEl, clear } from '../engine/svg_layer';
import { TrackMode } from '../engine/arrow_client';
import { decodeHybrid, appendHybridImage } from '../engine/hybrid_layer';
import { TrackRenderer, RenderContext } from './base';
import { renderAlignmentRows } from './_alignment_view';
import {
  pickAllowList,
  pickNumber,
  pickPaletteColor,
} from './style';

const MODE_COLOR_DEFAULTS: Record<string, string> = {
  'genome-guided': '#5ed6cf',
  'de-novo': '#a8d65e',
  default: '#888',
};

// Higher-saturation cycle for member view — cluster identity is the
// primary signal, so 8 colors give us enough headroom to differentiate
// neighboring clusters without recycling at small zoom levels.
const CLUSTER_PALETTE_CYCLE = [
  '#5e9cd6',
  '#d6755e',
  '#a4d65e',
  '#d65ed6',
  '#5ed6cf',
  '#d6a05e',
  '#7c5ed6',
  '#5ed694',
];

const renderer: TrackRenderer = {
  kind: 'cluster_pileup',
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
      label.textContent = `hybrid · ${frame.nItems.toLocaleString()} clusters`;
      ctx.svg.appendChild(label);
      ctx.svg.dataset.naturalHeight = String(ctx.heightPx);
      return;
    }

    // Members view — the wire schema carries `blocks` and per-row
    // `cluster_id` keyed coloring. Delegate to the shared helper so
    // read_pileup and cluster_pileup share the visual vocabulary.
    if (table.getChild('blocks') !== null) {
      renderMembers(table, ctx);
      return;
    }

    // Clusters view (the original rectangle-per-cluster rendering).
    renderClusters(table, ctx);
  },
};

export default renderer;


function renderClusters(table: Table, ctx: RenderContext): void {
  if (table.numRows === 0) {
    emitEmpty(ctx, 'no clusters in window');
    return;
  }

  const startCol = table.getChild('span_start');
  const endCol = table.getChild('span_end');
  const rowCol = table.getChild('row');
  const modeCol = table.getChild('mode');
  const nReadsCol = table.getChild('n_reads');
  const strandCol = table.getChild('strand');
  const idCol = table.getChild('cluster_id');
  if (!startCol || !endCol || !rowCol || !modeCol) {
    ctx.svg.dataset.naturalHeight = String(ctx.heightPx);
    return;
  }

  const minRowH = pickNumber(ctx.style, 'min_row_height_px', 4);
  const maxRowH = pickNumber(ctx.style, 'max_row_height_px', 10);
  const opacityMin = pickNumber(ctx.style, 'opacity_min', 0.4);
  const opacityMax = pickNumber(ctx.style, 'opacity_max', 1.0);
  const opacityRange = Math.max(0, opacityMax - opacityMin);

  const allowedModes = pickAllowList(ctx.filter, 'visible_modes');
  const allowedStrands = pickAllowList(ctx.filter, 'visible_strands');
  const minReads = pickNumber(ctx.filter, 'min_reads', 1);

  const admit: boolean[] = new Array(table.numRows);
  let maxRow = -1;
  let maxN = 1;
  let admittedRows = 0;
  for (let i = 0; i < table.numRows; i++) {
    const m = String(modeCol.get(i));
    if (allowedModes && !allowedModes.has(m)) {
      admit[i] = false;
      continue;
    }
    const strand = strandCol ? String(strandCol.get(i)) : '';
    if (allowedStrands && strand && !allowedStrands.has(strand)) {
      admit[i] = false;
      continue;
    }
    const n = nReadsCol ? Number(nReadsCol.get(i)) : 1;
    if (n < minReads) {
      admit[i] = false;
      continue;
    }
    admit[i] = true;
    admittedRows++;
    const r = Number(rowCol.get(i));
    if (r > maxRow) maxRow = r;
    if (n > maxN) maxN = n;
  }

  if (admittedRows === 0) {
    emitEmpty(ctx, 'no clusters in window');
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
    const row = Number(rowCol.get(i));
    const m = String(modeCol.get(i));
    const n = nReadsCol ? Number(nReadsCol.get(i)) : 1;
    const x0 = ctx.xScale(start);
    const x1 = ctx.xScale(end);
    const opacity =
      opacityMin + opacityRange * (Math.log2(1 + n) / Math.log2(1 + maxN));
    const fill = pickPaletteColor(
      ctx.style,
      m,
      MODE_COLOR_DEFAULTS[m] ?? MODE_COLOR_DEFAULTS.default,
    );
    const rect = svgEl('rect', {
      x: x0,
      y: 4 + row * rowH,
      width: Math.max(1, x1 - x0),
      height: Math.max(1, rowH - 2),
      fill,
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

  const naturalHeight = stackH > 0 ? 4 + stackH * rowH : ctx.heightPx;
  ctx.svg.dataset.naturalHeight = String(naturalHeight);
}


function renderMembers(table: Table, ctx: RenderContext): void {
  const result = renderAlignmentRows(table, ctx, {
    colorKey: 'cluster_id',
    paletteCycle: CLUSTER_PALETTE_CYCLE,
    glyphDataAttrs: [
      { attr: 'data-alignment-id', column: 'alignment_id' },
      { attr: 'data-cluster-id', column: 'cluster_id' },
    ],
  });

  if (result.admittedRows === 0) {
    emitEmpty(ctx, 'no member reads in window');
    return;
  }

  const label = svgEl('text', {
    x: ctx.widthPx - 4,
    y: 12,
    'font-size': '10',
    'text-anchor': 'end',
    fill: '#8a8a93',
  });
  label.textContent =
    `members · ${result.admittedRows.toLocaleString()} reads`;
  ctx.svg.appendChild(label);
  ctx.svg.dataset.naturalHeight = String(result.naturalHeight);
}


function emitEmpty(ctx: RenderContext, message: string): void {
  const text = svgEl('text', {
    x: ctx.widthPx / 2,
    y: ctx.heightPx / 2,
    'font-size': '11',
    fill: '#5a5a63',
    'text-anchor': 'middle',
  });
  text.textContent = message;
  ctx.svg.appendChild(text);
  ctx.svg.dataset.naturalHeight = String(ctx.heightPx);
}
