// cluster_pileup renderer — two views:
//
//   clusters (default) — one rectangle per cluster from span_start to
//     span_end, colored by `mode` (genome / kmer / em), with
//     log-scaled opacity by n_reads. Hybrid mode paints a datashader
//     PNG.
//
//   members — one rectangle per member alignment (with CIGAR-aware
//     exon blocks + intron connectors + per-base X glyphs), colored
//     by `cluster_id` so the user can see which reads got grouped
//     together. Delegates to the shared _alignment_view helper.
//
// The view is chosen by the `cluster_view` filter, which the kernel
// applies; the renderer tells which one it was sent by which columns
// are on the wire. The settings popover follows the same filter, so it
// offers each view the controls that view is drawn with.

import { Table } from 'apache-arrow';
import { svgEl, clear } from '../../../engine/svg_layer';
import { TrackMode } from '../../../engine/arrow_client';
import { decodeHybrid, appendHybridImage } from '../../../engine/hybrid_layer';
import { TrackRenderer, RenderContext } from './base';
import { clusterView, minMapq } from './pushdown';
import {
  ALIGNMENT_PALETTE,
  ALIGNMENT_STYLE_CONTROLS,
  MIN_MAPQ_CONTROL,
  renderAlignmentRows,
} from './_alignment_view';
import { SettingsEnv, SettingsSchema } from '../../../panels/settings_schema';
import {
  REFETCH_HINT,
  num,
  opacity,
  orFallback,
  plainOptions,
  shownWhen,
  stringList,
} from './settings_common';
import {
  pickAllowList,
  pickNumber,
  pickPaletteColor,
} from '../../../panels/style';

// Keyed on the `mode` column of clusters.parquet. The canonical names
// describe the mechanism (genome / kmer / em); the two pre-rename spellings
// are kept so clusters.parquet files written before the rename still colour
// correctly rather than falling through to grey.
const MODE_COLOR_DEFAULTS: Record<string, string> = {
  genome: '#5ed6cf',
  kmer: '#a8d65e',
  em: '#d6a85e',
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

/** Style and filter defaults for the clusters view, shared by the
 *  drawing code and the settings popover. */
const DEFAULTS = {
  min_row_height_px: 4,
  max_row_height_px: 10,
  opacity_min: 0.4,
  opacity_max: 1.0,
  min_reads: 1,
} as const;

/** Offered when the track's metadata names no modes. */
const KNOWN_MODES = ['genome', 'kmer', 'em'];

function modesOf(env: SettingsEnv): string[] {
  return orFallback(stringList(env.meta.modes_in_data), KNOWN_MODES);
}

/** True while the track shows member reads: that view is selected and
 *  the kernel can serve it (asked for it without the upstream align
 *  dir, the kernel answers with clusters). */
function membersView(env: SettingsEnv): boolean {
  return env.filter.cluster_view === 'members' && env.meta.cluster_view_supported === true;
}

function clustersView(env: SettingsEnv): boolean {
  return !membersView(env);
}

// The two views are drawn by different code from different columns, so
// each offers its own controls. Row height is one stored key with a
// per-view default; per-cluster colours and the `visible_clusters`
// filter the members view also honours have no control, because the
// track's metadata does not list cluster ids.
const SETTINGS: SettingsSchema = {
  sections: [
    {
      title: 'Style',
      controls: [
        ...shownWhen(clustersView, [
          {
            type: 'palette',
            entries: (env) =>
              modesOf(env).map((mode) => ({
                key: mode,
                label: mode,
                default: MODE_COLOR_DEFAULTS[mode] ?? MODE_COLOR_DEFAULTS.default,
              })),
          },
          num('style', 'min_row_height_px', 'Min row height (px)', DEFAULTS.min_row_height_px, 1, 20, 1),
          num('style', 'max_row_height_px', 'Max row height (px)', DEFAULTS.max_row_height_px, 2, 40, 1),
          opacity('opacity_min', 'Opacity min', DEFAULTS.opacity_min),
          opacity('opacity_max', 'Opacity max', DEFAULTS.opacity_max),
        ]),
        ...shownWhen(membersView, [
          { type: 'palette', entries: [...ALIGNMENT_PALETTE] },
          ...ALIGNMENT_STYLE_CONTROLS,
        ]),
      ],
    },
    {
      title: 'Filter',
      controls: [
        {
          // Offered only when the kernel can expand clusters into member
          // reads (it needs the upstream align dir). Applied by the kernel.
          type: 'select',
          target: 'filter',
          key: 'cluster_view',
          label: 'Cluster view',
          default: 'clusters',
          options: [
            { value: 'clusters', label: 'Clusters' },
            { value: 'members', label: 'Member reads' },
          ],
          when: (env) => env.meta.cluster_view_supported === true,
          hint: REFETCH_HINT,
        },
        ...shownWhen(clustersView, [
          { type: 'allowlist', target: 'filter', key: 'visible_modes', label: 'Visible modes', options: (env) => plainOptions(modesOf(env)) },
          { type: 'allowlist', target: 'filter', key: 'visible_strands', label: 'Visible strands', options: plainOptions(['+', '-', '.']) },
          num('filter', 'min_reads', 'Min reads', DEFAULTS.min_reads, 1, 1_000_000, 1),
        ]),
        ...shownWhen(membersView, [
          { type: 'allowlist', target: 'filter', key: 'visible_strands', label: 'Visible strands', options: plainOptions(['+', '-']) },
          MIN_MAPQ_CONTROL,
        ]),
      ],
    },
  ],
};

const renderer: TrackRenderer = {
  kind: 'cluster_pileup',
  order: 4,
  unit: ['cluster', 'clusters'],
  // The status line counts clusters in either view.
  pushdown: { min_mapq: minMapq, cluster_view: clusterView },
  settings: SETTINGS,
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

  const minRowH = pickNumber(ctx.style, 'min_row_height_px', DEFAULTS.min_row_height_px);
  const maxRowH = pickNumber(ctx.style, 'max_row_height_px', DEFAULTS.max_row_height_px);
  const opacityMin = pickNumber(ctx.style, 'opacity_min', DEFAULTS.opacity_min);
  const opacityMax = pickNumber(ctx.style, 'opacity_max', DEFAULTS.opacity_max);
  const opacityRange = Math.max(0, opacityMax - opacityMin);

  const allowedModes = pickAllowList(ctx.filter, 'visible_modes');
  const allowedStrands = pickAllowList(ctx.filter, 'visible_strands');
  const minReads = pickNumber(ctx.filter, 'min_reads', DEFAULTS.min_reads);

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
