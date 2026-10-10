// read_pileup renderer — vector mode delegates to the shared
// _alignment_view helper (per-block exon segments + dotted intron
// connectors + crossed-line mismatch glyphs, colored by sample_id);
// hybrid mode paints the server-rendered datashader PNG.
//
// Wire columns consumed in vector mode:
//   alignment_id, ref_start, ref_end, strand, mapq, row,
//   blocks: list<struct{ref_start, ref_end, n_match, n_mismatch}>,
//   mismatch_positions: list<int64>,
//   sample_id: int64?, sample_name: string?

import { Table } from 'apache-arrow';
import { svgEl, clear } from '../../../engine/svg_layer';
import { TrackMode } from '../../../engine/arrow_client';
import { decodeHybrid, appendHybridImage } from '../../../engine/hybrid_layer';
import { TrackRenderer, RenderContext } from './base';
import { minMapq } from './pushdown';
import { ALIGNMENT_DEFAULTS, renderAlignmentRows } from './_alignment_view';
import { SettingsSchema } from '../../../panels/settings_schema';
import {
  REFETCH_HINT,
  SAMPLE_PALETTE_CYCLE,
  generalSection,
  num,
  opacity,
  plainOptions,
  sampleOptions,
  samplePalette,
} from './settings_common';

const STRAND_FALLBACK: Record<string, string> = {
  '+': '#5e9cd6',
  '-': '#d6755e',
  default: '#888888',
};

const SETTINGS: SettingsSchema = {
  sections: [
    generalSection(1.0),
    {
      title: 'Style',
      controls: [
        // Per-sample exon colour — the primary colour key. Reads with no
        // sample fall back to the strand colours below.
        { type: 'palette', entries: samplePalette(), emptyHint: 'no samples in this source' },
        {
          type: 'palette',
          entries: [
            { key: '+', label: 'Forward strand (fallback)', default: STRAND_FALLBACK['+'] },
            { key: '-', label: 'Reverse strand (fallback)', default: STRAND_FALLBACK['-'] },
            { key: 'default', label: 'Unstranded / default', default: STRAND_FALLBACK.default },
            // Shared across every read: sample identity lives in the exon
            // fill, so the mismatch colour stays one consistent value.
            { key: 'intron', label: 'Intron connector', default: ALIGNMENT_DEFAULTS.intron_color },
            { key: 'mismatch', label: 'Mismatch glyph', default: ALIGNMENT_DEFAULTS.mismatch_color },
          ],
        },
        { type: 'text', target: 'style', key: 'intron_stroke_dasharray', label: 'Intron dasharray', default: ALIGNMENT_DEFAULTS.intron_stroke_dasharray },
        num('style', 'intron_stroke_width_px', 'Intron stroke (px)', ALIGNMENT_DEFAULTS.intron_stroke_width_px, 0.5, 4, 0.5),
        num('style', 'mismatch_glyph_size_px', 'Mismatch glyph size (px)', ALIGNMENT_DEFAULTS.mismatch_glyph_size_px, 2, 20, 1),
        num('style', 'min_row_height_px', 'Min row height (px)', ALIGNMENT_DEFAULTS.min_row_height_px, 1, 20, 1),
        num('style', 'max_row_height_px', 'Max row height (px)', ALIGNMENT_DEFAULTS.max_row_height_px, 2, 40, 1),
        opacity('read_opacity', 'Read opacity', ALIGNMENT_DEFAULTS.read_opacity),
      ],
    },
    {
      title: 'Filter',
      controls: [
        { type: 'allowlist', target: 'filter', key: 'visible_samples', label: 'Visible samples', options: sampleOptions() },
        { type: 'allowlist', target: 'filter', key: 'visible_strands', label: 'Visible strands', options: plainOptions(['+', '-']) },
        // Applied by the kernel, not from the cached table.
        num('filter', 'min_mapq', 'Min MAPQ', 0, 0, 60, 1, REFETCH_HINT),
      ],
    },
  ],
};

const renderer: TrackRenderer = {
  kind: 'read_pileup',
  order: 3,
  unit: ['read', 'reads'],
  pushdown: { min_mapq: minMapq },
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
      label.textContent = `hybrid · ${frame.nItems.toLocaleString()} reads`;
      ctx.svg.appendChild(label);
      ctx.svg.dataset.naturalHeight = String(ctx.heightPx);
      return;
    }

    const result = renderAlignmentRows(table, ctx, {
      colorKey: 'sample_id',
      paletteCycle: SAMPLE_PALETTE_CYCLE,
      strandFallback: STRAND_FALLBACK,
      glyphDataAttrs: [
        { attr: 'data-alignment-id', column: 'alignment_id' },
      ],
    });

    if (result.admittedRows === 0) {
      emitEmpty(ctx);
      return;
    }

    const label = svgEl('text', {
      x: ctx.widthPx - 4,
      y: 12,
      'font-size': '10',
      'text-anchor': 'end',
      fill: '#8a8a93',
    });
    label.textContent = `vector · ${result.admittedRows.toLocaleString()} reads`;
    ctx.svg.appendChild(label);
    ctx.svg.dataset.naturalHeight = String(result.naturalHeight);
  },
};

export default renderer;


function emitEmpty(ctx: RenderContext): void {
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
}
