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
import {
  ALIGNMENT_PALETTE,
  ALIGNMENT_STYLE_CONTROLS,
  MIN_MAPQ_CONTROL,
  renderAlignmentRows,
} from './_alignment_view';
import { SettingsSchema } from '../../../panels/settings_schema';
import {
  SAMPLE_PALETTE_CYCLE,
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
            ...ALIGNMENT_PALETTE,
          ],
        },
        ...ALIGNMENT_STYLE_CONTROLS,
      ],
    },
    {
      title: 'Filter',
      controls: [
        { type: 'allowlist', target: 'filter', key: 'visible_samples', label: 'Visible samples', options: sampleOptions() },
        { type: 'allowlist', target: 'filter', key: 'visible_strands', label: 'Visible strands', options: plainOptions(['+', '-']) },
        MIN_MAPQ_CONTROL,
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
