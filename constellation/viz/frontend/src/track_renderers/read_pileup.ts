// read_pileup renderer — vector mode draws per-block exon segments
// (solid rects, color keyed by sample_id) with dotted intron
// connectors between adjacent blocks and X glyphs at per-base
// substitution sites; hybrid mode paints the server-rendered
// datashader PNG.
//
// Wire columns consumed in vector mode:
//   alignment_id, ref_start, ref_end, strand, mapq, row,
//   blocks: list<struct{ref_start, ref_end, n_match, n_mismatch}>,
//   mismatch_positions: list<int64>,
//   sample_id: int64?, sample_name: string?
//
// `blocks` is always populated by the kernel (synthetic single-block
// fallback when alignment_blocks/ has no rows for an alignment), so
// the renderer's per-row loop never branches on its absence.
// `mismatch_positions` is empty when the viewport is too coarse to
// resolve per-base glyphs (the kernel decides via
// mismatch_glyph_bp_per_pixel_limit). `sample_id` keys a cycled
// palette so the user can compare coverage across samples at a glance;
// null sample_ids fall back to the strand-colored default.

import { Table } from 'apache-arrow';
import { svgEl, clear } from '../engine/svg_layer';
import { TrackMode } from '../engine/arrow_client';
import { decodeHybrid, appendHybridImage } from '../engine/hybrid_layer';
import { TrackRenderer, RenderContext } from './base';
import {
  pickAllowList,
  pickCycledColor,
  pickNumber,
  pickPaletteColor,
  pickString,
} from './style';

const STRAND_COLOR_DEFAULTS: Record<string, string> = {
  '+': '#5e9cd6',
  '-': '#d6755e',
  default: '#888888',
};

// Mirror of the coverage_histogram per-sample palette so multi-track
// views (coverage + pileup of the same sample) share a color identity.
const SAMPLE_PALETTE_CYCLE = [
  '#4f9efb',
  '#fb7c4f',
  '#a4d65e',
  '#d65eb6',
  '#5ed6cf',
];

const INTRON_DEFAULT = '#5a5a63';
const MISMATCH_DEFAULT = '#e3493a';

interface Block {
  refStart: number;
  refEnd: number;
}

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
      emitEmpty(ctx);
      return;
    }

    const startCol = table.getChild('ref_start');
    const endCol = table.getChild('ref_end');
    const strandCol = table.getChild('strand');
    const rowCol = table.getChild('row');
    const idCol = table.getChild('alignment_id');
    const blocksCol = table.getChild('blocks');
    const mmCol = table.getChild('mismatch_positions');
    const sampleIdCol = table.getChild('sample_id');
    if (!startCol || !endCol || !strandCol || !rowCol || !blocksCol) {
      ctx.svg.dataset.naturalHeight = String(ctx.heightPx);
      return;
    }

    const minRowH = pickNumber(ctx.style, 'min_row_height_px', 2);
    const maxRowH = pickNumber(ctx.style, 'max_row_height_px', 8);
    const readOpacity = pickNumber(ctx.style, 'read_opacity', 1.0);
    const intronDasharray = pickString(
      ctx.style,
      'intron_stroke_dasharray',
      '2,2',
    );
    const intronStrokeWidth = pickNumber(ctx.style, 'intron_stroke_width_px', 1);
    const mismatchSize = pickNumber(ctx.style, 'mismatch_glyph_size_px', 6);
    const mismatchColor = pickPaletteColor(
      ctx.style,
      'mismatch',
      MISMATCH_DEFAULT,
    );
    const intronColor = pickPaletteColor(ctx.style, 'intron', INTRON_DEFAULT);
    const allowedStrands = pickAllowList(ctx.filter, 'visible_strands');
    const allowedSamples = pickAllowList(ctx.filter, 'visible_samples');

    // Stable per-sample palette index — assign in encounter order so a
    // user dragging the viewport keeps the same color for the same
    // sample even as the visible alignments change. The metadata-side
    // `samples_in_data` is sorted, so colors are also stable across
    // sessions when the user hasn't overridden anything.
    const samplePaletteIdx = new Map<number, number>();

    const admit: boolean[] = new Array(table.numRows);
    let maxRow = -1;
    let admittedRows = 0;
    for (let i = 0; i < table.numRows; i++) {
      const strand = String(strandCol.get(i));
      if (allowedStrands && !allowedStrands.has(strand)) {
        admit[i] = false;
        continue;
      }
      const sampleVal = sampleIdCol ? sampleIdCol.get(i) : null;
      if (allowedSamples) {
        const key = sampleVal === null || sampleVal === undefined
          ? 'null'
          : String(sampleVal);
        if (!allowedSamples.has(key)) {
          admit[i] = false;
          continue;
        }
      }
      admit[i] = true;
      admittedRows++;
      const r = Number(rowCol.get(i));
      if (r > maxRow) maxRow = r;
      if (sampleVal !== null && sampleVal !== undefined) {
        const sid = Number(sampleVal);
        if (Number.isFinite(sid) && !samplePaletteIdx.has(sid)) {
          samplePaletteIdx.set(sid, samplePaletteIdx.size);
        }
      }
    }

    if (admittedRows === 0) {
      emitEmpty(ctx);
      return;
    }

    const stackH = maxRow + 1;
    const rowH = Math.max(
      minRowH,
      Math.min(maxRowH, (ctx.heightPx - 4) / Math.max(1, stackH)),
    );

    for (let i = 0; i < table.numRows; i++) {
      if (!admit[i]) continue;
      const strand = String(strandCol.get(i));
      const row = Number(rowCol.get(i));
      const yMid = 4 + row * rowH + rowH / 2;
      const rectH = Math.max(1, rowH - 1);

      // Resolve exon fill in priority order:
      //  1. user palette override keyed by sample_id (numeric)
      //  2. cycled palette default for this sample's index
      //  3. user palette override keyed by `exon`
      //  4. user palette override keyed by strand
      //  5. strand-based default
      // Null sample_id falls back to (3)/(4)/(5).
      const sampleVal = sampleIdCol ? sampleIdCol.get(i) : null;
      const strandFallback =
        STRAND_COLOR_DEFAULTS[strand] ?? STRAND_COLOR_DEFAULTS.default;
      const exonStyleFallback = pickPaletteColor(
        ctx.style,
        strand,
        pickPaletteColor(ctx.style, 'exon', strandFallback),
      );
      let exonFill = exonStyleFallback;
      if (sampleVal !== null && sampleVal !== undefined) {
        const sid = Number(sampleVal);
        if (Number.isFinite(sid)) {
          const idx = samplePaletteIdx.get(sid) ?? 0;
          exonFill = pickCycledColor(
            ctx.style,
            sid,
            SAMPLE_PALETTE_CYCLE,
            idx,
          );
        }
      }

      const blocks = readBlocks(blocksCol, i);
      // Sort blocks ascending — kernel emits in block_index order, but
      // the fallback single-block path uses the alignment's own
      // (start, end); defensive sort is cheap at viewport scale.
      blocks.sort((a, b) => a.refStart - b.refStart);

      // Exon rectangles per block.
      const alignmentId = idCol ? String(idCol.get(i)) : null;
      for (const block of blocks) {
        const x0 = ctx.xScale(block.refStart);
        const x1 = ctx.xScale(block.refEnd);
        const rect = svgEl('rect', {
          x: x0,
          y: 4 + row * rowH,
          width: Math.max(1, x1 - x0),
          height: rectH,
          fill: exonFill,
          opacity: String(readOpacity),
        });
        if (alignmentId !== null) {
          rect.setAttribute('data-alignment-id', alignmentId);
        }
        ctx.svg.appendChild(rect);
      }

      // Intron connector between each adjacent pair of blocks.
      for (let j = 0; j < blocks.length - 1; j++) {
        const intronX0 = ctx.xScale(blocks[j].refEnd);
        const intronX1 = ctx.xScale(blocks[j + 1].refStart);
        if (intronX1 <= intronX0) continue;
        const line = svgEl('line', {
          x1: intronX0,
          y1: yMid,
          x2: intronX1,
          y2: yMid,
          stroke: intronColor,
          'stroke-width': String(intronStrokeWidth),
          'stroke-dasharray': intronDasharray,
          opacity: String(readOpacity),
        });
        ctx.svg.appendChild(line);
      }

      // Per-position mismatch X glyphs (only populated at zooms where
      // the kernel decided per-base detail is resolvable). Drawn last
      // so they sit on top of the exon rectangles.
      if (mmCol) {
        const mmList = readNumberList(mmCol, i);
        if (mmList.length > 0) {
          drawMismatchGlyphs(
            ctx.svg,
            mmList,
            ctx.xScale,
            yMid,
            rectH,
            mismatchSize,
            mismatchColor,
          );
        }
      }
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


/** Pull the i-th alignment's block list out of the `blocks`
 *  `List<Struct>` column. Returns native JS numbers; Arrow int64s come
 *  back as BigInt and we narrow to Number for SVG math (read pile-up
 *  coordinates are well within Number.MAX_SAFE_INTEGER). */
function readBlocks(col: unknown, i: number): Block[] {
  const list = (col as { get: (i: number) => unknown }).get(i);
  if (!list) return [];
  const out: Block[] = [];
  // apache-arrow's List<Struct> element is a Vector you can iterate.
  // Each element is a row-proxy with `.toJSON()` returning an object.
  const iter = list as Iterable<unknown>;
  for (const elem of iter) {
    if (!elem) continue;
    const e = elem as { ref_start?: unknown; ref_end?: unknown };
    out.push({
      refStart: Number(e.ref_start ?? 0),
      refEnd: Number(e.ref_end ?? 0),
    });
  }
  return out;
}


/** Pull the i-th row's int64 list as a number[]. Empty list returns []. */
function readNumberList(col: unknown, i: number): number[] {
  const list = (col as { get: (i: number) => unknown }).get(i);
  if (!list) return [];
  const out: number[] = [];
  const iter = list as Iterable<unknown>;
  for (const v of iter) {
    if (v === null || v === undefined) continue;
    out.push(Number(v));
  }
  return out;
}


function drawMismatchGlyphs(
  svg: SVGSVGElement,
  positions: number[],
  xScale: (g: number) => number,
  yMid: number,
  rectH: number,
  glyphSize: number,
  color: string,
): void {
  // X glyph as two crossed line segments. Cheaper to render than a
  // <text>X</text> and scales cleanly under SVG zoom.
  const half = Math.max(2, Math.min(glyphSize, rectH + 2)) / 2;
  const stroke = '1.2';
  for (const pos of positions) {
    const x = xScale(pos);
    const ne = svgEl('line', {
      x1: x - half,
      y1: yMid - half,
      x2: x + half,
      y2: yMid + half,
      stroke: color,
      'stroke-width': stroke,
      'stroke-linecap': 'round',
    });
    const nw = svgEl('line', {
      x1: x - half,
      y1: yMid + half,
      x2: x + half,
      y2: yMid - half,
      stroke: color,
      'stroke-width': stroke,
      'stroke-linecap': 'round',
    });
    svg.appendChild(ne);
    svg.appendChild(nw);
  }
}


