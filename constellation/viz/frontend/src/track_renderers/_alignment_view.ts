// Shared per-alignment renderer used by both read_pileup and
// cluster_pileup (members view). Each row carries:
//   ref_start, ref_end, strand, row, blocks, mismatch_positions
// plus a kernel-specific colorKey column (`sample_id` for read_pileup,
// `cluster_id` for cluster_pileup's member view). The helper draws
// one solid <rect> per CIGAR exon block, dotted intron connectors
// between adjacent blocks, and crossed-line X glyphs at every
// mismatch position. Color is keyed off pickCycledColor(colorKey)
// with a stable per-key palette index that's assigned in encounter
// order — pan/zoom preserves the same color for the same key.

import { Table } from 'apache-arrow';
import { svgEl } from '../engine/svg_layer';
import { GenomicScale } from '../engine/scales';
import {
  pickAllowList,
  pickCycledColor,
  pickNumber,
  pickPaletteColor,
  pickString,
} from './style';

export interface AlignmentViewOptions {
  /** Wire column whose value keys the per-row exon-fill palette.
   *  read_pileup: 'sample_id'. cluster_pileup (members): 'cluster_id'. */
  colorKey: 'sample_id' | 'cluster_id';
  /** Default palette cycle for unset per-key overrides. Mirror the
   *  coverage_histogram cycle for read_pileup so a sample wears the
   *  same color across both tracks. */
  paletteCycle: readonly string[];
  /** Strand-based color fallback when colorKey is null on a row.
   *  Only meaningful when null colorKey values are admissible (read
   *  _pileup with escape-hatch demux; cluster_pileup members never
   *  null). When omitted, null keys fall back to the palette default
   *  entry under `palette.default` / `'#888'`. */
  strandFallback?: Record<string, string>;
  /** Per-row chrome data-attribute set on each glyph rect. The renderer
   *  uses this to attach things like data-cluster-id="..." or
   *  data-alignment-id="..." for inspector hover. Keyed by column
   *  name; values must be int64-or-stringifiable Arrow vectors. */
  glyphDataAttrs?: Array<{ attr: string; column: string }>;
}

const STRAND_FALLBACK_DEFAULT: Record<string, string> = {
  '+': '#5e9cd6',
  '-': '#d6755e',
  default: '#888888',
};


export interface AlignmentViewContext {
  svg: SVGSVGElement;
  widthPx: number;
  heightPx: number;
  xScale: GenomicScale;
  style?: Record<string, unknown>;
  filter?: Record<string, unknown>;
}


export interface AlignmentViewResult {
  /** Number of rows the renderer actually drew (i.e. survived
   *  client-side filters). Used by callers to set the track header
   *  label. */
  admittedRows: number;
  /** Vertical extent of the drawn stack, in pixels. Caller stamps this
   *  on svg.dataset.naturalHeight for SVG export sizing. */
  naturalHeight: number;
}


/** Draw the alignment rows into `ctx.svg`. The caller is responsible
 *  for clearing the SVG beforehand and stamping `svg.dataset.naturalHeight`
 *  with the returned value. Returns 0/heightPx when the table is empty
 *  or every row gets filtered out — caller can branch on that to draw
 *  an empty-state label. */
export function renderAlignmentRows(
  table: Table,
  ctx: AlignmentViewContext,
  opts: AlignmentViewOptions,
): AlignmentViewResult {
  if (table.numRows === 0) {
    return { admittedRows: 0, naturalHeight: ctx.heightPx };
  }

  const startCol = table.getChild('ref_start');
  const endCol = table.getChild('ref_end');
  const strandCol = table.getChild('strand');
  const rowCol = table.getChild('row');
  const blocksCol = table.getChild('blocks');
  const mmCol = table.getChild('mismatch_positions');
  const keyCol = table.getChild(opts.colorKey);
  if (!startCol || !endCol || !strandCol || !rowCol || !blocksCol) {
    return { admittedRows: 0, naturalHeight: ctx.heightPx };
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
  const mismatchColor = pickPaletteColor(ctx.style, 'mismatch', '#e3493a');
  const intronColor = pickPaletteColor(ctx.style, 'intron', '#5a5a63');
  const allowedStrands = pickAllowList(ctx.filter, 'visible_strands');
  // Sample / cluster filter — the same allowlist mechanism applies on
  // whichever colorKey is in use. Filter name follows the column.
  const allowedKeys =
    opts.colorKey === 'sample_id'
      ? pickAllowList(ctx.filter, 'visible_samples')
      : pickAllowList(ctx.filter, 'visible_clusters');

  const strandFallback = opts.strandFallback ?? STRAND_FALLBACK_DEFAULT;
  const keyPaletteIdx = new Map<number, number>();

  const admit: boolean[] = new Array(table.numRows);
  let maxRow = -1;
  let admittedRows = 0;
  for (let i = 0; i < table.numRows; i++) {
    const strand = String(strandCol.get(i));
    if (allowedStrands && !allowedStrands.has(strand)) {
      admit[i] = false;
      continue;
    }
    const keyVal = keyCol ? keyCol.get(i) : null;
    if (allowedKeys) {
      const k = keyVal === null || keyVal === undefined
        ? 'null'
        : String(keyVal);
      if (!allowedKeys.has(k)) {
        admit[i] = false;
        continue;
      }
    }
    admit[i] = true;
    admittedRows++;
    const r = Number(rowCol.get(i));
    if (r > maxRow) maxRow = r;
    if (keyVal !== null && keyVal !== undefined) {
      const num = Number(keyVal);
      if (Number.isFinite(num) && !keyPaletteIdx.has(num)) {
        keyPaletteIdx.set(num, keyPaletteIdx.size);
      }
    }
  }

  if (admittedRows === 0) {
    return { admittedRows: 0, naturalHeight: ctx.heightPx };
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

    // Color resolution priority:
    //   1. user palette override keyed by colorKey value (numeric)
    //   2. cycled palette default for this key's index
    //   3. user palette override keyed by 'exon'
    //   4. user palette override keyed by strand
    //   5. strand-based default
    const keyVal = keyCol ? keyCol.get(i) : null;
    const strandDefault =
      strandFallback[strand] ?? strandFallback.default ?? '#888888';
    const exonStyleFallback = pickPaletteColor(
      ctx.style,
      strand,
      pickPaletteColor(ctx.style, 'exon', strandDefault),
    );
    let exonFill = exonStyleFallback;
    if (keyVal !== null && keyVal !== undefined) {
      const num = Number(keyVal);
      if (Number.isFinite(num)) {
        const idx = keyPaletteIdx.get(num) ?? 0;
        exonFill = pickCycledColor(ctx.style, num, opts.paletteCycle, idx);
      }
    }

    const blocks = readBlocks(blocksCol, i);
    blocks.sort((a, b) => a.refStart - b.refStart);

    // Exon rectangles per CIGAR M/=/X block.
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
      for (const dataAttr of opts.glyphDataAttrs ?? []) {
        const col = table.getChild(dataAttr.column);
        if (col) {
          rect.setAttribute(dataAttr.attr, String(col.get(i)));
        }
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

    // Per-position mismatch X glyphs (only populated at zooms where the
    // kernel decided per-base detail is resolvable).
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

  const naturalHeight = stackH > 0 ? 4 + stackH * rowH : ctx.heightPx;
  return { admittedRows, naturalHeight };
}


interface Block {
  refStart: number;
  refEnd: number;
}


function readBlocks(col: unknown, i: number): Block[] {
  const list = (col as { get: (i: number) => unknown }).get(i);
  if (!list) return [];
  const out: Block[] = [];
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
  // X glyph as two crossed line segments. Cheaper than text and scales
  // cleanly under SVG zoom.
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
