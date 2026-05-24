// gene_annotation renderer — rectangles for exons + lines for introns
// + arrows for strand. Stacks parents over children when parent_id
// resolves within the visible set.

import { Table } from 'apache-arrow';
import { svgEl, clear } from '../engine/svg_layer';
import { TrackMode } from '../engine/arrow_client';
import { TrackRenderer, RenderContext } from './base';
import {
  pickAllowList,
  pickBool,
  pickNumber,
  pickPaletteColor,
  pickString,
} from './style';

const TYPE_COLOR_DEFAULTS: Record<string, string> = {
  gene: '#5e8cd6',
  mRNA: '#7a99e0',
  CDS: '#c5d65e',
  exon: '#9bc23a',
  three_prime_UTR: '#a89dc4',
  five_prime_UTR: '#a89dc4',
  repeat_region: '#d65e5e',
  default: '#888',
};

const renderer: TrackRenderer = {
  kind: 'gene_annotation',
  render(table: Table, _mode: TrackMode, ctx: RenderContext): void {
    clear(ctx.svg);

    const rowH = pickNumber(ctx.style, 'row_height_px', 14);
    const featureOpacity = pickNumber(ctx.style, 'feature_opacity', 0.85);
    const labelFontFamily = pickString(ctx.style, 'label_font_family', '');
    const labelFontSize = pickNumber(ctx.style, 'label_font_size_px', 10);
    const labelMinWidth = pickNumber(ctx.style, 'label_min_width_px', 24);
    const chevronMinWidth = pickNumber(
      ctx.style,
      'strand_chevron_min_width_px',
      12,
    );
    const showChevrons = pickBool(ctx.style, 'show_chevrons', true);
    const showLabelsStyle = ctx.style?.show_labels;
    const showLabels =
      typeof showLabelsStyle === 'boolean'
        ? showLabelsStyle
        : ctx.showLabels !== false;

    const allowedTypes = pickAllowList(ctx.filter, 'visible_types');
    const allowedStrands = pickAllowList(ctx.filter, 'visible_strands');
    const allowedSources = pickAllowList(ctx.filter, 'visible_sources');
    const minLengthBp = pickNumber(ctx.filter, 'min_length_bp', 0);

    if (table.numRows === 0) {
      const t = svgEl('text', {
        x: ctx.widthPx / 2,
        y: ctx.heightPx / 2,
        'font-size': '11',
        fill: '#5a5a63',
        'text-anchor': 'middle',
      });
      t.textContent = 'no features in window';
      ctx.svg.appendChild(t);
      ctx.svg.dataset.naturalHeight = String(ctx.heightPx);
      return;
    }

    // Build a per-row admission mask once so packing and label
    // calculation can both ignore filtered rows cheaply.
    const admit: boolean[] = new Array(table.numRows);
    const sourceCol = table.getChild('source');
    for (let i = 0; i < table.numRows; i++) {
      const row = table.get(i);
      if (!row) {
        admit[i] = false;
        continue;
      }
      const type = String(row.type);
      const strand = String(row.strand);
      const length = Number(row.end) - Number(row.start);
      const source = sourceCol
        ? String(sourceCol.get(i) ?? '')
        : '';
      if (allowedTypes && !allowedTypes.has(type)) {
        admit[i] = false;
        continue;
      }
      if (allowedStrands && !allowedStrands.has(strand)) {
        admit[i] = false;
        continue;
      }
      if (allowedSources && source && !allowedSources.has(source)) {
        admit[i] = false;
        continue;
      }
      if (minLengthBp > 0 && length < minLengthBp) {
        admit[i] = false;
        continue;
      }
      admit[i] = true;
    }

    const rows = packRows(table, admit);
    const nextStartOnRow = showLabels
      ? computeNextStartOnRow(table, rows, admit)
      : null;

    let maxRowIdx = -1;
    for (let i = 0; i < table.numRows; i++) {
      if (!admit[i]) continue;
      const row = table.get(i);
      if (!row) continue;
      const start = Number(row.start);
      const end = Number(row.end);
      const type = String(row.type);
      const strand = String(row.strand);
      const x0 = ctx.xScale(start);
      const x1 = ctx.xScale(end);
      const y = 4 + rows[i] * rowH;
      if (rows[i] > maxRowIdx) maxRowIdx = rows[i];
      const fill = pickPaletteColor(
        ctx.style,
        type,
        TYPE_COLOR_DEFAULTS[type] ?? TYPE_COLOR_DEFAULTS.default,
      );
      const rect = svgEl('rect', {
        x: x0,
        y,
        width: Math.max(1, x1 - x0),
        height: rowH - 4,
        fill,
        opacity: String(featureOpacity),
      });
      const featureId = row.feature_id;
      if (featureId !== null && featureId !== undefined) {
        rect.setAttribute('data-feature-id', String(featureId));
      }
      const name = row.name as string | null;
      if (name) {
        const title = svgEl('title');
        title.textContent = `${type} ${name} (${strand})`;
        rect.appendChild(title);
      }
      ctx.svg.appendChild(rect);

      // Strand chevron at the right edge if there's room.
      if (showChevrons && x1 - x0 > chevronMinWidth) {
        const chevron = svgEl('path', {
          d:
            strand === '-'
              ? `M ${x0 + 4} ${y + 2} L ${x0} ${y + (rowH - 4) / 2} L ${x0 + 4} ${y + rowH - 6}`
              : `M ${x1 - 4} ${y + 2} L ${x1} ${y + (rowH - 4) / 2} L ${x1 - 4} ${y + rowH - 6}`,
          stroke: '#fff',
          'stroke-width': 1.5,
          fill: 'none',
          opacity: '0.7',
        });
        ctx.svg.appendChild(chevron);
      }

      if (showLabels && name && nextStartOnRow) {
        const labelX = Math.max(0, x0) + 3;
        const nextX =
          nextStartOnRow[i] === Number.POSITIVE_INFINITY
            ? ctx.widthPx
            : Math.min(ctx.widthPx, ctx.xScale(nextStartOnRow[i]));
        const maxLabelW = Math.floor(nextX - labelX - 4);
        if (maxLabelW >= labelMinWidth) {
          // Approx glyph width at font-size 10 in our default UI font
          // hovers around 5.6 px; round up so longer names get clipped
          // rather than spilling across the gap.
          const maxChars = Math.max(
            2,
            Math.floor(maxLabelW / (labelFontSize * 0.6)),
          );
          const text =
            name.length > maxChars ? `${name.slice(0, maxChars - 1)}…` : name;
          const labelAttrs: Record<string, string | number> = {
            x: labelX,
            y: y + (rowH - 4) / 2 + 3,
            'font-size': String(labelFontSize),
            fill: '#e3e3e8',
            'pointer-events': 'none',
            'paint-order': 'stroke',
            stroke: '#0f0f12',
            'stroke-width': '2',
            'stroke-linejoin': 'round',
          };
          if (labelFontFamily) {
            labelAttrs['font-family'] = labelFontFamily;
          }
          const label = svgEl('text', labelAttrs);
          label.textContent = text;
          ctx.svg.appendChild(label);
        }
      }
    }

    const naturalHeight =
      maxRowIdx >= 0 ? 4 + (maxRowIdx + 1) * rowH : ctx.heightPx;
    ctx.svg.dataset.naturalHeight = String(naturalHeight);
  },
};

/**
 * For each feature index, find the start coordinate of the next feature
 * on the SAME packed row. Used to clip per-feature labels so they never
 * overlap a downstream neighbour. The last feature on each row gets
 * `+Infinity` (no neighbour).
 */
function computeNextStartOnRow(
  table: Table,
  rows: number[],
  admit: boolean[],
): number[] {
  const n = table.numRows;
  const startCol = table.getChild('start');
  const result = new Array<number>(n).fill(Number.POSITIVE_INFINITY);
  if (!startCol) return result;
  // Group feature indices by row, sorted by start.
  const byRow = new Map<number, number[]>();
  for (let i = 0; i < n; i++) {
    if (!admit[i]) continue;
    const r = rows[i];
    let bucket = byRow.get(r);
    if (!bucket) {
      bucket = [];
      byRow.set(r, bucket);
    }
    bucket.push(i);
  }
  for (const bucket of byRow.values()) {
    bucket.sort(
      (a, b) => Number(startCol.get(a)) - Number(startCol.get(b)),
    );
    for (let k = 0; k < bucket.length - 1; k++) {
      result[bucket[k]] = Number(startCol.get(bucket[k + 1]));
    }
  }
  return result;
}

function packRows(table: Table, admit: boolean[]): number[] {
  const n = table.numRows;
  const indexed: number[] = [];
  for (let i = 0; i < n; i++) {
    if (admit[i]) indexed.push(i);
  }
  const startCol = table.getChild('start');
  const endCol = table.getChild('end');
  if (!startCol || !endCol) return new Array(n).fill(0);
  indexed.sort(
    (a, b) =>
      Number(startCol.get(a)) - Number(startCol.get(b)) ||
      Number(endCol.get(a)) - Number(endCol.get(b)),
  );
  const rowEnds: number[] = [];
  const out = new Array<number>(n).fill(0);
  for (const i of indexed) {
    const s = Number(startCol.get(i));
    const e = Number(endCol.get(i));
    let placed = false;
    for (let r = 0; r < rowEnds.length; r++) {
      if (rowEnds[r] <= s) {
        rowEnds[r] = e;
        out[i] = r;
        placed = true;
        break;
      }
    }
    if (!placed) {
      out[i] = rowEnds.length;
      rowEnds.push(e);
    }
  }
  return out;
}

export default renderer;
