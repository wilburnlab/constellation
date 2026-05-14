// gene_annotation renderer — rectangles for exons + lines for introns
// + arrows for strand. Stacks parents over children when parent_id
// resolves within the visible set.

import { Table } from 'apache-arrow';
import { svgEl, clear } from '../engine/svg_layer';
import { TrackMode } from '../engine/arrow_client';
import { TrackRenderer, RenderContext } from './base';

const TYPE_COLOR: Record<string, string> = {
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
      return;
    }

    // Greedy row pack so child features don't visually merge.
    const rows = packRows(table);
    const showLabels = ctx.showLabels !== false;
    const nextStartOnRow = showLabels ? computeNextStartOnRow(table, rows) : null;

    const rowH = 14;
    for (let i = 0; i < table.numRows; i++) {
      const row = table.get(i);
      if (!row) continue;
      const start = Number(row.start);
      const end = Number(row.end);
      const type = String(row.type);
      const strand = String(row.strand);
      const x0 = ctx.xScale(start);
      const x1 = ctx.xScale(end);
      const y = 4 + rows[i] * rowH;
      const fill = TYPE_COLOR[type] ?? TYPE_COLOR.default;
      const rect = svgEl('rect', {
        x: x0,
        y,
        width: Math.max(1, x1 - x0),
        height: rowH - 4,
        fill,
        opacity: '0.85',
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
      if (x1 - x0 > 12) {
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
        if (maxLabelW >= 24) {
          // Approx glyph width at font-size 10 in our default UI font
          // hovers around 5.6 px; round up so longer names get clipped
          // rather than spilling across the gap.
          const maxChars = Math.max(2, Math.floor(maxLabelW / 6));
          const text =
            name.length > maxChars ? `${name.slice(0, maxChars - 1)}…` : name;
          const label = svgEl('text', {
            x: labelX,
            y: y + (rowH - 4) / 2 + 3,
            'font-size': '10',
            fill: '#e3e3e8',
            'pointer-events': 'none',
            'paint-order': 'stroke',
            stroke: '#0f0f12',
            'stroke-width': '2',
            'stroke-linejoin': 'round',
          });
          label.textContent = text;
          ctx.svg.appendChild(label);
        }
      }
    }
  },
};

/**
 * For each feature index, find the start coordinate of the next feature
 * on the SAME packed row. Used to clip per-feature labels so they never
 * overlap a downstream neighbour. The last feature on each row gets
 * `+Infinity` (no neighbour).
 */
function computeNextStartOnRow(table: Table, rows: number[]): number[] {
  const n = table.numRows;
  const startCol = table.getChild('start');
  const result = new Array<number>(n).fill(Number.POSITIVE_INFINITY);
  if (!startCol) return result;
  // Group feature indices by row, sorted by start.
  const byRow = new Map<number, number[]>();
  for (let i = 0; i < n; i++) {
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

function packRows(table: Table): number[] {
  const n = table.numRows;
  const indexed: number[] = Array.from({ length: n }, (_, i) => i);
  const startCol = table.getChild('start');
  const endCol = table.getChild('end');
  if (!startCol || !endCol) return new Array(n).fill(0);
  indexed.sort(
    (a, b) =>
      Number(startCol.get(a)) - Number(startCol.get(b)) ||
      Number(endCol.get(a)) - Number(endCol.get(b)),
  );
  const rowEnds: number[] = [];
  const out = new Array<number>(n);
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
