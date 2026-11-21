// Covariance arcs: each joins two candidate columns above the axis, its
// height proportional to the distance it spans, its colour the effect (φ;
// the log odds where φ is undefined), its opacity −log10 p and its width
// the reads co-observed at the pair. Pure geometry and colour, so the
// encoding is tested on its own.

import { svgEl } from '../../engine/svg_layer';
import type { ArcInfo } from './support_data';

/** Positive association (the columns depart together). */
export const ARC_POSITIVE = '#61ffca';
/** Negative (depletion: they never depart together). */
export const ARC_NEGATIVE = '#ff9e64';

/** The lowest an arc is drawn, in pixels. */
export const MIN_ARC_HEIGHT = 10;

export interface ArcGeometry {
  index: number;
  x1: number;
  x2: number;
  height: number;
  colour: string;
  opacity: number;
  width: number;
}

/** The effect an arc is coloured by, in [-1, 1]. */
export function arcEffect(arcs: ArcInfo, i: number): number {
  const phi = arcs.phi[i];
  if (Number.isFinite(phi)) return Math.max(-1, Math.min(1, phi));
  const lo = arcs.logOdds[i];
  if (Number.isFinite(lo)) return Math.tanh(lo / 4);
  return 1; // a self edge (a collapsed run): one clique
}

export function arcColour(effect: number): string {
  return effect >= 0 ? ARC_POSITIVE : ARC_NEGATIVE;
}

/** −log10 p mapped to [0.2, 1]: 1e-10 and stronger are opaque. */
export function arcOpacity(p: number): number {
  const s = p > 0 ? -Math.log10(p) : 300;
  return Math.max(0.2, Math.min(1, s / 10));
}

/** log10 of co-observed reads mapped to [0.75, 4] px. */
export function arcWidth(n: number): number {
  if (!(n > 0)) return 0.75;
  return Math.max(0.75, Math.min(4, 0.75 + Math.log10(n)));
}

/** Arcs in view as geometry. `x(col)` is a column's centre in pixels;
 *  an arc's height is proportional to the distance it spans on screen
 *  (half of it), at least `MIN_ARC_HEIGHT` so that arcs between nearby
 *  columns of a zoomed-out transcript are still seen (and separate as the
 *  view zooms in), at most `maxHeight`. Arcs whose both ends are out of
 *  view on one side are dropped; |effect| below `minEffect` too.
 *  `fullSpan` is unused and kept for callers. */
export function arcGeometry(
  arcs: ArcInfo,
  x: (col: number) => number,
  widthPx: number,
  maxHeight: number,
  _fullSpan: number,
  minEffect = 0,
): ArcGeometry[] {
  const out: ArcGeometry[] = [];
  for (let i = 0; i < arcs.n; i += 1) {
    const u = Math.min(arcs.u[i], arcs.v[i]);
    const v = Math.max(arcs.u[i], arcs.v[i]);
    const x1 = x(u);
    const x2 = x(v);
    if (x2 < 0 || x1 > widthPx) continue;
    const effect = arcEffect(arcs, i);
    if (Math.abs(effect) < minEffect) continue;
    out.push({
      index: i,
      x1,
      x2,
      height: Math.min(maxHeight, Math.max(MIN_ARC_HEIGHT, (x2 - x1) / 2)),
      colour: arcColour(effect),
      opacity: arcOpacity(arcs.pValue[i]),
      width: arcWidth(arcs.total[i]),
    });
  }
  // Tallest first, so short arcs draw on top and stay clickable.
  return out.sort((a, b) => b.height - a.height);
}

/** An arc as an SVG path from `(x1, base)` to `(x2, base)`, peaking
 *  `height` above. A self edge is a small loop. */
export function arcPath(g: ArcGeometry, base: number): string {
  if (Math.abs(g.x2 - g.x1) < 1) {
    const r = Math.max(2, g.height / 3);
    return `M ${g.x1} ${base} c ${-r} ${-2 * r} ${r} ${-2 * r} 0 0`;
  }
  const mid = (g.x1 + g.x2) / 2;
  // A quadratic Bézier peaks at half its control point's offset.
  return `M ${g.x1} ${base} Q ${mid} ${base - 2 * g.height} ${g.x2} ${base}`;
}

export function drawArcs(
  parent: SVGGElement,
  geoms: ArcGeometry[],
  base: number,
  onPick?: (index: number, event: MouseEvent) => void,
  onHover?: (index: number | null, event: MouseEvent) => void,
): void {
  for (const g of geoms) {
    const path = svgEl('path', {
      d: arcPath(g, base),
      fill: 'none',
      stroke: g.colour,
      'stroke-opacity': g.opacity.toFixed(2),
      'stroke-width': g.width.toFixed(2),
      class: 'tx-arc',
      'data-arc': g.index,
    });
    if (onPick) path.addEventListener('click', (e) => onPick(g.index, e));
    if (onHover) {
      path.addEventListener('mouseenter', (e) => onHover(g.index, e));
      path.addEventListener('mouseleave', (e) => onHover(null, e));
    }
    parent.appendChild(path);
  }
}
