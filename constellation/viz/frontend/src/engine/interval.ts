// Interval math for one continuous axis: retention time, m/z, a genomic
// coordinate. Pure functions, so the rules (clamping, minimum span, zoom
// about a point) are tested once and shared by every browser.

/** A closed range `[lo, hi]` on one axis, `lo < hi`. */
export type Interval = readonly [number, number];

/** What an axis allows. `round` (e.g. `Math.round` for whole base pairs)
 *  is applied to both ends of every interval produced. */
export interface IntervalLimits {
  min: number;
  max: number;
  /** Smallest span a zoom may reach. */
  minSpan: number;
  round?: (v: number) => number;
}

function rounded(iv: Interval, round?: (v: number) => number): Interval {
  return round ? [round(iv[0]), round(iv[1])] : iv;
}

/** Fit `iv` inside the limits, keeping its span when it fits and the
 *  whole range when it does not. */
export function clampInterval(iv: Interval, limits: IntervalLimits): Interval {
  const { min, max } = limits;
  const span = iv[1] - iv[0];
  if (span >= max - min) return rounded([min, max], limits.round);
  if (iv[0] < min) return rounded([min, min + span], limits.round);
  if (iv[1] > max) return rounded([max - span, max], limits.round);
  return rounded(iv, limits.round);
}

/** Scale the span by `factor` (> 1 zooms out) keeping the point at
 *  `anchor` (0 = left edge, 1 = right edge) where it is. */
export function zoomAbout(
  iv: Interval,
  factor: number,
  anchor: number,
  limits: IntervalLimits,
): Interval {
  const span = iv[1] - iv[0];
  const frac = Math.min(1, Math.max(0, anchor));
  let next = Math.max(limits.minSpan, span * factor);
  if (limits.round) next = Math.max(limits.minSpan, limits.round(next));
  const at = iv[0] + span * frac;
  const lo = limits.round ? limits.round(at - next * frac) : at - next * frac;
  return clampInterval([lo, lo + next], limits);
}

/** Shift by `fraction` of the span (positive moves the window right). */
export function panBy(iv: Interval, fraction: number, limits: IntervalLimits): Interval {
  const shift = (iv[1] - iv[0]) * fraction;
  return clampInterval([iv[0] + shift, iv[1] + shift], limits);
}

/** Whether two intervals are the same to within `eps`. */
export function sameInterval(a: Interval | null, b: Interval | null, eps = 1e-9): boolean {
  if (a === null || b === null) return a === b;
  return Math.abs(a[0] - b[0]) <= eps && Math.abs(a[1] - b[1]) <= eps;
}
