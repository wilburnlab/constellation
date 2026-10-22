// Colours of the mass-spec browser's data marks.
//
// Validated with the dataviz skill's palette checker against the browser's
// dark surface (#15141b): the categorical order passes every adjacent-pair
// gate (worst CVD ΔE 8.4, normal-vision 19.3, all ≥ 3:1); the isotope ramp
// passes as an ordinal ramp (one hue, monotone lightness); the ion-type
// trio passes the normal-vision floor all-pairs with CVD in the 6–8 band,
// which is legal because every annotated peak is also labelled (y7, b5^2).
// Colours follow the entity (a trace's slot is its place in the target's
// list, which does not change while the target is shown), never a rank.

/** Fragment and ad-hoc traces, in this fixed order; a ninth reuses none. */
export const SERIES = ['#3987e5', '#d95926', '#199e70', '#c98500', '#d55181', '#008300', '#9085e9', '#e66767'];

/** Precursor isotopes M, M+1, M+2: one hue, darker with each step. */
export const ISOTOPES = ['#9ec5f4', '#5598e7', '#1c5cab'];

/** Annotated peaks by ion series; the rest of a spectrum stays neutral. */
export const ION_COLORS: Record<string, string> = {
  b: '#3987e5',
  y: '#e66767',
  p: '#199e70',
};

/** Peaks no ion explains. */
export const UNANNOTATED = '#6d6d6d';

export function seriesColor(slot: number): string {
  return SERIES[slot % SERIES.length];
}

export function ionColor(ionType: string | null | undefined): string | null {
  if (!ionType) return null;
  return ION_COLORS[ionType] ?? '#c98500'; // a / c / x / z: the one slot left
}
