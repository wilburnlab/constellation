// Level of detail, chosen on each axis independently.
//
//   px / column ↓ \ px / row →   ≥ 10 (legible)   1-10 (thin)   < 1 (aggregated)
//   ≥ 8                          letters          ticks         allele fractions per row bin
//   1-8                          ticks + span     ticks         departure heat per row bin
//   < 1                          density per row  density       2-D density image
//
// So a user can zoom x onto one domain with every read a row of coloured
// ticks, or both axes to read bases; zoomed fully out a transcript is a
// density image whose bands are its haplotypes (when rows are ordered by
// pattern).

export type LodX = 'letters' | 'ticks' | 'density';
export type LodY = 'legible' | 'thin' | 'aggregated';

export const LETTERS_PX_PER_COL = 8;
export const LEGIBLE_PX_PER_ROW = 10;

export interface Lod {
  x: LodX;
  y: LodY;
}

export function chooseLod(pxPerCol: number, pxPerRow: number): Lod {
  const x: LodX = pxPerCol >= LETTERS_PX_PER_COL ? 'letters' : pxPerCol >= 1 ? 'ticks' : 'density';
  const y: LodY = pxPerRow >= LEGIBLE_PX_PER_ROW ? 'legible' : pxPerRow >= 1 ? 'thin' : 'aggregated';
  return { x, y };
}

/** Whether a LOD is drawn cell by cell (rects and letters) rather than as
 *  a binned image: every column and every row at least a pixel. */
export function isCellwise(lod: Lod): boolean {
  return lod.x !== 'density' && lod.y !== 'aggregated';
}
