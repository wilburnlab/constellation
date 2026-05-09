// d3-scale wrappers — one consistent linear scale per genome track.
//
// `xScale(domain, width)` returns a fresh d3 linear scale clamped to
// the panel width. The genome browser uses one scale for the whole
// viewport; tracks share it so brushes/zooms move them in lockstep.

import { scaleLinear, type ScaleLinear } from 'd3-scale';
import { axisBottom, type Axis } from 'd3-axis';
import { format } from 'd3-format';

export type GenomicScale = ScaleLinear<number, number>;

export function xScale(
  domain: [number, number],
  width: number,
): GenomicScale {
  return scaleLinear().domain(domain).range([0, width]);
}

export function makeAxis(scale: GenomicScale): Axis<number> {
  return axisBottom<number>(scale)
    .ticks(8)
    .tickSizeOuter(0)
    .tickFormat((d) => formatGenomic(Number(d)));
}

const _tickFmt = format(',');

export function formatGenomic(bp: number): string {
  if (Math.abs(bp) >= 1_000_000) {
    return `${(bp / 1_000_000).toFixed(2)} Mb`;
  }
  if (Math.abs(bp) >= 1_000) {
    return `${(bp / 1_000).toFixed(2)} kb`;
  }
  return `${_tickFmt(bp)} bp`;
}
