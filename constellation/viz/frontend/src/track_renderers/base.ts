// Track-renderer interface — mirror to TrackKernel.fetch on the Python side.
//
// Each renderer module exports a default object satisfying this
// interface. The GenomeBrowser widget composes track renderers based
// on the `kind` returned by /api/tracks; new track types land as new
// modules + a registry entry in `track_renderers/index.ts`.

import { Table } from 'apache-arrow';
import { TrackMode } from '../engine/arrow_client';
import { GenomicScale } from '../engine/scales';

export interface TrackMetadata {
  kind: string;
  binding_id: string;
  label: string;
  default_height_px: number;
  [extra: string]: unknown;
}

export interface RenderContext {
  /** Track-local SVG to draw into. Pre-sized by the host. */
  svg: SVGSVGElement;
  /** Pixel width of the track frame. */
  widthPx: number;
  /** Pixel height available to the renderer. */
  heightPx: number;
  /** Genomic-position → pixel-x scale. */
  xScale: GenomicScale;
  /** Decoded metadata payload. */
  meta: TrackMetadata;
}

export interface TrackRenderer {
  kind: string;
  /** Render a freshly fetched table into the provided SVG. The
   *  renderer is responsible for clearing previous contents. */
  render(table: Table, mode: TrackMode, ctx: RenderContext): void;
}
