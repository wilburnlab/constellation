// Hybrid-mode helper: turn the one-row HYBRID_SCHEMA payload into an
// SVG <image> element placed inside the track frame.
//
// Server-rasterized PNG bytes arrive as a `Uint8Array`. We base64-
// encode and slot the data URI into the `<image>` href. The Arrow
// envelope around the PNG is intentionally simple: one row per
// rendered viewport, no streaming chunks.

import { Table } from 'apache-arrow';
import { svgEl } from './svg_layer';

export interface HybridFrame {
  pngBytes: Uint8Array;
  extentStart: number;
  extentEnd: number;
  widthPx: number;
  heightPx: number;
  nItems: number;
}

export function decodeHybrid(table: Table): HybridFrame | null {
  if (table.numRows === 0) return null;
  const row = table.get(0);
  if (!row) return null;
  return {
    pngBytes: row.png_bytes as Uint8Array,
    extentStart: Number(row.extent_start),
    extentEnd: Number(row.extent_end),
    widthPx: Number(row.width_px),
    heightPx: Number(row.height_px),
    nItems: Number(row.n_items),
  };
}

export function appendHybridImage(
  parent: SVGSVGElement,
  frame: HybridFrame,
  options: { x?: number; y?: number } = {},
): SVGImageElement {
  const x = options.x ?? 0;
  const y = options.y ?? 0;
  const dataUri = pngToDataUri(frame.pngBytes);
  const img = svgEl('image', {
    x,
    y,
    width: frame.widthPx,
    height: frame.heightPx,
    href: dataUri,
    preserveAspectRatio: 'none',
  });
  parent.appendChild(img);
  return img;
}

function pngToDataUri(bytes: Uint8Array): string {
  // btoa works on binary strings; build a binary string from the bytes.
  let s = '';
  for (let i = 0; i < bytes.length; i++) {
    s += String.fromCharCode(bytes[i]);
  }
  return `data:image/png;base64,${btoa(s)}`;
}
