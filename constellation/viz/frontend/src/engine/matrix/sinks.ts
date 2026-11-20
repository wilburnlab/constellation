// The two places a matrix is drawn: a canvas on screen, and SVG elements
// for an export that stays vector.

import { svgEl } from '../svg_layer';
import type { MatrixSink } from './draw';

export function canvasSink(ctx: CanvasRenderingContext2D): MatrixSink {
  return {
    rect(x, y, w, h, fill, alpha) {
      ctx.globalAlpha = alpha;
      ctx.fillStyle = fill;
      ctx.fillRect(x, y, w, h);
      ctx.globalAlpha = 1;
    },
    text(x, y, s, fill, alpha, fontPx) {
      ctx.globalAlpha = alpha;
      ctx.fillStyle = fill;
      ctx.font = `${fontPx}px ui-monospace, monospace`;
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText(s, x, y);
      ctx.globalAlpha = 1;
    },
    image(rgba, w, h, x, y, dw, dh) {
      const off = document.createElement('canvas');
      off.width = w;
      off.height = h;
      const octx = off.getContext('2d');
      if (!octx) return;
      octx.putImageData(new ImageData(rgba as Uint8ClampedArray<ArrayBuffer>, w, h), 0, 0);
      ctx.imageSmoothingEnabled = false;
      ctx.drawImage(off, x, y, dw, dh);
    },
  };
}

/** Writes into `g`. An image needs a canvas to encode it, so where there
 *  is none (jsdom) it is counted in `skippedImages` instead. */
export function svgSink(g: SVGGElement): MatrixSink & { skippedImages: number } {
  const sink = {
    skippedImages: 0,
    rect(x: number, y: number, w: number, h: number, fill: string, alpha: number) {
      const attrs: Record<string, string | number> = { x: r2(x), y: r2(y), width: r2(w), height: r2(h), fill };
      if (alpha < 1) attrs['fill-opacity'] = r2(alpha);
      g.appendChild(svgEl('rect', attrs));
    },
    text(x: number, y: number, s: string, fill: string, alpha: number, fontPx: number) {
      const attrs: Record<string, string | number> = {
        x: r2(x),
        y: r2(y),
        fill,
        'font-size': r2(fontPx),
        'font-family': 'ui-monospace, monospace',
        'text-anchor': 'middle',
        'dominant-baseline': 'central',
      };
      if (alpha < 1) attrs['fill-opacity'] = r2(alpha);
      const t = svgEl('text', attrs);
      t.textContent = s;
      g.appendChild(t);
    },
    image(rgba: Uint8ClampedArray, w: number, h: number, x: number, y: number, dw: number, dh: number) {
      const href = rgbaToDataUrl(rgba, w, h);
      if (href === null) {
        sink.skippedImages += 1;
        return;
      }
      g.appendChild(
        svgEl('image', {
          x: r2(x),
          y: r2(y),
          width: r2(dw),
          height: r2(dh),
          href,
          preserveAspectRatio: 'none',
          style: 'image-rendering:pixelated',
        }),
      );
    },
  };
  return sink;
}

function r2(v: number): number {
  return Math.round(v * 100) / 100;
}

/** RGBA pixels as a PNG data URL, or null where there is no canvas. */
export function rgbaToDataUrl(rgba: Uint8ClampedArray, w: number, h: number): string | null {
  try {
    const c = document.createElement('canvas');
    c.width = w;
    c.height = h;
    const ctx = c.getContext('2d');
    if (!ctx || typeof ImageData === 'undefined') return null;
    ctx.putImageData(new ImageData(rgba as Uint8ClampedArray<ArrayBuffer>, w, h), 0, 0);
    return c.toDataURL('image/png');
  } catch {
    return null;
  }
}
