// A raster layer: a HiDPI canvas under an SVG overlay.
//
// For views that draw more than SVG can hold — a read matrix of 20,000
// reads — the bulk goes on a canvas and the decorations a user points at
// (a hover crosshair, a selected column) on the SVG above it. The canvas
// is sized in CSS pixels and backed at the device pixel ratio, so lines
// stay one physical pixel on a HiDPI screen. Redraws are coalesced to one
// per animation frame; the host calls `resize` from its own
// ResizeObserver, as every browser here already watches its pane.
//
// Export (engine/export.ts, `rasterOrVectorSvg`) re-renders the layer as
// vectors when it is under a glyph budget and embeds the canvas as an
// image otherwise.

import { svgEl } from './svg_layer';

export interface CanvasLayerSize {
  widthPx: number;
  heightPx: number;
  /** Device pixels per CSS pixel the canvas is backed at. */
  dpr: number;
}

export interface CanvasLayer {
  /** Positioned container; the host places it. */
  readonly root: HTMLDivElement;
  readonly canvas: HTMLCanvasElement;
  readonly overlay: SVGSVGElement;
  size(): CanvasLayerSize;
  /** Size in CSS pixels; redraws. */
  resize(widthPx: number, heightPx: number): void;
  /** Redraw on the next animation frame (several calls, one draw). */
  requestDraw(): void;
  /** Redraw now. */
  drawNow(): void;
  dispose(): void;
}

export interface CanvasLayerOptions {
  /** Device pixel ratio; `window.devicePixelRatio` when omitted. */
  dpr?: number;
  /** The 2-D context; `canvas.getContext('2d')` when omitted (tests pass a
   *  recording one, jsdom has none). */
  context?: (canvas: HTMLCanvasElement) => CanvasRenderingContext2D | null;
  /** Frame scheduler; `requestAnimationFrame` when omitted. */
  schedule?: (fn: () => void) => number;
  cancel?: (handle: number) => void;
}

export function createCanvasLayer(
  host: HTMLElement,
  draw: (ctx: CanvasRenderingContext2D, size: CanvasLayerSize) => void,
  opts: CanvasLayerOptions = {},
): CanvasLayer {
  const root = document.createElement('div');
  root.className = 'canvas-layer';
  root.style.position = 'relative';
  const canvas = document.createElement('canvas');
  canvas.className = 'canvas-layer-canvas';
  canvas.style.position = 'absolute';
  canvas.style.left = '0';
  canvas.style.top = '0';
  const overlay = svgEl('svg', { class: 'canvas-layer-overlay' }) as SVGSVGElement;
  overlay.style.position = 'absolute';
  overlay.style.left = '0';
  overlay.style.top = '0';
  overlay.style.pointerEvents = 'none';
  root.append(canvas, overlay);
  host.appendChild(root);

  const getContext =
    opts.context ?? ((c: HTMLCanvasElement) => (typeof c.getContext === 'function' ? safeContext(c) : null));
  const schedule =
    opts.schedule ??
    ((fn: () => void) =>
      typeof requestAnimationFrame === 'function'
        ? requestAnimationFrame(fn)
        : (setTimeout(fn, 0) as unknown as number));
  const cancel =
    opts.cancel ??
    ((h: number) =>
      typeof cancelAnimationFrame === 'function' ? cancelAnimationFrame(h) : clearTimeout(h));
  let size: CanvasLayerSize = { widthPx: 0, heightPx: 0, dpr: 1 };
  let pending: number | null = null;
  let disposed = false;

  const drawNow = (): void => {
    if (pending !== null) {
      cancel(pending);
      pending = null;
    }
    if (disposed || size.widthPx <= 0 || size.heightPx <= 0) return;
    const ctx = getContext(canvas);
    if (!ctx) return;
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.setTransform(size.dpr, 0, 0, size.dpr, 0, 0);
    draw(ctx, size);
  };

  return {
    root,
    canvas,
    overlay,
    size: () => size,
    resize(widthPx: number, heightPx: number): void {
      const dpr = opts.dpr ?? (typeof window !== 'undefined' ? window.devicePixelRatio || 1 : 1);
      const w = Math.max(0, Math.floor(widthPx));
      const h = Math.max(0, Math.floor(heightPx));
      size = { widthPx: w, heightPx: h, dpr };
      canvas.width = Math.round(w * dpr);
      canvas.height = Math.round(h * dpr);
      canvas.style.width = `${w}px`;
      canvas.style.height = `${h}px`;
      root.style.width = `${w}px`;
      root.style.height = `${h}px`;
      overlay.setAttribute('width', String(w));
      overlay.setAttribute('height', String(h));
      drawNow();
    },
    requestDraw(): void {
      if (pending !== null || disposed) return;
      pending = schedule(() => {
        pending = null;
        drawNow();
      });
    },
    drawNow,
    dispose(): void {
      disposed = true;
      if (pending !== null) cancel(pending);
      pending = null;
      root.remove();
    },
  };
}

/** jsdom implements `getContext` as a stub that logs "not implemented"
 *  and returns null; a real browser returns the context. */
function safeContext(canvas: HTMLCanvasElement): CanvasRenderingContext2D | null {
  try {
    return canvas.getContext('2d');
  } catch {
    return null;
  }
}
