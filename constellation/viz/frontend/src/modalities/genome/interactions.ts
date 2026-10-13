// Pan + zoom interactions for the genome browser.
//
// Wheel-scroll zooms; drag pans. We don't use d3-zoom directly because
// d3-zoom's transform model is geared toward an SVG-transform-based
// pan/zoom — but our tracks re-render against a new `Locus` rather
// than transforming a single SVG. So we hand-roll a small wheel/drag
// listener that emits new `Locus` values into the bus.

import { Locus, ViewportBus } from './viewport_bus';

export const ZOOM_STEP = 1.2;
export const MIN_WINDOW_BP = 20;

/**
 * Apply a zoom factor to a locus around a fractional anchor.
 *
 * `factor > 1` zooms out (span grows); `factor < 1` zooms in (span
 * shrinks). `anchorFraction` is the [0..1] position the user wants
 * fixed under the cursor (0.5 = window center). Returned locus is
 * already clamped to `[0, contigLength]` with span preserved.
 */
export function zoomLocus(
  locus: Locus,
  contigLength: number,
  factor: number,
  anchorFraction: number,
): Locus {
  const span = locus.end - locus.start;
  const newSpan = Math.max(MIN_WINDOW_BP, Math.round(span * factor));
  const frac = Math.min(1, Math.max(0, anchorFraction));
  const anchorBp = locus.start + span * frac;
  const newStart = Math.round(anchorBp - newSpan * frac);
  return clamp(
    {
      contig: locus.contig,
      start: newStart,
      end: newStart + newSpan,
    },
    contigLength,
  );
}

export interface PanZoomOptions {
  bus: ViewportBus;
  /** Element to attach listeners to (typically the browser frame). */
  surface: HTMLElement;
  /** Pixel width of the rendered track area; used to map drag-px → bp. */
  getWidthPx: () => number;
  /** Maximum allowed window (i.e. contig length). */
  getContigLength: () => number;
}

export function attachPanZoom(opts: PanZoomOptions): () => void {
  const { bus, surface, getWidthPx, getContigLength } = opts;

  let dragStart: { clientX: number; locus: Locus } | null = null;

  const onWheel = (event: WheelEvent) => {
    if (!event.ctrlKey && !event.shiftKey && Math.abs(event.deltaX) > Math.abs(event.deltaY)) {
      // Horizontal scroll = pan
      const widthPx = Math.max(1, getWidthPx());
      const locus = bus.locus;
      const span = locus.end - locus.start;
      const dxBp = (event.deltaX / widthPx) * span;
      bus.setLocus(clamp({
        contig: locus.contig,
        start: Math.round(locus.start + dxBp),
        end: Math.round(locus.end + dxBp),
      }, getContigLength()));
      event.preventDefault();
      return;
    }
    // Default: vertical scroll = zoom
    event.preventDefault();
    const factor = event.deltaY > 0 ? ZOOM_STEP : 1 / ZOOM_STEP;
    const widthPx = Math.max(1, getWidthPx());
    const rect = surface.getBoundingClientRect();
    const cursorPx = event.clientX - rect.left;
    const cursorFraction = widthPx > 0 ? cursorPx / widthPx : 0.5;
    bus.setLocus(zoomLocus(bus.locus, getContigLength(), factor, cursorFraction));
  };

  const onMouseDown = (event: MouseEvent) => {
    if (event.button !== 0) return;
    dragStart = { clientX: event.clientX, locus: bus.locus };
  };
  const onMouseMove = (event: MouseEvent) => {
    if (!dragStart) return;
    const widthPx = Math.max(1, getWidthPx());
    const span = dragStart.locus.end - dragStart.locus.start;
    const dxPx = event.clientX - dragStart.clientX;
    const dxBp = -(dxPx / widthPx) * span;
    bus.setLocus(clamp({
      contig: dragStart.locus.contig,
      start: Math.round(dragStart.locus.start + dxBp),
      end: Math.round(dragStart.locus.end + dxBp),
    }, getContigLength()));
  };
  const onMouseUp = () => {
    dragStart = null;
  };

  surface.addEventListener('wheel', onWheel, { passive: false });
  surface.addEventListener('mousedown', onMouseDown);
  window.addEventListener('mousemove', onMouseMove);
  window.addEventListener('mouseup', onMouseUp);

  return () => {
    surface.removeEventListener('wheel', onWheel);
    surface.removeEventListener('mousedown', onMouseDown);
    window.removeEventListener('mousemove', onMouseMove);
    window.removeEventListener('mouseup', onMouseUp);
  };
}

function clamp(locus: Locus, contigLength: number): Locus {
  const span = locus.end - locus.start;
  if (span >= contigLength) {
    return { contig: locus.contig, start: 0, end: contigLength };
  }
  if (locus.start < 0) {
    return { contig: locus.contig, start: 0, end: span };
  }
  if (locus.end > contigLength) {
    return {
      contig: locus.contig,
      start: contigLength - span,
      end: contigLength,
    };
  }
  return locus;
}
