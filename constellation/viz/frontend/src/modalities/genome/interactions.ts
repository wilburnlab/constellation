// Pan + zoom for the genome browser, on the engine's 1D controller.
//
// A locus is an interval on one contig in whole base pairs, so the
// genome's limits are `[0, contig length]` with ends rounded and a minimum
// window. Wheel zooms about the cursor, a mostly horizontal scroll pans,
// and a drag on the tracks pans. Presses on the browser's own controls
// (track headers, resize handles, the overview bar, form fields) are left
// to them. The engine's brush and double-click fit are off, so the genome
// browser behaves as it always has.

import type { Interval, IntervalLimits } from '../../engine/interval';
import { zoomAbout } from '../../engine/interval';
import { attachPanZoom1D } from '../../engine/pan_zoom';
import { Locus, ViewportBus } from './viewport_bus';

export const ZOOM_STEP = 1.2;
export const MIN_WINDOW_BP = 20;

/** Elements whose presses belong to the browser's own controls. */
const OWN_CONTROLS = '.overview, .track-header, .track-resize-handle, button, input, select, textarea, a';

function limitsFor(contigLength: number): IntervalLimits {
  return { min: 0, max: contigLength, minSpan: MIN_WINDOW_BP, round: Math.round };
}

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
  const [start, end] = zoomAbout([locus.start, locus.end], factor, anchorFraction, limitsFor(contigLength));
  return { contig: locus.contig, start, end };
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
  const { bus, surface } = opts;
  return attachPanZoom1D({
    surface,
    plotRect: () => ({ left: surface.getBoundingClientRect().left, width: Math.max(1, opts.getWidthPx()) }),
    get: () => [bus.locus.start, bus.locus.end],
    set: ([start, end]: Interval) => bus.setLocus({ contig: bus.locus.contig, start, end }),
    limits: () => limitsFor(opts.getContigLength()),
    wheelStep: ZOOM_STEP,
    ignore: (target) => target instanceof Element && target.closest(OWN_CONTROLS) !== null,
    brush: false,
    doubleClickFits: false,
  });
}
