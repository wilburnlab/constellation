// Pan, zoom, brush, click and hover on one continuous axis — and, in
// attachPanZoom2D below, on two axes zoomed independently (a read matrix:
// columns across, reads down).
//
// Wheel zooms about the cursor (a mostly horizontal scroll pans), a drag
// pans, shift-drag selects a range to zoom to, double-click fits, and a
// press that barely moves is a click at a data value (a browser picks a
// scan with it). Hovering reports the data value under the cursor (for a
// crosshair). Everything is expressed through callbacks on intervals, so
// the controller knows nothing of what is drawn.
//
// Listeners live on the surface; a drag uses pointer capture, so nothing
// is registered on the window.

import { Interval, IntervalLimits, clampInterval, panBy, zoomAbout } from './interval';

export type PanZoomCause = 'wheel' | 'drag' | 'brush' | 'reset';

export interface PanZoom1DOptions {
  /** Element that receives the pointer and wheel events. */
  surface: HTMLElement;
  /** The data area's left edge and width in client coordinates. */
  plotRect(): { left: number; width: number };
  get(): Interval;
  set(next: Interval, cause: PanZoomCause): void;
  limits(): IntervalLimits;
  /** What double-click fits to; the limits when omitted. */
  fit?(): Interval;
  /** Zoom factor per wheel notch (default 1.2). */
  wheelStep?: number;
  onClick?(value: number, event: PointerEvent): void;
  onHover?(value: number | null): void;
  /** A brush in progress (`null` when it ends), for drawing feedback. */
  onBrush?(range: Interval | null): void;
  /** Presses (and double-clicks) on these targets are left alone: the
   *  host's own controls (buttons, resize handles, a widget with its own
   *  drag). Any other press is taken, which cancels its compatibility
   *  mouse events. The wheel zooms everywhere on the surface. */
  ignore?(target: EventTarget | null): boolean;
  /** Shift-drag selects a range to zoom to (default true); off, shift-drag pans. */
  brush?: boolean;
  /** Double-click fits (default true). */
  doubleClickFits?: boolean;
}

/** Movement, in pixels, below which a press is a click rather than a drag. */
export const CLICK_SLOP_PX = 4;

export function attachPanZoom1D(opts: PanZoom1DOptions): () => void {
  const { surface } = opts;
  const step = opts.wheelStep ?? 1.2;
  let press: {
    pointerId: number;
    x0: number;
    iv: Interval;
    brush: boolean;
    moved: boolean;
  } | null = null;

  const fraction = (clientX: number): number => {
    const rect = opts.plotRect();
    return rect.width > 0 ? (clientX - rect.left) / rect.width : 0.5;
  };
  const valueAt = (clientX: number, iv: Interval = opts.get()): number =>
    iv[0] + fraction(clientX) * (iv[1] - iv[0]);

  const onWheel = (e: WheelEvent): void => {
    e.preventDefault();
    const limits = opts.limits();
    if (!e.ctrlKey && Math.abs(e.deltaX) > Math.abs(e.deltaY)) {
      const width = Math.max(1, opts.plotRect().width);
      opts.set(panBy(opts.get(), e.deltaX / width, limits), 'wheel');
      return;
    }
    const factor = e.deltaY > 0 ? step : 1 / step;
    opts.set(zoomAbout(opts.get(), factor, fraction(e.clientX), limits), 'wheel');
  };

  const brushes = opts.brush ?? true;
  const onDown = (e: PointerEvent): void => {
    if (e.button !== 0 || opts.ignore?.(e.target)) return;
    press = { pointerId: e.pointerId, x0: e.clientX, iv: opts.get(), brush: brushes && e.shiftKey, moved: false };
    surface.setPointerCapture?.(e.pointerId);
    e.preventDefault();
  };

  const brushRange = (clientX: number): Interval => {
    if (!press) return opts.get();
    const a = valueAt(press.x0, press.iv);
    const b = valueAt(clientX, press.iv);
    return a < b ? [a, b] : [b, a];
  };

  const onMove = (e: PointerEvent): void => {
    if (!press) {
      opts.onHover?.(valueAt(e.clientX));
      return;
    }
    if (Math.abs(e.clientX - press.x0) >= CLICK_SLOP_PX) press.moved = true;
    if (!press.moved) return;
    if (press.brush) {
      opts.onBrush?.(brushRange(e.clientX));
      return;
    }
    const width = Math.max(1, opts.plotRect().width);
    const fractionMoved = -(e.clientX - press.x0) / width;
    opts.set(panBy(press.iv, fractionMoved, opts.limits()), 'drag');
  };

  const finish = (e: PointerEvent): void => {
    if (!press || e.pointerId !== press.pointerId) return;
    const { moved, brush, x0, iv } = press;
    const range = brush ? brushRange(e.clientX) : null;
    surface.releasePointerCapture?.(press.pointerId);
    press = null;
    if (!moved) {
      // Where the press began: the release can drift by a few pixels.
      opts.onClick?.(valueAt(x0, iv), e);
      return;
    }
    if (brush && range) {
      opts.onBrush?.(null);
      const limits = opts.limits();
      const span = Math.max(limits.minSpan, range[1] - range[0]);
      const mid = (range[0] + range[1]) / 2;
      opts.set(clampInterval([mid - span / 2, mid + span / 2], limits), 'brush');
    }
  };

  const onCancel = (e: PointerEvent): void => {
    if (!press || e.pointerId !== press.pointerId) return;
    press = null;
    opts.onBrush?.(null);
  };

  const onLeave = (): void => {
    if (!press) opts.onHover?.(null);
  };

  const onDouble = (e: MouseEvent): void => {
    if (opts.doubleClickFits === false || opts.ignore?.(e.target)) return;
    const limits = opts.limits();
    opts.set(opts.fit?.() ?? [limits.min, limits.max], 'reset');
  };

  surface.addEventListener('wheel', onWheel, { passive: false });
  surface.addEventListener('pointerdown', onDown);
  surface.addEventListener('pointermove', onMove);
  surface.addEventListener('pointerup', finish);
  surface.addEventListener('pointercancel', onCancel);
  surface.addEventListener('pointerleave', onLeave);
  surface.addEventListener('dblclick', onDouble);
  return () => {
    surface.removeEventListener('wheel', onWheel);
    surface.removeEventListener('pointerdown', onDown);
    surface.removeEventListener('pointermove', onMove);
    surface.removeEventListener('pointerup', finish);
    surface.removeEventListener('pointercancel', onCancel);
    surface.removeEventListener('pointerleave', onLeave);
    surface.removeEventListener('dblclick', onDouble);
  };
}

// ── two axes ──────────────────────────────────────────────────────────

/** The x and y intervals of a 2-D view. y grows downward (a row index). */
export interface View2D {
  x: Interval;
  y: Interval;
}

export type PanZoom2DCause = 'wheel' | 'drag' | 'reset';

export interface PanZoom2DOptions {
  surface: HTMLElement;
  /** The data area in client coordinates. */
  plotRect(): { left: number; top: number; width: number; height: number };
  get(): View2D;
  set(next: View2D, cause: PanZoom2DCause): void;
  limitsX(): IntervalLimits;
  limitsY(): IntervalLimits;
  /** What double-click fits to; both axes' limits when omitted. */
  fit?(): View2D;
  /** Zoom factor per wheel notch (default 1.2). */
  wheelStep?: number;
  onClick?(x: number, y: number, event: PointerEvent): void;
  /** The data point under the cursor, `null` when it leaves. */
  onHover?(at: { x: number; y: number } | null): void;
  /** As in {@link PanZoom1DOptions.ignore}. */
  ignore?(target: EventTarget | null): boolean;
  /** Double-click fits (default true). */
  doubleClickFits?: boolean;
}

/** Which axes one wheel event zooms or pans, and by how much. Pure, so
 *  the gesture grammar is tested without events:
 *
 *  - plain wheel: zoom x about the cursor;
 *  - shift + wheel: zoom y. Tested BEFORE a horizontal delta, because
 *    Chrome (and Windows generally) deliver shift + wheel as `deltaX`
 *    with `deltaY` 0 — read as a horizontal scroll it would pan;
 *  - ctrl (or meta) + wheel: zoom both (a trackpad pinch arrives so);
 *  - a mostly horizontal delta with no modifier: pan x (a trackpad's
 *    sideways swipe).
 */
export function wheelGesture(e: {
  deltaX: number;
  deltaY: number;
  shiftKey: boolean;
  ctrlKey: boolean;
  metaKey?: boolean;
}): { kind: 'zoom'; axes: 'x' | 'y' | 'both'; outward: boolean } | { kind: 'pan-x'; delta: number } {
  if (e.shiftKey) {
    const d = e.deltaY !== 0 ? e.deltaY : e.deltaX;
    return { kind: 'zoom', axes: 'y', outward: d > 0 };
  }
  if (e.ctrlKey || e.metaKey) return { kind: 'zoom', axes: 'both', outward: e.deltaY > 0 };
  if (Math.abs(e.deltaX) > Math.abs(e.deltaY)) return { kind: 'pan-x', delta: e.deltaX };
  return { kind: 'zoom', axes: 'x', outward: e.deltaY > 0 };
}

export function attachPanZoom2D(opts: PanZoom2DOptions): () => void {
  const { surface } = opts;
  const step = opts.wheelStep ?? 1.2;
  let press: { pointerId: number; x0: number; y0: number; view: View2D; moved: boolean } | null =
    null;

  const fractions = (clientX: number, clientY: number): [number, number] => {
    const r = opts.plotRect();
    return [
      r.width > 0 ? (clientX - r.left) / r.width : 0.5,
      r.height > 0 ? (clientY - r.top) / r.height : 0.5,
    ];
  };
  const valueAt = (clientX: number, clientY: number, view: View2D = opts.get()) => {
    const [fx, fy] = fractions(clientX, clientY);
    return {
      x: view.x[0] + fx * (view.x[1] - view.x[0]),
      y: view.y[0] + fy * (view.y[1] - view.y[0]),
    };
  };

  const onWheel = (e: WheelEvent): void => {
    e.preventDefault();
    const view = opts.get();
    const g = wheelGesture(e);
    if (g.kind === 'pan-x') {
      const width = Math.max(1, opts.plotRect().width);
      opts.set({ x: panBy(view.x, g.delta / width, opts.limitsX()), y: view.y }, 'wheel');
      return;
    }
    const factor = g.outward ? step : 1 / step;
    const [fx, fy] = fractions(e.clientX, e.clientY);
    const x = g.axes === 'y' ? view.x : zoomAbout(view.x, factor, fx, opts.limitsX());
    const y = g.axes === 'x' ? view.y : zoomAbout(view.y, factor, fy, opts.limitsY());
    opts.set({ x, y }, 'wheel');
  };

  const onDown = (e: PointerEvent): void => {
    if (e.button !== 0 || opts.ignore?.(e.target)) return;
    press = { pointerId: e.pointerId, x0: e.clientX, y0: e.clientY, view: opts.get(), moved: false };
    surface.setPointerCapture?.(e.pointerId);
    e.preventDefault();
  };

  const onMove = (e: PointerEvent): void => {
    if (!press) {
      opts.onHover?.(valueAt(e.clientX, e.clientY));
      return;
    }
    const dx = e.clientX - press.x0;
    const dy = e.clientY - press.y0;
    if (Math.hypot(dx, dy) >= CLICK_SLOP_PX) press.moved = true;
    if (!press.moved) return;
    const r = opts.plotRect();
    opts.set(
      {
        x: panBy(press.view.x, -dx / Math.max(1, r.width), opts.limitsX()),
        y: panBy(press.view.y, -dy / Math.max(1, r.height), opts.limitsY()),
      },
      'drag',
    );
  };

  const finish = (e: PointerEvent): void => {
    if (!press || e.pointerId !== press.pointerId) return;
    const { moved, x0, y0, view } = press;
    surface.releasePointerCapture?.(press.pointerId);
    press = null;
    if (!moved) {
      const at = valueAt(x0, y0, view);
      opts.onClick?.(at.x, at.y, e);
    }
  };

  const onCancel = (e: PointerEvent): void => {
    if (press && e.pointerId === press.pointerId) press = null;
  };
  const onLeave = (): void => {
    if (!press) opts.onHover?.(null);
  };
  const onDouble = (e: MouseEvent): void => {
    if (opts.doubleClickFits === false || opts.ignore?.(e.target)) return;
    const lx = opts.limitsX();
    const ly = opts.limitsY();
    opts.set(opts.fit?.() ?? { x: [lx.min, lx.max], y: [ly.min, ly.max] }, 'reset');
  };

  surface.addEventListener('wheel', onWheel, { passive: false });
  surface.addEventListener('pointerdown', onDown);
  surface.addEventListener('pointermove', onMove);
  surface.addEventListener('pointerup', finish);
  surface.addEventListener('pointercancel', onCancel);
  surface.addEventListener('pointerleave', onLeave);
  surface.addEventListener('dblclick', onDouble);
  return () => {
    surface.removeEventListener('wheel', onWheel);
    surface.removeEventListener('pointerdown', onDown);
    surface.removeEventListener('pointermove', onMove);
    surface.removeEventListener('pointerup', finish);
    surface.removeEventListener('pointercancel', onCancel);
    surface.removeEventListener('pointerleave', onLeave);
    surface.removeEventListener('dblclick', onDouble);
  };
}
