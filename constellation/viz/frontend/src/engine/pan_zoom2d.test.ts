import { afterEach, describe, expect, it } from 'vitest';
import type { IntervalLimits } from './interval';
import { attachPanZoom2D, PanZoom2DCause, View2D, wheelGesture } from './pan_zoom';

const LX: IntervalLimits = { min: 0, max: 1000, minSpan: 1 };
const LY: IntervalLimits = { min: 0, max: 100, minSpan: 1 };

function pointer(type: string, clientX: number, clientY: number): Event {
  const Ctor = (globalThis as { PointerEvent?: typeof MouseEvent }).PointerEvent ?? MouseEvent;
  const event = new Ctor(type, { bubbles: true, cancelable: true, clientX, clientY, button: 0 });
  if ((event as PointerEvent).pointerId === undefined) Object.defineProperty(event, 'pointerId', { value: 1 });
  return event;
}

function wheel(init: WheelEventInit): WheelEvent {
  return new WheelEvent('wheel', { bubbles: true, cancelable: true, clientX: 200, clientY: 100, ...init });
}

interface Rig {
  surface: HTMLElement;
  view: View2D;
  sets: Array<{ view: View2D; cause: PanZoom2DCause }>;
  clicks: Array<[number, number]>;
  hovers: Array<{ x: number; y: number } | null>;
  detach: () => void;
}

/** A 200 x 100 px data area at client (100, 50), showing x [0, 1000], y [0, 100]. */
function rig(): Rig {
  const surface = document.createElement('div');
  document.body.appendChild(surface);
  const r = { surface, view: { x: [0, 1000], y: [0, 100] }, sets: [], clicks: [], hovers: [] } as unknown as Rig;
  r.detach = attachPanZoom2D({
    surface,
    plotRect: () => ({ left: 100, top: 50, width: 200, height: 100 }),
    get: () => r.view,
    set: (view, cause) => {
      r.view = view;
      r.sets.push({ view, cause });
    },
    limitsX: () => LX,
    limitsY: () => LY,
    onClick: (x, y) => r.clicks.push([x, y]),
    onHover: (at) => r.hovers.push(at),
  });
  return r;
}

let current: Rig | null = null;
afterEach(() => {
  current?.detach();
  current?.surface.remove();
  current = null;
});

describe('wheelGesture', () => {
  it('reads shift + wheel as a y zoom before a horizontal delta (Chrome sends it as deltaX)', () => {
    expect(wheelGesture({ deltaX: 100, deltaY: 0, shiftKey: true, ctrlKey: false })).toEqual({
      kind: 'zoom',
      axes: 'y',
      outward: true,
    });
    expect(wheelGesture({ deltaX: 0, deltaY: -100, shiftKey: true, ctrlKey: false })).toEqual({
      kind: 'zoom',
      axes: 'y',
      outward: false,
    });
  });

  it('plain wheel zooms x, ctrl zooms both, a sideways swipe pans x', () => {
    expect(wheelGesture({ deltaX: 0, deltaY: 100, shiftKey: false, ctrlKey: false })).toEqual({
      kind: 'zoom',
      axes: 'x',
      outward: true,
    });
    expect(wheelGesture({ deltaX: 0, deltaY: -5, shiftKey: false, ctrlKey: true })).toEqual({
      kind: 'zoom',
      axes: 'both',
      outward: false,
    });
    expect(wheelGesture({ deltaX: 30, deltaY: 2, shiftKey: false, ctrlKey: false })).toEqual({
      kind: 'pan-x',
      delta: 30,
    });
  });
});

describe('attachPanZoom2D', () => {
  it('the wheel zooms x about the cursor and leaves y alone', () => {
    const r = (current = rig());
    r.surface.dispatchEvent(wheel({ deltaY: -100 }));
    expect(r.view.y).toEqual([0, 100]);
    expect(r.view.x[1] - r.view.x[0]).toBeCloseTo(1000 / 1.2);
    expect((r.view.x[0] + r.view.x[1]) / 2).toBeCloseTo(500);
  });

  it('shift + wheel zooms y about the cursor and leaves x alone', () => {
    const r = (current = rig());
    // Cursor at the data area's vertical middle (client y 100 = 50 rows).
    r.surface.dispatchEvent(wheel({ deltaX: -100, deltaY: 0, shiftKey: true }));
    expect(r.view.x).toEqual([0, 1000]);
    expect(r.view.y[1] - r.view.y[0]).toBeCloseTo(100 / 1.2);
    expect((r.view.y[0] + r.view.y[1]) / 2).toBeCloseTo(50);
  });

  it('ctrl + wheel zooms both axes', () => {
    const r = (current = rig());
    r.surface.dispatchEvent(wheel({ deltaY: -100, ctrlKey: true }));
    expect(r.view.x[1] - r.view.x[0]).toBeCloseTo(1000 / 1.2);
    expect(r.view.y[1] - r.view.y[0]).toBeCloseTo(100 / 1.2);
  });

  it('a drag pans both axes by the distance moved', () => {
    const r = (current = rig());
    r.view = { x: [200, 400], y: [20, 40] };
    r.surface.dispatchEvent(pointer('pointerdown', 200, 100));
    r.surface.dispatchEvent(pointer('pointermove', 150, 75)); // 50 px left, 25 px up
    r.surface.dispatchEvent(pointer('pointerup', 150, 75));
    const last = r.sets.at(-1)!;
    expect(last.cause).toBe('drag');
    expect(last.view.x).toEqual([250, 450]); // +quarter of the 200 span
    expect(last.view.y).toEqual([25, 45]); // +quarter of the 20 span
    expect(r.clicks).toEqual([]);
  });

  it('a press that moves less than the slop in both axes together is a click', () => {
    const r = (current = rig());
    r.surface.dispatchEvent(pointer('pointerdown', 150, 60));
    r.surface.dispatchEvent(pointer('pointermove', 152, 62)); // hypot 2.8 < 4
    r.surface.dispatchEvent(pointer('pointerup', 152, 62));
    expect(r.clicks).toEqual([[250, 10]]);
    expect(r.sets).toEqual([]);
    // 3 px each way is under the slop per axis but not together.
    r.surface.dispatchEvent(pointer('pointerdown', 150, 60));
    r.surface.dispatchEvent(pointer('pointermove', 153, 63));
    r.surface.dispatchEvent(pointer('pointerup', 153, 63));
    expect(r.clicks).toHaveLength(1);
    expect(r.sets.at(-1)?.cause).toBe('drag');
  });

  it('hover reports both values, and null on leave', () => {
    const r = (current = rig());
    r.surface.dispatchEvent(pointer('pointermove', 300, 150));
    r.surface.dispatchEvent(pointer('pointerleave', 300, 150));
    expect(r.hovers).toEqual([{ x: 1000, y: 100 }, null]);
  });

  it('double-click fits both axes; detaching removes every listener', () => {
    const r = (current = rig());
    r.view = { x: [10, 20], y: [5, 6] };
    r.surface.dispatchEvent(new MouseEvent('dblclick', { bubbles: true }));
    expect(r.view).toEqual({ x: [0, 1000], y: [0, 100] });
    r.detach();
    r.sets = [];
    r.surface.dispatchEvent(new MouseEvent('dblclick', { bubbles: true }));
    r.surface.dispatchEvent(wheel({ deltaY: -100 }));
    expect(r.sets).toEqual([]);
  });

  it('zooms clamp to each axis own limits', () => {
    const r = (current = rig());
    for (let i = 0; i < 40; i += 1) r.surface.dispatchEvent(wheel({ deltaY: 100, shiftKey: true }));
    expect(r.view.y).toEqual([0, 100]);
    for (let i = 0; i < 80; i += 1) r.surface.dispatchEvent(wheel({ deltaY: -100, shiftKey: true }));
    expect(r.view.y[1] - r.view.y[0]).toBeCloseTo(1);
  });
});
