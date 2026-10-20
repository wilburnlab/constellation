import { afterEach, describe, expect, it } from 'vitest';
import type { Interval, IntervalLimits } from './interval';
import { attachPanZoom1D, PanZoomCause } from './pan_zoom';

const LIMITS: IntervalLimits = { min: 0, max: 1000, minSpan: 1 };

function pointer(type: string, clientX: number, extra: { shiftKey?: boolean } = {}): Event {
  const Ctor = (globalThis as { PointerEvent?: typeof MouseEvent }).PointerEvent ?? MouseEvent;
  const event = new Ctor(type, { bubbles: true, cancelable: true, clientX, button: 0, ...extra });
  if ((event as PointerEvent).pointerId === undefined) Object.defineProperty(event, 'pointerId', { value: 1 });
  return event;
}

interface Rig {
  surface: HTMLElement;
  iv: Interval;
  sets: Array<{ iv: Interval; cause: PanZoomCause }>;
  clicks: number[];
  hovers: Array<number | null>;
  brushes: Array<Interval | null>;
  detach: () => void;
}

/** A 200 px data area starting at client x = 100, showing [0, 1000]. */
function rig(): Rig {
  const surface = document.createElement('div');
  document.body.appendChild(surface);
  const r = { surface, iv: [0, 1000] as Interval, sets: [], clicks: [], hovers: [], brushes: [] } as unknown as Rig;
  r.detach = attachPanZoom1D({
    surface,
    plotRect: () => ({ left: 100, width: 200 }),
    get: () => r.iv,
    set: (iv, cause) => {
      r.iv = iv;
      r.sets.push({ iv, cause });
    },
    limits: () => LIMITS,
    onClick: (v) => r.clicks.push(v),
    onHover: (v) => r.hovers.push(v),
    onBrush: (b) => r.brushes.push(b),
  });
  return r;
}

let current: Rig | null = null;
afterEach(() => {
  current?.detach();
  current?.surface.remove();
  current = null;
});

describe('attachPanZoom1D', () => {
  it('a press that does not move is a click at the data value', () => {
    const r = (current = rig());
    r.surface.dispatchEvent(pointer('pointerdown', 150));
    r.surface.dispatchEvent(pointer('pointerup', 151));
    expect(r.clicks).toEqual([250]);
    expect(r.sets).toEqual([]);
  });

  it('a drag pans by the distance moved', () => {
    const r = (current = rig());
    r.iv = [200, 400];
    r.surface.dispatchEvent(pointer('pointerdown', 200));
    r.surface.dispatchEvent(pointer('pointermove', 150)); // 50 px left = +quarter span
    r.surface.dispatchEvent(pointer('pointerup', 150));
    expect(r.sets.at(-1)).toEqual({ iv: [250, 450], cause: 'drag' });
    expect(r.clicks).toEqual([]);
  });

  it('shift-drag zooms to the range swept', () => {
    const r = (current = rig());
    r.surface.dispatchEvent(pointer('pointerdown', 120, { shiftKey: true }));
    r.surface.dispatchEvent(pointer('pointermove', 180, { shiftKey: true }));
    expect(r.brushes.at(-1)).toEqual([100, 400]);
    r.surface.dispatchEvent(pointer('pointerup', 180, { shiftKey: true }));
    expect(r.sets.at(-1)).toEqual({ iv: [100, 400], cause: 'brush' });
    expect(r.brushes.at(-1)).toBeNull();
  });

  it('the wheel zooms about the cursor', () => {
    const r = (current = rig());
    r.surface.dispatchEvent(new WheelEvent('wheel', { deltaY: -100, clientX: 200, bubbles: true, cancelable: true }));
    const [lo, hi] = r.iv;
    expect(hi - lo).toBeCloseTo(1000 / 1.2);
    expect((lo + hi) / 2).toBeCloseTo(500); // cursor at the middle stays there
  });

  it('hover reports the value under the cursor, and null on leave', () => {
    const r = (current = rig());
    r.surface.dispatchEvent(pointer('pointermove', 300));
    r.surface.dispatchEvent(pointer('pointerleave', 300));
    expect(r.hovers).toEqual([1000, null]);
  });

  it('double-click fits and detaching removes every listener', () => {
    const r = (current = rig());
    r.iv = [10, 20];
    r.surface.dispatchEvent(new MouseEvent('dblclick', { bubbles: true }));
    expect(r.iv).toEqual([0, 1000]);
    r.detach();
    r.sets = [];
    r.surface.dispatchEvent(new MouseEvent('dblclick', { bubbles: true }));
    expect(r.sets).toEqual([]);
  });
});

describe('options', () => {
  it('leaves presses on ignored targets alone, and can turn off brush and fit', () => {
    const surface = document.createElement('div');
    const button = document.createElement('button');
    surface.appendChild(button);
    document.body.appendChild(surface);
    let iv: Interval = [0, 1000];
    const sets: Interval[] = [];
    const detach = attachPanZoom1D({
      surface,
      plotRect: () => ({ left: 100, width: 200 }),
      get: () => iv,
      set: (next) => {
        iv = next;
        sets.push(next);
      },
      limits: () => LIMITS,
      ignore: (t) => t instanceof HTMLButtonElement,
      brush: false,
      doubleClickFits: false,
    });
    button.dispatchEvent(pointer('pointerdown', 150));
    button.dispatchEvent(pointer('pointermove', 100));
    expect(sets).toEqual([]);
    // Without the brush, shift-drag pans.
    iv = [200, 400];
    surface.dispatchEvent(pointer('pointerdown', 200, { shiftKey: true }));
    surface.dispatchEvent(pointer('pointermove', 150, { shiftKey: true }));
    surface.dispatchEvent(pointer('pointerup', 150, { shiftKey: true }));
    expect(iv).toEqual([250, 450]);
    surface.dispatchEvent(new MouseEvent('dblclick', { bubbles: true }));
    expect(iv).toEqual([250, 450]);
    detach();
    surface.remove();
  });
});
