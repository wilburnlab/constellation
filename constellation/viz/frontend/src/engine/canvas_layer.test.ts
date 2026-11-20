import { afterEach, describe, expect, it } from 'vitest';
import { createCanvasLayer } from './canvas_layer';
import { EXPORT_GLYPH_BUDGET, layerSvg } from './export';

/** A context that records what was asked of it. */
function fakeContext() {
  const calls: string[] = [];
  const ctx = {
    setTransform: (...a: number[]) => calls.push(`setTransform(${a.join(',')})`),
    clearRect: () => calls.push('clearRect'),
  } as unknown as CanvasRenderingContext2D;
  return { ctx, calls };
}

let host: HTMLElement | null = null;
afterEach(() => {
  host?.remove();
  host = null;
});

describe('createCanvasLayer', () => {
  it('backs the canvas at the device pixel ratio and draws in CSS pixels', () => {
    host = document.createElement('div');
    document.body.appendChild(host);
    const { ctx, calls } = fakeContext();
    const sizes: number[][] = [];
    const layer = createCanvasLayer(host, (_c, s) => sizes.push([s.widthPx, s.heightPx, s.dpr]), {
      dpr: 2,
      context: () => ctx,
    });
    layer.resize(300.7, 120);
    expect([layer.canvas.width, layer.canvas.height]).toEqual([600, 240]);
    expect(layer.canvas.style.width).toBe('300px');
    expect(layer.overlay.getAttribute('height')).toBe('120');
    expect(sizes).toEqual([[300, 120, 2]]);
    expect(calls).toContain('setTransform(2,0,0,2,0,0)');
  });

  it('coalesces redraw requests to one per frame, and dispose cancels', () => {
    host = document.createElement('div');
    const queue: Array<() => void> = [];
    let draws = 0;
    const layer = createCanvasLayer(host, () => (draws += 1), {
      dpr: 1,
      context: () => fakeContext().ctx,
      schedule: (fn) => queue.push(fn),
      cancel: (h) => {
        queue[h - 1] = () => undefined;
      },
    });
    layer.resize(10, 10);
    expect(draws).toBe(1);
    layer.requestDraw();
    layer.requestDraw();
    layer.requestDraw();
    expect(queue).toHaveLength(1);
    queue.shift()!();
    expect(draws).toBe(2);
    layer.requestDraw();
    layer.dispose();
    queue.forEach((fn) => fn());
    expect(draws).toBe(2);
    expect(host.contains(layer.root)).toBe(false);
  });

  it('draws nothing without a context (jsdom) and nothing at zero size', () => {
    host = document.createElement('div');
    let draws = 0;
    const layer = createCanvasLayer(host, () => (draws += 1), { dpr: 1 });
    layer.resize(10, 10);
    const sized = createCanvasLayer(host, () => (draws += 1), { dpr: 1, context: () => fakeContext().ctx });
    sized.resize(0, 10);
    expect(draws).toBe(0);
  });
});

describe('layerSvg', () => {
  it('re-renders as vectors under the glyph budget', () => {
    let drawn = 0;
    const { svg, vector } = layerSvg({
      widthPx: 100,
      heightPx: 50,
      glyphs: 10,
      vector: (g) => {
        drawn += 1;
        g.appendChild(document.createElementNS('http://www.w3.org/2000/svg', 'rect'));
      },
      raster: () => 'data:image/png;base64,AAAA',
    });
    expect(vector).toBe(true);
    expect(drawn).toBe(1);
    expect(svg.querySelectorAll('rect')).toHaveLength(1);
    expect(svg.querySelector('image')).toBeNull();
  });

  it('embeds the canvas, sized in CSS pixels, and says so, over the budget', () => {
    let drawn = 0;
    const { svg, vector } = layerSvg({
      widthPx: 100,
      heightPx: 50,
      glyphs: EXPORT_GLYPH_BUDGET + 1,
      vector: () => (drawn += 1),
      raster: () => 'data:image/png;base64,AAAA',
    });
    expect(vector).toBe(false);
    expect(drawn).toBe(0);
    const img = svg.querySelector('image')!;
    expect([img.getAttribute('width'), img.getAttribute('height')]).toEqual(['100', '50']);
    expect(svg.querySelector('desc')!.textContent).toMatch(/Raster: .* over the 50,000 budget/);
  });
});
