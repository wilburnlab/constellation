import { describe, expect, it } from 'vitest';
import { MIN_WINDOW_BP, ZOOM_STEP, zoomLocus } from './interactions';

const CONTIG = 'chr1';
const LEN = 10_000;

describe('zoomLocus', () => {
  it('zooms in around the window centre', () => {
    const out = zoomLocus({ contig: CONTIG, start: 4000, end: 6000 }, LEN, 0.5, 0.5);
    expect(out).toEqual({ contig: CONTIG, start: 4500, end: 5500 });
  });

  it('keeps the base under the anchor fixed', () => {
    // Anchor at 25% of a 2 kb window = bp 4500; it stays at 25% afterwards.
    const out = zoomLocus({ contig: CONTIG, start: 4000, end: 6000 }, LEN, 0.5, 0.25);
    expect(out).toEqual({ contig: CONTIG, start: 4250, end: 5250 });
    expect(out.start + (out.end - out.start) * 0.25).toBe(4500);
  });

  it('zooms out by the step factor', () => {
    const out = zoomLocus({ contig: CONTIG, start: 4000, end: 5000 }, LEN, ZOOM_STEP, 0.5);
    expect(out.end - out.start).toBe(1200);
    expect(out).toEqual({ contig: CONTIG, start: 3900, end: 5100 });
  });

  it('never shrinks below the minimum window', () => {
    const out = zoomLocus({ contig: CONTIG, start: 5000, end: 5030 }, LEN, 0.1, 0.5);
    expect(out.end - out.start).toBe(MIN_WINDOW_BP);
  });

  it('clamps at the left edge with the span preserved', () => {
    const out = zoomLocus({ contig: CONTIG, start: 0, end: 1000 }, LEN, 2, 0.5);
    expect(out).toEqual({ contig: CONTIG, start: 0, end: 2000 });
  });

  it('clamps at the right edge with the span preserved', () => {
    const out = zoomLocus({ contig: CONTIG, start: 9000, end: 10_000 }, LEN, 2, 0.5);
    expect(out).toEqual({ contig: CONTIG, start: 8000, end: 10_000 });
  });

  it('collapses to the whole contig when the span would exceed it', () => {
    const out = zoomLocus({ contig: CONTIG, start: 2000, end: 9000 }, LEN, 2, 0.5);
    expect(out).toEqual({ contig: CONTIG, start: 0, end: LEN });
  });

  it('treats an out-of-range anchor as the nearest edge', () => {
    const left = zoomLocus({ contig: CONTIG, start: 4000, end: 6000 }, LEN, 0.5, -3);
    const right = zoomLocus({ contig: CONTIG, start: 4000, end: 6000 }, LEN, 0.5, 7);
    expect(left).toEqual({ contig: CONTIG, start: 4000, end: 5000 });
    expect(right).toEqual({ contig: CONTIG, start: 5000, end: 6000 });
  });
});
