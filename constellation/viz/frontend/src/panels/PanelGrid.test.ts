// PanelGrid under a toy host: two rows by two columns, a view that is a
// number, data that is a string. Nothing here is mass-spec specific.

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import {
  COL_HEADER_PX,
  COLLAPSED_COL_PX,
  COLLAPSED_ROW_PX,
  GridCell,
  GridDriver,
  PanelGrid,
  ROW_HEADER_PX,
  trackSizes,
} from './PanelGrid';
import { PanelKind } from './kind';

interface ToyView {
  t: number;
}

const KIND: PanelKind = { kind: 'toy', order: 0, unit: ['point', 'points'] };

interface Rig {
  grid: PanelGrid<ToyView, string>;
  host: HTMLElement;
  view: ToyView | null;
  fetched: string[];
  drawn: string[];
  /** Cells whose key is null. */
  blank: Set<string>;
}

function rig(storageKey?: string): Rig {
  const host = document.createElement('div');
  document.body.appendChild(host);
  const r = { host, view: { t: 1 }, fetched: [], drawn: [], blank: new Set() } as unknown as Rig;
  const driver: GridDriver<ToyView, string> = {
    view: () => r.view,
    dataKey: (cell, view) => (r.blank.has(cell.id) ? null : cell.col === 'chrom' ? 'whole' : `scan${view.t}`),
    fetch: async (cell, view) => {
      r.fetched.push(`${cell.id}@${view.t}`);
      return `${cell.id}:${view.t}`;
    },
    draw: (_cell, data, view) => {
      r.drawn.push(`${data}|${view.t}`);
      return 3;
    },
    placeholder: () => 'pick a scan',
  };
  r.grid = new PanelGrid<ToyView, string>({
    host,
    rows: [
      { id: 'ms1', label: 'MS1' },
      { id: 'ms2', label: 'MS2' },
    ],
    cols: [
      { id: 'chrom', label: 'Chromatogram' },
      { id: 'spec', label: 'Spectrum' },
    ],
    cellEntry: (row, col) => ({ kind: 'toy', binding_id: `${row}-${col}`, label: `${row} ${col}`, source_id: null }),
    driver,
    kindOf: () => KIND,
    fallbackSettings: { sections: [] },
    storageKey,
  });
  r.grid.resize({ widthPx: 1000, heightPx: 600 });
  return r;
}

let current: Rig | null = null;
beforeEach(() => window.localStorage.clear());
afterEach(() => {
  current?.grid.dispose();
  current?.host.remove();
  current = null;
  vi.useRealTimers();
});

describe('trackSizes', () => {
  it('shares space by weight around collapsed and hidden tracks', () => {
    const ids = ['a', 'b', 'c'];
    expect(trackSizes(ids, 600, { a: 2 }, new Set(), new Set(ids), 28)).toEqual({ a: 300, b: 150, c: 150 });
    expect(trackSizes(ids, 600, {}, new Set(['b']), new Set(ids), 28)).toEqual({ a: 286, b: 28, c: 286 });
    expect(trackSizes(ids, 600, {}, new Set(), new Set(['c']), 28)).toEqual({ a: 0, b: 0, c: 600 });
  });
});

describe('PanelGrid', () => {
  it('lays out headers and one panel per cell', () => {
    const r = (current = rig());
    expect(r.grid.cells.map((c) => c.id)).toEqual(['ms1/chrom', 'ms1/spec', 'ms2/chrom', 'ms2/spec']);
    expect(r.host.querySelectorAll('.panel-grid-col-header')).toHaveLength(2);
    expect(r.host.querySelectorAll('.panel-grid-row-header')).toHaveLength(2);
    const { rows, cols } = r.grid.trackPx;
    expect(cols).toEqual({ chrom: (1000 - ROW_HEADER_PX) / 2, spec: (1000 - ROW_HEADER_PX) / 2 });
    expect(rows).toEqual({ ms1: (600 - COL_HEADER_PX) / 2, ms2: (600 - COL_HEADER_PX) / 2 });
    // Grid cells have no drag grip, hide button or resize handle.
    const cell = r.grid.cell('ms1', 'spec')!;
    expect(cell.panel.element.querySelector('.track-handle')).toBeNull();
    expect(cell.panel.element.querySelector('.track-hide-btn')).toBeNull();
    expect(cell.panel.element.querySelector('.panel-grid-maximize')).not.toBeNull();
  });

  it('focusing a cell gives it everything and toggling restores the grid', () => {
    const r = (current = rig());
    r.grid.toggleFocus({ type: 'cell', row: 'ms2', col: 'spec' });
    expect(r.grid.trackPx.cols).toEqual({ chrom: 0, spec: 1000 - ROW_HEADER_PX });
    expect(r.grid.trackPx.rows).toEqual({ ms1: 0, ms2: 600 - COL_HEADER_PX });
    expect(r.grid.cells.filter((c) => !c.panel.element.hidden).map((c) => c.id)).toEqual(['ms2/spec']);
    r.grid.toggleFocus({ type: 'cell', row: 'ms2', col: 'spec' });
    expect(r.grid.focus).toEqual({ type: 'none' });
    expect(r.grid.cells.every((c) => !c.panel.element.hidden)).toBe(true);
  });

  it('focusing a row or a column keeps that whole row or column', () => {
    const r = (current = rig());
    r.grid.setFocus({ type: 'row', row: 'ms1' });
    expect(r.grid.cells.filter((c) => !c.panel.element.hidden).map((c) => c.id)).toEqual(['ms1/chrom', 'ms1/spec']);
    r.grid.setFocus({ type: 'col', col: 'chrom' });
    expect(r.grid.cells.filter((c) => !c.panel.element.hidden).map((c) => c.id)).toEqual(['ms1/chrom', 'ms2/chrom']);
  });

  it('places every header and cell explicitly, so hiding some moves none', () => {
    const r = (current = rig());
    r.grid.setFocus({ type: 'col', col: 'spec' });
    const at = (el: Element) => [(el as HTMLElement).style.gridRow, (el as HTMLElement).style.gridColumn];
    expect(r.grid.cells.map((c) => [c.id, ...at(c.panel.element)])).toEqual([
      ['ms1/chrom', '2', '2'],
      ['ms1/spec', '2', '3'],
      ['ms2/chrom', '3', '2'],
      ['ms2/spec', '3', '3'],
    ]);
    const root = r.grid.cells[0].panel.element.parentElement!;
    expect(at(root.querySelector('.panel-grid-corner')!)).toEqual(['1', '1']);
    expect(Array.from(root.querySelectorAll('.panel-grid-col-header'), at)).toEqual([['1', '2'], ['1', '3']]);
    expect(Array.from(root.querySelectorAll('.panel-grid-row-header'), at)).toEqual([['2', '1'], ['3', '1']]);
    // The column left out has no width, and its header is hidden too.
    expect(r.grid.trackPx.cols.chrom).toBe(0);
    expect((root.querySelector('.panel-grid-col-header') as HTMLElement).hidden).toBe(true);
  });

  it('a collapsed column keeps a strip and its cells are not fetched', async () => {
    const r = (current = rig());
    r.grid.setColCollapsed('spec', true);
    expect(r.grid.trackPx.cols).toEqual({ chrom: 1000 - ROW_HEADER_PX - COLLAPSED_COL_PX, spec: COLLAPSED_COL_PX });
    r.fetched = [];
    await r.grid.render();
    expect(r.fetched.sort()).toEqual(['ms1/chrom@1', 'ms2/chrom@1']);
    // Focus opens a collapsed column.
    r.grid.setFocus({ type: 'col', col: 'spec' });
    expect(r.grid.trackPx.cols.spec).toBe(1000 - ROW_HEADER_PX);
    expect(r.grid.isCellOpen(r.grid.cell('ms1', 'spec')!)).toBe(true);
  });

  it('a collapsed row keeps its header height', () => {
    const r = (current = rig());
    r.grid.setRowCollapsed('ms1', true);
    expect(r.grid.trackPx.rows.ms1).toBe(COLLAPSED_ROW_PX);
    expect(r.grid.cell('ms1', 'chrom')!.panel.element.classList.contains('collapsed-row')).toBe(true);
  });

  it('fetches only cells whose key changed and redraws the rest from cache', async () => {
    const r = (current = rig());
    await r.grid.render();
    expect(r.fetched.sort()).toEqual(['ms1/chrom@1', 'ms1/spec@1', 'ms2/chrom@1', 'ms2/spec@1']);
    r.fetched = [];
    r.drawn = [];
    r.view = { t: 2 }; // spectra change (their key has the scan), chromatograms do not
    await r.grid.render();
    expect(r.fetched.sort()).toEqual(['ms1/spec@2', 'ms2/spec@2']);
    // Cached chromatograms are drawn again for the new view.
    expect(r.drawn).toContain('ms1/chrom:1|2');
    expect(r.grid.cell('ms1', 'chrom')!.panel.statusEl.textContent).toBe('3 points');
  });

  it('a cell with nothing to show gets the placeholder', async () => {
    const r = (current = rig());
    r.blank.add('ms2/spec');
    await r.grid.render();
    const body = r.grid.cell('ms2', 'spec')!.panel.bodyHost;
    expect(body.querySelector('.panel-grid-placeholder')?.textContent).toBe('pick a scan');
  });

  it('remembers focus, collapse and weights', () => {
    vi.useFakeTimers();
    const r = (current = rig('grid-test'));
    r.grid.setColCollapsed('spec', true);
    r.grid.setFocus({ type: 'row', row: 'ms2' });
    vi.advanceTimersByTime(500);
    r.grid.dispose();
    r.host.remove();
    const again = (current = rig('grid-test'));
    expect(again.grid.focus).toEqual({ type: 'row', row: 'ms2' });
    expect(again.grid.isColCollapsed('spec')).toBe(true);
  });

  it('reports where each open cell is drawn', () => {
    const r = (current = rig());
    const rects = r.grid.layoutRects();
    expect(rects.map((x) => [x.cell.id, x.x, x.y])).toEqual([
      ['ms1/chrom', 0, 0],
      ['ms1/spec', 485, 0],
      ['ms2/chrom', 0, 288],
      ['ms2/spec', 485, 288],
    ]);
    const size = r.grid.cellSize(r.grid.cell('ms1', 'chrom') as GridCell<string>);
    expect(rects[0].width).toBe(size.widthPx);
    expect(rects[0].height).toBe(size.heightPx);
  });

  it('new rows rebuild the cells', () => {
    const r = (current = rig());
    r.grid.setRows([
      { id: 'ms1', label: 'MS1' },
      { id: 'ms2', label: 'MS2' },
      { id: 'ms3', label: 'MS3' },
    ]);
    expect(r.grid.cells).toHaveLength(6);
    expect(r.grid.rowSlot('ms3')).toBeInstanceOf(HTMLElement);
  });
});
