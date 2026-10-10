// The schema interpreter on its own, with a schema that belongs to no
// real panel kind. The genome kinds' popovers are pinned separately by
// their form-model snapshots.

import { afterEach, describe, expect, it, vi } from 'vitest';
import { SettingsPanel } from './SettingsPanel';
import { SettingsSchema } from './settings_schema';

type Dict = Record<string, unknown>;

const SCHEMA: SettingsSchema = {
  sections: [
    {
      title: 'Look',
      controls: [
        { type: 'number', target: 'style', key: 'size', label: 'Size', default: 3, min: 1, max: 9, step: 1 },
        { type: 'text', target: 'style', key: 'font', label: 'Font', default: 'mono' },
        { type: 'select', target: 'style', key: 'scale', label: 'Scale', default: 'lin',
          options: [{ value: 'lin', label: 'Linear' }, { value: 'log', label: 'Log' }], hint: 'refetches' },
        { type: 'toggle', target: 'style', key: 'labels', label: 'Labels',
          default: (env) => env.host.labelsOn !== false },
        { type: 'palette', entries: (env) =>
            (env.meta.series as string[]).map((s, i) => ({ key: s, label: `Series ${s}`, default: i ? '#00f' : '#f00' })),
          emptyHint: 'no series' },
      ],
    },
    {
      title: 'Rows',
      controls: [
        { type: 'allowlist', target: 'filter', key: 'ids', label: 'Ids',
          options: (env) => (env.meta.ids as number[]).map((id) => ({ value: id, label: `#${id}` })) },
        { type: 'allowlist', target: 'filter', key: 'tags', label: 'Tags',
          options: [{ value: 'a', label: 'A' }, { value: 'b', label: 'B' }], emptyHint: 'no tags' },
        { type: 'number', target: 'filter', key: 'min', label: 'Minimum', default: 0, min: 0, max: 10, step: 1,
          when: (env) => env.filter.advanced === true },
        { type: 'note', text: 'expert only', when: (env) => env.meta.expert === true },
      ],
    },
    { title: 'Hidden', when: () => false, controls: [{ type: 'note', text: 'never shown' }] },
  ],
};

const opened: SettingsPanel[] = [];

function open(opts: { meta?: Dict; style?: Dict; filter?: Dict; host?: Dict } = {}) {
  const anchor = document.createElement('button');
  document.body.appendChild(anchor);
  const cb = { onStyleChange: vi.fn(), onFilterChange: vi.fn(), onReset: vi.fn(), onClose: vi.fn() };
  const panel = new SettingsPanel({
    anchor,
    kind: 'toy',
    label: 'Toy panel',
    schema: SCHEMA,
    meta: { series: ['x', 'y'], ids: [7, 8], ...opts.meta },
    style: opts.style ?? {},
    filter: opts.filter ?? {},
    host: opts.host,
    ...cb,
  });
  panel.mount(document.body);
  opened.push(panel);
  const root = document.body.querySelector('.track-settings-popover:last-of-type') as HTMLElement;
  return { root, ...cb };
}

afterEach(() => {
  for (const p of opened.splice(0)) p.dispose();
  document.body.replaceChildren();
});

function titles(root: HTMLElement): string[] {
  return Array.from(root.querySelectorAll('.settings-section-title'), (t) => t.textContent ?? '');
}

function row(root: HTMLElement, label: string): HTMLElement {
  const hit = Array.from(root.querySelectorAll<HTMLElement>('.settings-row, .settings-checkbox-row')).find(
    (r) => (r.querySelector('.settings-row-label, span')?.textContent ?? '') === label,
  );
  if (!hit) throw new Error(`no row ${label}`);
  return hit;
}

function fire(el: HTMLInputElement | HTMLSelectElement, value: string, event = 'change'): void {
  el.value = value;
  el.dispatchEvent(new Event(event, { bubbles: true }));
}

describe('SettingsPanel', () => {
  it('renders the header and the sections whose condition holds', () => {
    const { root } = open();
    expect(root.querySelector('.settings-title')?.textContent).toBe('Toy panel');
    expect(root.querySelector('.settings-kind')?.textContent).toBe('toy');
    expect(titles(root)).toEqual(['Look', 'Rows']);
    expect(root.getAttribute('role')).toBe('dialog');
  });

  it('shows defaults, and stored values of the right type over them', () => {
    const fresh = open();
    expect(row(fresh.root, 'Size').querySelector('input')!.value).toBe('3');
    expect(row(fresh.root, 'Font').querySelector('input')!.value).toBe('mono');
    expect(row(fresh.root, 'Scale').querySelector('select')!.value).toBe('lin');
    expect(row(fresh.root, 'Series x').querySelector('input')!.value).toBe('#ff0000');

    const stored = open({ style: { size: 7, font: '', scale: 'log', 'palette.x': '#123456' } });
    expect(row(stored.root, 'Size').querySelector('input')!.value).toBe('7');
    expect(row(stored.root, 'Font').querySelector('input')!.value).toBe('');
    expect(row(stored.root, 'Scale').querySelector('select')!.value).toBe('log');
    expect(row(stored.root, 'Series x').querySelector('input')!.value).toBe('#123456');

    // A stored value of the wrong type is not shown.
    const wrong = open({ style: { size: '7', scale: 3, labels: 'yes' } });
    expect(row(wrong.root, 'Size').querySelector('input')!.value).toBe('3');
    expect(row(wrong.root, 'Scale').querySelector('select')!.value).toBe('lin');
    expect(row(wrong.root, 'Labels').querySelector('input')!.checked).toBe(true);
  });

  it('carries number bounds, select options and a hint', () => {
    const { root } = open();
    const size = row(root, 'Size').querySelector('input')!;
    expect([size.min, size.max, size.step]).toEqual(['1', '9', '1']);
    const scale = row(root, 'Scale');
    expect(Array.from(scale.querySelectorAll('option'), (o) => [o.value, o.textContent])).toEqual([
      ['lin', 'Linear'],
      ['log', 'Log'],
    ]);
    expect(scale.querySelector('.settings-row-hint')?.textContent).toBe('refetches');
  });

  it('takes a default from the host when the schema computes one', () => {
    expect(row(open().root, 'Labels').querySelector('input')!.checked).toBe(true);
    expect(row(open({ host: { labelsOn: false } }).root, 'Labels').querySelector('input')!.checked).toBe(false);
    // A stored value wins over the computed default.
    const stored = open({ host: { labelsOn: false }, style: { labels: true } });
    expect(row(stored.root, 'Labels').querySelector('input')!.checked).toBe(true);
  });

  it('builds palettes and allow-lists from metadata, with empty-state handling', () => {
    const none = open({ meta: { series: [], ids: [] } });
    expect(Array.from(none.root.querySelectorAll('.settings-empty'), (e) => e.textContent)).toEqual(['no series']);
    // An allow-list with no options and no empty hint is simply absent.
    expect(none.root.textContent).not.toContain('Ids');
    expect(none.root.textContent).toContain('Tags');
  });

  it('shows a control or a note only while its condition holds', () => {
    expect(open().root.textContent).not.toContain('Minimum');
    expect(open({ filter: { advanced: true } }).root.textContent).toContain('Minimum');
    expect(open().root.textContent).not.toContain('expert only');
    expect(open({ meta: { expert: true } }).root.textContent).toContain('expert only');
  });

  it('reports style edits as the whole dict', () => {
    const h = open({ style: { keep: 1 } });
    fire(row(h.root, 'Size').querySelector('input')!, '5');
    expect(h.onStyleChange).toHaveBeenLastCalledWith({ keep: 1, size: 5 });
    fire(row(h.root, 'Scale').querySelector('select')!, 'log');
    expect(h.onStyleChange).toHaveBeenLastCalledWith({ keep: 1, size: 5, scale: 'log' });
    fire(row(h.root, 'Series y').querySelector('input')!, '#abcdef', 'input');
    expect(h.onStyleChange).toHaveBeenLastCalledWith({ keep: 1, size: 5, scale: 'log', 'palette.y': '#abcdef' });
    expect(h.onFilterChange).not.toHaveBeenCalled();
  });

  it('removes a text key when the field is cleared', () => {
    const h = open();
    const font = row(h.root, 'Font').querySelector('input')!;
    fire(font, ' serif ');
    expect(h.onStyleChange).toHaveBeenLastCalledWith({ font: 'serif' });
    fire(font, '  ');
    expect(h.onStyleChange).toHaveBeenLastCalledWith({});
  });

  it('writes 0 when a number field is cleared (as the row builder always has)', () => {
    // An emptied number input has the value '', and Number('') is 0 — so
    // clearing a field stores 0 rather than restoring the default. Pinned
    // as current behavior; it is not obviously what a user means.
    const h = open();
    fire(row(h.root, 'Size').querySelector('input')!, '');
    expect(h.onStyleChange).toHaveBeenLastCalledWith({ size: 0 });
  });

  it('stores numeric allow-list options as numbers and string ones as strings', () => {
    const h = open();
    const boxes = Array.from(h.root.querySelectorAll<HTMLElement>('label.settings-checkbox-row'));
    const untick = (label: string): void => {
      const cb = boxes.find((b) => b.querySelector('span')?.textContent === label)!.querySelector('input')!;
      cb.checked = false;
      cb.dispatchEvent(new Event('change', { bubbles: true }));
    };
    untick('#7');
    expect(h.onFilterChange).toHaveBeenLastCalledWith({ ids: [8] });
    untick('A');
    expect(h.onFilterChange).toHaveBeenLastCalledWith({ ids: [8], tags: ['b'] });
    untick('B');
    expect(h.onFilterChange).toHaveBeenLastCalledWith({ ids: [8], tags: [] });
    expect(h.onStyleChange).not.toHaveBeenCalled();
  });

  it('treats an unset or "all" allow-list as everything ticked', () => {
    const ticked = (root: HTMLElement): boolean[] =>
      Array.from(root.querySelectorAll<HTMLInputElement>('label.settings-checkbox-row input'))
        .slice(1)
        .map((cb) => cb.checked);
    expect(ticked(open().root)).toEqual([true, true, true, true]);
    expect(ticked(open({ filter: { ids: 'all', tags: null } }).root)).toEqual([true, true, true, true]);
    expect(ticked(open({ filter: { ids: [8], tags: [] } }).root)).toEqual([false, true, false, false]);
  });

  it('reset empties both dicts, tells the host and redraws from defaults', () => {
    const h = open({ style: { size: 8 }, filter: { advanced: true } });
    expect(h.root.textContent).toContain('Minimum');
    (h.root.querySelector('.settings-reset-btn') as HTMLElement).click();
    expect(h.onReset).toHaveBeenCalledTimes(1);
    expect(row(h.root, 'Size').querySelector('input')!.value).toBe('3');
    expect(h.root.textContent).not.toContain('Minimum');
  });
});
