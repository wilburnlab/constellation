// The genome track kinds' settings schemas — form-model snapshots and
// edit propagation.
//
// The form model is what the gear popover offers for each track kind:
// sections, controls, their labels, initial values, bounds and options.
// It is pinned per kind so the settings code can be restructured with
// the popover's contents held fixed. Each case mounts the generic
// `SettingsPanel` with the schema the kind's renderer declares, which is
// what the browser's track stack does.

import { afterEach, describe, expect, it, vi } from 'vitest';
import { SettingsPanel } from '../../../panels/SettingsPanel';
import { loadMetadata } from '../__fixtures__/load';
import type { TrackMetadata } from './base';
import { getRenderer, registeredKinds } from './index';
import { FALLBACK_SETTINGS } from './settings_common';

type Dict = Record<string, unknown>;

interface Harness {
  panel: SettingsPanel;
  root: HTMLElement;
  anchor: HTMLElement;
  onStyleChange: ReturnType<typeof vi.fn>;
  onFilterChange: ReturnType<typeof vi.fn>;
  onReset: ReturnType<typeof vi.fn>;
  onClose: ReturnType<typeof vi.fn>;
}

const open: Harness[] = [];

function mount(
  kind: string,
  opts: { meta?: Dict; style?: Dict; filter?: Dict } = {},
): Harness {
  const all = loadMetadata();
  const key = Object.keys(all).find((k) => k.startsWith(`${kind}/`));
  const base: Dict = key
    ? all[key]
    : { kind, binding_id: `${kind}-0`, label: kind, default_height_px: 80 };
  const meta = { ...base, ...(opts.meta ?? {}) } as TrackMetadata;
  const anchor = document.createElement('button');
  document.body.appendChild(anchor);
  const h = {
    anchor,
    onStyleChange: vi.fn(),
    onFilterChange: vi.fn(),
    onReset: vi.fn(),
    onClose: vi.fn(),
  };
  const panel = new SettingsPanel({
    anchor,
    kind,
    label: meta.label,
    schema: getRenderer(kind)?.settings ?? FALLBACK_SETTINGS,
    meta,
    style: opts.style ?? {},
    filter: opts.filter ?? {},
    onStyleChange: h.onStyleChange,
    onFilterChange: h.onFilterChange,
    onReset: h.onReset,
    onClose: h.onClose,
  });
  panel.mount(document.body);
  const root = document.body.querySelector('.track-settings-popover:last-of-type') as HTMLElement;
  const harness = { ...h, panel, root };
  open.push(harness);
  return harness;
}

afterEach(() => {
  for (const h of open.splice(0)) {
    h.panel.dispose();
    h.anchor.remove();
  }
});

// ----------------------------------------------------------------------
// Form-model extraction
// ----------------------------------------------------------------------

function describeInput(input: HTMLInputElement): Dict {
  if (input.type === 'checkbox') return { control: 'checkbox', checked: input.checked };
  if (input.type === 'number') {
    return {
      control: 'number',
      value: input.value,
      min: input.min,
      max: input.max,
      step: input.step,
    };
  }
  if (input.type === 'color') return { control: 'color', value: input.value };
  return { control: input.type, value: input.value, placeholder: input.placeholder };
}

function describeRow(row: Element): Dict {
  const out: Dict = {
    label: row.querySelector('.settings-row-label')?.textContent ?? null,
  };
  const select = row.querySelector('select');
  const input = row.querySelector('input');
  if (select) {
    out.control = 'select';
    out.value = select.value;
    out.options = Array.from(select.options).map((o) => ({ value: o.value, label: o.textContent }));
  } else if (input) {
    Object.assign(out, describeInput(input));
  } else {
    out.control = 'none';
  }
  const hints = Array.from(row.querySelectorAll('.settings-row-hint')).map((h) => h.textContent);
  if (hints.length > 0) out.hints = hints;
  return out;
}

function describeChild(el: Element): Dict | null {
  if (el.classList.contains('settings-section-title')) return null;
  if (el.classList.contains('settings-empty')) return { control: 'empty', text: el.textContent };
  if (el.classList.contains('settings-row')) return describeRow(el);
  if (el.classList.contains('settings-checkbox-row')) {
    const cb = el.querySelector('input') as HTMLInputElement;
    return { label: el.querySelector('span')?.textContent ?? null, control: 'checkbox', checked: cb.checked };
  }
  if (el.classList.contains('settings-row-hint')) return { control: 'hint', text: el.textContent };
  // Allow-list: an unclassed wrapper holding a title + one checkbox per option.
  const boxes = Array.from(el.querySelectorAll(':scope > label.settings-checkbox-row'));
  if (el.tagName === 'DIV' && el.className === '' && boxes.length > 0) {
    return {
      label: el.querySelector(':scope > .settings-row-label')?.textContent ?? null,
      control: 'allowlist',
      options: boxes.map((b) => ({
        label: b.querySelector('span')?.textContent ?? null,
        checked: (b.querySelector('input') as HTMLInputElement).checked,
      })),
    };
  }
  // Anything unrecognized is kept verbatim so it cannot vanish unnoticed.
  return { control: 'unknown', html: el.outerHTML };
}

function formModel(root: HTMLElement): Dict {
  return {
    title: root.querySelector('.settings-title')?.textContent ?? null,
    kind: root.querySelector('.settings-kind')?.textContent ?? null,
    sections: Array.from(root.querySelectorAll('.settings-section')).map((section) => ({
      title: section.querySelector('.settings-section-title')?.textContent ?? null,
      controls: Array.from(section.children)
        .map(describeChild)
        .filter((c): c is Dict => c !== null),
    })),
    actions: Array.from(root.querySelectorAll('.settings-actions button')).map((b) => b.textContent),
  };
}

function snapshotJson(model: Dict): string {
  return `${JSON.stringify(model, null, 2)}\n`;
}

// ----------------------------------------------------------------------
// Snapshots
// ----------------------------------------------------------------------

describe('genome track settings: form model', () => {
  it.each(registeredKinds())('%s (defaults, kernel metadata)', async (kind) => {
    const { root } = mount(kind);
    await expect(snapshotJson(formModel(root))).toMatchFileSnapshot(
      `./__snapshots__/settings.${kind}.json`,
    );
  });

  it('gene_annotation with stored style and filter', async () => {
    const { root } = mount('gene_annotation', {
      style: {
        row_height_px: 20,
        feature_opacity: 0.5,
        label_font_family: 'Georgia',
        show_labels: false,
        'palette.gene': '#123456',
        palette: { CDS: '#abcdef', exon: '#abc' },
      },
      filter: { visible_types: ['gene', 'mRNA'], visible_strands: [], min_length_bp: 250 },
    });
    await expect(snapshotJson(formModel(root))).toMatchFileSnapshot(
      './__snapshots__/settings.gene_annotation.stored.json',
    );
  });

  it('read_pileup with stored style and filter', async () => {
    const { root } = mount('read_pileup', {
      style: { 'palette.1': '#008080', read_opacity: 0.6, intron_stroke_dasharray: '4,1' },
      filter: { visible_samples: [2], visible_strands: ['+'], min_mapq: 20 },
    });
    await expect(snapshotJson(formModel(root))).toMatchFileSnapshot(
      './__snapshots__/settings.read_pileup.stored.json',
    );
  });

  it('coverage_histogram before any samples are known', async () => {
    const { root } = mount('coverage_histogram', { meta: { samples_in_data: [] } });
    await expect(snapshotJson(formModel(root))).toMatchFileSnapshot(
      './__snapshots__/settings.coverage_histogram.no-samples.json',
    );
  });

  it('coverage_histogram with the unstratified (-1) sample', async () => {
    const { root } = mount('coverage_histogram', { meta: { samples_in_data: [-1] } });
    await expect(snapshotJson(formModel(root))).toMatchFileSnapshot(
      './__snapshots__/settings.coverage_histogram.unstratified.json',
    );
  });

  it('cluster_pileup when the members view is unavailable', async () => {
    const { root } = mount('cluster_pileup', { meta: { cluster_view_supported: false } });
    await expect(snapshotJson(formModel(root))).toMatchFileSnapshot(
      './__snapshots__/settings.cluster_pileup.no-members.json',
    );
  });

  it('cluster_pileup in the members view', async () => {
    const { root } = mount('cluster_pileup', { filter: { cluster_view: 'members' } });
    await expect(snapshotJson(formModel(root))).toMatchFileSnapshot(
      './__snapshots__/settings.cluster_pileup.members.json',
    );
  });

  it('an unregistered kind', async () => {
    const { root } = mount('not_a_kind');
    await expect(snapshotJson(formModel(root))).toMatchFileSnapshot(
      './__snapshots__/settings.unknown-kind.json',
    );
  });

  it('leaves no control the extractor cannot classify', () => {
    for (const kind of registeredKinds()) {
      const { root } = mount(kind);
      expect(snapshotJson(formModel(root))).not.toContain('"control": "unknown"');
    }
  });
});

// ----------------------------------------------------------------------
// Edit propagation
// ----------------------------------------------------------------------

function rowByLabel(root: HTMLElement, label: string): HTMLElement {
  const rows = Array.from(root.querySelectorAll('.settings-row, .settings-checkbox-row'));
  const hit = rows.find(
    (r) => (r.querySelector('.settings-row-label, span')?.textContent ?? '') === label,
  );
  if (!hit) throw new Error(`no settings row labelled ${label}`);
  return hit as HTMLElement;
}

function setValue(el: HTMLInputElement | HTMLSelectElement, value: string, event = 'change'): void {
  el.value = value;
  el.dispatchEvent(new Event(event, { bubbles: true }));
}

function toggle(row: HTMLElement): void {
  const cb = row.querySelector('input[type=checkbox]') as HTMLInputElement;
  cb.checked = !cb.checked;
  cb.dispatchEvent(new Event('change', { bubbles: true }));
}

function allowListBox(root: HTMLElement, listLabel: string, option: string): HTMLElement {
  const titles = Array.from(root.querySelectorAll('div > .settings-row-label'));
  const title = titles.find(
    (t) => t.textContent === listLabel && t.parentElement?.className === '',
  );
  if (!title) throw new Error(`no allow-list labelled ${listLabel}`);
  const boxes = Array.from(title.parentElement!.querySelectorAll('label.settings-checkbox-row'));
  const hit = boxes.find((b) => b.querySelector('span')?.textContent === option);
  if (!hit) throw new Error(`no option ${option} in ${listLabel}`);
  return hit as HTMLElement;
}

describe('genome track settings: edits', () => {
  it('writes a number row into style and reports the whole dict', () => {
    const h = mount('gene_annotation', { style: { feature_opacity: 0.5 } });
    setValue(rowByLabel(h.root, 'Row height (px)').querySelector('input')!, '22');
    expect(h.onStyleChange).toHaveBeenCalledTimes(1);
    expect(h.onStyleChange.mock.calls[0][0]).toEqual({ feature_opacity: 0.5, row_height_px: 22 });
    expect(h.onFilterChange).not.toHaveBeenCalled();
  });

  it('writes a number row even when the value equals the default', () => {
    const h = mount('gene_annotation');
    setValue(rowByLabel(h.root, 'Row height (px)').querySelector('input')!, '14');
    expect(h.onStyleChange.mock.calls[0][0]).toEqual({ row_height_px: 14 });
  });

  it('sets a text row and deletes the key when cleared', () => {
    const h = mount('gene_annotation');
    const input = rowByLabel(h.root, 'Label font family').querySelector('input')!;
    setValue(input, '  Georgia ');
    expect(h.onStyleChange.mock.calls[0][0]).toEqual({ label_font_family: 'Georgia' });
    setValue(input, '   ');
    expect(h.onStyleChange.mock.calls[1][0]).toEqual({});
  });

  it('writes a select row', () => {
    const h = mount('coverage_histogram');
    setValue(rowByLabel(h.root, 'Y scale').querySelector('select')!, 'log');
    expect(h.onStyleChange.mock.calls[0][0]).toEqual({ y_scale: 'log' });
  });

  it('writes a style checkbox', () => {
    const h = mount('gene_annotation');
    toggle(rowByLabel(h.root, 'Show chevrons'));
    expect(h.onStyleChange.mock.calls[0][0]).toEqual({ show_chevrons: false });
  });

  it('writes a colour under palette.<key> on every input event', () => {
    const h = mount('gene_annotation');
    setValue(rowByLabel(h.root, 'gene').querySelector('input')!, '#112233', 'input');
    expect(h.onStyleChange.mock.calls[0][0]).toEqual({ 'palette.gene': '#112233' });
  });

  it('stores a string allow-list as the remaining ticked options', () => {
    const h = mount('gene_annotation');
    toggle(allowListBox(h.root, 'Visible strands', '-'));
    expect(h.onFilterChange).toHaveBeenCalledTimes(1);
    expect(h.onFilterChange.mock.calls[0][0]).toEqual({ visible_strands: ['+', '.'] });
    toggle(allowListBox(h.root, 'Visible strands', '-'));
    // Re-ticking everything stores the full list, not a sentinel.
    expect(h.onFilterChange.mock.calls[1][0]).toEqual({ visible_strands: ['+', '.', '-'] });
  });

  it('stores an empty allow-list when the last option is unticked', () => {
    const h = mount('read_pileup', { filter: { visible_strands: ['+'] } });
    toggle(allowListBox(h.root, 'Visible strands', '+'));
    expect(h.onFilterChange.mock.calls[0][0]).toEqual({ visible_strands: [] });
  });

  it('stores visible_samples as numbers', () => {
    const h = mount('read_pileup');
    toggle(allowListBox(h.root, 'Visible samples', 'alpha (1)'));
    const filter = h.onFilterChange.mock.calls[0][0] as Dict;
    expect(filter.visible_samples).toEqual([2]);
  });

  it('writes filter number rows through onFilterChange', () => {
    const h = mount('read_pileup');
    setValue(rowByLabel(h.root, 'Min MAPQ').querySelector('input')!, '20');
    expect(h.onFilterChange.mock.calls[0][0]).toEqual({ min_mapq: 20 });
    expect(h.onStyleChange).not.toHaveBeenCalled();
  });

  it('writes the cluster view select into filter', () => {
    const h = mount('cluster_pileup');
    setValue(rowByLabel(h.root, 'Cluster view').querySelector('select')!, 'members');
    expect(h.onFilterChange.mock.calls[0][0]).toEqual({ cluster_view: 'members' });
  });

  it('reset clears both dicts, notifies the host and rebuilds the form', () => {
    const h = mount('gene_annotation', {
      style: { row_height_px: 30 },
      filter: { min_length_bp: 500 },
    });
    expect(rowByLabel(h.root, 'Row height (px)').querySelector('input')!.value).toBe('30');
    (h.root.querySelector('.settings-reset-btn') as HTMLButtonElement).click();
    expect(h.onReset).toHaveBeenCalledTimes(1);
    expect(rowByLabel(h.root, 'Row height (px)').querySelector('input')!.value).toBe('14');
    // Edits after a reset start from an empty dict.
    setValue(rowByLabel(h.root, 'Row height (px)').querySelector('input')!, '16');
    expect(h.onStyleChange.mock.calls[0][0]).toEqual({ row_height_px: 16 });
  });
});

describe('genome track settings: dismissal', () => {
  it('closes on Escape and on a mousedown outside the panel and anchor', () => {
    const h = mount('gene_annotation');
    h.root.dispatchEvent(new MouseEvent('mousedown', { bubbles: true }));
    h.anchor.dispatchEvent(new MouseEvent('mousedown', { bubbles: true }));
    expect(h.onClose).not.toHaveBeenCalled();
    document.body.dispatchEvent(new MouseEvent('mousedown', { bubbles: true }));
    expect(h.onClose).toHaveBeenCalledTimes(1);
    document.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape' }));
    expect(h.onClose).toHaveBeenCalledTimes(2);
  });

  it('stops listening after dispose', () => {
    const h = mount('gene_annotation');
    h.panel.dispose();
    document.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape' }));
    document.body.dispatchEvent(new MouseEvent('mousedown', { bubbles: true }));
    expect(h.onClose).not.toHaveBeenCalled();
    expect(document.body.contains(h.root)).toBe(false);
  });
});
