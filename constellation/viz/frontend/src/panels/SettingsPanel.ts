// SettingsPanel — the gear popover for one panel, rendered from the
// schema its kind declares (see settings_schema.ts).
//
// The popover knows nothing about any particular kind. It walks the
// schema's sections and controls, shows each control's current value
// (or its default), and reports edits through two callbacks:
// `onStyleChange` for visual knobs and `onFilterChange` for data
// filters. Each call carries the whole dict, not a patch. What happens
// next — an instant client-side restyle or a refetch — is the host's
// decision.
//
// It anchors itself to the gear button like the other popovers
// (engine/popover.ts).

import { attachDismiss, positionBelowRight } from '../engine/popover';
import {
  Bag,
  Control,
  SettingsEnv,
  SettingsSchema,
  resolve,
} from './settings_schema';
import './panels.css';

export interface SettingsPanelArgs {
  anchor: HTMLElement;
  /** Shown beside the title. */
  kind: string;
  label: string;
  schema: SettingsSchema;
  /** The panel's server-side metadata, for schema functions. */
  meta: Bag;
  style: Record<string, unknown>;
  filter: Record<string, unknown>;
  /** Host-level state a schema default may follow. */
  host?: Bag;
  onStyleChange(style: Record<string, unknown>): void;
  onFilterChange(filter: Record<string, unknown>): void;
  onReset(): void;
  onClose(): void;
}

export class SettingsPanel {
  private readonly opts: SettingsPanelArgs;
  private readonly root: HTMLElement;
  private style: Record<string, unknown>;
  private filter: Record<string, unknown>;
  private detachDismiss: (() => void) | null = null;

  constructor(opts: SettingsPanelArgs) {
    this.opts = opts;
    this.style = { ...opts.style };
    this.filter = { ...opts.filter };
    this.root = document.createElement('div');
    this.root.className = 'track-settings-popover';
    this.root.setAttribute('role', 'dialog');
    this.root.setAttribute('aria-label', `Track settings — ${opts.label}`);
    this.build();
    // Keep 8px off the right edge: the gear sits at the far right of
    // its panel header.
    positionBelowRight(this.root, opts.anchor, { minRightPx: 8 });
    this.detachDismiss = attachDismiss(this.root, opts.anchor, () =>
      this.opts.onClose(),
    );
  }

  mount(parent: HTMLElement): void {
    parent.appendChild(this.root);
  }

  dispose(): void {
    this.detachDismiss?.();
    this.detachDismiss = null;
    this.root.remove();
  }

  // --------------------------------------------------------------------
  // Build
  // --------------------------------------------------------------------

  private env(): SettingsEnv {
    return {
      meta: this.opts.meta,
      style: this.style,
      filter: this.filter,
      host: this.opts.host ?? {},
    };
  }

  private build(): void {
    this.root.replaceChildren();

    const header = document.createElement('div');
    header.className = 'settings-header';
    const title = document.createElement('span');
    title.className = 'settings-title';
    title.textContent = this.opts.label;
    title.title = this.opts.label;
    const kind = document.createElement('span');
    kind.className = 'settings-kind';
    kind.textContent = this.opts.kind;
    header.appendChild(title);
    header.appendChild(kind);
    this.root.appendChild(header);

    const env = this.env();
    for (const section of this.opts.schema.sections) {
      if (section.when && !section.when(env)) continue;
      const el = makeSection(section.title);
      for (const control of section.controls) {
        for (const node of this.renderControl(control, env)) el.appendChild(node);
      }
      this.root.appendChild(el);
    }

    const actions = document.createElement('div');
    actions.className = 'settings-actions';
    const resetBtn = document.createElement('button');
    resetBtn.type = 'button';
    resetBtn.className = 'settings-reset-btn';
    resetBtn.textContent = 'Reset to defaults';
    resetBtn.addEventListener('click', () => {
      this.style = {};
      this.filter = {};
      this.opts.onReset();
      this.build();
    });
    actions.appendChild(resetBtn);
    this.root.appendChild(actions);
  }

  /** The DOM for one control: usually one row, one per entry for a
   *  palette, none when the control is not applicable. */
  private renderControl(control: Control, env: SettingsEnv): HTMLElement[] {
    if (control.when && !control.when(env)) return [];

    if (control.type === 'note') return [emptyHint(control.text)];

    if (control.type === 'palette') {
      const entries = resolve(control.entries, env);
      if (entries.length === 0) {
        return control.emptyHint ? [emptyHint(control.emptyHint)] : [];
      }
      return entries.map((entry) =>
        paletteRow(
          entry.label,
          this.getPalette(entry.key) ?? entry.default,
          (hex) => this.setPalette(entry.key, hex),
        ),
      );
    }

    if (control.type === 'allowlist') {
      const options = resolve(control.options, env);
      if (options.length === 0) {
        return control.emptyHint ? [emptyHint(control.emptyHint)] : [];
      }
      const numeric = options.every((o) => typeof o.value === 'number');
      const labels = new Map(options.map((o) => [String(o.value), o.label]));
      return [
        allowListRow(
          control.label,
          options.map((o) => String(o.value)),
          this.getFilterArray(control.key),
          (selected) =>
            this.setFilterArray(
              control.key,
              numeric
                ? selected.map((s) => Number(s)).filter((n) => Number.isFinite(n))
                : selected,
            ),
          (key) => labels.get(key) ?? key,
        ),
      ];
    }

    const bag = control.target === 'style' ? this.style : this.filter;
    const write = (value: unknown): void => this.write(control.target, control.key, value);
    let row: HTMLElement;
    if (control.type === 'number') {
      row = numberRow(
        control.label,
        getNumber(bag, control.key, control.default),
        { min: control.min, max: control.max, step: control.step },
        write,
      );
    } else if (control.type === 'text') {
      row = textRow(control.label, getString(bag, control.key, control.default), (value) => {
        // An empty text field means "no override": drop the key.
        if (value) write(value);
        else this.remove(control.target, control.key);
      });
    } else if (control.type === 'select') {
      row = selectRow(
        control.label,
        getString(bag, control.key, control.default),
        control.options,
        write,
      );
    } else {
      row = checkboxRow(
        control.label,
        getBoolean(bag, control.key, resolve(control.default, env)),
        write,
      );
    }
    if (control.hint) row.appendChild(filterHint(control.hint));
    return [row];
  }

  // --------------------------------------------------------------------
  // Writes — mutate the panel's own copy and report the whole dict
  // --------------------------------------------------------------------

  private write(target: 'style' | 'filter', key: string, value: unknown): void {
    if (target === 'style') {
      this.style[key] = value;
      this.opts.onStyleChange(this.style);
    } else {
      this.filter[key] = value;
      this.opts.onFilterChange(this.filter);
    }
  }

  private remove(target: 'style' | 'filter', key: string): void {
    if (target === 'style') {
      delete this.style[key];
      this.opts.onStyleChange(this.style);
    } else {
      delete this.filter[key];
      this.opts.onFilterChange(this.filter);
    }
  }

  // Palette colours are stored under `palette.<key>` so the renderers'
  // pickPaletteColor finds them.

  private setPalette(key: string, hex: string): void {
    this.style[`palette.${key}`] = hex;
    this.opts.onStyleChange(this.style);
  }

  private getPalette(key: string): string | undefined {
    const dotted = this.style[`palette.${key}`];
    if (typeof dotted === 'string') return dotted;
    const nested = this.style.palette;
    if (nested && typeof nested === 'object') {
      const v = (nested as Record<string, unknown>)[key];
      if (typeof v === 'string') return v;
    }
    return undefined;
  }

  private getFilterArray(key: string): string[] | null {
    const v = this.filter[key];
    if (v === undefined || v === null || v === 'all') return null;
    if (!Array.isArray(v)) return null;
    return (v as unknown[]).map(String);
  }

  private setFilterArray(key: string, selected: Array<string | number>): void {
    // Every option ticked is stored as the full list (not a sentinel);
    // the renderers' pickAllowList treats that the same as "all".
    if (selected.length === 0) {
      this.filter[key] = [] as unknown[];
    } else {
      this.filter[key] = selected;
    }
    this.opts.onFilterChange(this.filter);
  }
}

// ----------------------------------------------------------------------
// Stored-value coercion — a value of the wrong type falls back to the
// control's default rather than being shown.
// ----------------------------------------------------------------------

function getNumber(source: Record<string, unknown>, key: string, fallback: number): number {
  const v = source[key];
  if (typeof v === 'number' && Number.isFinite(v)) return v;
  return fallback;
}

function getString(source: Record<string, unknown>, key: string, fallback: string): string {
  const v = source[key];
  return typeof v === 'string' ? v : fallback;
}

function getBoolean(source: Record<string, unknown>, key: string, fallback: boolean): boolean {
  const v = source[key];
  return typeof v === 'boolean' ? v : fallback;
}

// ----------------------------------------------------------------------
// Shared row builders (pure functions — no `this` capture).
// ----------------------------------------------------------------------

function makeSection(title: string): HTMLElement {
  const section = document.createElement('div');
  section.className = 'settings-section';
  const heading = document.createElement('div');
  heading.className = 'settings-section-title';
  heading.textContent = title;
  section.appendChild(heading);
  return section;
}

function numberRow(
  label: string,
  initial: number,
  bounds: { min: number; max: number; step: number },
  onChange: (value: number) => void,
): HTMLElement {
  const row = document.createElement('div');
  row.className = 'settings-row';
  const labelEl = document.createElement('span');
  labelEl.className = 'settings-row-label';
  labelEl.textContent = label;
  const input = document.createElement('input');
  input.type = 'number';
  input.min = String(bounds.min);
  input.max = String(bounds.max);
  input.step = String(bounds.step);
  input.value = String(initial);
  input.addEventListener('change', () => {
    const v = Number(input.value);
    if (Number.isFinite(v)) onChange(v);
  });
  row.appendChild(labelEl);
  row.appendChild(input);
  return row;
}

function textRow(
  label: string,
  initial: string,
  onChange: (value: string) => void,
): HTMLElement {
  const row = document.createElement('div');
  row.className = 'settings-row';
  const labelEl = document.createElement('span');
  labelEl.className = 'settings-row-label';
  labelEl.textContent = label;
  const input = document.createElement('input');
  input.type = 'text';
  input.value = initial;
  input.placeholder = '(inherit)';
  input.addEventListener('change', () => onChange(input.value.trim()));
  row.appendChild(labelEl);
  row.appendChild(input);
  return row;
}

function selectRow(
  label: string,
  initial: string,
  options: Array<{ value: string; label: string }>,
  onChange: (value: string) => void,
): HTMLElement {
  const row = document.createElement('div');
  row.className = 'settings-row';
  const labelEl = document.createElement('span');
  labelEl.className = 'settings-row-label';
  labelEl.textContent = label;
  const select = document.createElement('select');
  for (const o of options) {
    const opt = document.createElement('option');
    opt.value = o.value;
    opt.textContent = o.label;
    if (o.value === initial) opt.selected = true;
    select.appendChild(opt);
  }
  select.addEventListener('change', () => onChange(select.value));
  row.appendChild(labelEl);
  row.appendChild(select);
  return row;
}

function checkboxRow(
  label: string,
  initial: boolean,
  onChange: (value: boolean) => void,
): HTMLElement {
  const row = document.createElement('label');
  row.className = 'settings-checkbox-row';
  const cb = document.createElement('input');
  cb.type = 'checkbox';
  cb.checked = initial;
  cb.addEventListener('change', () => onChange(cb.checked));
  const span = document.createElement('span');
  span.textContent = label;
  row.appendChild(cb);
  row.appendChild(span);
  return row;
}

function paletteRow(
  label: string,
  initial: string,
  onChange: (hex: string) => void,
): HTMLElement {
  const row = document.createElement('div');
  row.className = 'settings-row';
  const labelEl = document.createElement('span');
  labelEl.className = 'settings-row-label';
  labelEl.textContent = label;
  const input = document.createElement('input');
  input.type = 'color';
  input.value = normalizeHexForInput(initial);
  input.addEventListener('input', () => onChange(input.value));
  row.appendChild(labelEl);
  row.appendChild(input);
  return row;
}

function allowListRow(
  label: string,
  options: string[],
  current: string[] | null,
  onChange: (selected: string[]) => void,
  labelFor?: (key: string) => string,
): HTMLElement {
  // `current === null` means "all" — render every checkbox checked.
  const wrap = document.createElement('div');
  const title = document.createElement('div');
  title.className = 'settings-row-label';
  title.textContent = label;
  wrap.appendChild(title);

  const selected = new Set<string>(current ?? options);
  for (const opt of options) {
    const row = document.createElement('label');
    row.className = 'settings-checkbox-row';
    const cb = document.createElement('input');
    cb.type = 'checkbox';
    cb.checked = selected.has(opt);
    cb.addEventListener('change', () => {
      if (cb.checked) selected.add(opt);
      else selected.delete(opt);
      // If the user has every option ticked, treat as "all" by emitting
      // the full list — the host stores it as-is and the renderer's
      // pickAllowList treats every-allowed identically.
      onChange(Array.from(selected));
    });
    const span = document.createElement('span');
    span.textContent = labelFor ? labelFor(opt) : opt;
    row.appendChild(cb);
    row.appendChild(span);
    wrap.appendChild(row);
  }
  return wrap;
}

function emptyHint(text: string): HTMLElement {
  const el = document.createElement('div');
  el.className = 'settings-empty';
  el.textContent = text;
  return el;
}


/** Inline italic note for filter rows that have side effects beyond
 *  the cached-payload restyle path (i.e. they round-trip to the
 *  server). Appended as a sibling under the row so it sits directly
 *  beneath the input without breaking the row's flex layout. */
function filterHint(text: string): HTMLElement {
  const el = document.createElement('div');
  el.className = 'settings-row-hint';
  el.textContent = text;
  return el;
}


/** <input type="color"> only accepts 7-char #rrggbb. Coerce 3-char or
 *  named-fallback values defensively. */
function normalizeHexForInput(value: string): string {
  if (/^#[0-9a-fA-F]{6}$/.test(value)) return value;
  if (/^#[0-9a-fA-F]{3}$/.test(value)) {
    const r = value[1];
    const g = value[2];
    const b = value[3];
    return `#${r}${r}${g}${g}${b}${b}`;
  }
  return '#888888';
}
