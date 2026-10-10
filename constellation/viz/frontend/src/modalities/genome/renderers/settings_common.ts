// Pieces the genome track kinds share when declaring their settings
// (see panels/settings_schema.ts).

import {
  Control,
  Opt,
  PaletteEntry,
  SettingsEnv,
  SettingsSchema,
  SettingsSection,
} from '../../../panels/settings_schema';

/** Per-sample colour cycle. Shared by coverage and read pile-up so the
 *  same sample has the same colour identity across tracks. */
export const SAMPLE_PALETTE_CYCLE: readonly string[] = [
  '#4f9efb',
  '#fb7c4f',
  '#a4d65e',
  '#d65eb6',
  '#5ed6cf',
];

/** A number control. */
export function num(
  target: 'style' | 'filter',
  key: string,
  label: string,
  fallback: number,
  min: number,
  max: number,
  step: number,
  hint?: string,
): Control {
  return { type: 'number', target, key, label, default: fallback, min, max, step, hint };
}

/** An opacity control (0–1 in steps of 0.05). */
export function opacity(key: string, label: string, fallback: number): Control {
  return num('style', key, label, fallback, 0, 1, 0.05);
}

export function toggle(
  target: 'style' | 'filter',
  key: string,
  label: string,
  fallback: boolean,
): Control {
  return { type: 'toggle', target, key, label, default: fallback };
}

/** Options whose label is the value itself. */
export function plainOptions(values: readonly string[]): Opt[] {
  return values.map((value) => ({ value, label: value }));
}

/** The "General" section every genome track's popover opens with. */
export function generalSection(defaultOpacity: number): SettingsSection {
  return {
    title: 'General',
    controls: [
      opacity('opacity', 'Opacity', defaultOpacity),
      { type: 'text', target: 'style', key: 'label_font_family', label: 'Label font family', default: '' },
      num('style', 'label_font_size_px', 'Label font size (px)', 10, 6, 24, 1),
      toggle('style', 'show_legend', 'Show legend / labels', true),
    ],
  };
}

/** What the popover shows for a kind with no renderer registered. */
export const FALLBACK_SETTINGS: SettingsSchema = {
  sections: [
    generalSection(1.0),
    { title: 'Style', controls: [{ type: 'note', text: 'no style controls for this track kind' }] },
    { title: 'Filter', controls: [{ type: 'note', text: 'no filter controls for this track kind' }] },
  ],
};

// ----------------------------------------------------------------------
// Metadata coercion
// ----------------------------------------------------------------------

export function numericList(source: unknown): number[] {
  if (!Array.isArray(source)) return [];
  const out: number[] = [];
  for (const v of source as unknown[]) {
    const n = Number(v);
    if (Number.isFinite(n)) out.push(n);
  }
  return out;
}

export function stringList(source: unknown): string[] {
  if (!Array.isArray(source)) return [];
  const out: string[] = [];
  for (const v of source as unknown[]) {
    if (v === null || v === undefined) continue;
    out.push(String(v));
  }
  return out;
}

/** Parallel-array companion to `numericList`: preserves explicit
 *  null/undefined slots so indices pair against another array
 *  (`samples_in_data[i]` <-> `sample_names[i]`). */
export function stringOrNullList(source: unknown): Array<string | null> {
  if (!Array.isArray(source)) return [];
  const out: Array<string | null> = [];
  for (const v of source as unknown[]) {
    if (v === null || v === undefined) {
      out.push(null);
    } else {
      out.push(String(v));
    }
  }
  return out;
}

/** `list`, or `fallback` when the metadata did not supply any. */
export function orFallback(list: string[], fallback: readonly string[]): string[] {
  return list.length > 0 ? list : [...fallback];
}

// ----------------------------------------------------------------------
// Samples — `samples_in_data` paired with the optional `sample_names`
// ----------------------------------------------------------------------

interface Sample {
  id: number;
  name: string | null;
}

function samplesOf(env: SettingsEnv): Sample[] {
  const ids = numericList(env.meta.samples_in_data);
  const names = stringOrNullList(env.meta.sample_names);
  return ids.map((id, idx) => ({ id, name: names[idx] ?? null }));
}

function sampleLabel(sample: Sample): string {
  return sample.name ? `${sample.name} (${sample.id})` : `sample ${sample.id}`;
}

/** One colour per sample. `unstratified` labels the `-1` pseudo-sample
 *  that an unstratified coverage table carries. */
export function samplePalette(options: { unstratified?: string } = {}) {
  return (env: SettingsEnv): PaletteEntry[] =>
    samplesOf(env).map((sample, idx) => ({
      key: String(sample.id),
      label:
        sample.id === -1 && options.unstratified !== undefined
          ? options.unstratified
          : sampleLabel(sample),
      default: SAMPLE_PALETTE_CYCLE[idx % SAMPLE_PALETTE_CYCLE.length],
    }));
}

/** One allow-list option per sample, valued by its numeric id. */
export function sampleOptions(options: { unstratified?: string } = {}) {
  return (env: SettingsEnv): Opt<number>[] =>
    samplesOf(env).map((sample) => ({
      value: sample.id,
      label:
        sample.id === -1 && options.unstratified !== undefined
          ? options.unstratified
          : sampleLabel(sample),
    }));
}

/** The note shown beside controls that are applied by the server. */
export const REFETCH_HINT = 'changing this triggers a refetch';
