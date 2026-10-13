// Declarative description of a panel's settings popover.
//
// A panel kind states which controls its gear popover offers; the
// generic `SettingsPanel` renders them. Values land in the panel's two
// opaque dicts — `style` (how it is drawn) and `filter` (which rows are
// drawn, or fetched) — which the layout layer persists without looking
// inside.
//
// The schema is data plus a few small functions: anything that depends
// on what the server reported about the panel (which samples exist,
// whether a view is available) is a function of the environment below,
// so option lists and visibility can follow the data without the
// popover knowing anything about the kind.

export type Bag = Readonly<Record<string, unknown>>;

/** What a schema function may look at. */
export interface SettingsEnv {
  /** The panel's server-side metadata. */
  meta: Bag;
  /** Current values. */
  style: Bag;
  filter: Bag;
  /** Host-level state a default may follow (e.g. a toolbar toggle). */
  host: Bag;
}

/** A fixed value, or one computed from the environment. */
export type Dyn<T> = T | ((env: SettingsEnv) => T);

export type When = (env: SettingsEnv) => boolean;

export interface Opt<V extends string | number = string> {
  value: V;
  label: string;
}

/** One colour of a palette, stored under `style["palette.<key>"]`. */
export interface PaletteEntry {
  key: string;
  label: string;
  default: string;
}

interface Keyed {
  /** Which dict the value is written to. */
  target: 'style' | 'filter';
  key: string;
  label: string;
  /** Shown only when this holds. */
  when?: When;
  /** Small note under the control, e.g. that a change refetches. */
  hint?: string;
}

export type Control =
  | (Keyed & { type: 'number'; default: number; min: number; max: number; step: number })
  /** Free text; clearing it removes the key. */
  | (Keyed & { type: 'text'; default: string })
  | (Keyed & { type: 'select'; default: string; options: Opt[] })
  | (Keyed & { type: 'toggle'; default: Dyn<boolean> })
  /** A checkbox per option; an unset key means "all". Options whose
   *  values are numbers are stored as numbers. With no options the
   *  control is omitted, or replaced by `emptyHint`. */
  | (Omit<Keyed, 'target'> & {
      type: 'allowlist';
      target: 'filter';
      options: Dyn<Opt<string | number>[]>;
      emptyHint?: string;
    })
  /** One colour row per entry. With no entries the control is omitted,
   *  or replaced by `emptyHint`. */
  | { type: 'palette'; entries: Dyn<PaletteEntry[]>; emptyHint?: string; when?: When }
  /** A line of explanatory text in place of controls. */
  | { type: 'note'; text: string; when?: When };

export interface SettingsSection {
  title: string;
  controls: Control[];
  when?: When;
}

export interface SettingsSchema {
  sections: SettingsSection[];
}

export function resolve<T>(value: Dyn<T>, env: SettingsEnv): T {
  return typeof value === 'function' ? (value as (env: SettingsEnv) => T)(env) : value;
}
