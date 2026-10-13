// What a panel host needs to know about a kind of panel, declared by the
// kind itself.
//
// A host used to carry this as tables keyed by kind name — a display
// order list, a unit-noun map, a set of server-side filter keys. Each
// renderer now states these for itself, so a host handles any kind
// without naming it and a new kind is one module.

import { SettingsSchema } from './settings_schema';

/** Turn a stored filter value into its query-string form, or `undefined`
 *  to leave the parameter out of the request. */
export type PushdownEncoder = (value: unknown) => string | undefined;

export interface PanelKind {
  /** Registry key; matches the server kernel's `kind`. */
  kind: string;

  /** Default position among kinds, lower first. Panels of one kind keep
   *  the order the server listed them in. */
  order: number;

  /** `[singular, plural]` noun for the status line ("showing 4 reads"). */
  unit: readonly [string, string];

  /** Filter keys the server applies rather than the renderer.
   *
   *  Presence of a key is what makes it "pushdown": a change to it means
   *  refetching, whereas any other filter or style key is re-applied
   *  client-side from the table already fetched. The encoder maps the
   *  stored value to the request parameter of the same name. Parameters
   *  are sent in the order the keys are declared. */
  pushdown?: Readonly<Record<string, PushdownEncoder>>;

  /** The controls this kind's settings popover offers. */
  settings?: SettingsSchema;
}

/** Sort rank for a kind no renderer is registered for: after every
 *  known kind. */
export const UNKNOWN_KIND_ORDER = 1_000_000;

/** The noun for `n` items of a kind (or the generic one). */
export function unitFor(kind: PanelKind | null | undefined, n: number): string {
  const [singular, plural] = kind?.unit ?? ['row', 'rows'];
  return n === 1 ? singular : plural;
}

/** Request parameters for a kind's pushdown filters, in declared order. */
export function encodePushdown(
  kind: PanelKind | null | undefined,
  filter: Readonly<Record<string, unknown>>,
): Record<string, string> {
  const out: Record<string, string> = {};
  for (const [key, encode] of Object.entries(kind?.pushdown ?? {})) {
    const value = encode(filter[key]);
    if (value !== undefined) out[key] = value;
  }
  return out;
}

/** True when any of a kind's pushdown filter values differs between the
 *  two filter dicts (including present ↔ absent). */
export function pushdownChanged(
  kind: PanelKind | null | undefined,
  before: Readonly<Record<string, unknown>>,
  after: Readonly<Record<string, unknown>>,
): boolean {
  for (const key of Object.keys(kind?.pushdown ?? {})) {
    if (!sameValue(before[key], after[key])) return true;
  }
  return false;
}

function sameValue(a: unknown, b: unknown): boolean {
  if (a === b) return true;
  if (a === undefined || b === undefined) return false;
  if (a === null || b === null) return false;
  return JSON.stringify(a) === JSON.stringify(b);
}
