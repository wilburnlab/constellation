// Per-binding style + filter helpers shared across renderers.
//
// Each renderer keeps its own defaults (the hardcoded palette / sizes
// that shipped before per-panel controls existed). These helpers exist
// only to coerce values out of the opaque `ctx.style` / `ctx.filter`
// dicts with type-safe fallbacks, so a missing override falls back to
// the renderer's literal default and a typo in the saved state can't
// blow up the render loop.

export function pickString(
  source: Record<string, unknown> | undefined,
  key: string,
  fallback: string,
): string {
  const v = source?.[key];
  return typeof v === 'string' && v.length > 0 ? v : fallback;
}

export function pickNumber(
  source: Record<string, unknown> | undefined,
  key: string,
  fallback: number,
): number {
  const v = source?.[key];
  if (typeof v === 'number' && Number.isFinite(v)) return v;
  if (typeof v === 'string') {
    const parsed = Number(v);
    if (Number.isFinite(parsed)) return parsed;
  }
  return fallback;
}

export function pickBool(
  source: Record<string, unknown> | undefined,
  key: string,
  fallback: boolean,
): boolean {
  const v = source?.[key];
  if (typeof v === 'boolean') return v;
  return fallback;
}

/** Resolve a palette lookup with per-key override + default. The `style`
 *  object may carry either a flat `palette.<key>` (one-level dotted, the
 *  shape we serialize) or a nested `palette: {<key>: ...}` (one-level
 *  TOML table). Both are accepted. */
export function pickPaletteColor(
  source: Record<string, unknown> | undefined,
  key: string,
  fallback: string,
): string {
  if (!source) return fallback;
  const dotted = source[`palette.${key}`];
  if (typeof dotted === 'string' && dotted.length > 0) return dotted;
  const nested = source.palette;
  if (nested && typeof nested === 'object') {
    const v = (nested as Record<string, unknown>)[key];
    if (typeof v === 'string' && v.length > 0) return v;
  }
  return fallback;
}

/** Resolve a category-based palette where the keys are numeric (sample
 *  ids etc.). Falls back to cycling through `cycle[idx % cycle.length]`. */
export function pickCycledColor(
  source: Record<string, unknown> | undefined,
  key: string | number,
  cycle: readonly string[],
  cycleIdx: number,
): string {
  const fallback = cycle[Math.max(0, cycleIdx) % cycle.length];
  return pickPaletteColor(source, String(key), fallback);
}

/** Read a filter that may be `"all"` (no constraint) or an array of
 *  allowed values. Returns null when there is no constraint. */
export function pickAllowList<T extends string | number>(
  source: Record<string, unknown> | undefined,
  key: string,
): Set<string> | null {
  const v = source?.[key];
  if (v === undefined || v === null || v === 'all') return null;
  if (!Array.isArray(v)) return null;
  const out = new Set<string>();
  for (const item of v as readonly (T | string | number)[]) {
    if (item === null || item === undefined) continue;
    out.add(String(item));
  }
  return out;
}
