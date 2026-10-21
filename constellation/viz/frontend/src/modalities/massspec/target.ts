// A target: the peptide precursor the browser is showing, identified by a
// search or typed by hand, with everything the panels draw for it.

export type ToleranceUnit = 'ppm' | 'Da';

/** One m/z to trace. */
export interface TraceTarget {
  mz: number;
  label: string;
  /** Whether the engine quantified from it; null when not known. */
  quantified?: boolean | null;
}

/** One theoretical ion (`massspec.spectra.annotate.ion_ladder`). */
export interface LadderRow {
  ion_type: string;
  ion_number: number;
  position: number;
  fragment_charge: number;
  loss: string | null;
  mz: number;
  annotation: string;
}

export interface Target {
  /** Identifies what is shown (an identification of a search, or a peptide). */
  key: string;
  /** "PEPTIDEK/2". */
  label: string;
  modifiedSequence: string;
  sequence: string;
  charge: number;
  precursorMz: number;
  /** Apex and integration window (seconds); null for a peptide typed by hand. */
  rt: number | null;
  rtStart: number | null;
  rtEnd: number | null;
  /** The searched library's RT mapped onto this run. */
  predictedRt: number | null;
  /** The DIA window isolating it (MS2 row), null for DDA or unknown. */
  window: number | null;
  apexScans: { ms1: number | null; ms2: number | null };
  ms1: TraceTarget[];
  ms2: TraceTarget[];
  ladder: LadderRow[];
  /** The engine's median chromatogram (arbitrary units). */
  median: Array<{ rt: number; intensity: number }>;
  /** The search's tolerances; null to use the analyzer's defaults. */
  fragmentTolerance: [number, ToleranceUnit] | null;
  precursorTolerance: [number, ToleranceUnit] | null;
  qvalue: number | null;
  proteins: string | null;
  /** The search it came from, for an identification. Ids are 64-bit, so
   *  they stay strings (a JavaScript number would round them). */
  source: { experiment: string; unit: string; precursorId: string } | null;
}

function tolerance(raw: unknown): [number, ToleranceUnit] | null {
  if (!raw || typeof raw !== 'object') return null;
  const { value, unit } = raw as { value?: unknown; unit?: unknown };
  if (typeof value !== 'number' || (unit !== 'ppm' && unit !== 'Da')) return null;
  return [value, unit];
}

const num = (v: unknown): number | null => (typeof v === 'number' && Number.isFinite(v) ? v : null);

/** A target from `…/acquisitions/{key}/precursor`, asked for `precursorId`
 *  (the id in the reply has been through JSON.parse and may be rounded)
 *  and, for a spectrum-centric match, its `scan`. */
export function targetFromPrecursor(json: Record<string, unknown>, precursorId: string, scan: number | null = null): Target {
  const id = json.identification as Record<string, unknown>;
  const located = (json.located ?? {}) as Record<string, unknown>;
  const modseq = String(id.modified_sequence ?? '');
  const charge = Number(id.charge);
  const ms1 = (json.ms1_targets as Array<Record<string, unknown>>) ?? [];
  const ms2 = (json.ms2_targets as Array<Record<string, unknown>>) ?? [];
  return {
    key: `${json.experiment}|${json.unit}|${precursorId}${scan !== null ? `|${scan}` : ''}`,
    label: `${modseq}/${charge}`,
    modifiedSequence: modseq,
    sequence: String(id.sequence ?? modseq),
    charge,
    precursorMz: Number(id.precursor_mz),
    rt: num(id.rt),
    rtStart: num(id.rt_start),
    rtEnd: num(id.rt_end),
    predictedRt: num(json.predicted_rt),
    window: num(located.window),
    apexScans: { ms1: num(located.parent_scan), ms2: num(located.ms_scan) },
    ms1: ms1.map((t) => ({ mz: Number(t.mz), label: String(t.label) })),
    ms2: ms2.map((t) => ({ mz: Number(t.mz), label: String(t.label), quantified: (t.quantified as boolean | null) ?? null })),
    ladder: (json.ladder as LadderRow[]) ?? [],
    median: (json.chromatogram as Array<{ rt: number; intensity: number }>) ?? [],
    fragmentTolerance: tolerance(json.fragment_tolerance) ?? tolerance(json.precursor_tolerance),
    precursorTolerance: tolerance(json.precursor_tolerance),
    qvalue: num(id.qvalue),
    proteins: (id.proteins as string | null) ?? null,
    source: { experiment: String(json.experiment), unit: String(json.unit), precursorId },
  };
}

/** A target from `…/resolve` for a peptide typed by hand. */
export function targetFromResolve(json: Record<string, unknown>): Target {
  const charge = Number(json.charge);
  const options = (json.charges as Array<{ charge: number; precursor_mz: number }>) ?? [];
  const chosen = options.find((c) => c.charge === charge) ?? options[0];
  const modseq = String(json.modified_sequence);
  const ms1 = (json.ms1_targets as Array<Record<string, unknown>>) ?? [];
  const ms2 = (json.ms2_targets as Array<Record<string, unknown>>) ?? [];
  return {
    key: `peptide|${modseq}/${charge}`,
    label: `${modseq}/${charge}`,
    modifiedSequence: modseq,
    sequence: String(json.sequence ?? modseq),
    charge,
    precursorMz: chosen ? chosen.precursor_mz : NaN,
    rt: null,
    rtStart: null,
    rtEnd: null,
    predictedRt: null,
    window: null,
    apexScans: { ms1: null, ms2: null },
    ms1: ms1.map((t) => ({ mz: Number(t.mz), label: String(t.label) })),
    ms2: ms2.map((t) => ({ mz: Number(t.mz), label: String(t.label), quantified: null })),
    ladder: (json.ladder as LadderRow[]) ?? [],
    median: [],
    fragmentTolerance: null,
    precursorTolerance: null,
    qvalue: null,
    proteins: null,
    source: null,
  };
}
