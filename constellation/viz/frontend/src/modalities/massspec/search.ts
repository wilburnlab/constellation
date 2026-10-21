// What a search-box entry asks for, and matching it against what was found.
//
// The grammar (the plan's "search bar"):
//   scan 12345 | #12345          → that scan
//   rt 35.2                      → centre the retention-time window there (minutes)
//   512.27 | mz 512.27           → an extracted-ion chromatogram at that m/z
//   PEPTIDEK, PEPTIDEK/2,
//   EM[Oxidation]EVTSESK/2       → a peptide: identified precursors of it,
//                                   else the peptide itself (the server works out its m/z)
//   anything else                → identifications whose protein, sequence or
//                                   modified sequence contains it

export type Query =
  | { kind: 'scan'; scan: number }
  | { kind: 'rt'; minutes: number }
  | { kind: 'mz'; mz: number }
  | { kind: 'peptide'; text: string; charge: number | null }
  | { kind: 'text'; text: string }
  | { kind: 'empty' };

const AMINO_ACIDS = /^[ACDEFGHIKLMNPQRSTVWY]{2,}$/;

/** Whether `text` (with its `/charge` removed) reads as a peptide: amino-acid
 *  letters, with ProForma modifications in brackets. A gene name made only
 *  of such letters (GAPDH) reads as one too, so callers also try the text
 *  as a protein. */
function asPeptide(text: string): string | null {
  const bare = text.replace(/\[[^\]]*\]-?/g, '').replace(/-$/, '');
  return AMINO_ACIDS.test(bare) ? text : null;
}

export function parseQuery(raw: string): Query {
  const text = raw.trim();
  if (!text) return { kind: 'empty' };
  let m = /^(?:scan\s*|#)(\d+)$/i.exec(text);
  if (m) return { kind: 'scan', scan: Number(m[1]) };
  m = /^rt\s*(\d+(?:\.\d+)?)$/i.exec(text);
  if (m) return { kind: 'rt', minutes: Number(m[1]) };
  m = /^(?:mz\s*)?(\d+\.\d+|\d{3,5})$/i.exec(text);
  if (m) return { kind: 'mz', mz: Number(m[1]) };
  m = /^(.*?)(?:\/(\d+))?$/.exec(text);
  const body = m ? m[1] : text;
  if (asPeptide(body)) return { kind: 'peptide', text: body, charge: m && m[2] ? Number(m[2]) : null };
  return { kind: 'text', text };
}

/** The fields of an identification a search matches against. */
export interface Searchable {
  modified_sequence: string | null;
  sequence: string | null;
  proteins: string | null;
  charge: number | null;
}

/** Rows of `rows` a query matches, best first, at most `limit`. A peptide
 *  query matches sequences exactly (and the charge, when given) before it
 *  matches them as a prefix; a text query matches anywhere, ignoring case. */
export function matchIdentifications<T extends Searchable>(rows: readonly T[], query: Query, limit = 20): T[] {
  if (query.kind === 'peptide') {
    const plain = query.text.replace(/\[[^\]]*\]-?/g, '');
    const scored: Array<[number, T]> = [];
    for (const r of rows) {
      if (query.charge !== null && r.charge !== query.charge) continue;
      if (r.modified_sequence === query.text) scored.push([0, r]);
      else if (r.sequence === plain) scored.push([1, r]);
      else if (r.sequence?.startsWith(plain)) scored.push([2, r]);
    }
    return scored.sort((a, b) => a[0] - b[0]).slice(0, limit).map(([, r]) => r);
  }
  if (query.kind === 'text') {
    const needle = query.text.toLowerCase();
    const out: T[] = [];
    for (const r of rows) {
      if (
        r.proteins?.toLowerCase().includes(needle) ||
        r.sequence?.toLowerCase().includes(needle) ||
        r.modified_sequence?.toLowerCase().includes(needle)
      ) {
        out.push(r);
        if (out.length >= limit) break;
      }
    }
    return out;
  }
  return [];
}
