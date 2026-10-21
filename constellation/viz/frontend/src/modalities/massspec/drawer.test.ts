// The drawer on its own: residues of a ProForma sequence, the ion table and
// coverage line, and the identifications table's sorting and filtering.

import { describe, expect, it } from 'vitest';
import { DRAWN_ROWS, Drawer, IdRow, residuesOf } from './drawer';
import type { Target } from './target';

function target(modseq: string, ladder: Target['ladder']): Target {
  return {
    key: 'k',
    label: `${modseq}/2`,
    modifiedSequence: modseq,
    sequence: modseq.replace(/\[[^\]]*\]-?/g, ''),
    charge: 2,
    precursorMz: 500,
    rt: null,
    rtStart: null,
    rtEnd: null,
    predictedRt: null,
    window: null,
    apexScans: { ms1: null, ms2: null },
    ms1: [],
    ms2: [],
    ladder,
    median: [],
    fragmentTolerance: null,
    precursorTolerance: null,
    qvalue: null,
    proteins: null,
    source: null,
  };
}

function row(i: number, extra: Partial<IdRow> = {}): IdRow {
  return {
    key: `e|u|${i}`,
    experiment: 'e',
    unit: 'u',
    precursor_id: String(i),
    modified_sequence: `PEP${i}K`,
    sequence: `PEP${i}K`,
    charge: 2,
    precursor_mz: 400 + i,
    rt: 60 * i,
    intensity: 1000 * i,
    qvalue: 0.01 / (i + 1),
    proteins: i % 2 ? 'P1' : 'P2',
    scan: null,
    ...extra,
  };
}

describe('residuesOf', () => {
  it('keeps each residue with its modifications and drops terminal ones', () => {
    expect(residuesOf('PEPTIDEK')).toEqual(['P', 'E', 'P', 'T', 'I', 'D', 'E', 'K']);
    expect(residuesOf('[Acetyl]-EM[Oxidation]C[UNIMOD:4]K')).toEqual(['E', 'M[Oxidation]', 'C[UNIMOD:4]', 'K']);
  });
});

describe('Drawer', () => {
  it('lists b ions down and y ions up, lighting what the spectrum matched', () => {
    const d = new Drawer({ onSelect: () => undefined });
    const ladder = [
      { ion_type: 'b', ion_number: 1, position: 0, fragment_charge: 1, loss: null, mz: 98.06, annotation: 'b1' },
      { ion_type: 'b', ion_number: 2, position: 1, fragment_charge: 1, loss: null, mz: 227.1, annotation: 'b2' },
      { ion_type: 'y', ion_number: 1, position: 2, fragment_charge: 1, loss: null, mz: 147.11, annotation: 'y1' },
      { ion_type: 'y', ion_number: 2, position: 1, fragment_charge: 1, loss: null, mz: 276.16, annotation: 'y2' },
    ];
    d.setIons(target('PEK', ladder), new Set(['b2', 'y1^2']));
    const rows = [...d.element.querySelectorAll('.ms-ion-table tbody tr')].map((tr) =>
      [...tr.querySelectorAll('td')].map((td) => td.textContent),
    );
    expect(rows).toEqual([
      ['1', '98.06', 'P', '', ''],
      ['2', '227.10', 'E', '276.16', '2'],
      ['', '', 'K', '147.11', '1'],
    ]);
    const lit = [...d.element.querySelectorAll('.ms-ion-table td.matched')].map((td) => td.textContent);
    // y1 is lit by its doubly charged form.
    expect(lit).toEqual(['227.10', '147.11']);
    const bonds = [...d.element.querySelectorAll<HTMLElement>('.ms-coverage-bond')].map((b) => [b.dataset.b, b.dataset.y]);
    expect(bonds).toEqual([
      ['false', 'false'],
      ['true', 'true'],
    ]);
  });

  it('sorts by q-value, filters by text and draws at most DRAWN_ROWS rows', () => {
    const picked: IdRow[] = [];
    const d = new Drawer({ onSelect: (r) => picked.push(r) });
    d.setIdentifications(Array.from({ length: DRAWN_ROWS + 50 }, (_, i) => row(i)));
    const body = d.element.querySelector('.ms-id-table tbody')!;
    expect(body.children.length).toBe(DRAWN_ROWS);
    // q descends with i, so the best (smallest) is the last row given.
    expect(d.order[0].precursor_id).toBe(String(DRAWN_ROWS + 49));
    expect(d.element.querySelector('.ms-drawer-count')!.textContent).toBe(`${DRAWN_ROWS + 50} of ${DRAWN_ROWS + 50}`);
    const filter = d.element.querySelector<HTMLInputElement>('.ms-drawer-controls input')!;
    filter.value = 'pep12k';
    filter.dispatchEvent(new Event('input'));
    expect(d.order.map((r) => r.precursor_id)).toEqual(['12']);
    (body.firstElementChild as HTMLElement).click();
    expect(picked.map((r) => r.precursor_id)).toEqual(['12']);
    // Sorting by RT, then again to reverse it.
    filter.value = '';
    filter.dispatchEvent(new Event('input'));
    const rtHeader = d.element.querySelector<HTMLElement>('th[data-key="rt"]')!;
    rtHeader.click();
    expect(d.order[0].precursor_id).toBe('0');
    rtHeader.click();
    expect(d.order[0].precursor_id).toBe(String(DRAWN_ROWS + 49));
  });

  it('says why there is nothing to list', () => {
    const d = new Drawer({ onSelect: () => undefined });
    d.setEmpty('No search of this run is attached.');
    expect(d.element.querySelector('.ms-id-table')!.hasAttribute('hidden')).toBe(true);
    expect(d.element.textContent).toContain('No search of this run is attached.');
  });
});
