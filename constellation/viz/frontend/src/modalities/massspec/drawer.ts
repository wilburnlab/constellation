// The browser's side drawer: what was identified, the ions of the target,
// and the header of the scan being shown.
//
// Identifications are held as plain rows (a 100-minute DIA run gives about
// 32,000) and only the first rows of the sorted, filtered list are drawn.
// The Ions tab is the usual ion table (b ions down one side, y ions up the
// other, residue between) with the ions the shown spectrum matched lit in
// their series colour, and a coverage line over the sequence.

import type { Table } from 'apache-arrow';
import { ionColor } from './palette';
import type { LadderRow, Target } from './target';

/** One identification, as the drawer and the search box hold it. 64-bit
 *  ids stay strings. */
export interface IdRow {
  key: string;
  experiment: string;
  unit: string;
  precursor_id: string;
  modified_sequence: string | null;
  sequence: string | null;
  charge: number | null;
  precursor_mz: number | null;
  rt: number | null;
  intensity: number | null;
  qvalue: number | null;
  proteins: string | null;
  scan: number | null;
}

/** Rows from the identifications route's Arrow table. */
export function idRows(table: Table): IdRow[] {
  const col = (name: string) => table.getChild(name);
  const get = (name: string, i: number): unknown => col(name)?.get(i) ?? null;
  const num = (v: unknown): number | null => (v === null || v === undefined ? null : Number(v));
  const out: IdRow[] = [];
  for (let i = 0; i < table.numRows; i++) {
    const experiment = String(get('experiment', i) ?? '');
    const unit = String(get('unit', i) ?? '');
    const precursor = get('precursor_id', i);
    const precursorId = precursor === null ? '' : String(precursor);
    const scan = num(get('scan', i));
    out.push({
      key: `${experiment}|${unit}|${precursorId}${scan !== null ? `|${scan}` : ''}`,
      experiment,
      unit,
      precursor_id: precursorId,
      modified_sequence: get('modified_sequence', i) as string | null,
      sequence: get('sequence', i) as string | null,
      charge: num(get('charge', i)),
      precursor_mz: num(get('precursor_mz', i)),
      rt: num(get('rt', i)),
      intensity: num(get('intensity', i)),
      qvalue: num(get('qvalue', i)),
      proteins: get('proteins', i) as string | null,
      scan,
    });
  }
  return out;
}

type SortKey = 'modified_sequence' | 'charge' | 'precursor_mz' | 'rt' | 'qvalue' | 'intensity' | 'proteins';
const COLUMNS: Array<{ key: SortKey; label: string; format: (r: IdRow) => string }> = [
  { key: 'modified_sequence', label: 'Peptide', format: (r) => r.modified_sequence ?? '' },
  { key: 'charge', label: 'z', format: (r) => (r.charge === null ? '' : String(r.charge)) },
  { key: 'precursor_mz', label: 'm/z', format: (r) => (r.precursor_mz === null ? '' : r.precursor_mz.toFixed(3)) },
  { key: 'rt', label: 'RT', format: (r) => (r.rt === null ? '' : (r.rt / 60).toFixed(2)) },
  { key: 'qvalue', label: 'q', format: (r) => (r.qvalue === null ? '' : r.qvalue.toExponential(1)) },
  { key: 'intensity', label: 'Intensity', format: (r) => (r.intensity === null ? '' : r.intensity.toExponential(2)) },
  { key: 'proteins', label: 'Proteins', format: (r) => r.proteins ?? '' },
];

/** Rows drawn at once; the rest are reached by filtering or sorting. */
export const DRAWN_ROWS = 300;

export type DrawerTab = 'identifications' | 'ions' | 'scan';

export interface DrawerOptions {
  onSelect(row: IdRow): void;
}

export class Drawer {
  readonly element: HTMLElement;
  private readonly opts: DrawerOptions;
  private readonly tabs = new Map<DrawerTab, { button: HTMLButtonElement; body: HTMLElement }>();
  private rows: IdRow[] = [];
  private sorted: IdRow[] = [];
  private sortKey: SortKey = 'qvalue';
  private ascending = true;
  private filterText = '';
  private selected: string | null = null;
  private readonly filterInput: HTMLInputElement;
  private readonly countEl: HTMLElement;
  private readonly emptyEl: HTMLElement;
  private readonly idTable: HTMLTableElement;
  private readonly tableBody: HTMLTableSectionElement;
  private readonly headRow: HTMLTableRowElement;

  constructor(opts: DrawerOptions) {
    this.opts = opts;
    this.element = document.createElement('aside');
    this.element.className = 'ms-drawer';
    const bar = document.createElement('div');
    bar.className = 'ms-drawer-tabs';
    this.element.appendChild(bar);
    for (const [tab, label] of [
      ['identifications', 'Identifications'],
      ['ions', 'Ions'],
      ['scan', 'Scan'],
    ] as Array<[DrawerTab, string]>) {
      const button = document.createElement('button');
      button.type = 'button';
      button.className = 'ms-drawer-tab';
      button.textContent = label;
      button.addEventListener('click', () => this.showTab(tab));
      bar.appendChild(button);
      const body = document.createElement('div');
      body.className = `ms-drawer-body ms-drawer-${tab}`;
      this.element.appendChild(body);
      this.tabs.set(tab, { button, body });
    }

    const ids = this.tabs.get('identifications')!.body;
    const controls = document.createElement('div');
    controls.className = 'ms-drawer-controls';
    this.filterInput = document.createElement('input');
    this.filterInput.type = 'search';
    this.filterInput.placeholder = 'Filter peptides or proteins';
    this.filterInput.addEventListener('input', () => {
      this.filterText = this.filterInput.value.trim().toLowerCase();
      this.refresh();
    });
    this.countEl = document.createElement('span');
    this.countEl.className = 'ms-drawer-count';
    controls.append(this.filterInput, this.countEl);
    const table = document.createElement('table');
    table.className = 'ms-id-table';
    const head = table.createTHead();
    this.headRow = head.insertRow();
    for (const c of COLUMNS) {
      const th = document.createElement('th');
      th.textContent = c.label;
      th.dataset.key = c.key;
      th.addEventListener('click', () => this.sortBy(c.key));
      this.headRow.appendChild(th);
    }
    this.tableBody = table.createTBody();
    this.idTable = table;
    this.emptyEl = note('');
    this.emptyEl.hidden = true;
    ids.append(controls, this.emptyEl, table);
    this.showTab('identifications');
    this.setIons(null, new Set());
    this.setScan(null);
  }

  showTab(tab: DrawerTab): void {
    for (const [name, { button, body }] of this.tabs) {
      button.classList.toggle('active', name === tab);
      body.hidden = name !== tab;
    }
  }

  setIdentifications(rows: IdRow[]): void {
    this.rows = rows;
    this.emptyEl.hidden = true;
    this.idTable.hidden = false;
    this.refresh();
  }

  /** Say why there is nothing to list. */
  setEmpty(message: string): void {
    this.rows = [];
    this.refresh();
    this.emptyEl.textContent = message;
    this.emptyEl.hidden = false;
    this.idTable.hidden = true;
  }

  /** The visible order (sorted and filtered), for stepping between identifications. */
  get order(): readonly IdRow[] {
    return this.sorted;
  }

  setSelected(key: string | null): void {
    this.selected = key;
    for (const tr of Array.from(this.tableBody.rows)) tr.classList.toggle('selected', tr.dataset.key === key);
  }

  private sortBy(key: SortKey): void {
    this.ascending = this.sortKey === key ? !this.ascending : key !== 'intensity';
    this.sortKey = key;
    this.refresh();
  }

  private refresh(): void {
    const needle = this.filterText;
    const kept = needle
      ? this.rows.filter(
          (r) =>
            r.modified_sequence?.toLowerCase().includes(needle) ||
            r.sequence?.toLowerCase().includes(needle) ||
            r.proteins?.toLowerCase().includes(needle),
        )
      : this.rows.slice();
    const k = this.sortKey;
    const sign = this.ascending ? 1 : -1;
    kept.sort((a, b) => {
      const va = a[k];
      const vb = b[k];
      if (va === vb) return 0;
      if (va === null) return 1;
      if (vb === null) return -1;
      return (va < vb ? -1 : 1) * sign;
    });
    this.sorted = kept;
    for (const th of Array.from(this.headRow.cells)) {
      th.classList.toggle('sorted', th.dataset.key === k);
      th.dataset.dir = th.dataset.key === k ? (this.ascending ? 'asc' : 'desc') : '';
    }
    this.countEl.textContent = `${kept.length.toLocaleString()} of ${this.rows.length.toLocaleString()}`;
    this.tableBody.replaceChildren();
    for (const r of kept.slice(0, DRAWN_ROWS)) {
      const tr = this.tableBody.insertRow();
      tr.dataset.key = r.key;
      tr.classList.toggle('selected', r.key === this.selected);
      for (const c of COLUMNS) {
        const td = tr.insertCell();
        td.textContent = c.format(r);
        if (c.key === 'proteins') td.title = r.proteins ?? '';
      }
      tr.addEventListener('click', () => this.opts.onSelect(r));
    }
  }

  /** The ion table of `target`, with the ions in `matched` (mzPAF labels) lit. */
  setIons(target: Target | null, matched: Set<string>): void {
    const body = this.tabs.get('ions')!.body;
    body.replaceChildren();
    if (!target) {
      body.appendChild(note('Pick an identification, or type a peptide in the search box.'));
      return;
    }
    const title = document.createElement('div');
    title.className = 'ms-ions-title';
    title.textContent = target.label;
    body.appendChild(title);
    body.appendChild(coverage(target, matched));
    const residues = residuesOf(target.modifiedSequence);
    const length = residues.length;
    const byLabel = new Map(target.ladder.map((r) => [r.annotation, r] as [string, LadderRow]));
    const table = document.createElement('table');
    table.className = 'ms-ion-table';
    const head = table.createTHead().insertRow();
    for (const h of ['#', 'b', 'residue', 'y', '#']) {
      const th = document.createElement('th');
      th.textContent = h;
      head.appendChild(th);
    }
    const tbody = table.createTBody();
    residues.forEach((residue, i) => {
      const tr = tbody.insertRow();
      const b = byLabel.get(`b${i + 1}`);
      const y = byLabel.get(`y${length - i}`);
      tr.insertCell().textContent = i + 1 < length ? String(i + 1) : '';
      ionCell(tr, i + 1 < length ? b : undefined, matched);
      const res = tr.insertCell();
      res.textContent = residue;
      res.className = 'ms-ion-residue';
      ionCell(tr, i > 0 ? y : undefined, matched);
      tr.insertCell().textContent = i > 0 ? String(length - i) : '';
    });
    body.appendChild(table);
    const counted = [...matched].filter((l) => !l.startsWith('p')).length;
    body.appendChild(note(`${counted} ion${counted === 1 ? '' : 's'} matched in the spectrum shown (any charge or loss).`));
  }

  setScan(header: Record<string, unknown> | null): void {
    const body = this.tabs.get('scan')!.body;
    body.replaceChildren();
    if (!header) {
      body.appendChild(note('No scan picked yet.'));
      return;
    }
    const dl = document.createElement('dl');
    dl.className = 'ms-scan-header';
    for (const [k, v] of Object.entries(header)) {
      if (v === null || typeof v === 'object') continue;
      const dt = document.createElement('dt');
      dt.textContent = k;
      const dd = document.createElement('dd');
      dd.textContent = typeof v === 'number' && !Number.isInteger(v) ? v.toPrecision(7) : String(v);
      dl.append(dt, dd);
    }
    body.appendChild(dl);
  }
}

function note(text: string): HTMLElement {
  const p = document.createElement('p');
  p.className = 'ms-drawer-note';
  p.textContent = text;
  return p;
}

/** Residues of a ProForma sequence, each with its modifications attached. */
export function residuesOf(modseq: string): string[] {
  const body = modseq.replace(/^\[[^\]]*\]-/, '').replace(/-\[[^\]]*\]$/, '');
  const out: string[] = [];
  const re = /([A-Z])((?:\[[^\]]*\])*)/g;
  let m: RegExpExecArray | null;
  while ((m = re.exec(body))) out.push(m[1] + m[2]);
  return out;
}

/** Whether any charge or loss of an ion was matched (y7 is lit by y7^2 too). */
function anyForm(matched: Set<string>, base: string): boolean {
  for (const l of matched) if (l === base || l.startsWith(`${base}^`) || l.startsWith(`${base}-`)) return true;
  return false;
}

function ionCell(tr: HTMLTableRowElement, row: LadderRow | undefined, matched: Set<string>): void {
  const td = tr.insertCell();
  if (!row) return;
  td.textContent = row.mz.toFixed(2);
  if (anyForm(matched, row.annotation)) {
    td.classList.add('matched');
    td.style.color = ionColor(row.ion_type) ?? '';
  }
}

function coverage(target: Target, matched: Set<string>): HTMLElement {
  const residues = residuesOf(target.modifiedSequence);
  const n = residues.length;
  const line = document.createElement('div');
  line.className = 'ms-coverage';
  residues.forEach((res, i) => {
    const span = document.createElement('span');
    span.className = 'ms-coverage-residue';
    span.textContent = res.replace(/\[[^\]]*\]/g, '·');
    span.title = res;
    line.appendChild(span);
    if (i < n - 1) {
      const bond = document.createElement('span');
      bond.className = 'ms-coverage-bond';
      const b = anyForm(matched, `b${i + 1}`);
      const y = anyForm(matched, `y${n - 1 - i}`);
      bond.dataset.b = String(b);
      bond.dataset.y = String(y);
      bond.title = `b${i + 1}${b ? ' ✓' : ''} · y${n - 1 - i}${y ? ' ✓' : ''}`;
      if (b) bond.style.borderTopColor = ionColor('b') ?? '';
      if (y) bond.style.borderBottomColor = ionColor('y') ?? '';
      line.appendChild(bond);
    }
  });
  return line;
}
