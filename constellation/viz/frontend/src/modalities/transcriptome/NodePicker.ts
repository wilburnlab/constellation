// The list a user picks a cluster from: a page of a run's cluster index at
// a time, sorted and filtered on the server (a run can hold 1.7M clusters),
// searched by cluster id, content id, protein fragment or read id.

import { tableFromIPC } from 'apache-arrow';

export interface ClusterRow {
  cluster_id: number;
  n_reads: number;
  length: number;
  protein_length: number;
  rebuilt: boolean;
  n_absorbed: number;
}

export interface NodePickerOptions {
  host: HTMLElement;
  /** `/api/transcriptome/sessions/{sid}/runs/{key}` */
  base: string;
  onSelect(row: ClusterRow): void;
  pageSize?: number;
}

const SORTS: Array<[string, string]> = [
  ['n_reads', 'reads'],
  ['length', 'length'],
  ['protein_length', 'protein'],
  ['n_absorbed', 'merged'],
  ['cluster_id', 'id'],
];

export class NodePicker {
  readonly root: HTMLDivElement;
  private readonly opts: NodePickerOptions;
  private readonly search: HTMLInputElement;
  private readonly sort: HTMLSelectElement;
  private readonly minReads: HTMLInputElement;
  private readonly list: HTMLTableSectionElement;
  private readonly more: HTMLButtonElement;
  private readonly status: HTMLDivElement;
  private rows: ClusterRow[] = [];
  private total = 0;
  private selected: number | null = null;
  private generation = 0;
  private debounce: ReturnType<typeof setTimeout> | null = null;

  constructor(opts: NodePickerOptions) {
    this.opts = opts;
    this.root = document.createElement('div');
    this.root.className = 'tx-picker';
    const controls = document.createElement('div');
    controls.className = 'tx-picker-controls';
    this.search = document.createElement('input');
    this.search.type = 'search';
    this.search.placeholder = 'cluster, protein fragment or read id';
    this.search.className = 'tx-input tx-picker-search';
    this.sort = document.createElement('select');
    this.sort.className = 'tx-select';
    for (const [value, label] of SORTS) {
      const o = document.createElement('option');
      o.value = value;
      o.textContent = `by ${label}`;
      this.sort.appendChild(o);
    }
    this.minReads = document.createElement('input');
    this.minReads.type = 'number';
    this.minReads.min = '0';
    this.minReads.value = '2';
    this.minReads.title = 'minimum reads';
    this.minReads.className = 'tx-input tx-picker-min';
    const minLabel = document.createElement('label');
    minLabel.className = 'tx-field';
    minLabel.append('≥', this.minReads, 'reads');
    controls.append(this.search, this.sort, minLabel);

    const table = document.createElement('table');
    table.className = 'tx-picker-table';
    table.innerHTML =
      '<thead><tr><th>cluster</th><th>reads</th><th>nt</th><th>aa</th><th></th></tr></thead>';
    this.list = document.createElement('tbody');
    table.appendChild(this.list);
    const scroller = document.createElement('div');
    scroller.className = 'tx-picker-scroll';
    scroller.appendChild(table);
    this.more = document.createElement('button');
    this.more.className = 'tx-btn tx-picker-more';
    this.more.textContent = 'More';
    this.more.hidden = true;
    scroller.appendChild(this.more);
    this.status = document.createElement('div');
    this.status.className = 'tx-picker-status';
    this.root.append(controls, this.status, scroller);
    opts.host.appendChild(this.root);

    const reload = () => void this.load(true);
    this.search.addEventListener('input', () => {
      if (this.debounce) clearTimeout(this.debounce);
      this.debounce = setTimeout(reload, 250);
    });
    this.sort.addEventListener('change', reload);
    this.minReads.addEventListener('change', reload);
    this.more.addEventListener('click', () => void this.load(false));
    this.list.addEventListener('click', (e) => {
      const tr = (e.target as Element).closest('tr[data-cluster]') as HTMLTableRowElement | null;
      if (!tr) return;
      const row = this.rows[Number(tr.dataset.index)];
      if (row) this.choose(row);
    });
  }

  /** The URL of the next page. */
  pageUrl(offset: number): string {
    const params = new URLSearchParams({
      sort: this.sort.value,
      descending: this.sort.value === 'cluster_id' ? 'false' : 'true',
      min_reads: String(Math.max(0, Number(this.minReads.value) || 0)),
      offset: String(offset),
      limit: String(this.opts.pageSize ?? 200),
    });
    const q = this.search.value.trim();
    if (q) params.set('q', q);
    return `${this.opts.base}/clusters?${params.toString()}`;
  }

  async load(reset: boolean): Promise<void> {
    const gen = ++this.generation;
    if (reset) {
      this.rows = [];
      this.list.innerHTML = '';
    }
    this.status.textContent = this.rows.length ? 'loading…' : 'loading clusters… (the first listing indexes the run)';
    try {
      const response = await fetch(this.pageUrl(this.rows.length));
      if (!response.ok) throw new Error(`${response.status} ${await response.text()}`);
      if (gen !== this.generation) return;
      this.total = Number(response.headers.get('X-Total-Rows') ?? '0');
      const table = tableFromIPC(await response.arrayBuffer());
      if (gen !== this.generation) return;
      const start = this.rows.length;
      for (let i = 0; i < table.numRows; i += 1) {
        const r = table.get(i)!;
        this.rows.push({
          cluster_id: Number(r.cluster_id),
          n_reads: Number(r.n_reads),
          length: Number(r.length),
          protein_length: Number(r.protein_length),
          rebuilt: Boolean(r.rebuilt),
          n_absorbed: Number(r.n_absorbed ?? 0),
        });
      }
      this.appendRows(start);
      this.status.textContent = `${this.total.toLocaleString('en-US')} cluster${this.total === 1 ? '' : 's'}`;
      this.more.hidden = this.rows.length >= this.total;
    } catch (err) {
      if (gen === this.generation) this.status.textContent = `failed: ${(err as Error).message}`;
    }
  }

  private appendRows(start: number): void {
    const frag = document.createDocumentFragment();
    for (let i = start; i < this.rows.length; i += 1) {
      const r = this.rows[i];
      const tr = document.createElement('tr');
      tr.dataset.cluster = String(r.cluster_id);
      tr.dataset.index = String(i);
      if (r.cluster_id === this.selected) tr.classList.add('selected');
      const merged = r.n_absorbed > 0 ? (r.rebuilt ? '⊕' : '⊘') : '';
      tr.innerHTML =
        `<td>${r.cluster_id}</td><td>${r.n_reads.toLocaleString('en-US')}</td>` +
        `<td>${r.length.toLocaleString('en-US')}</td><td>${r.protein_length || ''}</td>` +
        `<td title="${r.n_absorbed ? `merged ${r.n_absorbed}${r.rebuilt ? ', rebuilt' : ', rebuild failed'}` : ''}">${merged}</td>`;
      frag.appendChild(tr);
    }
    this.list.appendChild(frag);
  }

  private choose(row: ClusterRow): void {
    this.selected = row.cluster_id;
    for (const tr of Array.from(this.list.querySelectorAll('tr.selected'))) tr.classList.remove('selected');
    this.list.querySelector(`tr[data-cluster="${row.cluster_id}"]`)?.classList.add('selected');
    this.opts.onSelect(row);
  }

  dispose(): void {
    if (this.debounce) clearTimeout(this.debounce);
    this.generation += 1;
    this.root.remove();
  }
}
