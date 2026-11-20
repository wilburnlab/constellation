// The transcriptome browser (PR2 of docs/plans/transcriptome-browser.md):
// pick a cluster of an EM run on the left, see the reads behind it on the
// right. A template of any round can be opened by id, which is how a split
// is looked at (its pooled reads, against its pooled consensus).

import { fetchJson } from '../../engine/arrow_client';
import { NodePicker, type ClusterRow } from './NodePicker';
import { SupportView } from './SupportView';
import type { ClusterDetail, RunInfo } from './types';
import './tx.css';

export interface TranscriptomeBrowserOptions {
  host: HTMLElement;
  sessionId: string;
  /** What to open first (the standalone page reads it from its URL:
   *  `#cluster=ID` or `#template=ROUND:ID`). Nothing opens on its own: the
   *  run's largest cluster can take a minute to replay. */
  initial?: { cluster?: number; template?: { round: number; templateId: number } };
}

/** `#cluster=ID` / `#template=ROUND:ID` → what to open first. */
export function parseInitial(hash: string): TranscriptomeBrowserOptions['initial'] {
  const params = new URLSearchParams(hash.replace(/^#/, ''));
  const cluster = params.get('cluster');
  if (cluster !== null && /^\d+$/.test(cluster)) return { cluster: Number(cluster) };
  const template = params.get('template')?.match(/^(\d+):(\d+)$/);
  if (template) return { template: { round: Number(template[1]), templateId: Number(template[2]) } };
  return undefined;
}

function el<K extends keyof HTMLElementTagNameMap>(tag: K, cls?: string, text?: string): HTMLElementTagNameMap[K] {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text !== undefined) e.textContent = text;
  return e;
}

export class TranscriptomeBrowser {
  readonly root: HTMLDivElement;
  private readonly opts: TranscriptomeBrowserOptions;
  private readonly runSel: HTMLSelectElement;
  private readonly roundInput: HTMLInputElement;
  private readonly templateInput: HTMLInputElement;
  private readonly openTemplate: HTMLButtonElement;
  private readonly left: HTMLDivElement;
  private readonly detail: HTMLDivElement;
  private readonly right: HTMLDivElement;
  private readonly message: HTMLDivElement;
  private runs: RunInfo[] = [];
  private run: RunInfo | null = null;
  private picker: NodePicker | null = null;
  private support: SupportView | null = null;
  private detailGen = 0;

  constructor(opts: TranscriptomeBrowserOptions) {
    this.opts = opts;
    this.root = el('div', 'tx-browser');
    const bar = el('div', 'tx-toolbar tx-top');
    bar.appendChild(el('span', 'tx-brand', 'Transcriptome'));
    this.runSel = el('select', 'tx-select');
    this.runSel.title = 'EM run';
    this.roundInput = el('input', 'tx-input tx-num');
    this.roundInput.type = 'number';
    this.roundInput.min = '1';
    this.roundInput.placeholder = 'round';
    this.templateInput = el('input', 'tx-input tx-tid');
    this.templateInput.placeholder = 'template id';
    this.openTemplate = el('button', 'tx-btn', 'Open template');
    const tpl = el('span', 'tx-field');
    tpl.append(this.roundInput, this.templateInput, this.openTemplate);
    this.message = el('div', 'tx-message');
    bar.append(this.runSel, tpl, this.message);
    const main = el('div', 'tx-main');
    this.left = el('div', 'tx-left');
    this.detail = el('div', 'tx-detail');
    this.right = el('div', 'tx-right');
    main.append(this.left, this.right);
    this.root.append(bar, main);
    opts.host.appendChild(this.root);

    this.runSel.addEventListener('change', () => {
      const run = this.runs.find((r) => r.key === this.runSel.value);
      if (run) this.selectRun(run);
    });
    this.openTemplate.addEventListener('click', () => {
      const round = Number(this.roundInput.value);
      const tid = Number(this.templateInput.value);
      if (!this.support || !Number.isFinite(round) || !Number.isFinite(tid) || !this.templateInput.value) return;
      void this.support.show({ kind: 'template', round, templateId: tid }, 'pooled');
    });
  }

  private get base(): string {
    return `/api/transcriptome/sessions/${this.opts.sessionId}/runs/${this.run!.key}`;
  }

  async mount(): Promise<void> {
    try {
      this.runs = await fetchJson<RunInfo[]>(`/api/transcriptome/sessions/${this.opts.sessionId}/runs`);
    } catch (err) {
      this.message.textContent = `Could not list the session's runs: ${(err as Error).message}`;
      return;
    }
    this.runSel.innerHTML = '';
    for (const r of this.runs) {
      const o = el('option', undefined, `${r.label} · ${r.n_clusters.toLocaleString('en-US')} clusters`);
      o.value = r.key;
      this.runSel.appendChild(o);
    }
    this.runSel.hidden = this.runs.length < 2;
    if (this.runs.length) await this.selectRun(this.runs[0]);
    else this.message.textContent = 'This session has no EM run.';
  }

  private async selectRun(run: RunInfo): Promise<void> {
    this.run = run;
    this.runSel.value = run.key;
    this.picker?.dispose();
    this.support?.dispose();
    this.left.innerHTML = '';
    this.detail.innerHTML = '';
    this.roundInput.value = String(run.rounds.at(-1) ?? 1);
    this.support = new SupportView({ host: this.right, base: this.base, samples: run.samples });
    this.message.textContent = `${run.n_reads.toLocaleString('en-US')} reads · rounds ${run.rounds.join(', ') || '—'}`;
    if (!run.finished) {
      this.message.textContent += ' · no final clusters: open a template by round and id';
      this.picker = null;
      if (this.opts.initial?.template) {
        void this.support.show({ kind: 'template', ...this.opts.initial.template }, 'pooled');
      }
      return;
    }
    this.picker = new NodePicker({
      host: this.left,
      base: this.base,
      onSelect: (row) => void this.selectCluster(row),
    });
    this.left.appendChild(this.detail);
    const initial = this.opts.initial;
    if (initial?.template) {
      void this.support.show({ kind: 'template', ...initial.template }, 'pooled');
    } else if (initial?.cluster !== undefined) {
      void this.selectCluster({ cluster_id: initial.cluster } as ClusterRow);
    }
    await this.picker.load(true);
  }

  private async selectCluster(row: ClusterRow): Promise<void> {
    const gen = ++this.detailGen;
    void this.support?.show({ kind: 'cluster', clusterId: row.cluster_id });
    this.detail.textContent = 'loading…';
    try {
      const d = await fetchJson<ClusterDetail>(`${this.base}/clusters/${row.cluster_id}`);
      if (gen !== this.detailGen) return;
      this.renderDetail(d);
    } catch (err) {
      if (gen === this.detailGen) this.detail.textContent = `detail failed: ${(err as Error).message}`;
    }
  }

  private renderDetail(d: ClusterDetail): void {
    const box = this.detail;
    box.innerHTML = '';
    box.appendChild(el('div', 'tx-detail-title', `cluster ${d.cluster_id}`));
    const facts: Array<[string, string]> = [
      ['reads', d.n_reads.toLocaleString('en-US')],
      ['length', `${d.length.toLocaleString('en-US')} nt`],
      ['protein', d.protein_length ? `${d.protein_length} aa` : '—'],
      ['round node', d.round !== null ? `r${d.round} · template ${d.parent_template_id} · node ${d.haplotype_id}` : '—'],
      [
        'final merge',
        d.n_absorbed ? `absorbed ${d.n_absorbed}${d.rebuilt ? ', rebuilt' : ', rebuild failed'}` : '—',
      ],
      ['content id', d.consensus_content_id !== null ? String(d.consensus_content_id) : '—'],
    ];
    const dl = el('dl', 'tx-facts');
    for (const [k, v] of facts) dl.append(el('dt', undefined, k), el('dd', undefined, v));
    box.appendChild(dl);
    if (d.parent_template_id !== null && d.round !== null) {
      const b = el('button', 'tx-btn', 'Open its template (pooled)');
      b.title = 'every read of the template this cluster’s node came from, against its pooled consensus';
      b.addEventListener('click', () =>
        void this.support?.show({ kind: 'template', round: d.round!, templateId: d.parent_template_id! }, 'pooled'),
      );
      box.appendChild(b);
    }
    if (d.samples.length) {
      const max = Math.max(...d.samples.map((s) => s.count), 1);
      const list = el('div', 'tx-samples');
      for (const s of d.samples) {
        const rowEl = el('div', 'tx-sample');
        const name = el('span', 'tx-sample-name', s.sample ?? `sample ${s.sample_id}`);
        const bar = el('span', 'tx-sample-bar');
        bar.style.width = `${(100 * s.count) / max}%`;
        const n = el('span', 'tx-sample-n', s.count.toLocaleString('en-US'));
        rowEl.append(name, bar, n);
        list.appendChild(rowEl);
      }
      box.appendChild(list);
    }
    if (d.protein) box.appendChild(el('div', 'tx-protein', d.protein));
  }

  dispose(): void {
    this.detailGen += 1;
    this.picker?.dispose();
    this.support?.dispose();
    this.root.remove();
  }
}
