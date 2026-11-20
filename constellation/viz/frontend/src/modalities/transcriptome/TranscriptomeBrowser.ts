// The transcriptome browser (PR2 of docs/plans/transcriptome-browser.md):
// pick a cluster of an EM run on the left, see the reads behind it on the
// right. A template of any round can be opened by id, which is how a split
// is looked at (its pooled reads, against its pooled consensus).

import { fetchJson } from '../../engine/arrow_client';
import { NodePicker, type ClusterRow } from './NodePicker';
import { SupportView } from './SupportView';
import { type ClusterDetail, type RunInfo, TEMPLATE_ROUND_UNIT, templateIdOf, templateLabel, templateRound } from './types';
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
  const template = params.get('template')?.match(/^(\d+):#?(\d+)$/);
  if (template) {
    const round = Number(template[1]);
    const templateId = Number.isSafeInteger(round) ? parseTemplate(round, template[2]) : null;
    if (templateId !== null) return { template: { round, templateId } };
  }
  return undefined;
}

function el<K extends keyof HTMLElementTagNameMap>(tag: K, cls?: string, text?: string): HTMLElementTagNameMap[K] {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text !== undefined) e.textContent = text;
  return e;
}

/** A template typed by a person: its number within the round (`1234`,
 *  `#1234`) or its full id, `(round << 40) | number`. Null if neither, or
 *  if the id is not an integer a JavaScript number holds exactly. */
export function parseTemplate(round: number, text: string): number | null {
  const m = text.trim().match(/^#?(\d+)$/);
  if (!m) return null;
  const v = Number(m[1]);
  if (!Number.isSafeInteger(v)) return null;
  const id = v >= TEMPLATE_ROUND_UNIT ? v : templateIdOf(round, v);
  return Number.isSafeInteger(id) ? id : null;
}

const TEMPLATE_HELP =
  'A template is the sequence a round aligned reads to: in round 1 a seed read, ' +
  'later the consensus of a node of the round before. Its M-step builds one or more ' +
  'nodes from the reads aligned to it (several when it splits), and the last round’s ' +
  'nodes, after the final merge, are the clusters. Give its number within the round ' +
  '(as shown with a cluster: “round 12 · template #1234”) or its full id.';

export class TranscriptomeBrowser {
  readonly root: HTMLDivElement;
  private readonly opts: TranscriptomeBrowserOptions;
  private readonly runSel: HTMLSelectElement;
  private readonly roundSel: HTMLSelectElement;
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
    this.roundSel = el('select', 'tx-select tx-round');
    this.roundSel.title = 'a round the run finished';
    this.templateInput = el('input', 'tx-input tx-tid');
    this.templateInput.placeholder = 'template #';
    this.templateInput.title = TEMPLATE_HELP;
    this.openTemplate = el('button', 'tx-btn', 'Open');
    this.openTemplate.title = TEMPLATE_HELP;
    const tpl = el('span', 'tx-field');
    tpl.title = TEMPLATE_HELP;
    tpl.append('Template of', this.roundSel, this.templateInput, this.openTemplate);
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
    const open = () => {
      let round = Number(this.roundSel.value);
      const id = parseTemplate(round, this.templateInput.value);
      if (!this.support || id === null) {
        this.message.textContent = 'a template is its number within the round (e.g. 1234) or its full id';
        return;
      }
      if (id >= TEMPLATE_ROUND_UNIT && templateRound(id) !== round) {
        // A full id names its own round.
        round = templateRound(id);
        if (this.run?.rounds.includes(round)) this.roundSel.value = String(round);
      }
      void this.support.show({ kind: 'template', round, templateId: id }, 'pooled');
    };
    this.openTemplate.addEventListener('click', open);
    this.templateInput.addEventListener('keydown', (e) => {
      if (e.key === 'Enter') open();
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
    // A cluster detail still on its way is of the run being left.
    this.detailGen += 1;
    this.run = run;
    this.runSel.value = run.key;
    this.picker?.dispose();
    this.support?.dispose();
    this.left.innerHTML = '';
    this.detail.innerHTML = '';
    this.roundSel.innerHTML = '';
    for (const r of [...run.rounds].reverse()) {
      const o = el('option', undefined, `round ${r}`);
      o.value = String(r);
      this.roundSel.appendChild(o);
    }
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
    const run = this.run;
    const current = () => gen === this.detailGen && run === this.run;
    void this.support?.show({ kind: 'cluster', clusterId: row.cluster_id });
    this.detail.textContent = 'loading…';
    try {
      const d = await fetchJson<ClusterDetail>(`${this.base}/clusters/${row.cluster_id}`);
      if (!current()) return;
      this.renderDetail(d);
    } catch (err) {
      if (current()) this.detail.textContent = `detail failed: ${(err as Error).message}`;
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
      [
        'made by',
        d.parent_template_id !== null ? `node ${d.haplotype_id} of ${templateLabel(d.parent_template_id)}` : '—',
      ],
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
