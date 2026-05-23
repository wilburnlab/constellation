// Reference-cache-first entry form for the genome browser task panel.
//
// Phase-1 form for the `viz/genome` descriptor. The user:
//   1. Picks a reference from the cache dropdown (or clicks "Add new
//      genome…" to open a `reference fetch` task panel; or clicks
//      "Refresh catalog" to fire `catalog update --source all`).
//   2. Adds one-or-more source rows pointing at `transcriptome align` /
//      `transcriptome cluster` output dirs. Each row auto-detects its
//      kind from the source's `manifest.json` via /api/sessions/
//      inspect-source and surfaces an inline assembly-mismatch warning
//      when the source was produced against a different assembly.
//   3. Optionally names the configuration in "Save as…" so it persists
//      to ~/.constellation/sessions/.
//   4. Clicks Open → POST /api/sessions/open → the host transitions
//      this panel into the GenomeBrowser widget.
//
// "Load saved session…" populates every field from a previously saved
// configuration in one click.

import { CommandSchema } from './types';
import type {
  InstalledReference,
  OpenSessionResult,
  SavedSessionPayload,
  SavedSessionSummary,
  SourceInspection,
} from './types';
import { DashboardState } from './state';

interface SourceRow {
  path: string;
  kind: 'align' | 'cluster' | '';
  label: string;
  warning: string | null;
  error: string | null;
}

export interface GenomeBrowserFormOptions {
  state: DashboardState;
  onSubmit: (
    result: OpenSessionResult,
    saved: SavedSessionSummary | null,
  ) => Promise<void>;
}

const STORAGE_LAST_REF = 'constellation.dashboard.viz.genome.lastReference';

export class GenomeBrowserForm {
  private readonly state: DashboardState;
  private readonly onSubmit: GenomeBrowserFormOptions['onSubmit'];
  private host: HTMLElement | null = null;
  private references: InstalledReference[] = [];
  private savedSessions: SavedSessionSummary[] = [];
  private sources: SourceRow[] = [{ path: '', kind: '', label: '', warning: null, error: null }];
  private selectedReference: string = '';
  private saveAs: string = '';
  private errorBanner: HTMLElement | null = null;
  private submitBtn: HTMLButtonElement | null = null;
  private referenceSelect: HTMLSelectElement | null = null;
  private savedSelect: HTMLSelectElement | null = null;
  private saveAsInput: HTMLInputElement | null = null;
  private rowsContainer: HTMLElement | null = null;
  private unsubscribers: Array<() => void> = [];

  constructor(opts: GenomeBrowserFormOptions) {
    this.state = opts.state;
    this.onSubmit = opts.onSubmit;
  }

  async mount(host: HTMLElement): Promise<void> {
    this.host = host;
    host.classList.add('viz-form', 'genome-browser-form');
    host.innerHTML = '';
    this.buildSkeleton(host);
    // Subscribe to reference:installed so the dropdown hot-refreshes
    // when the user finishes a `reference fetch` in a sibling tab.
    this.unsubscribers.push(
      this.state.on('reference:installed', () => {
        void this.loadReferences();
      }),
    );
    await Promise.all([this.loadReferences(), this.loadSavedSessions()]);
  }

  destroy(): void {
    for (const u of this.unsubscribers) u();
    this.unsubscribers = [];
    if (this.host) this.host.innerHTML = '';
    this.host = null;
  }

  // ----------------------------------------------------------------
  // Skeleton
  // ----------------------------------------------------------------

  private buildSkeleton(host: HTMLElement): void {
    const title = document.createElement('h2');
    title.textContent = 'Genome browser';
    host.appendChild(title);

    const help = document.createElement('div');
    help.className = 'form-help';
    help.textContent =
      'Pick a reference from the cache, then attach one or more ' +
      'transcriptome align / cluster output directories.';
    host.appendChild(help);

    // Load saved session row
    {
      const section = document.createElement('div');
      section.className = 'form-section';
      const row = document.createElement('div');
      row.className = 'form-row';
      const lbl = document.createElement('label');
      lbl.textContent = 'Load saved session…';
      row.appendChild(lbl);
      const select = document.createElement('select');
      select.innerHTML = '<option value="">(none — start fresh)</option>';
      select.addEventListener('change', () =>
        void this.handleSavedSessionPicked(select.value),
      );
      this.savedSelect = select;
      row.appendChild(select);
      section.appendChild(row);
      host.appendChild(section);
    }

    // Reference dropdown + sub-actions
    {
      const section = document.createElement('div');
      section.className = 'form-section';

      const row = document.createElement('div');
      row.className = 'form-row';
      const lbl = document.createElement('label');
      const lblText = document.createElement('span');
      lblText.textContent = 'Reference';
      lbl.appendChild(lblText);
      const star = document.createElement('span');
      star.style.color = 'var(--danger)';
      star.textContent = '*';
      lbl.appendChild(star);
      row.appendChild(lbl);

      const select = document.createElement('select');
      select.addEventListener('change', () => {
        this.selectedReference = select.value;
        try {
          window.localStorage.setItem(STORAGE_LAST_REF, this.selectedReference);
        } catch {
          /* ignore */
        }
        this.updateSubmitButton();
        this.recheckMismatches();
      });
      this.referenceSelect = select;
      row.appendChild(select);
      section.appendChild(row);

      // Sub-actions: "Add new genome…" + "Refresh catalog"
      const actions = document.createElement('div');
      actions.className = 'form-subactions';
      actions.style.display = 'flex';
      actions.style.gap = '12px';
      actions.style.marginTop = '6px';

      const addGenome = document.createElement('button');
      addGenome.type = 'button';
      addGenome.className = 'link-btn';
      addGenome.textContent = 'Add new genome…';
      addGenome.addEventListener('click', () => this.openReferenceFetchPanel());
      actions.appendChild(addGenome);

      const refresh = document.createElement('button');
      refresh.type = 'button';
      refresh.className = 'link-btn';
      refresh.textContent = 'Refresh catalog (all sources)';
      refresh.addEventListener('click', () => void this.runRefreshCatalog());
      actions.appendChild(refresh);

      section.appendChild(actions);
      host.appendChild(section);
    }

    // Multi-row sources
    {
      const section = document.createElement('div');
      section.className = 'form-section';
      const header = document.createElement('label');
      const headerText = document.createElement('span');
      headerText.textContent = 'Add results';
      header.appendChild(headerText);
      const star = document.createElement('span');
      star.style.color = 'var(--danger)';
      star.textContent = '*';
      header.appendChild(star);
      section.appendChild(header);

      const rows = document.createElement('div');
      rows.className = 'sources-rows';
      this.rowsContainer = rows;
      section.appendChild(rows);

      const addBtn = document.createElement('button');
      addBtn.type = 'button';
      addBtn.className = 'link-btn';
      addBtn.textContent = '+ Add row';
      addBtn.addEventListener('click', () => this.addSourceRow());
      section.appendChild(addBtn);

      host.appendChild(section);
      this.renderRows();
    }

    // Save as…
    {
      const section = document.createElement('div');
      section.className = 'form-section';
      const row = document.createElement('div');
      row.className = 'form-row';
      const lbl = document.createElement('label');
      lbl.textContent = 'Save as…';
      row.appendChild(lbl);
      const input = document.createElement('input');
      input.type = 'text';
      input.placeholder = 'optional — names the configuration in your session cache';
      input.addEventListener('input', () => {
        this.saveAs = input.value.trim();
      });
      this.saveAsInput = input;
      row.appendChild(input);
      section.appendChild(row);
      host.appendChild(section);
    }

    // Error banner + submit
    {
      const errBanner = document.createElement('div');
      errBanner.className = 'form-error';
      errBanner.style.minHeight = '1.2em';
      this.errorBanner = errBanner;
      host.appendChild(errBanner);

      const actions = document.createElement('div');
      actions.className = 'form-actions';
      const submit = document.createElement('button');
      submit.type = 'button';
      submit.className = 'run';
      submit.textContent = 'Open';
      submit.addEventListener('click', () => void this.submit());
      actions.appendChild(submit);
      this.submitBtn = submit;
      host.appendChild(actions);
      this.updateSubmitButton();
    }
  }

  // ----------------------------------------------------------------
  // Data loaders
  // ----------------------------------------------------------------

  private async loadReferences(): Promise<void> {
    try {
      const r = await fetch('/api/references');
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      this.references = (await r.json()) as InstalledReference[];
    } catch (err) {
      this.showError(`Could not load references: ${(err as Error).message}`);
      this.references = [];
    }
    this.repopulateReferenceSelect();
  }

  private repopulateReferenceSelect(): void {
    if (!this.referenceSelect) return;
    const prev =
      this.selectedReference ||
      (() => {
        try {
          return window.localStorage.getItem(STORAGE_LAST_REF) ?? '';
        } catch {
          return '';
        }
      })();
    this.referenceSelect.innerHTML = '';
    const placeholder = document.createElement('option');
    placeholder.value = '';
    placeholder.textContent = this.references.length
      ? '(pick a reference)'
      : '(no references installed — click "Add new genome…")';
    this.referenceSelect.appendChild(placeholder);
    let firstDefault: string | null = null;
    for (const ref of this.references) {
      const opt = document.createElement('option');
      opt.value = ref.handle;
      const star = ref.is_default ? ' ★' : '';
      opt.textContent = `${ref.handle}${star} — ${ref.organism} · ${ref.source} ${ref.release}`;
      this.referenceSelect.appendChild(opt);
      if (ref.is_default && firstDefault === null) firstDefault = ref.handle;
    }
    // Prefer the previously-selected handle when still installed; else
    // the per-organism default; else leave on placeholder.
    const handles = new Set(this.references.map((r) => r.handle));
    if (prev && handles.has(prev)) {
      this.referenceSelect.value = prev;
      this.selectedReference = prev;
    } else if (firstDefault !== null) {
      this.referenceSelect.value = firstDefault;
      this.selectedReference = firstDefault;
    } else {
      this.referenceSelect.value = '';
      this.selectedReference = '';
    }
    this.updateSubmitButton();
    this.recheckMismatches();
  }

  private async loadSavedSessions(): Promise<void> {
    try {
      const r = await fetch('/api/saved-sessions');
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      this.savedSessions = (await r.json()) as SavedSessionSummary[];
    } catch (err) {
      // Don't surface — saved-sessions are optional, the form still
      // works without them. Log to console for diagnostics.
      console.warn('saved-sessions load failed:', (err as Error).message);
      this.savedSessions = [];
    }
    this.repopulateSavedSelect();
  }

  private repopulateSavedSelect(): void {
    if (!this.savedSelect) return;
    this.savedSelect.innerHTML = '';
    const placeholder = document.createElement('option');
    placeholder.value = '';
    placeholder.textContent = '(none — start fresh)';
    this.savedSelect.appendChild(placeholder);
    for (const s of this.savedSessions) {
      const opt = document.createElement('option');
      opt.value = s.slug;
      opt.textContent = `${s.label} — ${s.reference_handle} (${s.n_sources} sources)`;
      this.savedSelect.appendChild(opt);
    }
  }

  // ----------------------------------------------------------------
  // Saved-session prefill
  // ----------------------------------------------------------------

  private async handleSavedSessionPicked(slug: string): Promise<void> {
    if (!slug) return;
    try {
      const r = await fetch(`/api/saved-sessions/${encodeURIComponent(slug)}`);
      if (!r.ok) {
        let detail = `HTTP ${r.status}`;
        try {
          const body = (await r.json()) as { detail?: unknown };
          if (typeof body.detail === 'string') detail = body.detail;
        } catch {
          /* fall through */
        }
        throw new Error(detail);
      }
      const payload = (await r.json()) as SavedSessionPayload;
      this.selectedReference = payload.reference_handle;
      if (this.referenceSelect) this.referenceSelect.value = payload.reference_handle;
      this.sources = payload.sources.map((s) => ({
        path: s.path,
        kind: (s.kind === 'align' || s.kind === 'cluster' ? s.kind : '') as
          | 'align'
          | 'cluster'
          | '',
        label: s.label,
        warning: null,
        error: null,
      }));
      if (this.sources.length === 0) {
        this.sources.push({ path: '', kind: '', label: '', warning: null, error: null });
      }
      this.saveAs = payload.label;
      if (this.saveAsInput) this.saveAsInput.value = payload.label;
      this.renderRows();
      this.updateSubmitButton();
      this.recheckMismatches();
      // Inspect each prefilled source to refresh its kind / warning.
      for (let i = 0; i < this.sources.length; i++) {
        void this.inspectRow(i);
      }
    } catch (err) {
      this.showError(`Failed to load saved session: ${(err as Error).message}`);
    }
  }

  // ----------------------------------------------------------------
  // Source rows
  // ----------------------------------------------------------------

  private addSourceRow(): void {
    this.sources.push({ path: '', kind: '', label: '', warning: null, error: null });
    this.renderRows();
  }

  private renderRows(): void {
    if (!this.rowsContainer) return;
    this.rowsContainer.innerHTML = '';
    this.sources.forEach((src, idx) => {
      const row = document.createElement('div');
      row.className = 'source-row';
      row.style.display = 'grid';
      row.style.gridTemplateColumns = '1fr 110px 1fr auto';
      row.style.gap = '8px';
      row.style.marginBottom = '4px';

      const path = document.createElement('input');
      path.type = 'text';
      path.placeholder = '/path/to/align/output';
      path.value = src.path;
      path.spellcheck = false;
      path.autocomplete = 'off';
      path.addEventListener('change', () => {
        this.sources[idx].path = path.value.trim();
        void this.inspectRow(idx);
      });
      row.appendChild(path);

      const kind = document.createElement('input');
      kind.type = 'text';
      kind.readOnly = true;
      kind.value = src.kind;
      kind.placeholder = '(auto)';
      kind.style.background = 'rgba(255,255,255,0.04)';
      row.appendChild(kind);

      const label = document.createElement('input');
      label.type = 'text';
      label.value = src.label;
      label.placeholder = '(label — defaults to dir name)';
      label.addEventListener('input', () => {
        this.sources[idx].label = label.value.trim();
      });
      row.appendChild(label);

      const remove = document.createElement('button');
      remove.type = 'button';
      remove.className = 'link-btn';
      remove.textContent = '✕';
      remove.title = 'Remove row';
      remove.addEventListener('click', () => {
        this.sources.splice(idx, 1);
        if (this.sources.length === 0) {
          this.sources.push({
            path: '', kind: '', label: '', warning: null, error: null,
          });
        }
        this.renderRows();
        this.updateSubmitButton();
      });
      row.appendChild(remove);

      if (src.warning || src.error) {
        const note = document.createElement('div');
        note.style.gridColumn = '1 / -1';
        note.style.fontSize = '0.85em';
        if (src.error) {
          note.style.color = 'var(--danger)';
          note.textContent = src.error;
        } else if (src.warning) {
          note.style.color = 'var(--warn)';
          note.textContent = src.warning;
        }
        row.appendChild(note);
      }

      this.rowsContainer!.appendChild(row);
    });
  }

  private async inspectRow(idx: number): Promise<void> {
    const src = this.sources[idx];
    if (!src.path) {
      src.kind = '';
      src.warning = null;
      src.error = null;
      this.renderRows();
      this.updateSubmitButton();
      return;
    }
    try {
      const r = await fetch('/api/sessions/inspect-source', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ path: src.path }),
      });
      if (!r.ok) {
        let detail = `HTTP ${r.status}`;
        try {
          const body = (await r.json()) as { detail?: unknown };
          if (typeof body.detail === 'string') detail = body.detail;
        } catch {
          /* fall through */
        }
        src.kind = '';
        src.warning = null;
        src.error = detail;
        this.renderRows();
        this.updateSubmitButton();
        return;
      }
      const inspection = (await r.json()) as SourceInspection;
      src.kind = inspection.kind;
      src.error = null;
      if (!src.label) {
        // Default the label to the manifest's directory basename.
        const parts = inspection.path.split('/');
        src.label = parts[parts.length - 1] || inspection.path;
      }
      // Mismatch warning: source carries no handle (escape-hatch run)
      // → hard error (form refuses submit). Same as below but
      // structurally distinct from the assembly-mismatch case.
      if (!inspection.reference_handle) {
        src.error =
          'this source was produced with --reference-dir (no cache ' +
          'handle). Import the reference via ' +
          '`constellation reference import` before opening.';
        src.warning = null;
      } else {
        src.warning = this.computeWarning(inspection);
      }
      this.renderRows();
      this.updateSubmitButton();
    } catch (err) {
      src.error = `inspect failed: ${(err as Error).message}`;
      src.kind = '';
      this.renderRows();
      this.updateSubmitButton();
    }
  }

  private recheckMismatches(): void {
    // Refresh per-row warnings without re-hitting the inspect-source
    // endpoint. Called when the reference dropdown changes.
    const refAssembly = this.currentAssembly();
    let mutated = false;
    for (const src of this.sources) {
      if (src.error) continue;
      if (!src.path || !src.kind) continue;
      const prev = src.warning;
      src.warning = this.warningForAssembly(refAssembly, /* source assembly */ undefined);
      if (prev !== src.warning) mutated = true;
    }
    if (mutated) this.renderRows();
  }

  private currentAssembly(): string | null {
    const ref = this.references.find((r) => r.handle === this.selectedReference);
    return ref?.assembly_accession ?? null;
  }

  private computeWarning(inspection: SourceInspection): string | null {
    const refAssembly = this.currentAssembly();
    return this.warningForAssembly(refAssembly, inspection.assembly_accession);
  }

  private warningForAssembly(
    refAssembly: string | null,
    sourceAssembly: string | null | undefined,
  ): string | null {
    if (!refAssembly || sourceAssembly === undefined || sourceAssembly === null) {
      return null;
    }
    if (refAssembly === sourceAssembly) return null;
    return (
      `Source assembly ${sourceAssembly} differs from the chosen ` +
      `reference's ${refAssembly}; coordinates may not align.`
    );
  }

  // ----------------------------------------------------------------
  // Side actions
  // ----------------------------------------------------------------

  private openReferenceFetchPanel(): void {
    // Synthesize a thin CommandSchema and emit command:open so the
    // dashboard shell opens a sibling task panel for `reference fetch`.
    // The shell will introspect the full argument list via /api/cli/
    // schema; here we only need to identify the path.
    const cmd: CommandSchema = {
      name: 'fetch',
      path: ['reference', 'fetch'],
      help: null,
      arguments: [],
      subcommands: [],
    };
    this.state.emit('command:open', cmd);
  }

  private async runRefreshCatalog(): Promise<void> {
    this.showError('');
    try {
      const r = await fetch('/api/commands', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ argv: ['catalog', 'update', '--source', 'all'] }),
      });
      if (!r.ok) {
        let detail = `HTTP ${r.status}`;
        try {
          const body = (await r.json()) as { detail?: unknown };
          if (typeof body.detail === 'string') detail = body.detail;
        } catch {
          /* fall through */
        }
        if (r.status === 409) {
          this.state.emit('job:rejected', detail);
        }
        throw new Error(detail);
      }
    } catch (err) {
      this.showError(`Refresh catalog failed: ${(err as Error).message}`);
    }
  }

  // ----------------------------------------------------------------
  // Submit
  // ----------------------------------------------------------------

  private updateSubmitButton(): void {
    if (!this.submitBtn) return;
    const validSources = this.sources.filter(
      (s) => s.path && s.kind && !s.error,
    );
    const ok = this.selectedReference !== '' && validSources.length > 0;
    this.submitBtn.disabled = !ok;
  }

  private async submit(): Promise<void> {
    if (!this.submitBtn) return;
    this.showError('');
    if (!this.selectedReference) {
      this.showError('Pick a reference first.');
      return;
    }
    const validSources = this.sources.filter(
      (s) => s.path && s.kind && !s.error,
    );
    if (validSources.length === 0) {
      this.showError('Add at least one source directory.');
      return;
    }
    for (const src of this.sources) {
      if (src.error) {
        this.showError(`Fix source errors before opening: ${src.error}`);
        return;
      }
    }
    this.submitBtn.disabled = true;
    try {
      const r = await fetch('/api/sessions/open', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          reference_handle: this.selectedReference,
          sources: validSources.map((s) => ({
            path: s.path,
            kind: s.kind,
            label: s.label || undefined,
          })),
          label: this.saveAs || undefined,
          saved_as: this.saveAs || undefined,
        }),
      });
      if (!r.ok) {
        let detail = `HTTP ${r.status}`;
        try {
          const body = (await r.json()) as { detail?: unknown };
          if (typeof body.detail === 'string') detail = body.detail;
        } catch {
          /* fall through */
        }
        throw new Error(detail);
      }
      const result = (await r.json()) as OpenSessionResult;

      // If the user supplied a Save-as name, persist the configuration
      // alongside opening. We POST after the session open succeeds, so
      // a broken open doesn't poison the saved-session list.
      let saved: SavedSessionSummary | null = null;
      if (this.saveAs) {
        try {
          const sr = await fetch('/api/saved-sessions', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              label: this.saveAs,
              reference_handle: this.selectedReference,
              sources: validSources.map((s) => ({
                path: s.path,
                kind: s.kind,
                label: s.label || s.path,
              })),
            }),
          });
          if (sr.ok) {
            saved = (await sr.json()) as SavedSessionSummary;
          }
        } catch (err) {
          // Non-fatal — opening succeeded, the saved-session write
          // failure is a side issue.
          console.warn('saved-session write failed', err);
        }
      }
      await this.onSubmit(result, saved);
    } catch (err) {
      this.showError((err as Error).message);
      this.submitBtn.disabled = false;
    }
  }

  private showError(message: string): void {
    if (this.errorBanner) this.errorBanner.textContent = message;
  }
}
