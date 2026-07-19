// DatasetManagerPopover — toolbar popover that lists the session's
// reference and attached sources, with per-binding visibility toggles,
// per-source remove buttons, and an inline "Add dataset" sub-form.
//
// The popover is purely a controller: it reads the current source +
// binding state from the host (GenomeBrowser) and emits callbacks for
// each mutation. The host owns track-state persistence and the actual
// /api/sessions/{id}/sources roundtrip.

import { PathInput } from './PathInput';

export interface SourceRow {
  source_id: string;
  label: string;
  kind: 'align' | 'cluster';
  path: string;
  warning?: string | null;
}

export interface BindingRow {
  binding_id: string;
  kind: string;          // kernel kind
  label: string;
  source_id: string | null;  // null for reference-only bindings
  visible: boolean;
}

export interface DatasetManagerHandlers {
  onToggleBinding(binding_id: string, visible: boolean): void;
  onRemoveSource(source_id: string): Promise<void>;
  onAddSource(path: string): Promise<{ ok: true } | { ok: false; error: string }>;
}

export interface DatasetManagerOptions {
  anchor: HTMLElement;
  referenceLabel: string;
  referenceBindings: BindingRow[];
  sources: SourceRow[];
  bindingsBySource: Map<string, BindingRow[]>;
  handlers: DatasetManagerHandlers;
  onClose(): void;
}

export class DatasetManagerPopover {
  private readonly opts: DatasetManagerOptions;
  private readonly root: HTMLElement;
  private addPathInput: PathInput | null = null;
  private addError: HTMLElement | null = null;
  private addBtn: HTMLButtonElement | null = null;
  private outsideHandler: ((e: MouseEvent) => void) | null = null;
  private escHandler: ((e: KeyboardEvent) => void) | null = null;

  constructor(opts: DatasetManagerOptions) {
    this.opts = opts;
    this.root = document.createElement('div');
    this.root.className = 'dataset-popover';
    this.root.setAttribute('role', 'dialog');
    this.root.setAttribute('aria-label', 'Loaded datasets and visible tracks');
    this.build();
    this.position();
    this.attachDismiss();
  }

  mount(parent: HTMLElement): void {
    parent.appendChild(this.root);
  }

  dispose(): void {
    if (this.outsideHandler) {
      document.removeEventListener('mousedown', this.outsideHandler);
      this.outsideHandler = null;
    }
    if (this.escHandler) {
      document.removeEventListener('keydown', this.escHandler);
      this.escHandler = null;
    }
    this.addPathInput?.destroy();
    this.addPathInput = null;
    this.root.remove();
  }

  private build(): void {
    this.root.replaceChildren();

    // Reference section
    this.root.appendChild(
      sectionHeader(`Reference: ${this.opts.referenceLabel}`, true),
    );
    for (const b of this.opts.referenceBindings) {
      this.root.appendChild(this.bindingRow(b));
    }

    // Each attached source
    for (const src of this.opts.sources) {
      const header = document.createElement('div');
      header.className = 'dataset-source-header';

      const title = document.createElement('div');
      title.className = 'dataset-source-title';
      const kindBadge = document.createElement('span');
      kindBadge.className = `dataset-kind-badge dataset-kind-${src.kind}`;
      kindBadge.textContent = src.kind;
      const labelEl = document.createElement('span');
      labelEl.className = 'dataset-source-label';
      labelEl.textContent = src.label;
      labelEl.title = src.path;
      title.appendChild(kindBadge);
      title.appendChild(labelEl);

      const removeBtn = document.createElement('button');
      removeBtn.type = 'button';
      removeBtn.className = 'dataset-remove-btn';
      removeBtn.textContent = '✕';
      removeBtn.title = `Remove ${src.label}`;
      removeBtn.addEventListener('click', () => {
        void this.handleRemove(src.source_id);
      });

      header.appendChild(title);
      header.appendChild(removeBtn);
      this.root.appendChild(header);

      if (src.warning) {
        const warn = document.createElement('div');
        warn.className = 'dataset-warning';
        warn.textContent = src.warning;
        this.root.appendChild(warn);
      }

      const childBindings = this.opts.bindingsBySource.get(src.source_id) ?? [];
      if (childBindings.length === 0) {
        const none = document.createElement('div');
        none.className = 'dataset-empty-bindings';
        none.textContent = '(no tracks emitted for this source)';
        this.root.appendChild(none);
      } else {
        for (const b of childBindings) {
          this.root.appendChild(this.bindingRow(b));
        }
      }
    }

    // Add dataset section
    const addWrap = document.createElement('div');
    addWrap.className = 'dataset-add-wrap';
    const addLabel = document.createElement('label');
    addLabel.className = 'dataset-add-label';
    addLabel.textContent = '+ Add dataset';
    addWrap.appendChild(addLabel);

    const inputRow = document.createElement('div');
    inputRow.className = 'dataset-add-row';
    const pathInput = new PathInput({
      kind: 'dir',
      placeholder: 'align/ or cluster/ output dir',
    });
    const piEl = pathInput.render();
    piEl.style.flex = '1 1 auto';
    piEl.style.minWidth = '0';
    const addBtn = document.createElement('button');
    addBtn.type = 'button';
    addBtn.className = 'dataset-add-btn';
    addBtn.textContent = 'Add';
    addBtn.addEventListener('click', () => {
      void this.handleAdd();
    });
    inputRow.appendChild(piEl);
    inputRow.appendChild(addBtn);
    addWrap.appendChild(inputRow);

    const err = document.createElement('div');
    err.className = 'dataset-add-error';
    err.hidden = true;
    addWrap.appendChild(err);

    this.root.appendChild(addWrap);

    this.addPathInput = pathInput;
    this.addError = err;
    this.addBtn = addBtn;
  }

  private bindingRow(b: BindingRow): HTMLElement {
    const row = document.createElement('label');
    row.className = 'dataset-binding-row';
    const cb = document.createElement('input');
    cb.type = 'checkbox';
    cb.checked = b.visible;
    cb.addEventListener('change', () => {
      this.opts.handlers.onToggleBinding(b.binding_id, cb.checked);
    });
    const span = document.createElement('span');
    span.className = 'dataset-binding-label';
    span.textContent = b.label;
    row.appendChild(cb);
    row.appendChild(span);
    return row;
  }

  private async handleAdd(): Promise<void> {
    if (!this.addPathInput || !this.addError || !this.addBtn) return;
    const path = this.addPathInput.getValue().trim();
    if (!path) return;
    this.addError.hidden = true;
    this.addBtn.disabled = true;
    this.addBtn.textContent = 'Adding…';
    try {
      const result = await this.opts.handlers.onAddSource(path);
      if (result.ok) {
        this.addPathInput.setValue('');
        // The host will dispose + re-open us; nothing more to do.
      } else {
        this.addError.textContent = result.error;
        this.addError.hidden = false;
      }
    } catch (err) {
      this.addError.textContent = (err as Error).message;
      this.addError.hidden = false;
    } finally {
      if (this.addBtn) {
        this.addBtn.disabled = false;
        this.addBtn.textContent = 'Add';
      }
    }
  }

  private async handleRemove(sourceId: string): Promise<void> {
    try {
      await this.opts.handlers.onRemoveSource(sourceId);
      // Host re-opens us with the rebuilt source list.
    } catch (err) {
      console.warn('failed to remove source', err);
    }
  }

  private position(): void {
    const rect = this.opts.anchor.getBoundingClientRect();
    this.root.style.position = 'fixed';
    this.root.style.top = `${rect.bottom + 4}px`;
    this.root.style.right = `${window.innerWidth - rect.right}px`;
    this.root.style.zIndex = '30';
  }

  private attachDismiss(): void {
    this.outsideHandler = (e: MouseEvent): void => {
      const target = e.target as Node;
      if (this.root.contains(target)) return;
      if (this.opts.anchor.contains(target)) return;
      this.opts.onClose();
    };
    this.escHandler = (e: KeyboardEvent): void => {
      if (e.key === 'Escape') this.opts.onClose();
    };
    document.addEventListener('mousedown', this.outsideHandler);
    document.addEventListener('keydown', this.escHandler);
  }
}

function sectionHeader(text: string, readOnly: boolean): HTMLElement {
  const el = document.createElement('div');
  el.className = 'dataset-source-header';
  const title = document.createElement('div');
  title.className = 'dataset-source-title';
  const label = document.createElement('span');
  label.className = 'dataset-source-label';
  label.textContent = text;
  title.appendChild(label);
  if (readOnly) {
    const tag = document.createElement('span');
    tag.className = 'dataset-readonly-tag';
    tag.textContent = 'read-only';
    title.appendChild(tag);
  }
  el.appendChild(title);
  return el;
}
