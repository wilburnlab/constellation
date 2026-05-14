// VizForm — phase-1 form for visualization task panels.
//
// Renders the descriptor's fields, recalls `remember:true` fields from
// localStorage, and invokes the provided onSubmit on confirm.
// onSubmit is responsible for mounting the actual widget (the task
// panel passes its own host element + a transition callback).

import type { VizDescriptor } from './viz_registry';

export interface VizFormOptions {
  descriptor: VizDescriptor;
  prefill?: Record<string, string>;
  onSubmit: (values: Record<string, string>) => void | Promise<void>;
}

const STORAGE_PREFIX = 'constellation.dashboard.viz.';

export class VizForm {
  private readonly descriptor: VizDescriptor;
  private readonly prefill: Record<string, string>;
  private readonly onSubmit: (
    values: Record<string, string>,
  ) => void | Promise<void>;
  private element: HTMLElement | null = null;
  private inputs = new Map<string, HTMLInputElement>();
  private submitBtn: HTMLButtonElement | null = null;
  private errorEl: HTMLElement | null = null;

  constructor(opts: VizFormOptions) {
    this.descriptor = opts.descriptor;
    this.prefill = opts.prefill ?? {};
    this.onSubmit = opts.onSubmit;
  }

  mount(element: HTMLElement): void {
    this.element = element;
    element.classList.add('viz-form');
    element.innerHTML = '';

    const title = document.createElement('h2');
    title.textContent = this.descriptor.label;
    element.appendChild(title);

    if (this.descriptor.helpText) {
      const help = document.createElement('div');
      help.className = 'form-help';
      help.textContent = this.descriptor.helpText;
      element.appendChild(help);
    }

    const section = document.createElement('div');
    section.className = 'form-section';
    element.appendChild(section);

    for (const field of this.descriptor.fields) {
      const row = document.createElement('div');
      row.className = 'form-row';

      const label = document.createElement('label');
      const labelText = document.createElement('span');
      labelText.textContent = field.label;
      label.appendChild(labelText);
      if (field.required) {
        const star = document.createElement('span');
        star.style.color = 'var(--danger)';
        star.textContent = '*';
        label.appendChild(star);
      }
      row.appendChild(label);

      const input = document.createElement('input');
      input.type = 'text';
      input.placeholder = field.placeholder ?? '';
      input.spellcheck = false;
      input.autocomplete = 'off';

      const initial =
        this.prefill[field.name] ??
        (field.remember ? readStored(field.name) : '');
      input.value = initial;

      input.addEventListener('keydown', (e) => {
        if (e.key === 'Enter') {
          e.preventDefault();
          void this.submit();
        }
      });
      input.addEventListener('input', () => this.updateSubmitButton());
      this.inputs.set(field.name, input);
      row.appendChild(input);

      section.appendChild(row);
    }

    const actions = document.createElement('div');
    actions.className = 'form-actions';
    const submit = document.createElement('button');
    submit.type = 'button';
    submit.className = 'run';
    submit.textContent = this.descriptor.submitLabel;
    submit.addEventListener('click', () => void this.submit());
    actions.appendChild(submit);
    const err = document.createElement('span');
    err.className = 'form-error';
    actions.appendChild(err);
    element.appendChild(actions);

    this.submitBtn = submit;
    this.errorEl = err;
    this.updateSubmitButton();
  }

  destroy(): void {
    this.inputs.clear();
    this.submitBtn = null;
    this.errorEl = null;
    if (this.element) this.element.innerHTML = '';
    this.element = null;
  }

  /** Public entry so the welcome-panel quick-launch path can trigger
   *  the same flow as clicking the form's submit button. */
  async submit(): Promise<void> {
    if (!this.submitBtn) return;
    const values: Record<string, string> = {};
    for (const field of this.descriptor.fields) {
      const input = this.inputs.get(field.name);
      const raw = (input?.value ?? '').trim();
      values[field.name] = raw;
      if (field.required && !raw) {
        this.showError(`${field.label} is required.`);
        input?.focus();
        return;
      }
    }
    for (const field of this.descriptor.fields) {
      if (field.remember) writeStored(field.name, values[field.name]);
    }
    this.showError('');
    this.submitBtn.disabled = true;
    try {
      await this.onSubmit(values);
    } catch (err) {
      this.showError(err instanceof Error ? err.message : String(err));
      this.submitBtn.disabled = false;
    }
  }

  private updateSubmitButton(): void {
    if (!this.submitBtn) return;
    const ok = this.descriptor.fields.every((f) => {
      if (!f.required) return true;
      const input = this.inputs.get(f.name);
      return Boolean(input && input.value.trim());
    });
    this.submitBtn.disabled = !ok;
    if (this.errorEl) this.errorEl.textContent = '';
  }

  private showError(message: string): void {
    if (this.errorEl) this.errorEl.textContent = message;
  }
}

function readStored(name: string): string {
  try {
    return window.localStorage.getItem(STORAGE_PREFIX + name) ?? '';
  } catch {
    return '';
  }
}

function writeStored(name: string, value: string): void {
  try {
    window.localStorage.setItem(STORAGE_PREFIX + name, value);
  } catch {
    /* ignore — private mode etc. */
  }
}
