// Auto-generated form for one CLI subcommand.
//
// Drives off the ArgumentSchema list emitted by the introspect walker.
// Constructs an argv array from form state and POSTs to /api/commands.
// On 200, invokes the host-provided `onJobStarted` callback so the
// TaskPanel can swap its content to a terminal in-place. On 409,
// surfaces the rejection message inline.

import type { ArgumentSchema, CommandResponse, CommandSchema } from './types';
import { DashboardState } from './state';

export interface CommandFormOptions {
  command: CommandSchema;
  state: DashboardState;
  /** Host callback fired once /api/commands returns 200 with a job_id.
   *  The TaskPanel uses this to transition the panel into terminal
   *  mode without spawning a separate dock panel. */
  onJobStarted?: (event: { jobId: string; argv: string[] }) => void;
}

type ArgState = Map<string, string | boolean | string[]>;

const ADVANCED_DEST_PATTERNS = [
  /^threads$/,
  /^chunk_size$/,
  /^batch_size$/,
  /^resume$/,
  /^progress$/,
  /^min_/,
  /^max_/,
  /^intron_/,
  /_limit$/,
  /_priority$/,
  /_tolerance/,
  /^matrix_min/,
  /^cache_dir$/,
  /^repo$/,
  /^emit_/,
  /^allow_/,
];

export class CommandForm {
  private readonly command: CommandSchema;
  private readonly state: DashboardState;
  private readonly onJobStarted?: (
    event: { jobId: string; argv: string[] },
  ) => void;
  private readonly values: ArgState = new Map();
  private element: HTMLElement | null = null;
  private runButton: HTMLButtonElement | null = null;
  private errorEl: HTMLElement | null = null;
  private destroyFn: (() => void) | null = null;

  constructor(opts: CommandFormOptions) {
    this.command = opts.command;
    this.state = opts.state;
    this.onJobStarted = opts.onJobStarted;
    for (const arg of opts.command.arguments) {
      this.values.set(arg.dest, defaultFormValue(arg));
    }
  }

  mount(element: HTMLElement): void {
    this.element = element;
    element.classList.add('command-form');
    element.innerHTML = '';

    const title = document.createElement('h2');
    title.textContent = formatLabel(this.command);
    element.appendChild(title);

    const cmdLine = document.createElement('div');
    cmdLine.className = 'form-cmd';
    cmdLine.textContent = 'constellation ' + this.command.path.join(' ');
    element.appendChild(cmdLine);

    if (this.command.help) {
      const help = document.createElement('div');
      help.className = 'form-help';
      help.textContent = this.command.help;
      help.style.marginBottom = '12px';
      element.appendChild(help);
    }

    const sections = this.bucketArguments();
    if (sections.required.length > 0) {
      element.appendChild(this.renderSection('Required', sections.required, false));
    }
    if (sections.optional.length > 0) {
      element.appendChild(this.renderSection('Optional', sections.optional, false));
    }
    if (sections.advanced.length > 0) {
      element.appendChild(this.renderSection('Advanced', sections.advanced, true));
    }

    const actions = document.createElement('div');
    actions.className = 'form-actions';
    const run = document.createElement('button');
    run.className = 'run';
    run.textContent = 'Run';
    run.type = 'button';
    run.addEventListener('click', () => void this.submit());
    actions.appendChild(run);
    const err = document.createElement('span');
    err.className = 'form-error';
    actions.appendChild(err);
    element.appendChild(actions);

    this.runButton = run;
    this.errorEl = err;
    this.updateRunButton();

    // Subscribe to rejections — the form surfaces 409s inline.
    this.destroyFn = this.state.on('job:rejected', (message) => {
      this.showError(message);
    });
  }

  destroy(): void {
    this.destroyFn?.();
    this.destroyFn = null;
    if (this.element) this.element.innerHTML = '';
  }

  // -------------------------------------------------------------------
  // Argument bucketing
  // -------------------------------------------------------------------

  private bucketArguments(): {
    required: ArgumentSchema[];
    optional: ArgumentSchema[];
    advanced: ArgumentSchema[];
  } {
    const required: ArgumentSchema[] = [];
    const optional: ArgumentSchema[] = [];
    const advanced: ArgumentSchema[] = [];
    for (const arg of this.command.arguments) {
      if (arg.required && !arg.is_positional) {
        required.push(arg);
      } else if (arg.is_positional) {
        required.push(arg);
      } else if (isAdvanced(arg)) {
        advanced.push(arg);
      } else {
        optional.push(arg);
      }
    }
    return { required, optional, advanced };
  }

  private renderSection(
    title: string,
    args: ArgumentSchema[],
    collapsible: boolean,
  ): HTMLElement {
    const section = document.createElement('div');
    section.className = 'form-section';
    const header = document.createElement('div');
    header.className = 'form-section-header';
    header.textContent = title;
    if (collapsible) {
      header.classList.add('collapsible');
    }
    section.appendChild(header);

    const body = document.createElement('div');
    section.appendChild(body);

    if (collapsible) {
      let expanded = false;
      header.classList.add('collapsed');
      body.style.display = 'none';
      header.addEventListener('click', () => {
        expanded = !expanded;
        header.classList.toggle('collapsed', !expanded);
        body.style.display = expanded ? '' : 'none';
      });
    }

    for (const arg of args) {
      body.appendChild(this.renderArgument(arg));
    }
    return section;
  }

  private renderArgument(arg: ArgumentSchema): HTMLElement {
    const row = document.createElement('div');
    row.className = 'form-row';

    const label = document.createElement('label');
    const flagName = primaryFlag(arg);
    const labelText = document.createElement('span');
    labelText.textContent = displayName(arg);
    label.appendChild(labelText);
    if (flagName) {
      const code = document.createElement('span');
      code.className = 'flag-name';
      code.textContent = flagName;
      label.appendChild(code);
    }
    if (arg.required) {
      const star = document.createElement('span');
      star.style.color = 'var(--danger)';
      star.textContent = '*';
      label.appendChild(star);
    }
    row.appendChild(label);

    const input = this.renderInput(arg);
    row.appendChild(input);

    if (arg.help) {
      const help = document.createElement('div');
      help.className = 'form-help';
      help.textContent = arg.help;
      row.appendChild(help);
    }
    return row;
  }

  private renderInput(arg: ArgumentSchema): HTMLElement {
    const dest = arg.dest;
    if (arg.type === 'flag') {
      const cb = document.createElement('input');
      cb.type = 'checkbox';
      cb.checked = Boolean(this.values.get(dest));
      cb.addEventListener('change', () => {
        this.values.set(dest, cb.checked);
        this.updateRunButton();
      });
      const wrap = document.createElement('label');
      wrap.style.display = 'flex';
      wrap.style.gap = '6px';
      wrap.style.alignItems = 'center';
      wrap.appendChild(cb);
      const note = document.createElement('span');
      note.style.color = 'var(--fg-dim)';
      note.style.fontSize = '11px';
      note.textContent = cb.checked ? 'enabled' : 'disabled';
      wrap.appendChild(note);
      cb.addEventListener('change', () => {
        note.textContent = cb.checked ? 'enabled' : 'disabled';
      });
      return wrap;
    }
    if (arg.type === 'enum') {
      const sel = document.createElement('select');
      const current = String(this.values.get(dest) ?? '');
      const choices = (arg.choices ?? []).map((c) => String(c));
      if (!arg.required && !choices.includes(current)) {
        const placeholder = document.createElement('option');
        placeholder.value = '';
        placeholder.textContent = '(default)';
        sel.appendChild(placeholder);
      }
      for (const choice of choices) {
        const opt = document.createElement('option');
        opt.value = choice;
        opt.textContent = choice;
        if (choice === current) opt.selected = true;
        sel.appendChild(opt);
      }
      sel.addEventListener('change', () => {
        this.values.set(dest, sel.value);
        this.updateRunButton();
      });
      return sel;
    }
    if (arg.type === 'multi') {
      const ta = document.createElement('textarea');
      ta.rows = 3;
      ta.style.background = 'var(--bg-elev)';
      ta.style.color = 'var(--fg)';
      ta.style.border = '1px solid var(--border)';
      ta.style.borderRadius = '4px';
      ta.style.padding = '4px 8px';
      ta.style.fontFamily =
        'ui-monospace, SF Mono, Menlo, monospace';
      ta.style.fontSize = '12px';
      ta.style.width = '100%';
      ta.placeholder = 'one value per line';
      const current = this.values.get(dest);
      ta.value = Array.isArray(current) ? current.join('\n') : '';
      ta.addEventListener('input', () => {
        const lines = ta.value
          .split('\n')
          .map((s) => s.trim())
          .filter((s) => s.length > 0);
        this.values.set(dest, lines);
        this.updateRunButton();
      });
      return ta;
    }
    // text / int / float / path all render as a single-line input
    const input = document.createElement('input');
    if (arg.type === 'int' || arg.type === 'float') {
      input.type = 'number';
      if (arg.type === 'float') input.step = 'any';
    } else {
      input.type = 'text';
    }
    if (arg.metavar) input.placeholder = arg.metavar;
    else if (arg.type === 'path') input.placeholder = '/path/to/...';
    const current = this.values.get(dest);
    input.value = current === null || current === undefined ? '' : String(current);
    input.addEventListener('input', () => {
      this.values.set(dest, input.value);
      this.validateInput(input, arg);
      this.updateRunButton();
    });
    return input;
  }

  private validateInput(input: HTMLInputElement, arg: ArgumentSchema): void {
    let invalid = false;
    const v = input.value.trim();
    if (arg.required && v === '') {
      invalid = true;
    } else if (v !== '') {
      if (arg.type === 'int' && !/^-?\d+$/.test(v)) invalid = true;
      if (arg.type === 'float' && Number.isNaN(Number(v))) invalid = true;
    }
    input.classList.toggle('invalid', invalid);
  }

  private updateRunButton(): void {
    if (!this.runButton) return;
    const ok = this.command.arguments.every((arg) => {
      if (!arg.required) return true;
      const v = this.values.get(arg.dest);
      if (arg.type === 'flag') return true;
      if (Array.isArray(v)) return v.length > 0;
      return v !== undefined && String(v).trim() !== '';
    });
    this.runButton.disabled = !ok;
    if (this.errorEl) this.errorEl.textContent = '';
  }

  private showError(message: string): void {
    if (!this.errorEl) return;
    this.errorEl.textContent = message;
  }

  // -------------------------------------------------------------------
  // argv assembly + submit
  // -------------------------------------------------------------------

  private assembleArgv(): string[] {
    const argv: string[] = [...this.command.path];
    // First, all non-positional flags
    for (const arg of this.command.arguments) {
      if (arg.is_positional) continue;
      const v = this.values.get(arg.dest);
      const flag = primaryFlag(arg);
      if (!flag) continue;
      if (arg.type === 'flag') {
        const negFlag = negativeFlag(arg);
        const checked = Boolean(v);
        // For BooleanOptionalAction (has both --flag and --no-flag),
        // emit --flag for true and --no-flag for false ONLY when the
        // state differs from the default. For store_true (positive-
        // only), emit --flag only when true. For store_false
        // (negative-only), emit when false.
        if (negFlag) {
          if (checked && arg.default !== true) {
            argv.push(flag);
          } else if (!checked && arg.default === true) {
            argv.push(negFlag);
          }
        } else if (flag.startsWith('--no-')) {
          // store_false-style: only emit when actually disabling
          if (checked) argv.push(flag);
        } else {
          if (checked) argv.push(flag);
        }
        continue;
      }
      if (arg.type === 'multi') {
        const items = Array.isArray(v) ? v : [];
        if (items.length === 0) continue;
        argv.push(flag, ...items);
        continue;
      }
      const s = String(v ?? '').trim();
      if (s === '') continue;
      argv.push(flag, s);
    }
    // Then positionals — appended in declaration order
    for (const arg of this.command.arguments) {
      if (!arg.is_positional) continue;
      const v = this.values.get(arg.dest);
      if (Array.isArray(v)) {
        argv.push(...v);
        continue;
      }
      const s = String(v ?? '').trim();
      if (s) argv.push(s);
    }
    return argv;
  }

  private async submit(): Promise<void> {
    if (!this.runButton) return;
    this.runButton.disabled = true;
    this.showError('');
    const argv = this.assembleArgv();
    try {
      const response = await fetch('/api/commands', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ argv }),
      });
      if (response.status === 409) {
        const body = (await response.json().catch(() => ({}))) as {
          detail?: string;
        };
        const message = body.detail ?? 'another job is running';
        this.showError(message);
        this.state.emit('job:rejected', message);
        this.runButton.disabled = false;
        return;
      }
      if (!response.ok) {
        const text = await response.text().catch(() => '');
        this.showError(`error: ${response.status} ${text}`);
        this.runButton.disabled = false;
        return;
      }
      const data = (await response.json()) as CommandResponse;
      this.onJobStarted?.({ jobId: data.job_id, argv: data.argv });
    } catch (err) {
      this.showError((err as Error).message);
    } finally {
      this.runButton.disabled = false;
      this.updateRunButton();
    }
  }
}

// ---------------------------------------------------------------------
// Helpers (module-level so tests can import them if needed)
// ---------------------------------------------------------------------

function defaultFormValue(arg: ArgumentSchema): string | boolean | string[] {
  if (arg.type === 'flag') return Boolean(arg.default);
  if (arg.type === 'multi') {
    return Array.isArray(arg.default) ? arg.default.map(String) : [];
  }
  return arg.default === null || arg.default === undefined
    ? ''
    : String(arg.default);
}

function primaryFlag(arg: ArgumentSchema): string {
  if (arg.option_strings.length === 0) return ''; // positional
  // BooleanOptionalAction puts `--flag` first and `--no-flag` second;
  // pick whichever doesn't start with `--no-`.
  const positive = arg.option_strings.find((f) => !f.startsWith('--no-'));
  return positive ?? arg.option_strings[0];
}

function negativeFlag(arg: ArgumentSchema): string | null {
  // BooleanOptionalAction stores both forms — return the negative one
  // if present.
  return arg.option_strings.find((f) => f.startsWith('--no-')) ?? null;
}

function displayName(arg: ArgumentSchema): string {
  return arg.dest.replace(/_/g, ' ').replace(/\b\w/g, (c) => c.toUpperCase());
}

function isAdvanced(arg: ArgumentSchema): boolean {
  return ADVANCED_DEST_PATTERNS.some((re) => re.test(arg.dest));
}

function formatLabel(cmd: CommandSchema): string {
  return cmd.path.map((p) => p.replace(/-/g, ' ')).join(' / ');
}
