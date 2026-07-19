// PathInput — an editable path text box plus a "Browse…" button that
// opens the FilePicker. The hybrid pattern: typing / pasting still works
// (so a not-yet-existing output dir or a copy-pasted path is fine), and
// Browse fills the field from the picker. This is the single reusable
// widget every path field across the GUI drops in — CommandForm's
// `path` args, the genome-browser source rows, the Datasets popover's
// add-dataset form, and VizForm path fields.

import { FilePicker, type PathKindMode } from './FilePicker';

export interface PathInputOptions {
  value?: string;
  /** 'dir' opens the picker in folder-select mode; 'file' in file-pick
   *  mode. */
  kind: PathKindMode;
  /** File filters (file mode), e.g. ['*.tsv', '*.bed']. */
  globs?: string[];
  placeholder?: string;
  /** Fired on every keystroke and on Browse-select — cheap live value
   *  updates (run-button enable, storing the value). */
  onChange?(value: string): void;
  /** Fired on blur (input `change`) and on Browse-select — the hook for
   *  expensive side-effects like directory inspection. */
  onCommit?(value: string): void;
}

export class PathInput {
  private readonly opts: PathInputOptions;
  private readonly row: HTMLElement;
  private readonly input: HTMLInputElement;
  private readonly browseBtn: HTMLButtonElement;
  private picker: FilePicker | null = null;

  constructor(opts: PathInputOptions) {
    this.opts = opts;

    this.row = document.createElement('div');
    this.row.className = 'path-input-row';

    this.input = document.createElement('input');
    this.input.type = 'text';
    this.input.className = 'path-input-field';
    this.input.spellcheck = false;
    this.input.autocomplete = 'off';
    this.input.placeholder =
      opts.placeholder ?? (opts.kind === 'dir' ? '/path/to/dir' : '/path/to/file');
    this.input.value = opts.value ?? '';
    this.input.addEventListener('input', () => {
      this.opts.onChange?.(this.input.value);
    });
    this.input.addEventListener('change', () => {
      this.opts.onCommit?.(this.input.value);
    });

    this.browseBtn = document.createElement('button');
    this.browseBtn.type = 'button';
    this.browseBtn.className = 'path-input-browse';
    this.browseBtn.textContent = 'Browse…';
    this.browseBtn.title =
      opts.kind === 'dir' ? 'Browse for a folder' : 'Browse for a file';
    this.browseBtn.addEventListener('click', () => this.openPicker());

    this.row.appendChild(this.input);
    this.row.appendChild(this.browseBtn);
  }

  /** The row element to insert into a form. */
  render(): HTMLElement {
    return this.row;
  }

  getValue(): string {
    return this.input.value;
  }

  setValue(value: string): void {
    this.input.value = value;
    this.opts.onChange?.(value);
  }

  focus(): void {
    this.input.focus();
  }

  destroy(): void {
    this.closePicker();
    this.row.remove();
  }

  private openPicker(): void {
    if (this.picker) {
      this.closePicker();
      return;
    }
    this.picker = new FilePicker({
      anchor: this.browseBtn,
      kind: this.opts.kind,
      initialPath: this.input.value.trim() || undefined,
      globs: this.opts.globs,
      onSelect: (path) => {
        this.setValue(path);
        this.opts.onCommit?.(path);
        this.closePicker();
      },
      onClose: () => this.closePicker(),
    });
    // Mount inside the row so a host popover's outside-click dismiss
    // doesn't fire while the picker is open (position:fixed keeps layout
    // anchored to the button regardless of DOM parent).
    this.picker.mount(this.row);
  }

  private closePicker(): void {
    if (this.picker) {
      this.picker.dispose();
      this.picker = null;
    }
  }
}
