// OptionsPopover — toolbar popover for browser-wide preferences that
// aren't per-track (those live in TrackSettingsPanel on each gear
// button). The host owns the options blob; we just render rows and
// emit per-key changes via callbacks.
//
// v1 contents: "Clip SVG export to viewport". Designed so additional
// boolean toggles slot in without restructuring (default-label
// visibility, axis-font globals, etc.).

export interface BrowserOptions {
  /** When true, "Save SVG" wraps each panel in a clipPath of the
   *  configured heightPx — matching exactly what's on screen. When
   *  false (default) the SVG auto-grows vertically per panel to
   *  capture every drawn feature. */
  clip_svg: boolean;
}

export const DEFAULT_BROWSER_OPTIONS: BrowserOptions = {
  clip_svg: false,
};

export interface OptionsPopoverHandlers {
  onChange<K extends keyof BrowserOptions>(
    key: K,
    value: BrowserOptions[K],
  ): void;
}

export interface OptionsPopoverArgs {
  anchor: HTMLElement;
  options: BrowserOptions;
  handlers: OptionsPopoverHandlers;
  onClose(): void;
}

export class OptionsPopover {
  private readonly opts: OptionsPopoverArgs;
  private readonly root: HTMLElement;
  private outsideHandler: ((e: MouseEvent) => void) | null = null;
  private escHandler: ((e: KeyboardEvent) => void) | null = null;

  constructor(opts: OptionsPopoverArgs) {
    this.opts = opts;
    this.root = document.createElement('div');
    this.root.className = 'options-popover';
    this.root.setAttribute('role', 'dialog');
    this.root.setAttribute('aria-label', 'Browser options');
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
    this.root.remove();
  }

  private build(): void {
    this.root.replaceChildren();

    const exportTitle = document.createElement('div');
    exportTitle.className = 'options-section-title';
    exportTitle.textContent = 'SVG export';
    this.root.appendChild(exportTitle);

    this.root.appendChild(
      this.booleanRow(
        'Clip to viewport',
        'When on, exported SVG matches what the browser shows; ' +
          'overflowing features are cut off. When off (default), the ' +
          'SVG grows vertically to capture every feature.',
        this.opts.options.clip_svg,
        (v) => this.opts.handlers.onChange('clip_svg', v),
      ),
    );
  }

  private booleanRow(
    label: string,
    hint: string,
    initial: boolean,
    onChange: (value: boolean) => void,
  ): HTMLElement {
    const row = document.createElement('label');
    row.className = 'options-row';
    const cb = document.createElement('input');
    cb.type = 'checkbox';
    cb.checked = initial;
    cb.addEventListener('change', () => onChange(cb.checked));
    const text = document.createElement('div');
    text.className = 'options-row-text';
    const labelEl = document.createElement('span');
    labelEl.className = 'options-row-label';
    labelEl.textContent = label;
    const hintEl = document.createElement('span');
    hintEl.className = 'options-row-hint';
    hintEl.textContent = hint;
    text.appendChild(labelEl);
    text.appendChild(hintEl);
    row.appendChild(cb);
    row.appendChild(text);
    return row;
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
