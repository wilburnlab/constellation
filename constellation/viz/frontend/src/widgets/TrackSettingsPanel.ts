// TrackSettingsPanel — per-binding gear popover. Three sections:
//
//   General — title override, opacity, label font, show/hide legend.
//             Same fields for every kernel.
//   Style   — kernel-specific palette + size knobs.
//   Filter  — kernel-specific dataset slicing (samples/modes/motifs,
//             strand toggles, min thresholds).
//
// All values flow back to the host via two callbacks: `onStyleChange`
// for visual knobs and `onFilterChange` for data filters. The host
// re-renders client-side from cached Arrow data (no server round-trip)
// so palette changes are instant.
//
// The panel positions itself anchored to the gear button using the
// same fixed-position pattern as DatasetManagerPopover / OptionsPopover.
// Categorical lists (samples/modes/motifs) come from the binding's
// metadata payload which is already fetched at mount time.

import { TrackMetadata } from '../track_renderers/base';

export interface TrackSettingsArgs {
  anchor: HTMLElement;
  kind: string;
  label: string;
  meta: TrackMetadata;
  style: Record<string, unknown>;
  filter: Record<string, unknown>;
  onStyleChange(style: Record<string, unknown>): void;
  onFilterChange(filter: Record<string, unknown>): void;
  onReset(): void;
  onClose(): void;
}

export class TrackSettingsPanel {
  private readonly opts: TrackSettingsArgs;
  private readonly root: HTMLElement;
  private style: Record<string, unknown>;
  private filter: Record<string, unknown>;
  private outsideHandler: ((e: MouseEvent) => void) | null = null;
  private escHandler: ((e: KeyboardEvent) => void) | null = null;

  constructor(opts: TrackSettingsArgs) {
    this.opts = opts;
    this.style = { ...opts.style };
    this.filter = { ...opts.filter };
    this.root = document.createElement('div');
    this.root.className = 'track-settings-popover';
    this.root.setAttribute('role', 'dialog');
    this.root.setAttribute('aria-label', `Track settings — ${opts.label}`);
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

  // --------------------------------------------------------------------
  // Build
  // --------------------------------------------------------------------

  private build(): void {
    this.root.replaceChildren();

    const header = document.createElement('div');
    header.className = 'settings-header';
    const title = document.createElement('span');
    title.className = 'settings-title';
    title.textContent = this.opts.label;
    title.title = this.opts.label;
    const kind = document.createElement('span');
    kind.className = 'settings-kind';
    kind.textContent = this.opts.kind;
    header.appendChild(title);
    header.appendChild(kind);
    this.root.appendChild(header);

    this.root.appendChild(this.buildGeneralSection());
    const styleSection = this.buildStyleSection();
    if (styleSection) this.root.appendChild(styleSection);
    const filterSection = this.buildFilterSection();
    if (filterSection) this.root.appendChild(filterSection);

    const actions = document.createElement('div');
    actions.className = 'settings-actions';
    const resetBtn = document.createElement('button');
    resetBtn.type = 'button';
    resetBtn.className = 'settings-reset-btn';
    resetBtn.textContent = 'Reset to defaults';
    resetBtn.addEventListener('click', () => {
      this.style = {};
      this.filter = {};
      this.opts.onReset();
      this.build();
    });
    actions.appendChild(resetBtn);
    this.root.appendChild(actions);
  }

  // --------------------------------------------------------------------
  // General section — common to every kernel
  // --------------------------------------------------------------------

  private buildGeneralSection(): HTMLElement {
    const section = makeSection('General');

    section.appendChild(
      this.numberRow('Opacity', 'opacity', defaultOpacity(this.opts.kind), {
        min: 0,
        max: 1,
        step: 0.05,
      }),
    );
    section.appendChild(
      this.textRow('Label font family', 'label_font_family', ''),
    );
    section.appendChild(
      this.numberRow('Label font size (px)', 'label_font_size_px', 10, {
        min: 6,
        max: 24,
        step: 1,
      }),
    );
    section.appendChild(
      this.checkboxStyleRow('Show legend / labels', 'show_legend', true),
    );
    return section;
  }

  // --------------------------------------------------------------------
  // Style — kernel-specific
  // --------------------------------------------------------------------

  private buildStyleSection(): HTMLElement | null {
    const section = makeSection('Style');
    const kind = this.opts.kind;
    if (kind === 'reference_sequence') {
      const paletteRows = paletteSection(
        ['A', 'C', 'G', 'T', 'U', 'N'],
        {
          A: '#5cd66e',
          C: '#5e9cd6',
          G: '#d6c95e',
          T: '#d65e5e',
          U: '#d65e5e',
          N: '#666',
        },
        (key, hex) => this.setPalette(key, hex),
        (key) => this.getPalette(key),
      );
      paletteRows.forEach((row) => section.appendChild(row));
      section.appendChild(
        this.numberRow(
          'Letter font size (px)',
          'letter_font_size_px',
          11,
          { min: 6, max: 20, step: 1 },
        ),
      );
      section.appendChild(
        this.numberRow(
          'Letter threshold (px/bp)',
          'letter_threshold_px_per_bp',
          6,
          { min: 1, max: 40, step: 1 },
        ),
      );
    } else if (kind === 'gene_annotation') {
      const paletteRows = paletteSection(
        [
          'gene',
          'mRNA',
          'CDS',
          'exon',
          'three_prime_UTR',
          'five_prime_UTR',
          'repeat_region',
          'default',
        ],
        {
          gene: '#5e8cd6',
          mRNA: '#7a99e0',
          CDS: '#c5d65e',
          exon: '#9bc23a',
          three_prime_UTR: '#a89dc4',
          five_prime_UTR: '#a89dc4',
          repeat_region: '#d65e5e',
          default: '#888888',
        },
        (key, hex) => this.setPalette(key, hex),
        (key) => this.getPalette(key),
      );
      paletteRows.forEach((row) => section.appendChild(row));
      section.appendChild(
        this.numberRow('Row height (px)', 'row_height_px', 14, {
          min: 6,
          max: 40,
          step: 1,
        }),
      );
      section.appendChild(
        this.numberRow('Feature opacity', 'feature_opacity', 0.85, {
          min: 0,
          max: 1,
          step: 0.05,
        }),
      );
      section.appendChild(
        this.numberRow('Label min width (px)', 'label_min_width_px', 24, {
          min: 0,
          max: 200,
          step: 1,
        }),
      );
      section.appendChild(
        this.checkboxStyleRow('Show chevrons', 'show_chevrons', true),
      );
      section.appendChild(
        this.checkboxStyleRow('Show labels', 'show_labels', true),
      );
    } else if (kind === 'coverage_histogram') {
      const samples = numericList(this.opts.meta.samples);
      const paletteCycle = [
        '#4f9efb',
        '#fb7c4f',
        '#a4d65e',
        '#d65eb6',
        '#5ed6cf',
      ];
      if (samples.length === 0) {
        section.appendChild(emptyHint('no samples in window yet'));
      } else {
        samples.forEach((sampleId, idx) => {
          const fallback = paletteCycle[idx % paletteCycle.length];
          const label =
            sampleId === -1 ? 'all (unstratified)' : `sample ${sampleId}`;
          section.appendChild(
            paletteRow(
              label,
              this.getPalette(String(sampleId)) ?? fallback,
              (hex) => this.setPalette(String(sampleId), hex),
            ),
          );
        });
      }
      section.appendChild(
        this.numberRow('Fill opacity', 'fill_opacity', 0.4, {
          min: 0,
          max: 1,
          step: 0.05,
        }),
      );
      section.appendChild(
        this.numberRow('Stroke width (px)', 'stroke_width_px', 1, {
          min: 0,
          max: 6,
          step: 0.5,
        }),
      );
      section.appendChild(
        this.selectRow('Y scale', 'y_scale', 'linear', [
          { value: 'linear', label: 'linear' },
          { value: 'log', label: 'log' },
        ]),
      );
      section.appendChild(
        this.checkboxStyleRow(
          'Show sample labels',
          'show_sample_labels',
          true,
        ),
      );
      section.appendChild(
        this.checkboxStyleRow('Show max depth', 'show_max_depth', true),
      );
    } else if (kind === 'read_pileup') {
      ['+', '-', 'default'].forEach((strand) => {
        const fallback = strand === '+' ? '#5e9cd6' : strand === '-' ? '#d6755e' : '#888888';
        const label =
          strand === '+' ? 'Forward strand'
          : strand === '-' ? 'Reverse strand'
          : 'Unstranded / default';
        section.appendChild(
          paletteRow(label, this.getPalette(strand) ?? fallback, (hex) =>
            this.setPalette(strand, hex),
          ),
        );
      });
      section.appendChild(
        this.numberRow('Min row height (px)', 'min_row_height_px', 2, {
          min: 1,
          max: 20,
          step: 1,
        }),
      );
      section.appendChild(
        this.numberRow('Max row height (px)', 'max_row_height_px', 8, {
          min: 2,
          max: 40,
          step: 1,
        }),
      );
      section.appendChild(
        this.numberRow('Read opacity', 'read_opacity', 1.0, {
          min: 0,
          max: 1,
          step: 0.05,
        }),
      );
    } else if (kind === 'cluster_pileup') {
      const modes = stringList(this.opts.meta.modes);
      const defaults: Record<string, string> = {
        'genome-guided': '#5ed6cf',
        'de-novo': '#a8d65e',
        default: '#888888',
      };
      if (modes.length === 0) {
        // Fall back to the known modes so the picker isn't empty.
        modes.push('genome-guided', 'de-novo');
      }
      modes.forEach((mode) => {
        const fallback = defaults[mode] ?? '#888888';
        section.appendChild(
          paletteRow(mode, this.getPalette(mode) ?? fallback, (hex) =>
            this.setPalette(mode, hex),
          ),
        );
      });
      section.appendChild(
        this.numberRow('Min row height (px)', 'min_row_height_px', 4, {
          min: 1,
          max: 20,
          step: 1,
        }),
      );
      section.appendChild(
        this.numberRow('Max row height (px)', 'max_row_height_px', 10, {
          min: 2,
          max: 40,
          step: 1,
        }),
      );
      section.appendChild(
        this.numberRow('Opacity min', 'opacity_min', 0.4, {
          min: 0,
          max: 1,
          step: 0.05,
        }),
      );
      section.appendChild(
        this.numberRow('Opacity max', 'opacity_max', 1.0, {
          min: 0,
          max: 1,
          step: 0.05,
        }),
      );
    } else if (kind === 'splice_junctions') {
      const motifs = stringList(this.opts.meta.motifs);
      const defaults: Record<string, string> = {
        'GT-AG': '#5e9cd6',
        'GC-AG': '#9b5ed6',
        'AT-AC': '#d6a05e',
        default: '#888888',
      };
      if (motifs.length === 0) {
        motifs.push('GT-AG', 'GC-AG', 'AT-AC');
      }
      motifs.forEach((motif) => {
        const fallback = defaults[motif] ?? '#888888';
        section.appendChild(
          paletteRow(motif, this.getPalette(motif) ?? fallback, (hex) =>
            this.setPalette(motif, hex),
          ),
        );
      });
      section.appendChild(
        this.numberRow('Min stroke (px)', 'arc_stroke_min_px', 1, {
          min: 0.5,
          max: 10,
          step: 0.5,
        }),
      );
      section.appendChild(
        this.numberRow('Max stroke (px)', 'arc_stroke_max_px', 4, {
          min: 0.5,
          max: 12,
          step: 0.5,
        }),
      );
      section.appendChild(
        this.numberRow('Arc opacity', 'arc_opacity', 0.85, {
          min: 0,
          max: 1,
          step: 0.05,
        }),
      );
    } else {
      section.appendChild(emptyHint('no style controls for this track kind'));
    }
    return section;
  }

  // --------------------------------------------------------------------
  // Filter — kernel-specific dataset-slice controls
  // --------------------------------------------------------------------

  private buildFilterSection(): HTMLElement | null {
    const kind = this.opts.kind;
    if (kind === 'reference_sequence') return null;
    const section = makeSection('Filter');

    if (kind === 'gene_annotation') {
      section.appendChild(
        allowListRow(
          'Visible feature types',
          [
            'gene',
            'mRNA',
            'CDS',
            'exon',
            'three_prime_UTR',
            'five_prime_UTR',
            'repeat_region',
          ],
          this.getFilterArray('visible_types'),
          (selected) => this.setFilterArray('visible_types', selected),
        ),
      );
      section.appendChild(
        allowListRow(
          'Visible strands',
          ['+', '-', '.'],
          this.getFilterArray('visible_strands'),
          (selected) => this.setFilterArray('visible_strands', selected),
        ),
      );
      section.appendChild(
        allowListRow(
          'Visible sources',
          ['reference', 'derived'],
          this.getFilterArray('visible_sources'),
          (selected) => this.setFilterArray('visible_sources', selected),
        ),
      );
      section.appendChild(
        this.numberFilterRow('Min length (bp)', 'min_length_bp', 0, {
          min: 0,
          max: 1_000_000,
          step: 10,
        }),
      );
    } else if (kind === 'coverage_histogram') {
      const samples = numericList(this.opts.meta.samples);
      if (samples.length === 0) {
        section.appendChild(emptyHint('no samples in window yet'));
      } else {
        section.appendChild(
          allowListRow(
            'Visible samples',
            samples.map(String),
            this.getFilterArray('visible_samples'),
            (selected) => {
              const asNums = selected
                .map((s) => Number(s))
                .filter((n) => Number.isFinite(n));
              this.setFilterArray('visible_samples', asNums);
            },
            (key) => (key === '-1' ? 'all' : `sample ${key}`),
          ),
        );
      }
      section.appendChild(
        this.numberFilterRow('Min depth', 'min_depth', 0, {
          min: 0,
          max: 1_000_000,
          step: 1,
        }),
      );
    } else if (kind === 'read_pileup') {
      section.appendChild(
        allowListRow(
          'Visible strands',
          ['+', '-'],
          this.getFilterArray('visible_strands'),
          (selected) => this.setFilterArray('visible_strands', selected),
        ),
      );
    } else if (kind === 'cluster_pileup') {
      const modes = stringList(this.opts.meta.modes);
      if (modes.length === 0) modes.push('genome-guided', 'de-novo');
      section.appendChild(
        allowListRow(
          'Visible modes',
          modes,
          this.getFilterArray('visible_modes'),
          (selected) => this.setFilterArray('visible_modes', selected),
        ),
      );
      section.appendChild(
        allowListRow(
          'Visible strands',
          ['+', '-', '.'],
          this.getFilterArray('visible_strands'),
          (selected) => this.setFilterArray('visible_strands', selected),
        ),
      );
      section.appendChild(
        this.numberFilterRow('Min reads', 'min_reads', 1, {
          min: 1,
          max: 1_000_000,
          step: 1,
        }),
      );
    } else if (kind === 'splice_junctions') {
      const motifs = stringList(this.opts.meta.motifs);
      if (motifs.length === 0) motifs.push('GT-AG', 'GC-AG', 'AT-AC');
      section.appendChild(
        allowListRow(
          'Visible motifs',
          motifs,
          this.getFilterArray('visible_motifs'),
          (selected) => this.setFilterArray('visible_motifs', selected),
        ),
      );
      section.appendChild(
        this.numberFilterRow('Min support', 'min_support', 1, {
          min: 1,
          max: 1_000_000,
          step: 1,
        }),
      );
      section.appendChild(
        this.checkboxFilterRow(
          'Only annotated junctions',
          'annotated_only',
          false,
        ),
      );
    } else {
      section.appendChild(emptyHint('no filter controls for this track kind'));
    }
    return section;
  }

  // --------------------------------------------------------------------
  // Row builders
  // --------------------------------------------------------------------

  private numberRow(
    label: string,
    key: string,
    fallback: number,
    bounds: { min: number; max: number; step: number },
  ): HTMLElement {
    return numberRow(
      label,
      this.getNumber(this.style, key, fallback),
      bounds,
      (value) => {
        this.style[key] = value;
        this.opts.onStyleChange(this.style);
      },
    );
  }

  private numberFilterRow(
    label: string,
    key: string,
    fallback: number,
    bounds: { min: number; max: number; step: number },
  ): HTMLElement {
    return numberRow(
      label,
      this.getNumber(this.filter, key, fallback),
      bounds,
      (value) => {
        this.filter[key] = value;
        this.opts.onFilterChange(this.filter);
      },
    );
  }

  private textRow(label: string, key: string, fallback: string): HTMLElement {
    return textRow(label, this.getString(this.style, key, fallback), (value) => {
      if (value) {
        this.style[key] = value;
      } else {
        delete this.style[key];
      }
      this.opts.onStyleChange(this.style);
    });
  }

  private selectRow(
    label: string,
    key: string,
    fallback: string,
    options: Array<{ value: string; label: string }>,
  ): HTMLElement {
    return selectRow(
      label,
      this.getString(this.style, key, fallback),
      options,
      (value) => {
        this.style[key] = value;
        this.opts.onStyleChange(this.style);
      },
    );
  }

  private checkboxStyleRow(
    label: string,
    key: string,
    fallback: boolean,
  ): HTMLElement {
    return checkboxRow(
      label,
      this.getBoolean(this.style, key, fallback),
      (value) => {
        this.style[key] = value;
        this.opts.onStyleChange(this.style);
      },
    );
  }

  private checkboxFilterRow(
    label: string,
    key: string,
    fallback: boolean,
  ): HTMLElement {
    return checkboxRow(
      label,
      this.getBoolean(this.filter, key, fallback),
      (value) => {
        this.filter[key] = value;
        this.opts.onFilterChange(this.filter);
      },
    );
  }

  // --------------------------------------------------------------------
  // Palette helpers — store under `palette.<key>` so the renderer's
  // pickPaletteColor finds them.
  // --------------------------------------------------------------------

  private setPalette(key: string, hex: string): void {
    this.style[`palette.${key}`] = hex;
    this.opts.onStyleChange(this.style);
  }

  private getPalette(key: string): string | undefined {
    const dotted = this.style[`palette.${key}`];
    if (typeof dotted === 'string') return dotted;
    const nested = this.style.palette;
    if (nested && typeof nested === 'object') {
      const v = (nested as Record<string, unknown>)[key];
      if (typeof v === 'string') return v;
    }
    return undefined;
  }

  // --------------------------------------------------------------------
  // Coerce helpers
  // --------------------------------------------------------------------

  private getNumber(
    source: Record<string, unknown>,
    key: string,
    fallback: number,
  ): number {
    const v = source[key];
    if (typeof v === 'number' && Number.isFinite(v)) return v;
    return fallback;
  }

  private getString(
    source: Record<string, unknown>,
    key: string,
    fallback: string,
  ): string {
    const v = source[key];
    return typeof v === 'string' ? v : fallback;
  }

  private getBoolean(
    source: Record<string, unknown>,
    key: string,
    fallback: boolean,
  ): boolean {
    const v = source[key];
    return typeof v === 'boolean' ? v : fallback;
  }

  private getFilterArray(key: string): string[] | null {
    const v = this.filter[key];
    if (v === undefined || v === null || v === 'all') return null;
    if (!Array.isArray(v)) return null;
    return (v as unknown[]).map(String);
  }

  private setFilterArray(key: string, selected: Array<string | number>): void {
    // If the user has every option checked, persist as "all" so the
    // renderer can short-circuit and the saved state stays compact.
    // The caller decides what "all" means by passing the full list.
    if (selected.length === 0) {
      this.filter[key] = [] as unknown[];
    } else {
      this.filter[key] = selected;
    }
    this.opts.onFilterChange(this.filter);
  }

  // --------------------------------------------------------------------
  // Positioning + dismiss
  // --------------------------------------------------------------------

  private position(): void {
    const rect = this.opts.anchor.getBoundingClientRect();
    this.root.style.position = 'fixed';
    this.root.style.top = `${rect.bottom + 4}px`;
    // Anchor to the right edge of the gear so the panel grows leftward
    // and doesn't fall off the right edge of the viewport.
    const right = Math.max(8, window.innerWidth - rect.right);
    this.root.style.right = `${right}px`;
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

// ----------------------------------------------------------------------
// Shared row builders (pure functions — no `this` capture).
// ----------------------------------------------------------------------

function makeSection(title: string): HTMLElement {
  const section = document.createElement('div');
  section.className = 'settings-section';
  const heading = document.createElement('div');
  heading.className = 'settings-section-title';
  heading.textContent = title;
  section.appendChild(heading);
  return section;
}

function numberRow(
  label: string,
  initial: number,
  bounds: { min: number; max: number; step: number },
  onChange: (value: number) => void,
): HTMLElement {
  const row = document.createElement('div');
  row.className = 'settings-row';
  const labelEl = document.createElement('span');
  labelEl.className = 'settings-row-label';
  labelEl.textContent = label;
  const input = document.createElement('input');
  input.type = 'number';
  input.min = String(bounds.min);
  input.max = String(bounds.max);
  input.step = String(bounds.step);
  input.value = String(initial);
  input.addEventListener('change', () => {
    const v = Number(input.value);
    if (Number.isFinite(v)) onChange(v);
  });
  row.appendChild(labelEl);
  row.appendChild(input);
  return row;
}

function textRow(
  label: string,
  initial: string,
  onChange: (value: string) => void,
): HTMLElement {
  const row = document.createElement('div');
  row.className = 'settings-row';
  const labelEl = document.createElement('span');
  labelEl.className = 'settings-row-label';
  labelEl.textContent = label;
  const input = document.createElement('input');
  input.type = 'text';
  input.value = initial;
  input.placeholder = '(inherit)';
  input.addEventListener('change', () => onChange(input.value.trim()));
  row.appendChild(labelEl);
  row.appendChild(input);
  return row;
}

function selectRow(
  label: string,
  initial: string,
  options: Array<{ value: string; label: string }>,
  onChange: (value: string) => void,
): HTMLElement {
  const row = document.createElement('div');
  row.className = 'settings-row';
  const labelEl = document.createElement('span');
  labelEl.className = 'settings-row-label';
  labelEl.textContent = label;
  const select = document.createElement('select');
  for (const o of options) {
    const opt = document.createElement('option');
    opt.value = o.value;
    opt.textContent = o.label;
    if (o.value === initial) opt.selected = true;
    select.appendChild(opt);
  }
  select.addEventListener('change', () => onChange(select.value));
  row.appendChild(labelEl);
  row.appendChild(select);
  return row;
}

function checkboxRow(
  label: string,
  initial: boolean,
  onChange: (value: boolean) => void,
): HTMLElement {
  const row = document.createElement('label');
  row.className = 'settings-checkbox-row';
  const cb = document.createElement('input');
  cb.type = 'checkbox';
  cb.checked = initial;
  cb.addEventListener('change', () => onChange(cb.checked));
  const span = document.createElement('span');
  span.textContent = label;
  row.appendChild(cb);
  row.appendChild(span);
  return row;
}

function paletteRow(
  label: string,
  initial: string,
  onChange: (hex: string) => void,
): HTMLElement {
  const row = document.createElement('div');
  row.className = 'settings-row';
  const labelEl = document.createElement('span');
  labelEl.className = 'settings-row-label';
  labelEl.textContent = label;
  const input = document.createElement('input');
  input.type = 'color';
  input.value = normalizeHexForInput(initial);
  input.addEventListener('input', () => onChange(input.value));
  row.appendChild(labelEl);
  row.appendChild(input);
  return row;
}

function paletteSection(
  keys: readonly string[],
  defaults: Record<string, string>,
  setPalette: (key: string, hex: string) => void,
  getPalette: (key: string) => string | undefined,
): HTMLElement[] {
  return keys.map((key) =>
    paletteRow(
      key,
      getPalette(key) ?? defaults[key] ?? '#888888',
      (hex) => setPalette(key, hex),
    ),
  );
}

function allowListRow(
  label: string,
  options: string[],
  current: string[] | null,
  onChange: (selected: string[]) => void,
  labelFor?: (key: string) => string,
): HTMLElement {
  // `current === null` means "all" — render every checkbox checked.
  const wrap = document.createElement('div');
  const title = document.createElement('div');
  title.className = 'settings-row-label';
  title.textContent = label;
  wrap.appendChild(title);

  const selected = new Set<string>(current ?? options);
  for (const opt of options) {
    const row = document.createElement('label');
    row.className = 'settings-checkbox-row';
    const cb = document.createElement('input');
    cb.type = 'checkbox';
    cb.checked = selected.has(opt);
    cb.addEventListener('change', () => {
      if (cb.checked) selected.add(opt);
      else selected.delete(opt);
      // If the user has every option ticked, treat as "all" by emitting
      // the full list — the host stores it as-is and the renderer's
      // pickAllowList treats every-allowed identically.
      onChange(Array.from(selected));
    });
    const span = document.createElement('span');
    span.textContent = labelFor ? labelFor(opt) : opt;
    row.appendChild(cb);
    row.appendChild(span);
    wrap.appendChild(row);
  }
  return wrap;
}

function emptyHint(text: string): HTMLElement {
  const el = document.createElement('div');
  el.className = 'settings-empty';
  el.textContent = text;
  return el;
}

function numericList(source: unknown): number[] {
  if (!Array.isArray(source)) return [];
  const out: number[] = [];
  for (const v of source as unknown[]) {
    const n = Number(v);
    if (Number.isFinite(n)) out.push(n);
  }
  return out;
}

function stringList(source: unknown): string[] {
  if (!Array.isArray(source)) return [];
  const out: string[] = [];
  for (const v of source as unknown[]) {
    if (v === null || v === undefined) continue;
    out.push(String(v));
  }
  return out;
}

function defaultOpacity(kind: string): number {
  if (kind === 'gene_annotation') return 0.85;
  if (kind === 'splice_junctions') return 0.85;
  if (kind === 'coverage_histogram') return 0.4;
  return 1.0;
}

/** <input type="color"> only accepts 7-char #rrggbb. Coerce 3-char or
 *  named-fallback values defensively. */
function normalizeHexForInput(value: string): string {
  if (/^#[0-9a-fA-F]{6}$/.test(value)) return value;
  if (/^#[0-9a-fA-F]{3}$/.test(value)) {
    const r = value[1];
    const g = value[2];
    const b = value[3];
    return `#${r}${r}${g}${g}${b}${b}`;
  }
  return '#888888';
}
