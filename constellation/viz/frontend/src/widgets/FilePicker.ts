// FilePicker — a directory/file browser popover over GET /api/fs/list.
//
// Anchored to a "Browse…" button (see PathInput), it lets the user
// navigate the sandboxed filesystem and pick a directory or a file
// instead of typing a full path. Follows the DatasetManagerPopover
// house style: a plain ES class that builds its DOM imperatively,
// mounts under a host element, positions itself fixed relative to the
// anchor, and dismisses on outside-click / Escape.
//
// Selection paths by mode:
//   dir  — "Select this folder" (current dir) or a typed new-subfolder
//          name (current/<name>, NOT created on disk — the CLI creates
//          output dirs; the viz server stays read-only).
//   file — click a file row.
// In both modes a "Go to" box navigates to a pasted path (Windows
// C:\… paths are normalized server-side under WSL).

import './FilePicker.css';

export type PathKindMode = 'dir' | 'file' | 'either';

function modeTitle(kind: PathKindMode): string {
  if (kind === 'dir') return 'Select a folder';
  if (kind === 'file') return 'Select a file';
  return 'Select a file or folder';
}

export interface FilePickerOptions {
  anchor: HTMLElement;
  kind: PathKindMode;
  /** Seed location — the current field value. Falls back to its parent,
   *  then the first sandbox root, if it can't be listed. */
  initialPath?: string;
  /** File filters (file mode only), e.g. ['*.tsv', '*.bed']. */
  globs?: string[];
  onSelect(path: string): void;
  onClose(): void;
}

interface FsEntry {
  name: string;
  path: string;
  is_dir: boolean;
  size: number | null;
  mtime: number | null;
}

interface FsRoot {
  label: string;
  path: string;
}

interface FsListing {
  path: string;
  parent: string | null;
  is_root: boolean;
  roots: FsRoot[];
  entries: FsEntry[];
  truncated: boolean;
}

export class FilePicker {
  private readonly opts: FilePickerOptions;
  private readonly root: HTMLElement;
  private rootsSelect: HTMLSelectElement | null = null;
  private breadcrumb: HTMLElement | null = null;
  private listEl: HTMLElement | null = null;
  private errorEl: HTMLElement | null = null;
  private gotoInput: HTMLInputElement | null = null;

  private current = '';
  private roots: FsRoot[] = [];
  private outsideHandler: ((e: MouseEvent) => void) | null = null;
  private escHandler: ((e: KeyboardEvent) => void) | null = null;

  constructor(opts: FilePickerOptions) {
    this.opts = opts;
    this.root = document.createElement('div');
    this.root.className = 'file-picker-popover';
    this.root.setAttribute('role', 'dialog');
    this.root.setAttribute('aria-label', modeTitle(opts.kind));
    this.build();
    this.position();
    this.attachDismiss();
  }

  /** Append the popover under `parent`. Mounting inside the anchor's own
   *  popover (rather than document.body) keeps a host popover's
   *  outside-click dismiss from firing while the picker is open —
   *  position:fixed means DOM parentage doesn't affect layout. */
  mount(parent: HTMLElement): void {
    parent.appendChild(this.root);
    void this.initialLoad();
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

  // -------------------------------------------------------------------
  // DOM
  // -------------------------------------------------------------------

  private build(): void {
    const header = document.createElement('div');
    header.className = 'fp-header';
    const title = document.createElement('span');
    title.className = 'fp-title';
    title.textContent = modeTitle(this.opts.kind);
    const close = document.createElement('button');
    close.type = 'button';
    close.className = 'fp-close';
    close.textContent = '✕';
    close.title = 'Close';
    close.addEventListener('click', () => this.opts.onClose());
    header.appendChild(title);
    header.appendChild(close);
    this.root.appendChild(header);

    const rootsRow = document.createElement('div');
    rootsRow.className = 'fp-roots-row';
    const rootsLabel = document.createElement('span');
    rootsLabel.className = 'fp-roots-label';
    rootsLabel.textContent = 'Root:';
    const sel = document.createElement('select');
    sel.className = 'fp-roots-select';
    sel.addEventListener('change', () => {
      if (sel.value) void this.load(sel.value);
    });
    rootsRow.appendChild(rootsLabel);
    rootsRow.appendChild(sel);
    this.root.appendChild(rootsRow);
    this.rootsSelect = sel;

    const crumb = document.createElement('div');
    crumb.className = 'fp-breadcrumb';
    this.root.appendChild(crumb);
    this.breadcrumb = crumb;

    const list = document.createElement('div');
    list.className = 'fp-list';
    this.root.appendChild(list);
    this.listEl = list;

    const err = document.createElement('div');
    err.className = 'fp-error';
    err.hidden = true;
    this.root.appendChild(err);
    this.errorEl = err;

    const footer = document.createElement('div');
    footer.className = 'fp-footer';

    // "Go to" navigation box (both modes).
    const gotoRow = document.createElement('div');
    gotoRow.className = 'fp-goto-row';
    const gotoInput = document.createElement('input');
    gotoInput.type = 'text';
    gotoInput.className = 'fp-goto-input';
    gotoInput.placeholder = 'Go to path… (paste to jump)';
    gotoInput.spellcheck = false;
    gotoInput.autocomplete = 'off';
    gotoInput.addEventListener('keydown', (e) => {
      if (e.key === 'Enter') {
        e.preventDefault();
        const v = gotoInput.value.trim();
        if (v) void this.load(v);
      }
    });
    gotoRow.appendChild(gotoInput);
    footer.appendChild(gotoRow);
    this.gotoInput = gotoInput;

    const showFolderSelect =
      this.opts.kind === 'dir' || this.opts.kind === 'either';
    const showNewFolder = this.opts.kind === 'dir';

    if (showFolderSelect) {
      const selectRow = document.createElement('div');
      selectRow.className = 'fp-select-row';
      const selectBtn = document.createElement('button');
      selectBtn.type = 'button';
      selectBtn.className = 'fp-select-btn';
      selectBtn.textContent = 'Select this folder';
      selectBtn.addEventListener('click', () => {
        if (this.current) this.select(this.current);
      });
      selectRow.appendChild(selectBtn);
      footer.appendChild(selectRow);
    }

    if (showNewFolder) {
      const newRow = document.createElement('div');
      newRow.className = 'fp-newfolder-row';
      const newLabel = document.createElement('span');
      newLabel.className = 'fp-newfolder-label';
      newLabel.textContent = 'New subfolder:';
      const newInput = document.createElement('input');
      newInput.type = 'text';
      newInput.className = 'fp-newfolder-input';
      newInput.placeholder = 'name';
      newInput.spellcheck = false;
      newInput.autocomplete = 'off';
      const useBtn = document.createElement('button');
      useBtn.type = 'button';
      useBtn.className = 'fp-newfolder-btn';
      useBtn.textContent = 'Use';
      const useNewFolder = (): void => {
        const name = newInput.value.trim();
        if (!name || !this.current) return;
        // A subfolder is a single path segment — reject separators and
        // dot-segments so a name can't climb out of the current dir.
        if (name === '.' || name === '..' || /[\\/]/.test(name)) {
          this.showError('Subfolder name must be a single name (no "/" or "..").');
          return;
        }
        this.select(joinPath(this.current, name));
      };
      useBtn.addEventListener('click', useNewFolder);
      newInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter') {
          e.preventDefault();
          useNewFolder();
        }
      });
      newRow.appendChild(newLabel);
      newRow.appendChild(newInput);
      newRow.appendChild(useBtn);
      footer.appendChild(newRow);
    }

    if (!showNewFolder) {
      const hint = document.createElement('div');
      hint.className = 'fp-hint';
      hint.textContent =
        this.opts.kind === 'either'
          ? 'Click a file, or select the current folder.'
          : 'Click a file to select it.';
      footer.appendChild(hint);
    }

    this.root.appendChild(footer);
  }

  // -------------------------------------------------------------------
  // Data
  // -------------------------------------------------------------------

  private async initialLoad(): Promise<void> {
    const seed = this.opts.initialPath?.trim();
    if (seed) {
      if (await this.tryLoad(seed)) return;
      // Fall back to the parent — handles a not-yet-existing output dir
      // or a full file path (e.g. a seeded `C:\Users\me\data.bam` in
      // file mode lands the user in `C:\Users\me`; the server normalizes
      // the Windows drive path on load).
      const parent = parentPath(seed);
      if (parent && parent !== seed && (await this.tryLoad(parent))) return;
    }
    await this.tryLoad(undefined);
  }

  /** Load without surfacing errors; returns whether it succeeded. */
  private async tryLoad(path: string | undefined): Promise<boolean> {
    try {
      await this.load(path);
      return true;
    } catch {
      return false;
    }
  }

  private async load(path: string | undefined): Promise<void> {
    if (!this.errorEl) return;
    try {
      const listing = await fetchFsListing({
        path,
        globs: this.opts.kind !== 'dir' ? this.opts.globs : undefined,
      });
      this.current = listing.path;
      this.roots = listing.roots;
      this.errorEl.hidden = true;
      this.renderRoots();
      this.renderBreadcrumb();
      this.renderEntries(listing);
      if (this.gotoInput) this.gotoInput.value = listing.path;
    } catch (err) {
      this.showError((err as Error).message);
      throw err;
    }
  }

  private renderRoots(): void {
    if (!this.rootsSelect) return;
    this.rootsSelect.replaceChildren();
    const containing = containingRoot(this.current, this.roots);
    for (const r of this.roots) {
      const opt = document.createElement('option');
      opt.value = r.path;
      opt.textContent = r.label;
      if (containing && r.path === containing.path) opt.selected = true;
      this.rootsSelect.appendChild(opt);
    }
  }

  private renderBreadcrumb(): void {
    if (!this.breadcrumb) return;
    this.breadcrumb.replaceChildren();
    const rootMatch = containingRoot(this.current, this.roots);
    const base = rootMatch ?? { label: this.current, path: this.current };
    const rel = this.current
      .slice(base.path.length)
      .split('/')
      .filter(Boolean);
    const segments: FsRoot[] = [{ label: base.label, path: base.path }];
    let acc = base.path;
    for (const seg of rel) {
      acc = joinPath(acc, seg);
      segments.push({ label: seg, path: acc });
    }
    segments.forEach((seg, i) => {
      if (i > 0) {
        const sep = document.createElement('span');
        sep.className = 'fp-crumb-sep';
        sep.textContent = '/';
        this.breadcrumb!.appendChild(sep);
      }
      const crumb = document.createElement('button');
      crumb.type = 'button';
      crumb.className = 'fp-crumb';
      crumb.textContent = seg.label;
      crumb.addEventListener('click', () => void this.load(seg.path));
      this.breadcrumb!.appendChild(crumb);
    });
  }

  private renderEntries(listing: FsListing): void {
    if (!this.listEl) return;
    this.listEl.replaceChildren();
    if (listing.entries.length === 0) {
      const empty = document.createElement('div');
      empty.className = 'fp-empty';
      empty.textContent = '(empty)';
      this.listEl.appendChild(empty);
      return;
    }
    for (const entry of listing.entries) {
      const row = document.createElement('div');
      row.className = 'fp-entry';
      const icon = document.createElement('span');
      icon.className = 'fp-icon';
      icon.textContent = entry.is_dir ? '📁' : '📄';
      const name = document.createElement('span');
      name.className = 'fp-name';
      name.textContent = entry.name;
      row.appendChild(icon);
      row.appendChild(name);

      if (entry.is_dir) {
        row.classList.add('fp-dir');
        row.addEventListener('click', () => void this.load(entry.path));
      } else if (this.opts.kind === 'file' || this.opts.kind === 'either') {
        row.classList.add('fp-file');
        row.addEventListener('click', () => this.select(entry.path));
      } else {
        // dir mode — files shown for context, not selectable.
        row.classList.add('fp-file', 'fp-disabled');
      }
      this.listEl.appendChild(row);
    }
    if (listing.truncated) {
      const more = document.createElement('div');
      more.className = 'fp-empty';
      more.textContent = '… (listing truncated)';
      this.listEl.appendChild(more);
    }
  }

  private select(path: string): void {
    this.opts.onSelect(path);
  }

  private showError(message: string): void {
    if (!this.errorEl) return;
    this.errorEl.textContent = message;
    this.errorEl.hidden = false;
  }

  private position(): void {
    const rect = this.opts.anchor.getBoundingClientRect();
    this.root.style.position = 'fixed';
    // Prefer opening below-left of the anchor; clamp into the viewport.
    const top = Math.min(rect.bottom + 4, window.innerHeight - 380);
    const left = Math.min(rect.left, window.innerWidth - 380);
    this.root.style.top = `${Math.max(8, top)}px`;
    this.root.style.left = `${Math.max(8, left)}px`;
    this.root.style.zIndex = '40';
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

// ---------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------

async function fetchFsListing(params: {
  path?: string;
  globs?: string[];
}): Promise<FsListing> {
  const qs = new URLSearchParams();
  if (params.path) qs.set('path', params.path);
  if (params.globs && params.globs.length > 0) {
    qs.set('globs', params.globs.join(','));
  }
  const resp = await fetch(`/api/fs/list?${qs.toString()}`);
  if (!resp.ok) {
    let detail = `HTTP ${resp.status}`;
    try {
      const body = (await resp.json()) as { detail?: unknown };
      if (typeof body.detail === 'string') detail = body.detail;
    } catch {
      /* not JSON */
    }
    throw new Error(detail);
  }
  return (await resp.json()) as FsListing;
}

/** The longest sandbox root that contains `path` (or null). */
function containingRoot(path: string, roots: FsRoot[]): FsRoot | null {
  let best: FsRoot | null = null;
  for (const r of roots) {
    if (path === r.path || path.startsWith(r.path.replace(/\/$/, '') + '/')) {
      if (best === null || r.path.length > best.path.length) best = r;
    }
  }
  return best;
}

function joinPath(base: string, name: string): string {
  return base.endsWith('/') ? base + name : `${base}/${name}`;
}

/** The parent of a path, handling both POSIX (`/`) and Windows (`\`)
 *  separators. Returns null when there is no separator to strip. */
function parentPath(p: string): string | null {
  const trimmed = p.replace(/[\\/]+$/, '');
  const idx = Math.max(trimmed.lastIndexOf('/'), trimmed.lastIndexOf('\\'));
  if (idx < 0) return null;
  const parent = trimmed.slice(0, idx);
  if (parent === '') return '/';
  // Bare drive root ("C:") → "C:\" so the server resolves it correctly.
  if (/^[A-Za-z]:$/.test(parent)) return `${parent}\\`;
  return parent;
}
