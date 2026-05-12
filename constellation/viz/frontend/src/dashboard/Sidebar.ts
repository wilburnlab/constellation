// Sidebar — Common ↔ All mode toggle + search + command tree.

import type { CliSchema, CommandSchema, CuratedEntry } from './types';
import { DashboardState, getStored, setStored, SidebarMode } from './state';

export interface SidebarOptions {
  schema: CliSchema;
  state: DashboardState;
}

interface LeafEntry {
  // Leaf nodes are commands with no further subcommands (or that have
  // arguments — top-level subcommand groups like `transcriptome` aren't
  // runnable themselves).
  command: CommandSchema;
  group?: string;
  label?: string;
  hint?: string;
}

export class Sidebar {
  private readonly schema: CliSchema;
  private readonly state: DashboardState;
  private mode: SidebarMode;
  private searchTerm = '';
  private container: HTMLElement | null = null;
  private treeEl: HTMLElement | null = null;
  private activePath: string | null = null;

  constructor(opts: SidebarOptions) {
    this.schema = opts.schema;
    this.state = opts.state;
    this.mode = getStored<SidebarMode>('sidebar_mode', 'common');
  }

  mount(element: HTMLElement): void {
    this.container = element;
    element.classList.add('sidebar');
    element.innerHTML = '';

    const toolbar = document.createElement('div');
    toolbar.className = 'sidebar-toolbar';

    const modeRow = document.createElement('div');
    modeRow.className = 'sidebar-mode';
    const commonBtn = document.createElement('button');
    commonBtn.type = 'button';
    commonBtn.textContent = 'Common';
    commonBtn.dataset.mode = 'common';
    const allBtn = document.createElement('button');
    allBtn.type = 'button';
    allBtn.textContent = 'All commands';
    allBtn.dataset.mode = 'all';
    modeRow.append(commonBtn, allBtn);
    commonBtn.addEventListener('click', () => this.setMode('common'));
    allBtn.addEventListener('click', () => this.setMode('all'));
    toolbar.appendChild(modeRow);

    const search = document.createElement('input');
    search.className = 'sidebar-search';
    search.type = 'search';
    search.placeholder = 'Search commands…';
    search.addEventListener('input', () => {
      this.searchTerm = search.value.trim().toLowerCase();
      this.renderTree();
    });
    toolbar.appendChild(search);

    element.appendChild(toolbar);

    const tree = document.createElement('div');
    tree.className = 'sidebar-tree';
    this.treeEl = tree;
    element.appendChild(tree);

    this.applyModeToButtons();
    this.renderTree();
  }

  private setMode(mode: SidebarMode): void {
    if (mode === this.mode) return;
    this.mode = mode;
    setStored('sidebar_mode', mode);
    this.applyModeToButtons();
    this.renderTree();
  }

  private applyModeToButtons(): void {
    if (!this.container) return;
    for (const btn of this.container.querySelectorAll<HTMLButtonElement>(
      '.sidebar-mode button',
    )) {
      btn.classList.toggle('active', btn.dataset.mode === this.mode);
    }
  }

  setActivePath(path: string[] | null): void {
    this.activePath = path ? path.join(' ') : null;
    if (!this.treeEl) return;
    for (const el of this.treeEl.querySelectorAll<HTMLElement>(
      '.sidebar-entry',
    )) {
      el.classList.toggle(
        'active',
        el.dataset.path === this.activePath,
      );
    }
  }

  private renderTree(): void {
    if (!this.treeEl) return;
    this.treeEl.innerHTML = '';
    const entries =
      this.mode === 'common'
        ? this.commonEntries()
        : this.allEntries();
    const filtered = this.filterEntries(entries);
    this.renderGroups(filtered);
  }

  private filterEntries(entries: LeafEntry[]): LeafEntry[] {
    if (!this.searchTerm) return entries;
    const needle = this.searchTerm;
    return entries.filter((e) => {
      const haystack = [
        e.label ?? '',
        e.command.path.join(' '),
        e.command.help ?? '',
        e.hint ?? '',
      ]
        .join(' ')
        .toLowerCase();
      return haystack.includes(needle);
    });
  }

  private renderGroups(entries: LeafEntry[]): void {
    if (!this.treeEl) return;
    if (entries.length === 0) {
      const empty = document.createElement('div');
      empty.className = 'sidebar-group';
      empty.textContent =
        this.searchTerm.length > 0
          ? 'No matches'
          : 'No commands available';
      this.treeEl.appendChild(empty);
      return;
    }
    const byGroup = new Map<string, LeafEntry[]>();
    for (const entry of entries) {
      const group = entry.group ?? 'Other';
      if (!byGroup.has(group)) byGroup.set(group, []);
      byGroup.get(group)!.push(entry);
    }
    for (const [group, items] of byGroup) {
      const header = document.createElement('div');
      header.className = 'sidebar-group';
      header.textContent = group;
      this.treeEl.appendChild(header);
      for (const entry of items) {
        this.treeEl.appendChild(this.entryNode(entry));
      }
    }
  }

  private entryNode(entry: LeafEntry): HTMLElement {
    const node = document.createElement('div');
    node.className = 'sidebar-entry';
    const pathKey = entry.command.path.join(' ');
    node.dataset.path = pathKey;
    if (pathKey === this.activePath) {
      node.classList.add('active');
    }

    const label = document.createElement('div');
    label.className = 'sidebar-entry-label';
    label.textContent = entry.label ?? entry.command.name;
    node.appendChild(label);

    const subPath = document.createElement('div');
    subPath.className = 'sidebar-entry-path';
    subPath.textContent = 'constellation ' + entry.command.path.join(' ');
    node.appendChild(subPath);

    const hintText = entry.hint ?? entry.command.help ?? '';
    if (hintText) {
      const hint = document.createElement('div');
      hint.className = 'sidebar-entry-hint';
      hint.textContent = hintText;
      node.appendChild(hint);
    }

    node.addEventListener('click', () => {
      this.state.emit('command:open', entry.command);
    });
    return node;
  }

  // -------------------------------------------------------------------
  // Tree → leaf-list flattening
  // -------------------------------------------------------------------

  private allEntries(): LeafEntry[] {
    const out: LeafEntry[] = [];
    const walk = (cmd: CommandSchema, groupPrefix: string): void => {
      if (cmd.subcommands.length === 0) {
        out.push({ command: cmd, group: groupPrefix || 'Commands' });
        return;
      }
      const group = cmd.path.length > 0 ? cmd.path[0] : groupPrefix;
      for (const child of cmd.subcommands) {
        walk(child, capitalize(group));
      }
    };
    for (const child of this.schema.subcommands) {
      walk(child, '');
    }
    return out;
  }

  private commonEntries(): LeafEntry[] {
    const out: LeafEntry[] = [];
    for (const entry of this.schema.curated) {
      const command = this.resolveCommand(entry.path);
      if (!command) continue; // curated path no longer exists — silently skip
      out.push({
        command,
        group: entry.group ?? 'Common',
        label: entry.label,
        hint: entry.hint,
      });
    }
    return out;
  }

  private resolveCommand(path: string[]): CommandSchema | null {
    let nodes: CommandSchema[] = this.schema.subcommands;
    let match: CommandSchema | null = null;
    for (const segment of path) {
      const next = nodes.find((c) => c.name === segment);
      if (!next) return null;
      match = next;
      nodes = next.subcommands;
    }
    return match;
  }
}

function capitalize(s: string): string {
  if (!s) return s;
  return s[0].toUpperCase() + s.slice(1);
}

// Convenience curated-entry type re-export so callers don't import twice.
export type { CuratedEntry };
