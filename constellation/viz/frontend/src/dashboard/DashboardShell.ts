// Top-level dashboard shell — wires dockview, sidebar, and status bar.
//
// Owns the DockviewComponent instance and the lifecycle of every panel.
// On `command:open` from the sidebar: focus the matching form panel or
// add a new one. On `job:started` from a CommandForm: open a Terminal
// panel positioned right of the form.
//
// Layout persistence: serialize the dockview state to localStorage on
// every layout change, deserialize on next boot.

import {
  DockviewComponent,
  type CreateComponentOptions,
  type GroupPanelPartInitParameters,
  type IContentRenderer,
  type SerializedDockview,
} from 'dockview-core';

import 'dockview-core/dist/styles/dockview.css';

import { CommandForm } from './CommandForm';
import { Sidebar } from './Sidebar';
import { StatusBar } from './StatusBar';
import { TerminalPanel } from './Terminal';
import { DashboardState, getStored, setStored } from './state';
import type { CliSchema, CommandSchema } from './types';

const LAYOUT_KEY = 'layout.v1';
const LAST_SESSION_PATH_KEY = 'last_session_path';

export interface DashboardShellOptions {
  schema: CliSchema;
  dockRoot: HTMLElement;
  statusRoot: HTMLElement;
}

interface PanelParams extends Record<string, unknown> {
  kind: 'sidebar' | 'welcome' | 'form' | 'terminal' | 'genome';
  command?: CommandSchema;
  jobId?: string;
  argv?: string[];
  sessionId?: string;
  label?: string;
}

interface SessionSummary {
  session_id: string;
  label: string;
  root: string;
  stages_present: Record<string, boolean>;
}

// Renderer that delegates pane resizes to an inner TerminalPanel.
// Used by the active-panel-change handler to refit xterm.
interface ResizeAware extends IContentRenderer {
  onResize(): void;
}

export class DashboardShell {
  private readonly schema: CliSchema;
  private readonly state = new DashboardState();
  private readonly sidebar: Sidebar;
  private readonly statusBar: StatusBar;
  private dock: DockviewComponent | null = null;
  // Form panels keyed by command path so reopens focus the same panel.
  private readonly formPanels = new Map<string, string>();

  constructor(opts: DashboardShellOptions) {
    this.schema = opts.schema;
    this.sidebar = new Sidebar({ schema: opts.schema, state: this.state });
    this.statusBar = new StatusBar(opts.statusRoot, this.state);
  }

  mount(dockRoot: HTMLElement): void {
    this.dock = new DockviewComponent(dockRoot, {
      createComponent: (opts) => this.createRenderer(opts),
    });

    // Wire shell-level event handlers BEFORE the first panel is added
    // so layout restoration triggers don't fire into a dead listener.
    this.state.on('command:open', (cmd) => this.openCommandPanel(cmd));
    this.state.on('job:started', ({ jobId, argv }) =>
      this.openTerminalPanel(jobId, argv),
    );

    // Initial layout: try to restore, otherwise build defaults.
    const stored = getStored<SerializedDockview | null>(LAYOUT_KEY, null);
    let restored = false;
    if (stored !== null) {
      try {
        this.dock.fromJSON(stored);
        restored = true;
      } catch {
        // Stale or incompatible layout — fall through to defaults.
        restored = false;
      }
    }
    if (!restored) {
      this.buildDefaultLayout();
    }

    // Persist on every layout change.
    this.dock.onDidLayoutChange(() => {
      if (this.dock) setStored(LAYOUT_KEY, this.dock.toJSON());
    });

    // Refit terminals when their pane becomes active (sizes finalize
    // after the dock layout settles).
    this.dock.onDidActivePanelChange((panel) => {
      const renderer = panel?.view?.content as ResizeAware | undefined;
      renderer?.onResize?.();
    });

    this.statusBar.start();
  }

  destroy(): void {
    this.statusBar.stop();
    this.dock?.dispose();
    this.dock = null;
  }

  // -------------------------------------------------------------------
  // dockview-core renderer factory
  //
  // dockview-core 3.x asks for a single `createComponent(opts)` factory
  // that returns an IContentRenderer per panel. We switch on the
  // declared component name to instantiate the right inner widget.
  // -------------------------------------------------------------------

  private createRenderer(opts: CreateComponentOptions): IContentRenderer {
    const sidebarRef = this.sidebar;
    const stateRef = this.state;
    const shell = this;

    switch (opts.name) {
      case 'sidebar':
        return new (class implements IContentRenderer {
          readonly element = document.createElement('div');
          init(_params: GroupPanelPartInitParameters): void {
            sidebarRef.mount(this.element);
          }
        })();

      case 'welcome':
        return new (class implements IContentRenderer {
          readonly element = (() => {
            const e = document.createElement('div');
            e.className = 'welcome';
            return e;
          })();
          init(_params: GroupPanelPartInitParameters): void {
            this.element.innerHTML = `
              <h1>Constellation</h1>
              <p>Pick a command from the sidebar to fill in a form,
              then press Run. Each running job streams into its own
              terminal panel. Drag panels by their tab to split or
              stack them.</p>
              <div class="session-launcher">
                <h2>Open a session</h2>
                <p>Point at a pipeline run directory (the parent of
                <code>genome/</code>, <code>S2_align/</code>, ...).</p>
                <div class="session-launcher-row">
                  <input class="session-launcher-input"
                    type="text"
                    placeholder="/path/to/run"
                    spellcheck="false"
                    autocomplete="off" />
                  <button class="session-launcher-button"
                    type="button">Open Genome Browser</button>
                </div>
                <div class="session-launcher-error" hidden></div>
              </div>
            `;
            const input = this.element.querySelector<HTMLInputElement>(
              '.session-launcher-input',
            )!;
            const button = this.element.querySelector<HTMLButtonElement>(
              '.session-launcher-button',
            )!;
            const error = this.element.querySelector<HTMLDivElement>(
              '.session-launcher-error',
            )!;
            input.value = getStored<string>(LAST_SESSION_PATH_KEY, '');
            const showError = (msg: string): void => {
              error.textContent = msg;
              error.hidden = false;
            };
            const clearError = (): void => {
              error.textContent = '';
              error.hidden = true;
            };
            const submit = async (): Promise<void> => {
              const path = input.value.trim();
              if (!path) {
                showError('Enter a session directory path.');
                return;
              }
              clearError();
              button.disabled = true;
              try {
                const summary = await shell.registerSession(path);
                setStored(LAST_SESSION_PATH_KEY, path);
                shell.openGenomePanel(summary.session_id, summary.label);
              } catch (err) {
                showError(err instanceof Error ? err.message : String(err));
              } finally {
                button.disabled = false;
              }
            };
            button.addEventListener('click', () => {
              void submit();
            });
            input.addEventListener('keydown', (ev) => {
              if (ev.key === 'Enter') {
                ev.preventDefault();
                void submit();
              }
            });
          }
        })();

      case 'form':
        return new (class implements IContentRenderer {
          readonly element = (() => {
            const e = document.createElement('div');
            e.style.height = '100%';
            return e;
          })();
          private form: CommandForm | null = null;
          init(params: GroupPanelPartInitParameters): void {
            const p = params.params as PanelParams | undefined;
            let cmd = p?.command;
            if (!cmd) {
              const path = (params.api.id ?? '')
                .replace(/^form:/, '')
                .split(' ');
              cmd = path[0] ? shell.findCommand(path) ?? undefined : undefined;
            }
            if (!cmd) {
              this.element.textContent =
                '(form unavailable — command no longer exists)';
              return;
            }
            this.form = new CommandForm({ command: cmd, state: stateRef });
            this.form.mount(this.element);
          }
        })();

      case 'terminal':
        return new (class implements ResizeAware {
          readonly element = (() => {
            const e = document.createElement('div');
            e.style.height = '100%';
            return e;
          })();
          private term: TerminalPanel | null = null;
          init(params: GroupPanelPartInitParameters): void {
            const p = params.params as PanelParams | undefined;
            if (!p?.jobId || !p?.argv) {
              this.element.textContent =
                '(terminal unavailable — job state was not restored)';
              return;
            }
            this.term = new TerminalPanel({
              jobId: p.jobId,
              argv: p.argv,
            });
            this.term.mount(this.element);
          }
          onResize(): void {
            this.term?.resize();
          }
        })();

      case 'genome':
        return new (class implements IContentRenderer {
          readonly element = (() => {
            const e = document.createElement('div');
            e.className = 'genome-panel';
            return e;
          })();
          init(params: GroupPanelPartInitParameters): void {
            const p = params.params as PanelParams | undefined;
            const sessionId = p?.sessionId;
            if (!sessionId) {
              this.element.textContent =
                '(genome panel unavailable — no session bound)';
              return;
            }
            // Lazy-import keeps the d3 / apache-arrow / track-renderer
            // chain out of the initial dashboard payload — only paid
            // when a genome panel actually opens.
            void import('../widgets/GenomeBrowser')
              .then(async ({ GenomeBrowser }) => {
                const browser = new GenomeBrowser({
                  host: this.element,
                  sessionId,
                });
                await browser.mount();
              })
              .catch((err: unknown) => {
                this.element.textContent = `Failed to mount genome browser: ${
                  err instanceof Error ? err.message : String(err)
                }`;
              });
          }
        })();

      default:
        // Fallback for unknown panel names — show an error rather than
        // crashing the whole shell.
        return new (class implements IContentRenderer {
          readonly element = (() => {
            const e = document.createElement('div');
            e.style.padding = '24px';
            e.style.color = 'var(--danger)';
            e.textContent = `Unknown panel kind: ${opts.name}`;
            return e;
          })();
          init(_params: GroupPanelPartInitParameters): void {}
        })();
    }
  }

  // -------------------------------------------------------------------
  // Default layout + panel management
  // -------------------------------------------------------------------

  private buildDefaultLayout(): void {
    if (!this.dock) return;
    this.dock.addPanel({
      id: 'sidebar',
      component: 'sidebar',
      title: 'Commands',
      params: { kind: 'sidebar' } satisfies PanelParams,
    });
    this.dock.addPanel({
      id: 'welcome',
      component: 'welcome',
      title: 'Welcome',
      params: { kind: 'welcome' } satisfies PanelParams,
      position: { referencePanel: 'sidebar', direction: 'right' },
    });
  }

  private openCommandPanel(cmd: CommandSchema): void {
    if (!this.dock) return;
    const key = cmd.path.join(' ');
    const existingId = this.formPanels.get(key);
    if (existingId) {
      const panel = this.dock.getGroupPanel(existingId);
      if (panel) {
        panel.api.setActive();
        this.sidebar.setActivePath(cmd.path);
        return;
      }
      this.formPanels.delete(key);
    }
    const id = `form:${key}`;
    this.dock.addPanel({
      id,
      component: 'form',
      title: cmd.path.join(' '),
      params: { kind: 'form', command: cmd } satisfies PanelParams,
      position: { referencePanel: 'sidebar', direction: 'right' },
    });
    this.formPanels.set(key, id);
    this.sidebar.setActivePath(cmd.path);
  }

  private openTerminalPanel(jobId: string, argv: string[]): void {
    if (!this.dock) return;
    const short = argv.slice(0, 2).join(' ') || 'job';
    this.dock.addPanel({
      id: `terminal:${jobId}`,
      component: 'terminal',
      title: `terminal · ${short}`,
      params: { kind: 'terminal', jobId, argv } satisfies PanelParams,
    });
  }

  // -------------------------------------------------------------------
  // Session registration + genome panel
  // -------------------------------------------------------------------

  async registerSession(path: string): Promise<SessionSummary> {
    const response = await fetch('/api/sessions', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ path }),
    });
    if (!response.ok) {
      let detail = `HTTP ${response.status}`;
      try {
        const body = (await response.json()) as { detail?: unknown };
        if (typeof body.detail === 'string') detail = body.detail;
      } catch {
        // fall through with the default message
      }
      throw new Error(detail);
    }
    return (await response.json()) as SessionSummary;
  }

  openGenomePanel(sessionId: string, label: string): void {
    if (!this.dock) return;
    const id = `genome:${sessionId}`;
    const existing = this.dock.getGroupPanel(id);
    if (existing) {
      existing.api.setActive();
      return;
    }
    this.dock.addPanel({
      id,
      component: 'genome',
      title: `genome · ${label}`,
      params: { kind: 'genome', sessionId, label } satisfies PanelParams,
      position: { referencePanel: 'sidebar', direction: 'right' },
    });
  }

  // -------------------------------------------------------------------
  // Schema lookup helpers
  // -------------------------------------------------------------------

  private findCommand(path: string[]): CommandSchema | null {
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
