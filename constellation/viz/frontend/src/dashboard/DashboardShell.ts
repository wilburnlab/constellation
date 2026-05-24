// Top-level dashboard shell — wires the persistent left rail (sidebar +
// logo collapse toggle), the dockview workspace, and the status bar.
//
// The sidebar is NOT a dockview panel — it lives in a fixed-width host
// (`#shell-sidebar-body`) so it can't be accidentally closed. Dockview
// owns only the right-hand workspace, where every command (compute or
// visualization) opens in the same panel kind: a `'task'` panel that
// transitions in-place from form → either terminal or visualization
// widget.

import {
  DockviewComponent,
  type CreateComponentOptions,
  type GroupPanelPartInitParameters,
  type IContentRenderer,
  type SerializedDockview,
} from 'dockview-core';

import 'dockview-core/dist/styles/dockview.css';

import { Sidebar } from './Sidebar';
import { StatusBar } from './StatusBar';
import { TaskPanel, taskInitForCommand, type TaskInit } from './TaskPanel';
import { DashboardState, getStored, setStored, type SidebarMode } from './state';
import type { CliSchema, CommandSchema } from './types';

// Bumped from layout.v1 when the sidebar moved out of dockview — a stale
// v1 blob that referenced a 'sidebar' panel would fail the factory.
const LAYOUT_KEY = 'layout.v2';
const SIDEBAR_COLLAPSED_KEY = 'sidebar.collapsed.v1';

export interface DashboardShellOptions {
  schema: CliSchema;
  dockRoot: HTMLElement;
  statusRoot: HTMLElement;
  sidebarBody: HTMLElement;
  sidebarHeader: HTMLElement;
  sidebarRoot: HTMLElement;
  sidebarRail: HTMLElement;
}

interface TaskPanelParams extends Record<string, unknown> {
  kind: 'task';
  // Persistable identity — the shell rebuilds the init on layout
  // restore by walking `schema.subcommands` to find the command.
  commandPath?: string[];
}

interface WelcomePanelParams extends Record<string, unknown> {
  kind: 'welcome';
}

type PanelParams = TaskPanelParams | WelcomePanelParams;

interface TaskAware extends IContentRenderer {
  onResize?: () => void;
  taskInit?: TaskInit;
}

export class DashboardShell {
  private readonly schema: CliSchema;
  private readonly state = new DashboardState();
  private readonly sidebar: Sidebar;
  private readonly statusBar: StatusBar;
  private readonly sidebarRoot: HTMLElement;
  private readonly sidebarBody: HTMLElement;
  private readonly sidebarHeader: HTMLElement;
  private readonly sidebarRail: HTMLElement;
  private dock: DockviewComponent | null = null;
  // Pending task inits keyed by panel id — added via openTask before
  // dockview asks for the renderer, consumed when createComponent
  // instantiates the panel.
  private readonly pendingInits = new Map<string, TaskInit>();
  // Task panels keyed by command path so reopens focus the same panel.
  private readonly taskPanels = new Map<string, string>();

  constructor(opts: DashboardShellOptions) {
    this.schema = opts.schema;
    this.sidebar = new Sidebar({ schema: opts.schema, state: this.state });
    this.statusBar = new StatusBar(opts.statusRoot, this.state);
    this.sidebarRoot = opts.sidebarRoot;
    this.sidebarBody = opts.sidebarBody;
    this.sidebarHeader = opts.sidebarHeader;
    this.sidebarRail = opts.sidebarRail;
  }

  mount(dockRoot: HTMLElement): void {
    // Sidebar lives outside dockview now — mount once into the
    // persistent shell host.
    this.sidebar.mount(this.sidebarBody);
    this.applyCollapsed(getStored<boolean>(SIDEBAR_COLLAPSED_KEY, false));
    this.sidebarHeader.addEventListener('click', () => this.toggleCollapsed());
    this.wireRailButtons();

    this.dock = new DockviewComponent(dockRoot, {
      createComponent: (opts) => this.createRenderer(opts),
    });

    this.state.on('command:open', (cmd) => this.openTaskForCommand(cmd));

    const stored = getStored<SerializedDockview | null>(LAYOUT_KEY, null);
    let restored = false;
    if (stored !== null) {
      try {
        this.dock.fromJSON(stored);
        restored = true;
      } catch {
        restored = false;
      }
    }
    if (!restored) {
      this.buildDefaultLayout();
    }

    this.dock.onDidLayoutChange(() => {
      if (this.dock) setStored(LAYOUT_KEY, this.dock.toJSON());
    });

    this.dock.onDidActivePanelChange((panel) => {
      const renderer = panel?.view?.content as TaskAware | undefined;
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
  // Sidebar collapse + rail wiring
  // -------------------------------------------------------------------

  private toggleCollapsed(): void {
    this.applyCollapsed(!this.sidebarRoot.classList.contains('collapsed'));
  }

  private applyCollapsed(collapsed: boolean): void {
    this.sidebarRoot.classList.toggle('collapsed', collapsed);
    setStored(SIDEBAR_COLLAPSED_KEY, collapsed);
  }

  private wireRailButtons(): void {
    const buttons = this.sidebarRail.querySelectorAll<HTMLButtonElement>(
      '.shell-rail-btn',
    );
    for (const btn of buttons) {
      btn.addEventListener('click', (e) => {
        e.stopPropagation();
        const rail = btn.dataset.rail;
        // Expand the sidebar so the chosen mode / search is visible.
        this.applyCollapsed(false);
        if (rail === 'common' || rail === 'all') {
          this.sidebar.setMode(rail as SidebarMode);
        } else if (rail === 'search') {
          // Defer focus until after the expand transition begins — the
          // input is display:none while collapsed.
          requestAnimationFrame(() => this.sidebar.focusSearch());
        }
      });
    }
  }

  // -------------------------------------------------------------------
  // dockview-core renderer factory
  // -------------------------------------------------------------------

  private createRenderer(opts: CreateComponentOptions): IContentRenderer {
    const stateRef = this.state;
    const shell = this;

    switch (opts.name) {
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
              <p class="welcome-tagline">Integrative bioinformatics
              platform — pick a command from the sidebar to get
              started.</p>
              <div class="welcome-links">
                <a class="welcome-link"
                  href="https://github.com/wilburnlab/constellation"
                  target="_blank" rel="noopener">
                  <div class="welcome-link-title">GitHub</div>
                  <div class="welcome-link-desc">Source repository</div>
                </a>
                <a class="welcome-link"
                  href="https://github.com/wilburnlab/constellation/blob/main/README.md"
                  target="_blank" rel="noopener">
                  <div class="welcome-link-title">Documentation</div>
                  <div class="welcome-link-desc">Project README</div>
                </a>
                <a class="welcome-link"
                  href="https://github.com/wilburnlab/constellation/issues"
                  target="_blank" rel="noopener">
                  <div class="welcome-link-title">Issues</div>
                  <div class="welcome-link-desc">Report a bug or request a feature</div>
                </a>
                <button class="welcome-link welcome-link-button" type="button"
                  data-command="doctor">
                  <div class="welcome-link-title">Doctor</div>
                  <div class="welcome-link-desc">Check install status &amp; tool discovery</div>
                </button>
              </div>
            `;
            const doctorBtn = this.element.querySelector<HTMLButtonElement>(
              '.welcome-link-button[data-command="doctor"]',
            );
            doctorBtn?.addEventListener('click', () => {
              shell.openCommandByPath(['doctor']);
            });
          }
        })();

      case 'task':
        return new (class implements TaskAware {
          readonly element = (() => {
            const e = document.createElement('div');
            e.style.height = '100%';
            return e;
          })();
          private panel: TaskPanel | null = null;
          private panelKey: string | null = null;
          init(params: GroupPanelPartInitParameters): void {
            const id = params.api.id;
            const init = shell.consumeInit(id, params.params as TaskPanelParams);
            if (!init) {
              this.element.textContent =
                '(task panel unavailable — command no longer exists)';
              return;
            }
            this.panelKey = id.replace(/^task:/, '');
            this.panel = new TaskPanel({
              state: stateRef,
              setTitle: (title: string) => params.api.setTitle(title),
            });
            this.panel.mount(this.element, init);
          }
          onResize(): void {
            this.panel?.onResize();
          }
          dispose(): void {
            this.panel?.destroy();
            this.panel = null;
            if (this.panelKey !== null) {
              shell.releaseTaskPanel(this.panelKey);
              this.panelKey = null;
            }
          }
        })();

      default:
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
  // Default layout + task-panel management
  // -------------------------------------------------------------------

  private buildDefaultLayout(): void {
    if (!this.dock) return;
    this.dock.addPanel({
      id: 'welcome',
      component: 'welcome',
      title: 'Home',
      params: { kind: 'welcome' } satisfies PanelParams,
    });
  }

  private openTaskForCommand(cmd: CommandSchema): void {
    if (!this.dock) return;
    const init = taskInitForCommand(cmd);
    this.addTaskPanel(cmd.path, init);
  }

  /** Open a sidebar command by its dotted path. Used by the home
   *  page's in-app shortcuts (e.g. the Doctor tile) — reuses the
   *  same code path the sidebar takes. */
  openCommandByPath(path: string[]): void {
    const cmd = this.findCommand(path);
    if (cmd) this.openTaskForCommand(cmd);
  }

  private addTaskPanel(commandPath: string[], init: TaskInit): void {
    if (!this.dock) return;
    const key = commandPath.join(' ');
    const existingId = this.taskPanels.get(key);
    if (existingId) {
      const panel = this.dock.getGroupPanel(existingId);
      if (panel) {
        panel.api.setActive();
        this.sidebar.setActivePath(commandPath);
        return;
      }
      this.taskPanels.delete(key);
    }
    const id = `task:${key}`;
    this.pendingInits.set(id, init);
    // Anchor new tabs to whichever group currently holds the home
    // panel so commands stack as tabs instead of slicing the
    // workspace into thinner and thinner splits. If home has been
    // closed, let dockview place the new panel in its active group
    // (or create one) — there's no longer a sidebar panel to
    // side-split against.
    const home = this.dock.getGroupPanel('welcome');
    this.dock.addPanel({
      id,
      component: 'task',
      title: titleFor(init, commandPath),
      params: {
        kind: 'task',
        commandPath,
      } satisfies PanelParams,
      ...(home ? { position: { referenceGroup: home.group } } : {}),
    });
    this.taskPanels.set(key, id);
    this.sidebar.setActivePath(commandPath);
  }

  /** Called from a task panel renderer's `dispose` so reopens of the
   *  same command get a fresh panel id instead of trying to focus a
   *  panel that dockview has already torn down. */
  releaseTaskPanel(key: string): void {
    this.taskPanels.delete(key);
  }

  private consumeInit(
    panelId: string,
    params: TaskPanelParams,
  ): TaskInit | null {
    const pending = this.pendingInits.get(panelId);
    if (pending) {
      this.pendingInits.delete(panelId);
      return pending;
    }
    // Layout-restore path: dockview ran `fromJSON` and is asking us to
    // recreate a panel whose init we never stashed. Rebuild from the
    // serialized commandPath.
    const path = params.commandPath;
    if (!Array.isArray(path) || path.length === 0) return null;
    const command = this.findCommand(path);
    if (!command) return null;
    return taskInitForCommand(command);
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

function titleFor(init: TaskInit, commandPath: string[]): string {
  if (init.kind === 'viz') return init.descriptor.label;
  return commandPath.join(' ');
}
