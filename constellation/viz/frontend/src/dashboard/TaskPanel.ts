// TaskPanel — single panel kind that transitions in-place across
// phases. Phase 1 is a form (CommandForm for compute, VizForm for
// visualization). Phase 2 swaps the form's content out for either a
// terminal streaming the subprocess (compute) or the embedded
// visualization widget (viz). The dock panel keeps its identity so the
// tab the user invoked the command from *becomes* the running tool.

import { CommandForm } from './CommandForm';
import { TerminalPanel } from './Terminal';
import { VizForm } from './VizForm';
import { DashboardState } from './state';
import type { CommandSchema } from './types';
import {
  findVizDescriptor,
  type VizDescriptor,
  type VizMountResult,
} from './viz_registry';

export type TaskInit =
  | { kind: 'command'; command: CommandSchema }
  | {
      kind: 'viz';
      descriptor: VizDescriptor;
      prefill?: Record<string, string>;
      autoSubmit?: boolean;
    };

export interface TaskPanelOptions {
  state: DashboardState;
  setTitle: (title: string) => void;
}

interface PhaseHandle {
  /** Tear down the active child. Idempotent. */
  destroy(): void;
  /** Forward dockview's resize hook (xterm needs explicit fit). */
  onResize?: () => void;
}

export class TaskPanel {
  private readonly state: DashboardState;
  private readonly setTitle: (title: string) => void;
  private host: HTMLElement | null = null;
  private active: PhaseHandle | null = null;
  private widgetDisposer: (() => void) | null = null;

  constructor(opts: TaskPanelOptions) {
    this.state = opts.state;
    this.setTitle = opts.setTitle;
  }

  mount(host: HTMLElement, init: TaskInit): void {
    this.host = host;
    host.classList.add('task-panel');
    host.innerHTML = '';
    if (init.kind === 'command') {
      this.mountCommandForm(init.command);
    } else {
      this.mountVizForm(init.descriptor, init.prefill, init.autoSubmit);
    }
  }

  destroy(): void {
    this.active?.destroy();
    this.active = null;
    if (this.widgetDisposer) {
      try {
        this.widgetDisposer();
      } catch (err) {
        console.warn('viz dispose failed', err);
      }
      this.widgetDisposer = null;
    }
    if (this.host) this.host.innerHTML = '';
    this.host = null;
  }

  /** Forward dockview's `onDidActivePanelChange` resize signal. */
  onResize(): void {
    this.active?.onResize?.();
  }

  // -------------------------------------------------------------------
  // Phase 1: form
  // -------------------------------------------------------------------

  private mountCommandForm(command: CommandSchema): void {
    if (!this.host) return;
    this.setTitle(command.path.join(' '));
    if (isUnknownBrowser(command)) {
      // A browser this dashboard build has no descriptor for: run as a
      // command it would start a second server in a terminal. Say why.
      const note = document.createElement('div');
      note.className = 'task-notice';
      note.style.padding = '16px';
      note.style.color = 'var(--warn, #ffca85)';
      note.textContent =
        `\`constellation ${command.path.join(' ')}\` is a browser this dashboard's frontend ` +
        'bundle predates, so it cannot open here. Rebuild the frontend ' +
        '(python -m constellation.viz.frontend.build, or constellation viz install-frontend) ' +
        'and reload the page.';
      this.host.appendChild(note);
      this.active = { destroy: () => note.remove() };
      return;
    }
    const form = new CommandForm({
      command,
      state: this.state,
      onJobStarted: ({ jobId, argv }) => this.transitionToTerminal(jobId, argv),
    });
    form.mount(this.host);
    this.active = {
      destroy: () => form.destroy(),
    };
  }

  private mountVizForm(
    descriptor: VizDescriptor,
    prefill?: Record<string, string>,
    autoSubmit?: boolean,
  ): void {
    if (!this.host) return;
    this.setTitle(descriptor.label);
    if (descriptor.customForm) {
      this.mountCustomForm(descriptor);
      return;
    }
    if (!descriptor.fields || !descriptor.open) {
      throw new Error(
        `viz descriptor at ${descriptor.path.join(' ')} is missing either ` +
          'customForm or (fields + open)',
      );
    }
    const open = descriptor.open;
    const form = new VizForm({
      descriptor: {
        ...descriptor,
        // VizForm only consumes the simple-form fields/submitLabel
        fields: descriptor.fields,
        submitLabel: descriptor.submitLabel ?? 'Open',
      },
      prefill,
      onSubmit: (values) => this.transitionToViz(descriptor, open, values),
    });
    form.mount(this.host);
    this.active = {
      destroy: () => form.destroy(),
    };
    if (autoSubmit) {
      void form.submit();
    }
  }

  private mountCustomForm(descriptor: VizDescriptor): void {
    if (!this.host || !descriptor.customForm) return;
    const host = this.host;
    const handlePromise = Promise.resolve(
      descriptor.customForm({
        host,
        state: this.state,
        setTitle: this.setTitle,
        transitionToWidget: async (mount) => {
          if (!this.host) return;
          this.active?.destroy();
          this.clearFormChrome();
          try {
            const result = await mount(this.host);
            this.setTitle(result.title);
            if (result.dispose) this.widgetDisposer = result.dispose;
          } catch (err) {
            this.renderTaskError(descriptor, err);
            throw err;
          }
          this.active = {
            destroy: () => {
              if (this.widgetDisposer) {
                try {
                  this.widgetDisposer();
                } catch (e) {
                  console.warn('viz dispose failed', e);
                }
                this.widgetDisposer = null;
              }
            },
          };
        },
      }),
    );
    // Capture a destroyer that resolves the handle's destroy once it
    // exists. If the user closes the panel before the customForm's
    // async mount returns, the handle is destroyed as soon as it's
    // available.
    let destroyed = false;
    let resolvedDestroy: (() => void) | null = null;
    void handlePromise.then((h) => {
      if (destroyed) {
        try {
          h.destroy();
        } catch (err) {
          console.warn('viz form dispose failed', err);
        }
      } else {
        resolvedDestroy = () => h.destroy();
      }
    });
    this.active = {
      destroy: () => {
        destroyed = true;
        if (resolvedDestroy) {
          try {
            resolvedDestroy();
          } catch (err) {
            console.warn('viz form dispose failed', err);
          }
        }
      },
    };
  }

  private renderTaskError(descriptor: VizDescriptor, err: unknown): void {
    if (!this.host) return;
    this.host.innerHTML = '';
    const msg = document.createElement('div');
    msg.className = 'task-error';
    msg.textContent = `Failed to open ${descriptor.label}: ${
      err instanceof Error ? err.message : String(err)
    }`;
    this.host.appendChild(msg);
  }

  // -------------------------------------------------------------------
  // Phase 2: terminal (compute) or viz (visualization)
  // -------------------------------------------------------------------

  private transitionToTerminal(jobId: string, argv: string[]): void {
    if (!this.host) return;
    this.active?.destroy();
    this.host.innerHTML = '';
    const short = argv.slice(0, 2).join(' ') || 'job';
    this.setTitle(`terminal · ${short}`);
    const term = new TerminalPanel({ jobId, argv });
    term.mount(this.host);
    this.active = {
      destroy: () => term.destroy(),
      onResize: () => term.resize(),
    };
  }

  /** Empty the host for a widget, dropping the classes the entry forms
   *  put on it (their padding and input rules would otherwise style the
   *  widget). */
  private clearFormChrome(): void {
    if (!this.host) return;
    this.host.innerHTML = '';
    this.host.classList.remove('viz-form', 'genome-browser-form');
  }

  private async transitionToViz(
    descriptor: VizDescriptor,
    open: (
      host: HTMLElement,
      values: Record<string, string>,
    ) => Promise<VizMountResult>,
    values: Record<string, string>,
  ): Promise<void> {
    if (!this.host) return;
    this.active?.destroy();
    this.clearFormChrome();
    try {
      const result = await open(this.host, values);
      this.setTitle(result.title);
      if (result.dispose) this.widgetDisposer = result.dispose;
    } catch (err) {
      this.renderTaskError(descriptor, err);
      throw err;
    }
    this.active = {
      destroy: () => {
        if (this.widgetDisposer) {
          try {
            this.widgetDisposer();
          } catch (err) {
            console.warn('viz dispose failed', err);
          }
          this.widgetDisposer = null;
        }
      },
    };
  }
}

/** A `viz` subcommand that serves a browser (it takes `--no-browser`)
 *  but has no descriptor in this build — a server newer than its bundle. */
export function isUnknownBrowser(command: CommandSchema): boolean {
  return (
    command.path[0] === 'viz' &&
    findVizDescriptor(command.path) === null &&
    command.arguments.some((a) => a.option_strings.includes('--no-browser'))
  );
}

/** Resolve a sidebar `command:open` event into a TaskInit. Compute
 *  subcommands stay as `command` inits; viz subcommands route through
 *  the registry into `viz` inits. */
export function taskInitForCommand(command: CommandSchema): TaskInit {
  const descriptor = findVizDescriptor(command.path);
  if (descriptor) {
    return { kind: 'viz', descriptor };
  }
  return { kind: 'command', command };
}
