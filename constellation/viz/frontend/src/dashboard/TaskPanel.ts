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
  private vizDescriptor: VizDescriptor | null = null;

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
    if (this.vizDescriptor?.dispose) {
      try {
        this.vizDescriptor.dispose();
      } catch (err) {
        console.warn('viz dispose failed', err);
      }
    }
    this.vizDescriptor = null;
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
    this.vizDescriptor = descriptor;
    this.setTitle(descriptor.label);
    const form = new VizForm({
      descriptor,
      prefill,
      onSubmit: (values) => this.transitionToViz(descriptor, values),
    });
    form.mount(this.host);
    this.active = {
      destroy: () => form.destroy(),
    };
    if (autoSubmit) {
      void form.submit();
    }
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

  private async transitionToViz(
    descriptor: VizDescriptor,
    values: Record<string, string>,
  ): Promise<void> {
    if (!this.host) return;
    this.active?.destroy();
    this.host.innerHTML = '';
    try {
      const result = await descriptor.open(this.host, values);
      this.setTitle(result.title);
    } catch (err) {
      this.host.innerHTML = '';
      const msg = document.createElement('div');
      msg.className = 'task-error';
      msg.textContent = `Failed to open ${descriptor.label}: ${
        err instanceof Error ? err.message : String(err)
      }`;
      this.host.appendChild(msg);
      throw err;
    }
    // Viz widgets dispose themselves through `descriptor.dispose`,
    // which the descriptor sets in its `open` body.
    this.active = {
      destroy: () => {
        if (descriptor.dispose) {
          try {
            descriptor.dispose();
          } catch (err) {
            console.warn('viz dispose failed', err);
          }
          descriptor.dispose = undefined;
        }
      },
    };
  }
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
