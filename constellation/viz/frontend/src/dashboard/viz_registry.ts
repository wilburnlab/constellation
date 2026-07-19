// Visualization-tool registry.
//
// The dashboard's task panel runs in one of two phase-2 modes: a
// terminal (compute jobs) or a visualization widget. For each viz
// subcommand we register a descriptor here. A descriptor either:
//   - declares `fields`, in which case the standard VizForm renders a
//     simple text/path-input form and submits to `open(host, values)`;
//   - declares `customForm`, in which case TaskPanel mounts the
//     returned form-handle and lets the form drive the phase-2 mount
//     itself (used by the reference-cache-first genome browser entry,
//     which needs a richer multi-row UI than the generic VizForm).
//
// Adding a second viz tool (spectrum viewer, structure browser, etc.)
// is one entry; no introspect or schema changes required.

import { GenomeBrowserForm } from './GenomeBrowserForm';
import { DashboardState } from './state';
import type {
  CommandSchema,
  OpenSessionResult,
  SavedSessionSummary,
  TrackLayoutEntry,
} from './types';

export type VizFieldKind = 'path' | 'dir' | 'file' | 'text';

export interface VizFormField {
  name: string;
  label: string;
  kind: VizFieldKind;
  placeholder?: string;
  remember?: boolean;
  required?: boolean;
}

export interface VizMountResult {
  title: string;
  /** Optional disposer for the mounted widget. TaskPanel calls this on
   *  destroy(). */
  dispose?: () => void;
}

/** Handle returned by a `customForm` mount — the task panel uses this
 *  for lifecycle integration (resize / dispose). */
export interface CustomFormHandle {
  destroy(): void;
  onResize?(): void;
}

/** Custom-form mount callback. The form drives the form-phase UI and
 *  calls `transitionToWidget` when the user is ready to swap into the
 *  phase-2 widget. */
export interface CustomFormContext {
  host: HTMLElement;
  state: DashboardState;
  setTitle: (title: string) => void;
  transitionToWidget: (
    mount: (host: HTMLElement) => Promise<VizMountResult>,
  ) => Promise<void>;
}

export interface VizDescriptor {
  path: string[];
  label: string;
  helpText?: string;
  submitLabel?: string;
  /** Simple-form fields — used when `customForm` is absent. */
  fields?: VizFormField[];
  /** Simple-form submit handler — used when `customForm` is absent. */
  open?: (
    host: HTMLElement,
    values: Record<string, string>,
  ) => Promise<VizMountResult>;
  /** Custom-form mount — overrides `fields` + `open` when present. */
  customForm?: (ctx: CustomFormContext) => CustomFormHandle | Promise<CustomFormHandle>;
}

const VIZ_DESCRIPTORS: VizDescriptor[] = [
  {
    path: ['viz', 'genome'],
    label: 'Genome browser',
    helpText:
      'Pick a reference from the cache, then attach one or more ' +
      'transcriptome align / cluster output directories.',
    customForm: (ctx) => {
      const form = new GenomeBrowserForm({
        state: ctx.state,
        async onSubmit(
          result: OpenSessionResult,
          _saved: SavedSessionSummary | null,
          initialLayout: TrackLayoutEntry[] | null,
        ) {
          await ctx.transitionToWidget(async (widgetHost) => {
            const { GenomeBrowser } = await import('../widgets/GenomeBrowser');
            widgetHost.innerHTML = '';
            const browser = new GenomeBrowser({
              host: widgetHost,
              sessionId: result.session_id,
              initialLayout: initialLayout ?? undefined,
            });
            await browser.mount();
            return {
              title: `genome · ${result.label}`,
              dispose: () => browser.dispose(),
            };
          });
        },
      });
      // Async mount — return the handle synchronously so TaskPanel can
      // wire lifecycle hooks; kick off the async mount() in the
      // background. The form renders progressively as data loads.
      void form.mount(ctx.host);
      return {
        destroy() {
          form.destroy();
        },
      };
    },
  },
];

export function findVizDescriptor(path: string[]): VizDescriptor | null {
  const key = path.join('');
  for (const d of VIZ_DESCRIPTORS) {
    if (d.path.join('') === key) return d;
  }
  return null;
}

export function isVizCommand(cmd: CommandSchema): boolean {
  return findVizDescriptor(cmd.path) !== null;
}
