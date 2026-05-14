// Visualization-tool registry.
//
// The dashboard's task panel runs in one of two phase-2 modes: a
// terminal (compute jobs) or a visualization widget. For each viz
// subcommand we register a descriptor here: the form fields the user
// fills in phase 1, plus an `open` callback that mounts the actual
// widget on the panel's DOM root in phase 2.
//
// Adding a second viz tool (spectrum viewer, structure browser, etc.)
// is one entry; no introspect or schema changes required.

import type { CommandSchema } from './types';

export type VizFieldKind = 'path' | 'text';

export interface VizFormField {
  name: string;
  label: string;
  kind: VizFieldKind;
  placeholder?: string;
  remember?: boolean;          // persist last value to localStorage
  required?: boolean;
}

export interface VizMountResult {
  title: string;
}

export interface VizDescriptor {
  path: string[];                                  // ['viz', 'genome']
  label: string;
  helpText?: string;
  submitLabel: string;
  fields: VizFormField[];
  open: (
    host: HTMLElement,
    values: Record<string, string>,
  ) => Promise<VizMountResult>;
  /** Disposer for the most-recently-mounted widget. Owners may call
   *  this when the task panel is closed or transitions out. */
  dispose?: () => void;
}

interface SessionSummary {
  session_id: string;
  label: string;
  root: string;
  stages_present: Record<string, boolean>;
}

async function registerSession(path: string): Promise<SessionSummary> {
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
      /* fall through */
    }
    throw new Error(detail);
  }
  return (await response.json()) as SessionSummary;
}

const VIZ_DESCRIPTORS: VizDescriptor[] = [
  {
    path: ['viz', 'genome'],
    label: 'Genome browser',
    helpText:
      'Open the embedded genome browser over a Constellation pipeline session ' +
      '(parent of `genome/`, `annotation/`, `S2_align/`, …).',
    submitLabel: 'Open',
    fields: [
      {
        name: 'session',
        label: 'Session directory',
        kind: 'path',
        placeholder: '/path/to/run',
        remember: true,
        required: true,
      },
    ],
    async open(host, values) {
      const sessionPath = (values.session ?? '').trim();
      if (!sessionPath) throw new Error('Session directory is required.');
      const summary = await registerSession(sessionPath);
      const { GenomeBrowser } = await import('../widgets/GenomeBrowser');
      host.innerHTML = '';
      const browser = new GenomeBrowser({
        host,
        sessionId: summary.session_id,
      });
      await browser.mount();
      // Store the disposer so the task panel can tear the widget down
      // cleanly when the panel closes.
      this.dispose = () => browser.dispose();
      return { title: `genome · ${summary.label}` };
    },
  },
];

export function findVizDescriptor(path: string[]): VizDescriptor | null {
  const key = path.join('');
  for (const d of VIZ_DESCRIPTORS) {
    if (d.path.join('') === key) return d;
  }
  return null;
}

export function isVizCommand(cmd: CommandSchema): boolean {
  return findVizDescriptor(cmd.path) !== null;
}
