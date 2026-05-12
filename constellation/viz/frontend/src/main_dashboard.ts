// PR 2 entry point — boots the JupyterLab-style dashboard SPA.
//
// On load:
// 1. Fetch /api/cli/schema once for the sidebar's command tree
// 2. Instantiate DashboardShell against #dock + #statusbar
// 3. The shell mounts sidebar + welcome panels via dockview-core

import { DashboardShell } from './dashboard/DashboardShell';
import { fetchJson } from './engine/arrow_client';
import type { CliSchema } from './dashboard/types';

async function main(): Promise<void> {
  const dockRoot = document.getElementById('dock');
  const statusRoot = document.getElementById('statusbar');
  if (!dockRoot || !statusRoot) {
    throw new Error('expected #dock and #statusbar root elements');
  }

  let schema: CliSchema;
  try {
    schema = await fetchJson<CliSchema>('/api/cli/schema');
  } catch (err) {
    dockRoot.innerHTML = `
      <div style="padding:24px;color:#d65e5e">
        Failed to load CLI schema: ${(err as Error).message}
      </div>`;
    return;
  }

  const shell = new DashboardShell({
    schema,
    dockRoot,
    statusRoot,
  });
  shell.mount(dockRoot);
}

void main();
