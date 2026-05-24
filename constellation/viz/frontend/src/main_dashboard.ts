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
  const sidebarRoot = document.getElementById('shell-sidebar');
  const sidebarBody = document.getElementById('shell-sidebar-body');
  const sidebarHeader = document.getElementById('shell-sidebar-header');
  const sidebarRail = document.getElementById('shell-sidebar-rail');
  if (
    !dockRoot ||
    !statusRoot ||
    !sidebarRoot ||
    !sidebarBody ||
    !sidebarHeader ||
    !sidebarRail
  ) {
    throw new Error(
      'expected #dock, #statusbar, #shell-sidebar, #shell-sidebar-body, ' +
        '#shell-sidebar-header, and #shell-sidebar-rail root elements',
    );
  }

  let schema: CliSchema;
  try {
    schema = await fetchJson<CliSchema>('/api/cli/schema');
  } catch (err) {
    dockRoot.innerHTML = `
      <div style="padding:24px;color:#ff6767">
        Failed to load CLI schema: ${(err as Error).message}
      </div>`;
    return;
  }

  const shell = new DashboardShell({
    schema,
    dockRoot,
    statusRoot,
    sidebarRoot,
    sidebarBody,
    sidebarHeader,
    sidebarRail,
  });
  shell.mount(dockRoot);
}

void main();
