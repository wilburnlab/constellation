// PR 1 entry point — bootstraps the genome browser against the
// session the FastAPI server announces via /api/sessions.
//
// PR 1 ships a single session per process, so we pick the first
// session we find. PR 2's dashboard handles multi-session selection
// via a separate entry.

import { fetchJson } from './engine/arrow_client';
import { GenomeBrowser } from './widgets/GenomeBrowser';

interface SessionSummary {
  session_id: string;
  label: string;
  root: string;
  stages_present: Record<string, boolean>;
}

async function main(): Promise<void> {
  const root = document.getElementById('app');
  if (!root) {
    throw new Error('expected #app root element');
  }
  root.innerHTML =
    '<div style="padding:16px;color:#8a8a93">Loading session…</div>';

  let sessions: SessionSummary[] = [];
  try {
    sessions = await fetchJson<SessionSummary[]>('/api/sessions');
  } catch (err) {
    root.innerHTML = `<div style="padding:16px;color:#d65e5e">
      Failed to reach /api/sessions: ${(err as Error).message}
    </div>`;
    return;
  }

  if (sessions.length === 0) {
    root.innerHTML = `<div style="padding:16px;color:#d65e5e">
      No sessions registered with the server.
    </div>`;
    return;
  }

  const session = sessions[0];
  document.title = `Constellation · ${session.label}`;

  const browser = new GenomeBrowser({
    host: root,
    sessionId: session.session_id,
  });
  await browser.mount();
}

void main();
