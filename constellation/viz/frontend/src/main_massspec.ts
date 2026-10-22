// Standalone entry for `constellation viz massspec`: opens the browser on
// the session the server was started with.

import { fetchJson } from './engine/arrow_client';
import { MassSpecBrowser } from './modalities/massspec/MassSpecBrowser';

interface SessionSummary {
  session_id: string;
  label: string;
  modality: string;
}

async function main(): Promise<void> {
  const root = document.getElementById('app');
  if (!root) throw new Error('expected #app root element');
  root.innerHTML = '<div style="padding:16px;color:#adacae">Loading session…</div>';

  let sessions: SessionSummary[] = [];
  try {
    sessions = await fetchJson<SessionSummary[]>('/api/sessions');
  } catch (err) {
    root.innerHTML = `<div style="padding:16px;color:#ff6767">Failed to reach /api/sessions: ${(err as Error).message}</div>`;
    return;
  }
  const session = sessions.find((s) => s.modality === 'massspec');
  if (!session) {
    root.innerHTML = '<div style="padding:16px;color:#ff6767">No mass-spec session registered with the server.</div>';
    return;
  }
  document.title = `Constellation · ${session.label}`;
  root.innerHTML = '';
  const browser = new MassSpecBrowser({ host: root, sessionId: session.session_id });
  await browser.mount();
}

void main();
