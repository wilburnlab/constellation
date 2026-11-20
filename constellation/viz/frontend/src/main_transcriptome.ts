// Standalone entry for `constellation viz transcriptome`: opens the browser on
// the session the server was started with.

import { fetchJson } from './engine/arrow_client';
import { TranscriptomeBrowser, parseInitial } from './modalities/transcriptome/TranscriptomeBrowser';

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
  const session = sessions.find((s) => s.modality === 'transcriptome');
  if (!session) {
    root.innerHTML = '<div style="padding:16px;color:#ff6767">No transcriptome session registered with the server.</div>';
    return;
  }
  document.title = `Constellation · ${session.label}`;
  root.innerHTML = '';
  const browser = new TranscriptomeBrowser({
    host: root,
    sessionId: session.session_id,
    initial: parseInitial(window.location.hash),
  });
  await browser.mount();
}

void main();
