// StatusBar — current-job indicator at the bottom of the window.
//
// Polls /api/commands/active every 2s and reflects the result in
// the existing #statusbar element (its DOM is created by
// index.dashboard.html, not by this component). Also surfaces toast-
// style transient messages for lock rejections (job:rejected).

import type { JobSnapshot } from './types';
import { DashboardState } from './state';

const POLL_INTERVAL_MS = 2000;
const TOAST_DURATION_MS = 5000;

export class StatusBar {
  private readonly element: HTMLElement;
  private readonly state: DashboardState;
  private readonly textEl: HTMLElement;
  private toastEl: HTMLElement | null = null;
  private toastTimer: number | null = null;
  private pollTimer: number | null = null;
  private unsubscribe: (() => void) | null = null;

  constructor(element: HTMLElement, state: DashboardState) {
    this.element = element;
    this.state = state;
    let text = element.querySelector<HTMLElement>('.status-text');
    if (!text) {
      text = document.createElement('span');
      text.className = 'status-text status-idle';
      text.textContent = 'Idle';
      element.appendChild(text);
    }
    this.textEl = text;
  }

  start(): void {
    void this.poll();
    this.pollTimer = window.setInterval(() => void this.poll(), POLL_INTERVAL_MS);
    this.unsubscribe = this.state.on('job:rejected', (message) =>
      this.showToast(message, 'warn'),
    );
    this.state.on('job:started', () => {
      // Refresh the status bar immediately on start — don't wait for
      // the next poll tick.
      void this.poll();
    });
  }

  stop(): void {
    if (this.pollTimer !== null) {
      window.clearInterval(this.pollTimer);
      this.pollTimer = null;
    }
    this.unsubscribe?.();
    if (this.toastTimer !== null) {
      window.clearTimeout(this.toastTimer);
      this.toastTimer = null;
    }
  }

  private async poll(): Promise<void> {
    try {
      const response = await fetch('/api/commands/active');
      if (!response.ok) {
        this.renderError(`HTTP ${response.status}`);
        return;
      }
      const data = (await response.json()) as JobSnapshot | null;
      this.state.emit('job:active', data);
      if (data === null) {
        this.renderIdle();
      } else {
        this.renderRunning(data);
      }
    } catch (err) {
      this.renderError((err as Error).message);
    }
  }

  private renderIdle(): void {
    this.textEl.className = 'status-text status-idle';
    this.textEl.textContent = 'Idle';
  }

  private renderRunning(job: JobSnapshot): void {
    const elapsed = elapsedSeconds(job.started_at);
    const argv = job.argv.join(' ');
    const truncated = argv.length > 80 ? argv.slice(0, 77) + '…' : argv;
    this.textEl.className = 'status-text status-running';
    this.textEl.textContent = `Running · constellation ${truncated} · ${elapsed.toFixed(0)}s`;
  }

  private renderError(message: string): void {
    this.textEl.className = 'status-text status-failed';
    this.textEl.textContent = `Status error: ${message}`;
  }

  private showToast(message: string, kind: 'warn' | 'error'): void {
    if (this.toastTimer !== null) window.clearTimeout(this.toastTimer);
    if (!this.toastEl) {
      this.toastEl = document.createElement('span');
      this.toastEl.style.marginLeft = 'auto';
      this.toastEl.style.padding = '2px 8px';
      this.toastEl.style.borderRadius = '4px';
      this.element.appendChild(this.toastEl);
    }
    this.toastEl.style.background =
      kind === 'warn' ? 'rgba(214, 160, 94, 0.2)' : 'rgba(214, 94, 94, 0.2)';
    this.toastEl.style.color =
      kind === 'warn' ? 'var(--warn)' : 'var(--danger)';
    this.toastEl.textContent = message;
    this.toastTimer = window.setTimeout(() => {
      if (this.toastEl) this.toastEl.textContent = '';
    }, TOAST_DURATION_MS);
  }
}

function elapsedSeconds(startedAtIso: string): number {
  const t = Date.parse(startedAtIso);
  if (Number.isNaN(t)) return 0;
  return (Date.now() - t) / 1000;
}
