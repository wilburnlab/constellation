// Terminal panel — xterm.js wrapping a WebSocket stream of OutputFrame
// JSON frames from /api/commands/{job_id}/stream.
//
// Each frame is `{stream: 'stdout'|'stderr'|'exit', line: string}`.
// stderr renders red; the final `exit` frame closes the stream and
// stops the elapsed-timer.

import { FitAddon } from '@xterm/addon-fit';
import { Terminal as XTerm } from '@xterm/xterm';
import '@xterm/xterm/css/xterm.css';

import type { OutputFrame } from './types';

export interface TerminalPanelOptions {
  jobId: string;
  argv: string[];
}

const ANSI_RESET = '\x1b[0m';
const ANSI_RED = '\x1b[31m';
const ANSI_DIM = '\x1b[2m';
const ANSI_GREEN = '\x1b[32m';

export class TerminalPanel {
  readonly jobId: string;
  readonly argv: string[];
  private container: HTMLElement | null = null;
  private term: XTerm | null = null;
  private fit: FitAddon | null = null;
  private ws: WebSocket | null = null;
  private elapsedEl: HTMLElement | null = null;
  private cancelBtn: HTMLButtonElement | null = null;
  private startedAt = Date.now();
  private elapsedTimer: number | null = null;
  private done = false;

  constructor(opts: TerminalPanelOptions) {
    this.jobId = opts.jobId;
    this.argv = opts.argv;
  }

  mount(element: HTMLElement): void {
    this.container = element;
    element.classList.add('terminal-panel');
    element.innerHTML = '';

    const header = document.createElement('div');
    header.className = 'terminal-header';
    const cmd = document.createElement('div');
    cmd.className = 'terminal-cmd';
    cmd.textContent = 'constellation ' + this.argv.join(' ');
    cmd.title = cmd.textContent;
    header.appendChild(cmd);

    const elapsed = document.createElement('div');
    elapsed.className = 'terminal-elapsed';
    elapsed.textContent = '0.0s';
    header.appendChild(elapsed);
    this.elapsedEl = elapsed;

    const cancel = document.createElement('button');
    cancel.type = 'button';
    cancel.textContent = 'Cancel';
    cancel.addEventListener('click', () => void this.cancel());
    header.appendChild(cancel);
    this.cancelBtn = cancel;

    element.appendChild(header);

    const body = document.createElement('div');
    body.className = 'terminal-body';
    element.appendChild(body);

    this.term = new XTerm({
      convertEol: true,
      fontFamily: 'ui-monospace, SF Mono, Menlo, monospace',
      fontSize: 12,
      theme: {
        background: '#0a0a0d',
        foreground: '#e3e3e8',
        cursor: '#4f9efb',
      },
      cursorBlink: false,
      disableStdin: true,
    });
    this.fit = new FitAddon();
    this.term.loadAddon(this.fit);
    this.term.open(body);
    // Fit after the dom is laid out — dockview panels measure async.
    requestAnimationFrame(() => this.fit?.fit());

    this.openStream();
    this.startedAt = Date.now();
    this.elapsedTimer = window.setInterval(() => this.tickElapsed(), 100);
  }

  resize(): void {
    requestAnimationFrame(() => this.fit?.fit());
  }

  destroy(): void {
    if (this.elapsedTimer !== null) {
      window.clearInterval(this.elapsedTimer);
      this.elapsedTimer = null;
    }
    if (this.ws) {
      try {
        this.ws.close();
      } catch {
        /* ignore */
      }
      this.ws = null;
    }
    this.term?.dispose();
    this.term = null;
    if (this.container) this.container.innerHTML = '';
  }

  private openStream(): void {
    const proto = window.location.protocol === 'https:' ? 'wss' : 'ws';
    const url = `${proto}://${window.location.host}/api/commands/${this.jobId}/stream`;
    const ws = new WebSocket(url);
    this.ws = ws;
    ws.addEventListener('message', (e) => this.handleFrame(e.data));
    ws.addEventListener('error', () => {
      this.term?.write(
        `\r\n${ANSI_RED}[ws error]${ANSI_RESET}\r\n`,
      );
    });
    ws.addEventListener('close', () => {
      if (!this.done) {
        this.term?.write(
          `\r\n${ANSI_DIM}[stream closed]${ANSI_RESET}\r\n`,
        );
        this.markDone();
      }
    });
  }

  private handleFrame(data: string | ArrayBuffer): void {
    if (typeof data !== 'string') return;
    let frame: OutputFrame;
    try {
      frame = JSON.parse(data) as OutputFrame;
    } catch {
      this.term?.write(String(data) + '\r\n');
      return;
    }
    if (frame.stream === 'exit') {
      const code = Number(frame.line);
      const color = code === 0 ? ANSI_GREEN : ANSI_RED;
      this.term?.write(
        `\r\n${color}[exit ${frame.line}]${ANSI_RESET}\r\n`,
      );
      this.markDone();
      return;
    }
    const color = frame.stream === 'stderr' ? ANSI_RED : '';
    const tail = frame.stream === 'stderr' ? ANSI_RESET : '';
    this.term?.write(color + frame.line + tail + '\r\n');
  }

  private markDone(): void {
    this.done = true;
    if (this.cancelBtn) this.cancelBtn.disabled = true;
    if (this.elapsedTimer !== null) {
      window.clearInterval(this.elapsedTimer);
      this.elapsedTimer = null;
    }
  }

  private tickElapsed(): void {
    if (!this.elapsedEl) return;
    const seconds = (Date.now() - this.startedAt) / 1000;
    this.elapsedEl.textContent = seconds.toFixed(1) + 's';
  }

  private async cancel(): Promise<void> {
    if (this.done) return;
    try {
      await fetch(`/api/commands/${this.jobId}`, { method: 'DELETE' });
    } catch (err) {
      this.term?.write(
        `\r\n${ANSI_RED}[cancel failed: ${(err as Error).message}]${ANSI_RESET}\r\n`,
      );
    }
  }
}
