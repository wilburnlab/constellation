// Shared event bus + persisted UI state for the dashboard.
//
// Mirrors the `ViewportBus` pattern from PR 1 (engine/viewport_bus.ts):
// a thin EventTarget wrapper plus per-key localStorage persistence.
// Components don't depend on each other directly — they subscribe to
// events.

import type { CommandSchema, JobSnapshot } from './types';

export type SidebarMode = 'common' | 'all';

export interface DashboardEvents {
  // The user picked a command from the sidebar — open / focus a task panel.
  'command:open': CommandSchema;
  // The currently-running job snapshot (or null). Polled by StatusBar.
  'job:active': JobSnapshot | null;
  // A POST /api/commands returned 409 — the form surfaces the message.
  'job:rejected': string;
}

export class DashboardState {
  private readonly bus = new EventTarget();

  emit<K extends keyof DashboardEvents>(
    event: K,
    detail: DashboardEvents[K],
  ): void {
    this.bus.dispatchEvent(new CustomEvent(event, { detail }));
  }

  on<K extends keyof DashboardEvents>(
    event: K,
    handler: (detail: DashboardEvents[K]) => void,
  ): () => void {
    const listener = (e: Event) => {
      handler((e as CustomEvent<DashboardEvents[K]>).detail);
    };
    this.bus.addEventListener(event, listener as EventListener);
    return () => this.bus.removeEventListener(event, listener as EventListener);
  }
}

// ---------------------------------------------------------------------
// localStorage helpers — keyed by namespace+version so future schema
// changes don't deserialize against stale shapes.
// ---------------------------------------------------------------------

const STORAGE_PREFIX = 'constellation.dashboard.';

export function getStored<T>(key: string, fallback: T): T {
  try {
    const raw = window.localStorage.getItem(STORAGE_PREFIX + key);
    if (raw === null) return fallback;
    return JSON.parse(raw) as T;
  } catch {
    return fallback;
  }
}

export function setStored(key: string, value: unknown): void {
  try {
    window.localStorage.setItem(STORAGE_PREFIX + key, JSON.stringify(value));
  } catch {
    // Quota or private-mode failure — ignore; UI just won't persist.
  }
}
