// TypedBus — a small typed publish/subscribe channel.
//
// The mechanism is generic; the events belong to whoever creates the
// bus (a browser declares its own vocabulary as an interface mapping
// event names to payloads). Handlers run synchronously in subscription
// order; a handler that throws does not stop the others.

export class TypedBus<E extends object> {
  private handlers = new Map<keyof E, Set<(payload: never) => void>>();

  /** Subscribe; returns the unsubscribe function. */
  on<K extends keyof E>(event: K, handler: (payload: E[K]) => void): () => void {
    let set = this.handlers.get(event);
    if (!set) {
      set = new Set();
      this.handlers.set(event, set);
    }
    set.add(handler as (payload: never) => void);
    return () => {
      set?.delete(handler as (payload: never) => void);
    };
  }

  emit<K extends keyof E>(event: K, payload: E[K]): void {
    const set = this.handlers.get(event);
    if (!set) return;
    for (const handler of [...set]) {
      try {
        (handler as (payload: E[K]) => void)(payload);
      } catch (err) {
        console.warn(`bus handler for ${String(event)} failed`, err);
      }
    }
  }

  /** Number of live subscriptions (for leak checks in tests). */
  get size(): number {
    let n = 0;
    for (const set of this.handlers.values()) n += set.size;
    return n;
  }

  dispose(): void {
    this.handlers.clear();
  }
}
