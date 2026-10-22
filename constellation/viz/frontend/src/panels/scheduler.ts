// RenderScheduler — fetch panels in parallel and draw only what is current.
//
// A render pass hands over one plan per panel: the key of the data the
// panel should show now, and how to fetch it. For each panel the
// scheduler then
//   - redraws from the cache when the cached key is the planned one (no
//     request: a pan or a style change that the data already covers);
//   - keeps a request already running for the planned key;
//   - otherwise aborts the panel's request for an older key and starts
//     one for the new key.
// A result is drawn only if its key is still the panel's latest planned
// key ("latest wins"): passes may overlap, and a slow response for a view
// the user has already left is dropped rather than drawn over the new one.
// At most `concurrency` requests run at once (browsers allow six per
// origin over HTTP/1.1; leave room for the page's other requests).

export interface FetchPlan<P, D> {
  target: P;
  /** What the panel should show; `null` when it should show nothing. */
  key: string | null;
  fetch(signal: AbortSignal): Promise<D>;
}

export interface RenderHooks<P, D> {
  cached(target: P): { key: string; data: D } | undefined;
  store(target: P, key: string, data: D): void;
  draw(target: P, data: D): void;
  loading?(target: P): void;
  failed(target: P, error: Error): void;
  /** The plan's key was null. */
  empty?(target: P): void;
}

interface Running {
  key: string;
  controller: AbortController;
  done: Promise<void>;
}

export class RenderScheduler<P, D> {
  private readonly concurrency: number;
  private readonly latest = new Map<P, string | null>();
  private readonly running = new Map<P, Running>();
  private active = 0;
  private readonly waiting: Array<() => void> = [];

  constructor(opts: { concurrency?: number } = {}) {
    this.concurrency = Math.max(1, opts.concurrency ?? 4);
  }

  /** Plan, fetch and draw. Resolves once every fetch this pass started
   *  (or kept) has settled. */
  async pass(plans: readonly FetchPlan<P, D>[], hooks: RenderHooks<P, D>): Promise<void> {
    const pending: Promise<void>[] = [];
    for (const plan of plans) {
      this.latest.set(plan.target, plan.key);
      const running = this.running.get(plan.target);
      if (plan.key === null) {
        running?.controller.abort();
        hooks.empty?.(plan.target);
        continue;
      }
      const cached = hooks.cached(plan.target);
      if (cached && cached.key === plan.key) {
        if (running && running.key !== plan.key) running.controller.abort();
        hooks.draw(plan.target, cached.data);
        continue;
      }
      if (running && running.key === plan.key) {
        pending.push(running.done);
        continue;
      }
      running?.controller.abort();
      pending.push(this.start(plan as FetchPlan<P, D> & { key: string }, hooks));
    }
    await Promise.all(pending);
  }

  /** Abort everything and forget what was planned. */
  cancelAll(): void {
    for (const running of this.running.values()) running.controller.abort();
    this.running.clear();
    this.latest.clear();
  }

  /** Requests running now (for tests). */
  get inFlight(): number {
    return this.running.size;
  }

  private start(plan: FetchPlan<P, D> & { key: string }, hooks: RenderHooks<P, D>): Promise<void> {
    const controller = new AbortController();
    const record: Running = { key: plan.key, controller, done: Promise.resolve() };
    hooks.loading?.(plan.target);
    record.done = (async () => {
      await this.slot();
      try {
        if (controller.signal.aborted) return;
        const data = await plan.fetch(controller.signal);
        if (controller.signal.aborted || this.latest.get(plan.target) !== plan.key) return;
        hooks.store(plan.target, plan.key, data);
        hooks.draw(plan.target, data);
      } catch (err) {
        const error = err as Error;
        if (error?.name === 'AbortError' || controller.signal.aborted) return;
        if (this.latest.get(plan.target) === plan.key) hooks.failed(plan.target, error);
      } finally {
        this.release();
        if (this.running.get(plan.target) === record) this.running.delete(plan.target);
      }
    })();
    this.running.set(plan.target, record);
    return record.done;
  }

  private slot(): Promise<void> {
    if (this.active < this.concurrency) {
      this.active += 1;
      return Promise.resolve();
    }
    return new Promise((resolve) => {
      this.waiting.push(() => {
        this.active += 1;
        resolve();
      });
    });
  }

  private release(): void {
    this.active -= 1;
    this.waiting.shift()?.();
  }
}
