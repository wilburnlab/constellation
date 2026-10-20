import { describe, expect, it } from 'vitest';
import { FetchPlan, RenderHooks, RenderScheduler } from './scheduler';

interface Deferred<T> {
  promise: Promise<T>;
  resolve(value: T): void;
  reject(err: Error): void;
}

function deferred<T>(): Deferred<T> {
  let resolve!: (v: T) => void;
  let reject!: (e: Error) => void;
  const promise = new Promise<T>((res, rej) => {
    resolve = res;
    reject = rej;
  });
  return { promise, resolve, reject };
}

/** A fake panel with a cache, and a log of what happened to it. */
class Target {
  cache?: { key: string; data: string };
  log: string[] = [];
  constructor(readonly name: string) {}
}

function hooks(): RenderHooks<Target, string> {
  return {
    cached: (t) => t.cache,
    store: (t, key, data) => {
      t.cache = { key, data };
    },
    draw: (t, data) => t.log.push(`draw ${data}`),
    loading: (t) => t.log.push('loading'),
    failed: (t, err) => t.log.push(`failed ${err.message}`),
    empty: (t) => t.log.push('empty'),
  };
}

/** A plan whose fetch waits on `gate`, recording each request. */
function plan(
  target: Target,
  key: string | null,
  gate: Deferred<string>,
  requests: string[],
): FetchPlan<Target, string> {
  return {
    target,
    key,
    fetch: (signal) => {
      requests.push(`${target.name}:${key}`);
      return new Promise<string>((resolve, reject) => {
        signal.addEventListener('abort', () => reject(Object.assign(new Error('aborted'), { name: 'AbortError' })));
        gate.promise.then(resolve, reject);
      });
    },
  };
}

describe('RenderScheduler', () => {
  it('fetches panels in parallel and draws each when it arrives', async () => {
    const s = new RenderScheduler<Target, string>();
    const a = new Target('a');
    const b = new Target('b');
    const ga = deferred<string>();
    const gb = deferred<string>();
    const requests: string[] = [];
    const pass = s.pass([plan(a, 'k1', ga, requests), plan(b, 'k1', gb, requests)], hooks());
    await Promise.resolve();
    expect(requests).toEqual(['a:k1', 'b:k1']);
    gb.resolve('B');
    ga.resolve('A');
    await pass;
    expect(a.log).toEqual(['loading', 'draw A']);
    expect(b.log).toEqual(['loading', 'draw B']);
    expect(a.cache).toEqual({ key: 'k1', data: 'A' });
  });

  it('draws from the cache when the key has not changed', async () => {
    const s = new RenderScheduler<Target, string>();
    const a = new Target('a');
    a.cache = { key: 'k1', data: 'cached' };
    const requests: string[] = [];
    await s.pass([plan(a, 'k1', deferred(), requests)], hooks());
    expect(requests).toEqual([]);
    expect(a.log).toEqual(['draw cached']);
  });

  it('drops a slow result for a view that has been left (latest wins)', async () => {
    const s = new RenderScheduler<Target, string>();
    const a = new Target('a');
    const slow = deferred<string>();
    const fast = deferred<string>();
    const requests: string[] = [];
    const first = s.pass([plan(a, 'old', slow, requests)], hooks());
    const second = s.pass([plan(a, 'new', fast, requests)], hooks());
    fast.resolve('NEW');
    slow.resolve('OLD');
    await Promise.all([first, second]);
    expect(a.log).toEqual(['loading', 'loading', 'draw NEW']);
    expect(a.cache?.key).toBe('new');
  });

  it('keeps a request already running for the same key', async () => {
    const s = new RenderScheduler<Target, string>();
    const a = new Target('a');
    const gate = deferred<string>();
    const requests: string[] = [];
    const first = s.pass([plan(a, 'k', gate, requests)], hooks());
    const second = s.pass([plan(a, 'k', gate, requests)], hooks());
    gate.resolve('K');
    await Promise.all([first, second]);
    expect(requests).toEqual(['a:k']);
    expect(a.log).toEqual(['loading', 'draw K']);
  });

  it('a null key aborts and shows nothing', async () => {
    const s = new RenderScheduler<Target, string>();
    const a = new Target('a');
    const gate = deferred<string>();
    const requests: string[] = [];
    const first = s.pass([plan(a, 'k', gate, requests)], hooks());
    await s.pass([plan(a, null, deferred(), requests)], hooks());
    gate.resolve('late');
    await first;
    expect(a.log).toEqual(['loading', 'empty']);
  });

  it('reports a failure only for the current key', async () => {
    const s = new RenderScheduler<Target, string>();
    const a = new Target('a');
    const gate = deferred<string>();
    const requests: string[] = [];
    const pass = s.pass([plan(a, 'k', gate, requests)], hooks());
    gate.reject(new Error('500'));
    await pass;
    expect(a.log).toEqual(['loading', 'failed 500']);
  });

  it('runs no more than its concurrency at once', async () => {
    const s = new RenderScheduler<Target, string>({ concurrency: 2 });
    const targets = ['a', 'b', 'c'].map((n) => new Target(n));
    const gates = targets.map(() => deferred<string>());
    const requests: string[] = [];
    const pass = s.pass(targets.map((t, i) => plan(t, 'k', gates[i], requests)), hooks());
    await Promise.resolve();
    await Promise.resolve();
    expect(requests).toEqual(['a:k', 'b:k']);
    gates[0].resolve('A');
    await gates[0].promise;
    await new Promise((r) => setTimeout(r, 0));
    expect(requests).toEqual(['a:k', 'b:k', 'c:k']);
    gates[1].resolve('B');
    gates[2].resolve('C');
    await pass;
    expect(targets.map((t) => t.log.at(-1))).toEqual(['draw A', 'draw B', 'draw C']);
  });
});
