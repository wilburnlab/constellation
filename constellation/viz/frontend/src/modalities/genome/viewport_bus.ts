// Shared viewport + cross-track event bus.
//
// The genome browser's load-bearing shared state is the locus
// (`{contig, start, end}`). Pan/zoom interactions update the locus and
// every track re-renders against the new range. The events travel on the
// engine's `TypedBus`, the channel every browser uses; this class adds the
// locus it carries, so setting an unchanged locus emits nothing.

import { TypedBus } from '../../engine/bus';

export interface Locus {
  contig: string;
  start: number;
  end: number;
}

export interface ViewportEvents {
  'locus:changed': Locus;
  'selection:feature': { kind: string; binding: string; id: number | string };
}

export class ViewportBus {
  private readonly bus = new TypedBus<ViewportEvents>();
  private locusInternal: Locus = { contig: '', start: 0, end: 1 };

  get locus(): Locus {
    return this.locusInternal;
  }

  setLocus(next: Locus): void {
    if (
      next.contig === this.locusInternal.contig &&
      next.start === this.locusInternal.start &&
      next.end === this.locusInternal.end
    ) {
      return;
    }
    this.locusInternal = next;
    this.bus.emit('locus:changed', next);
  }

  emitFeatureSelection(detail: ViewportEvents['selection:feature']): void {
    this.bus.emit('selection:feature', detail);
  }

  on<K extends keyof ViewportEvents>(
    event: K,
    handler: (detail: ViewportEvents[K]) => void,
  ): () => void {
    return this.bus.on(event, handler);
  }
}
