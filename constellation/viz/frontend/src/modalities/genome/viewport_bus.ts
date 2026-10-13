// Shared viewport + cross-track event bus.
//
// The genome browser's load-bearing shared state is the locus
// (`{contig, start, end}`). Pan/zoom interactions update the locus and
// every track re-renders against the new range. The bus is a thin
// wrapper around the standard `EventTarget` so future cross-modality
// panels (Vitessce-style coordination) can subscribe to the same
// channel without depending on a frontend framework.

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
  private readonly bus = new EventTarget();
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
    this.bus.dispatchEvent(
      new CustomEvent('locus:changed', { detail: next }),
    );
  }

  emitFeatureSelection(detail: ViewportEvents['selection:feature']): void {
    this.bus.dispatchEvent(
      new CustomEvent('selection:feature', { detail }),
    );
  }

  on<K extends keyof ViewportEvents>(
    event: K,
    handler: (detail: ViewportEvents[K]) => void,
  ): () => void {
    const listener = (e: Event) => {
      const ce = e as CustomEvent<ViewportEvents[K]>;
      handler(ce.detail);
    };
    this.bus.addEventListener(event, listener as EventListener);
    return () => this.bus.removeEventListener(event, listener as EventListener);
  }
}
