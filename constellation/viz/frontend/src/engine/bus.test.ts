import { describe, expect, it, vi } from 'vitest';
import { TypedBus } from './bus';

interface Events {
  moved: { at: number };
  cleared: null;
}

describe('TypedBus', () => {
  it('delivers to subscribers in order and unsubscribes', () => {
    const bus = new TypedBus<Events>();
    const seen: string[] = [];
    const off = bus.on('moved', (p) => seen.push(`a${p.at}`));
    bus.on('moved', (p) => seen.push(`b${p.at}`));
    bus.emit('moved', { at: 1 });
    off();
    bus.emit('moved', { at: 2 });
    expect(seen).toEqual(['a1', 'b1', 'b2']);
    expect(bus.size).toBe(1);
  });

  it('keeps going when a handler throws', () => {
    const bus = new TypedBus<Events>();
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    const after = vi.fn();
    bus.on('cleared', () => {
      throw new Error('boom');
    });
    bus.on('cleared', after);
    bus.emit('cleared', null);
    expect(after).toHaveBeenCalledOnce();
    warn.mockRestore();
  });

  it('forgets everything on dispose', () => {
    const bus = new TypedBus<Events>();
    bus.on('moved', () => {});
    bus.dispose();
    expect(bus.size).toBe(0);
  });
});
