// VizForm's remembered fields: stored per tool, with values remembered
// under the older un-namespaced key still read.

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { VizForm } from './VizForm';
import type { VizDescriptor } from './viz_registry';

function descriptor(tool: string): VizDescriptor {
  return {
    path: ['viz', tool],
    label: tool,
    fields: [
      { name: 'input', label: 'Input', kind: 'text', remember: true, required: true },
      { name: 'note', label: 'Note', kind: 'text' },
    ],
  };
}

function mount(tool: string, onSubmit = vi.fn()) {
  const host = document.createElement('div');
  document.body.appendChild(host);
  const form = new VizForm({ descriptor: descriptor(tool), onSubmit });
  form.mount(host);
  const [input, note] = Array.from(host.querySelectorAll<HTMLInputElement>('input[type=text]'));
  return { form, host, input, note, onSubmit };
}

beforeEach(() => {
  window.localStorage.clear();
});

afterEach(() => {
  document.body.replaceChildren();
});

describe('VizForm remembered fields', () => {
  it('stores a remembered field under its tool, and only fields marked remember', async () => {
    const h = mount('spectra');
    h.input.value = '/data/run.raw';
    h.note.value = 'scratch';
    await h.form.submit();

    expect(h.onSubmit).toHaveBeenCalledWith({ input: '/data/run.raw', note: 'scratch' });
    expect(Object.keys(window.localStorage)).toEqual(['constellation.dashboard.viz.spectra.input']);
    expect(window.localStorage.getItem('constellation.dashboard.viz.spectra.input')).toBe('/data/run.raw');
  });

  it('recalls the value for the same tool and not for another with the same field name', async () => {
    const first = mount('spectra');
    first.input.value = '/data/run.raw';
    await first.form.submit();

    expect(mount('spectra').input.value).toBe('/data/run.raw');
    expect(mount('structure').input.value).toBe('');
  });

  it('reads a value remembered under the old shared key, then keeps its own', async () => {
    window.localStorage.setItem('constellation.dashboard.viz.input', '/old/path');
    const h = mount('spectra');
    expect(h.input.value).toBe('/old/path');

    h.input.value = '/new/path';
    await h.form.submit();
    // The old key is left as it was; the tool's own key now wins.
    expect(window.localStorage.getItem('constellation.dashboard.viz.input')).toBe('/old/path');
    expect(mount('spectra').input.value).toBe('/new/path');
  });

  it('lets a prefill win over anything remembered', () => {
    window.localStorage.setItem('constellation.dashboard.viz.spectra.input', '/remembered');
    const host = document.createElement('div');
    const form = new VizForm({ descriptor: descriptor('spectra'), prefill: { input: '/given' }, onSubmit: vi.fn() });
    form.mount(host);
    expect(host.querySelector<HTMLInputElement>('input[type=text]')!.value).toBe('/given');
  });
});
