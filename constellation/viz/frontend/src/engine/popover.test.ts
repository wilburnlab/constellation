import { afterEach, describe, expect, it, vi } from 'vitest';
import { attachDismiss, positionBelowRight } from './popover';

function mount(): { root: HTMLElement; anchor: HTMLElement; outside: HTMLElement } {
  const root = document.createElement('div');
  const inner = document.createElement('span');
  root.appendChild(inner);
  const anchor = document.createElement('button');
  const outside = document.createElement('p');
  document.body.append(root, anchor, outside);
  return { root, anchor, outside };
}

afterEach(() => {
  document.body.replaceChildren();
  vi.restoreAllMocks();
});

describe('attachDismiss', () => {
  it('closes on a mousedown outside the popover and its anchor', () => {
    const { root, anchor, outside } = mount();
    const onClose = vi.fn();
    attachDismiss(root, anchor, onClose);

    root.firstElementChild!.dispatchEvent(new MouseEvent('mousedown', { bubbles: true }));
    anchor.dispatchEvent(new MouseEvent('mousedown', { bubbles: true }));
    expect(onClose).not.toHaveBeenCalled();

    outside.dispatchEvent(new MouseEvent('mousedown', { bubbles: true }));
    expect(onClose).toHaveBeenCalledTimes(1);
  });

  it('closes on Escape and ignores other keys', () => {
    const { root, anchor } = mount();
    const onClose = vi.fn();
    attachDismiss(root, anchor, onClose);
    document.dispatchEvent(new KeyboardEvent('keydown', { key: 'Enter' }));
    expect(onClose).not.toHaveBeenCalled();
    document.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape' }));
    expect(onClose).toHaveBeenCalledTimes(1);
  });

  it('stops listening once detached', () => {
    const { root, anchor, outside } = mount();
    const onClose = vi.fn();
    const detach = attachDismiss(root, anchor, onClose);
    detach();
    outside.dispatchEvent(new MouseEvent('mousedown', { bubbles: true }));
    document.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape' }));
    expect(onClose).not.toHaveBeenCalled();
  });
});

describe('positionBelowRight', () => {
  function anchorAt(right: number, bottom: number): HTMLElement {
    const anchor = document.createElement('button');
    vi.spyOn(anchor, 'getBoundingClientRect').mockReturnValue({
      right,
      bottom,
      left: right - 80,
      top: bottom - 24,
      width: 80,
      height: 24,
      x: right - 80,
      y: bottom - 24,
      toJSON: () => ({}),
    });
    return anchor;
  }

  it('pins the popover 4px below the anchor with right edges aligned', () => {
    vi.spyOn(window, 'innerWidth', 'get').mockReturnValue(1000);
    const root = document.createElement('div');
    positionBelowRight(root, anchorAt(900, 40));
    expect(root.style.position).toBe('fixed');
    expect(root.style.top).toBe('44px');
    expect(root.style.right).toBe('100px');
    expect(root.style.zIndex).toBe('30');
  });

  it('follows an anchor past the right edge unless a minimum is given', () => {
    vi.spyOn(window, 'innerWidth', 'get').mockReturnValue(1000);
    const free = document.createElement('div');
    positionBelowRight(free, anchorAt(1010, 40));
    expect(free.style.right).toBe('-10px');

    const clamped = document.createElement('div');
    positionBelowRight(clamped, anchorAt(1010, 40), { minRightPx: 8 });
    expect(clamped.style.right).toBe('8px');
    positionBelowRight(clamped, anchorAt(900, 40), { minRightPx: 8 });
    expect(clamped.style.right).toBe('100px');
  });
});
