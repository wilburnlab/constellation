// Anchored-popover primitives shared by every popover in the GUI
// (dataset manager, options, per-panel settings, file picker).
//
// A popover here is a fixed-position element owned by a widget and
// anchored to a button. These two helpers are the parts that were
// identical in each: where it goes, and when it closes itself.

/** Close a popover on Escape, or on a mousedown that lands outside both
 *  the popover and its anchor (a click on the anchor is left to the
 *  anchor's own toggle handler). Returns a function that removes the
 *  listeners; call it when the popover is disposed. */
export function attachDismiss(
  root: HTMLElement,
  anchor: HTMLElement,
  onClose: () => void,
): () => void {
  const onMouseDown = (e: MouseEvent): void => {
    const target = e.target as Node;
    if (root.contains(target)) return;
    if (anchor.contains(target)) return;
    onClose();
  };
  const onKeyDown = (e: KeyboardEvent): void => {
    if (e.key === 'Escape') onClose();
  };
  document.addEventListener('mousedown', onMouseDown);
  document.addEventListener('keydown', onKeyDown);
  return () => {
    document.removeEventListener('mousedown', onMouseDown);
    document.removeEventListener('keydown', onKeyDown);
  };
}

export interface BelowRightOptions {
  /** Smallest allowed distance from the viewport's right edge. Unset
   *  leaves the popover exactly under the anchor even when that is
   *  partly off-screen. */
  minRightPx?: number;
}

/** Pin `root` just below `anchor` with their right edges aligned, so the
 *  popover grows leftward from a toolbar or header button. */
export function positionBelowRight(
  root: HTMLElement,
  anchor: HTMLElement,
  options: BelowRightOptions = {},
): void {
  const rect = anchor.getBoundingClientRect();
  const right = window.innerWidth - rect.right;
  root.style.position = 'fixed';
  root.style.top = `${rect.bottom + 4}px`;
  root.style.right = `${
    options.minRightPx === undefined ? right : Math.max(options.minRightPx, right)
  }px`;
  root.style.zIndex = '30';
}
