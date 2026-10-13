// Panel — one panel of a browser: its chrome, its state, and the last
// data it was drawn from.
//
// The chrome is a header (drag grip, collapse, label, status line, gear,
// hide), a body that holds the panel's SVG, and a bottom resize handle.
// A panel does not know what it shows or how it is arranged with other
// panels; a container (`PanelStack` for a vertical stack) wires the
// buttons and decides order and visibility. A different container can
// lay the same panels out differently.
//
// The class names are the ones the panel CSS has always used.

import { Bag } from './settings_schema';
import { LayoutPanel } from './layout';
import './panels.css';

/** How the server lists one renderable thing. */
export interface PanelEntry {
  kind: string;
  binding_id: string;
  label: string;
  /** The attached source it belongs to, if any. */
  source_id: string | null;
}

export interface PanelInit {
  entry: PanelEntry;
  meta: Bag;
  visible: boolean;
  collapsed: boolean;
  heightPx: number;
  displayOrder: number;
  style: Record<string, unknown>;
  filter: Record<string, unknown>;
}

export class Panel<D = unknown> implements LayoutPanel {
  readonly entry: PanelEntry;
  readonly meta: Bag;

  /** The panel's root element. */
  readonly element: HTMLElement;
  readonly headerLabelEl: HTMLElement;
  readonly statusEl: HTMLElement;
  /** Holds the panel's `svg.track-canvas`. */
  readonly bodyHost: HTMLElement;
  readonly collapseBtn: HTMLButtonElement;
  readonly settingsBtn: HTMLButtonElement;
  readonly hideBtn: HTMLButtonElement;
  readonly resizeHandle: HTMLElement;

  visible: boolean;
  collapsed: boolean;
  heightPx: number;
  displayOrder: number;
  style: Record<string, unknown>;
  filter: Record<string, unknown>;

  /** The last data successfully fetched and the view it was fetched for.
   *  A style or client-side filter change redraws from this without a
   *  request, as long as the view has not moved. */
  lastFetched?: { data: D; viewKey: string };
  /** Aborts this panel's in-flight fetch. */
  cancel?: AbortController;

  constructor(init: PanelInit) {
    this.entry = init.entry;
    this.meta = init.meta;
    this.visible = init.visible;
    this.collapsed = init.collapsed;
    this.heightPx = init.heightPx;
    this.displayOrder = init.displayOrder;
    this.style = { ...init.style };
    this.filter = { ...init.filter };

    const panel = document.createElement('div');
    panel.className = 'track';
    panel.dataset.bindingId = init.entry.binding_id;
    panel.dataset.kind = init.entry.kind;
    panel.draggable = true;

    const header = document.createElement('div');
    header.className = 'track-header';

    const dragHandle = document.createElement('span');
    dragHandle.className = 'track-handle';
    dragHandle.textContent = '⋮⋮';
    dragHandle.title = 'Drag to reorder';

    const collapseBtn = document.createElement('button');
    collapseBtn.type = 'button';
    collapseBtn.className = 'track-collapse-btn';

    const labelEl = document.createElement('span');
    labelEl.className = 'track-header-label';
    labelEl.textContent = init.entry.label;

    const statusEl = document.createElement('span');
    statusEl.className = 'track-header-status';
    statusEl.textContent = '—';

    const settingsBtn = document.createElement('button');
    settingsBtn.type = 'button';
    settingsBtn.className = 'track-settings-btn';
    settingsBtn.textContent = '⚙';
    settingsBtn.title = 'Style and filter controls';

    const hideBtn = document.createElement('button');
    hideBtn.type = 'button';
    hideBtn.className = 'track-hide-btn';
    hideBtn.textContent = '👁';
    hideBtn.title = 'Hide track (re-enable from the Datasets menu)';

    header.appendChild(dragHandle);
    header.appendChild(collapseBtn);
    header.appendChild(labelEl);
    header.appendChild(statusEl);
    header.appendChild(settingsBtn);
    header.appendChild(hideBtn);

    const body = document.createElement('div');
    body.className = 'track-body';

    const resizeHandle = document.createElement('div');
    resizeHandle.className = 'track-resize-handle';
    resizeHandle.title = 'Drag to resize';

    panel.appendChild(header);
    panel.appendChild(body);
    panel.appendChild(resizeHandle);

    this.element = panel;
    this.headerLabelEl = labelEl;
    this.statusEl = statusEl;
    this.bodyHost = body;
    this.collapseBtn = collapseBtn;
    this.settingsBtn = settingsBtn;
    this.hideBtn = hideBtn;
    this.resizeHandle = resizeHandle;
    this.syncCollapseButton();
  }

  /** Bring the collapse button in line with `collapsed`. */
  syncCollapseButton(): void {
    this.collapseBtn.textContent = this.collapsed ? '▸' : '▾';
    this.collapseBtn.title = this.collapsed ? 'Expand track' : 'Collapse track';
  }
}
