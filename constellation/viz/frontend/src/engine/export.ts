// Vector-export helper.
//
// Serializes the live composite SVG (ruler + every track's <svg>) into
// a single self-contained SVG document the user can save. When the
// composite contains <image> elements (hybrid-mode tracks), we leave
// them in place — the result is technically a hybrid SVG, not a pure
// vector one. The frontend distinguishes the two via the per-track
// mode and emits a "hybrid view" warning when the user clicks the
// vector-export button on a hybrid composition (see GenomeBrowser).
//
// Per-panel composition uses `svg.dataset.naturalHeight` (set by every
// renderer) so panels with stacked content (gene_annotation,
// read_pileup, cluster_pileup) that overflow their configured heightPx
// don't bleed into the panel below. Two modes:
//
//   clip=false (default) — total SVG height grows to fit the natural
//   content of every panel; the live browser still shows the configured
//   height. Best for publication output where users want every feature
//   captured.
//
//   clip=true — each panel is wrapped in a per-panel <clipPath> of the
//   configured heightPx, matching what the browser viewport shows.

const SVG_NS = 'http://www.w3.org/2000/svg';

export interface ExportOptions {
  title: string;
  trackPanels: HTMLElement[];
  rulerSvg: SVGSVGElement | null;
  totalWidthPx: number;
  /** Whether to clip each panel to its configured heightPx. When false
   *  (default) panels grow vertically to fit overflowing content. */
  clip?: boolean;
}

export interface CompositeResult {
  svg: string;
  totalHeightPx: number;
}

/** Build a composite SVG document from the live ruler + track panels.
 *  Returns both the serialized string and the total height (so the
 *  caller can verify the auto-grow case sized things correctly). */
export function buildCompositeSvg(opts: ExportOptions): CompositeResult {
  const clip = opts.clip === true;
  const composite = document.createElementNS(SVG_NS, 'svg');
  composite.setAttribute('xmlns', SVG_NS);
  composite.setAttribute('width', String(opts.totalWidthPx));

  const titleEl = document.createElementNS(SVG_NS, 'title');
  titleEl.textContent = opts.title;
  composite.appendChild(titleEl);

  const defs = document.createElementNS(SVG_NS, 'defs');
  composite.appendChild(defs);

  let yOffset = 0;
  if (opts.rulerSvg) {
    const rulerH = parseInt(opts.rulerSvg.getAttribute('height') ?? '0', 10);
    const cloned = opts.rulerSvg.cloneNode(true) as SVGSVGElement;
    const g = document.createElementNS(SVG_NS, 'g');
    g.setAttribute('transform', `translate(0 ${yOffset})`);
    while (cloned.firstChild) {
      g.appendChild(cloned.firstChild);
    }
    composite.appendChild(g);
    yOffset += rulerH;
  }

  let clipId = 0;
  for (const panel of opts.trackPanels) {
    const trackSvg = panel.querySelector(
      'svg.track-canvas',
    ) as SVGSVGElement | null;
    if (!trackSvg) continue;
    const configuredH = parseInt(trackSvg.getAttribute('height') ?? '0', 10);
    const naturalAttr = trackSvg.dataset.naturalHeight;
    const naturalH = naturalAttr !== undefined ? Number(naturalAttr) : NaN;
    const cloned = trackSvg.cloneNode(true) as SVGSVGElement;
    const g = document.createElementNS(SVG_NS, 'g');
    g.setAttribute('transform', `translate(0 ${yOffset})`);
    while (cloned.firstChild) {
      g.appendChild(cloned.firstChild);
    }
    if (clip) {
      const id = `panel-clip-${clipId++}`;
      const clipPath = document.createElementNS(SVG_NS, 'clipPath');
      clipPath.setAttribute('id', id);
      const rect = document.createElementNS(SVG_NS, 'rect');
      rect.setAttribute('x', '0');
      rect.setAttribute('y', '0');
      rect.setAttribute('width', String(opts.totalWidthPx));
      rect.setAttribute('height', String(configuredH));
      clipPath.appendChild(rect);
      defs.appendChild(clipPath);
      g.setAttribute('clip-path', `url(#${id})`);
      yOffset += configuredH;
    } else {
      const panelH = Number.isFinite(naturalH)
        ? Math.max(configuredH, Math.ceil(naturalH))
        : configuredH;
      yOffset += panelH;
    }
    composite.appendChild(g);
  }

  composite.setAttribute('height', String(yOffset));
  composite.setAttribute(
    'viewBox',
    `0 0 ${opts.totalWidthPx} ${yOffset}`,
  );

  const svgString = new XMLSerializer().serializeToString(composite);
  return { svg: svgString, totalHeightPx: yOffset };
}

/** One cell of a grid export: its SVG and where it goes. */
export interface ExportCell {
  svg: SVGSVGElement;
  x: number;
  y: number;
  width: number;
  height: number;
  /** Drawn above the cell when the grid has room for titles. */
  title?: string;
}

/** Height of a cell title row in a grid export. */
export const GRID_TITLE_PX = 16;

/** Prefix every id in `root` and rewrite the references to them, so the
 *  clip paths of several cells cannot collide in one document. */
function prefixIds(root: Element, prefix: string): void {
  const renamed = new Map<string, string>();
  for (const el of Array.from(root.querySelectorAll('[id]'))) {
    const next = `${prefix}${el.id}`;
    renamed.set(el.id, next);
    el.id = next;
  }
  if (renamed.size === 0) return;
  const rewrite = (value: string): string =>
    value.replace(/url\(#([^)]+)\)/g, (whole, id: string) => {
      const next = renamed.get(id);
      return next ? `url(#${next})` : whole;
    });
  for (const el of [root, ...Array.from(root.querySelectorAll('*'))]) {
    for (const attr of Array.from(el.attributes)) {
      if (attr.value.includes('url(#')) el.setAttribute(attr.name, rewrite(attr.value));
      else if ((attr.name === 'href' || attr.name === 'xlink:href') && attr.value.startsWith('#')) {
        const next = renamed.get(attr.value.slice(1));
        if (next) el.setAttribute(attr.name, `#${next}`);
      }
    }
  }
}

/** Compose cells laid out on a grid into one SVG document. Each cell is
 *  placed at its rectangle (shifted down by a title row when any cell has
 *  a title) and clipped to it. */
export function buildGridSvg(
  cells: readonly ExportCell[],
  opts: { title: string; widthPx: number; heightPx: number },
): CompositeResult {
  const titled = cells.some((c) => c.title);
  const titlePx = titled ? GRID_TITLE_PX : 0;
  // No explicit xmlns attribute: the serializer writes the namespace of an
  // element created in it, and setting it as well duplicates it in jsdom.
  const composite = document.createElementNS(SVG_NS, 'svg');
  // Text inherits the page's typeface: a standalone file has no stylesheet.
  composite.setAttribute('font-family', pageFontFamily());
  const titleEl = document.createElementNS(SVG_NS, 'title');
  titleEl.textContent = opts.title;
  composite.appendChild(titleEl);
  const defs = document.createElementNS(SVG_NS, 'defs');
  composite.appendChild(defs);

  let bottom = 0;
  let right = 0;
  cells.forEach((cell, i) => {
    const prefix = `c${i}-`;
    const y = cell.y + (titled ? rowOffset(cells, cell, titlePx) : 0);
    const clipId = `${prefix}cell`;
    const clipPath = document.createElementNS(SVG_NS, 'clipPath');
    clipPath.setAttribute('id', clipId);
    const rect = document.createElementNS(SVG_NS, 'rect');
    rect.setAttribute('x', '0');
    rect.setAttribute('y', '0');
    rect.setAttribute('width', String(cell.width));
    rect.setAttribute('height', String(cell.height));
    clipPath.appendChild(rect);
    defs.appendChild(clipPath);

    if (cell.title) {
      const text = document.createElementNS(SVG_NS, 'text');
      text.setAttribute('x', String(cell.x + 4));
      text.setAttribute('y', String(y - 4));
      text.setAttribute('class', 'export-cell-title');
      text.setAttribute('font-size', '11');
      text.setAttribute('fill', '#adacae');
      text.textContent = cell.title;
      composite.appendChild(text);
    }
    const cloned = cell.svg.cloneNode(true) as SVGSVGElement;
    prefixIds(cloned, prefix);
    const g = document.createElementNS(SVG_NS, 'g');
    g.setAttribute('transform', `translate(${cell.x} ${y})`);
    g.setAttribute('clip-path', `url(#${clipId})`);
    while (cloned.firstChild) g.appendChild(cloned.firstChild);
    composite.appendChild(g);
    bottom = Math.max(bottom, y + cell.height);
    right = Math.max(right, cell.x + cell.width);
  });

  const width = Math.max(opts.widthPx, right);
  const height = Math.max(opts.heightPx, bottom);
  composite.setAttribute('width', String(width));
  composite.setAttribute('height', String(height));
  composite.setAttribute('viewBox', `0 0 ${width} ${height}`);
  return { svg: new XMLSerializer().serializeToString(composite), totalHeightPx: height };
}

/** The typeface the page draws text in. */
function pageFontFamily(): string {
  const family = typeof getComputedStyle === 'function' ? getComputedStyle(document.body).fontFamily : '';
  return family || 'system-ui, sans-serif';
}

/** Title rows above a cell: one per distinct row start at or above it. */
function rowOffset(cells: readonly ExportCell[], cell: ExportCell, titlePx: number): number {
  const starts = new Set(cells.map((c) => c.y));
  let n = 0;
  for (const start of starts) if (start <= cell.y) n += 1;
  return n * titlePx;
}

export function downloadSvg(name: string, svgString: string): void {
  const blob = new Blob([svgString], { type: 'image/svg+xml' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = name;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

/** Quick element-count cost estimate for the cost-confirm dialog. */
export function estimateGlyphCount(panels: HTMLElement[]): number {
  let total = 0;
  for (const panel of panels) {
    const svg = panel.querySelector('svg.track-canvas');
    if (!svg) continue;
    total += svg.querySelectorAll(
      'rect, path, circle, line, polygon, polyline, text',
    ).length;
  }
  return total;
}
