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
