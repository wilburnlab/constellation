// Vector-export helper.
//
// Serializes the live composite SVG (ruler + every track's <svg>) into
// a single self-contained SVG document the user can save. When the
// composite contains <image> elements (hybrid-mode tracks), we leave
// them in place — the result is technically a hybrid SVG, not a pure
// vector one. The frontend distinguishes the two via the per-track
// mode and emits a "hybrid view" warning when the user clicks the
// vector-export button on a hybrid composition (see GenomeBrowser).

const SVG_NS = 'http://www.w3.org/2000/svg';

export interface ExportOptions {
  title: string;
  trackPanels: HTMLElement[];
  rulerSvg: SVGSVGElement | null;
  totalWidthPx: number;
  /** Total height; computed as sum of ruler + track heights. */
  totalHeightPx: number;
}

export function buildCompositeSvg(opts: ExportOptions): string {
  const composite = document.createElementNS(SVG_NS, 'svg');
  composite.setAttribute('xmlns', SVG_NS);
  composite.setAttribute('width', String(opts.totalWidthPx));
  composite.setAttribute('height', String(opts.totalHeightPx));
  composite.setAttribute(
    'viewBox',
    `0 0 ${opts.totalWidthPx} ${opts.totalHeightPx}`,
  );

  const titleEl = document.createElementNS(SVG_NS, 'title');
  titleEl.textContent = opts.title;
  composite.appendChild(titleEl);

  let yOffset = 0;
  if (opts.rulerSvg) {
    const cloned = opts.rulerSvg.cloneNode(true) as SVGSVGElement;
    const g = document.createElementNS(SVG_NS, 'g');
    g.setAttribute('transform', `translate(0 ${yOffset})`);
    while (cloned.firstChild) {
      g.appendChild(cloned.firstChild);
    }
    composite.appendChild(g);
    yOffset += parseInt(opts.rulerSvg.getAttribute('height') ?? '0', 10);
  }

  for (const panel of opts.trackPanels) {
    const trackSvg = panel.querySelector('svg.track-canvas') as SVGSVGElement | null;
    if (!trackSvg) continue;
    const cloned = trackSvg.cloneNode(true) as SVGSVGElement;
    const g = document.createElementNS(SVG_NS, 'g');
    g.setAttribute('transform', `translate(0 ${yOffset})`);
    while (cloned.firstChild) {
      g.appendChild(cloned.firstChild);
    }
    composite.appendChild(g);
    yOffset += parseInt(trackSvg.getAttribute('height') ?? '0', 10);
  }

  return new XMLSerializer().serializeToString(composite);
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
