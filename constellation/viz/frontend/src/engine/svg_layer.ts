// SVG layer primitives — small helpers for building track-local SVG
// without dragging in a framework. Each track's renderer takes a
// container <svg> and an `xScale`, then mutates the DOM.
//
// We don't use d3-selection's data-binding for the bulk path —
// React-style data joins are cute but overkill for a static-per-render
// track. Each `render(table, xScale)` call clears and rewrites; the
// surrounding GenomeBrowser orchestrates re-render cadence.

const SVG_NS = 'http://www.w3.org/2000/svg';

export function svgEl<K extends keyof SVGElementTagNameMap>(
  tag: K,
  attrs?: Record<string, string | number>,
): SVGElementTagNameMap[K] {
  const node = document.createElementNS(SVG_NS, tag);
  if (attrs) {
    for (const [key, value] of Object.entries(attrs)) {
      node.setAttribute(key, String(value));
    }
  }
  return node;
}

export function clear(el: Element): void {
  while (el.firstChild) {
    el.removeChild(el.firstChild);
  }
}

export function ensureSvg(
  host: HTMLElement,
  width: number,
  height: number,
): SVGSVGElement {
  let svg = host.querySelector('svg.track-canvas') as SVGSVGElement | null;
  if (!svg) {
    svg = svgEl('svg', {
      class: 'track-canvas',
      width,
      height,
    }) as SVGSVGElement;
    host.appendChild(svg);
  } else {
    svg.setAttribute('width', String(width));
    svg.setAttribute('height', String(height));
  }
  return svg;
}
