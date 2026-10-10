import { describe, expect, it } from 'vitest';
import { buildCompositeSvg, estimateGlyphCount } from './export';
import { ensureSvg, svgEl } from './svg_layer';

const WIDTH = 900;

/** A track panel shaped like the browser's: div > svg.track-canvas. */
function panel(heightPx: number, naturalHeight: number | null, glyphs = 1): HTMLElement {
  const host = document.createElement('div');
  const svg = ensureSvg(host, WIDTH, heightPx);
  for (let i = 0; i < glyphs; i++) {
    svg.appendChild(svgEl('rect', { x: i, y: 0, width: 1, height: 1 }));
  }
  if (naturalHeight !== null) svg.dataset.naturalHeight = String(naturalHeight);
  return host;
}

function ruler(heightPx: number): SVGSVGElement {
  const host = document.createElement('div');
  const svg = ensureSvg(host, WIDTH, heightPx);
  svg.appendChild(svgEl('g', { transform: 'translate(0 4)' }));
  return svg;
}

function count(haystack: string, needle: string): number {
  return haystack.split(needle).length - 1;
}

describe('buildCompositeSvg', () => {
  it('grows each panel to its natural height when clip is off', () => {
    const { svg, totalHeightPx } = buildCompositeSvg({
      title: 'chr1:0-100',
      trackPanels: [panel(80, 130), panel(60, 40)],
      rulerSvg: ruler(28),
      totalWidthPx: WIDTH,
    });
    // ruler 28 + max(80, 130) + max(60, 40)
    expect(totalHeightPx).toBe(28 + 130 + 60);
    expect(svg).toContain(`height="${totalHeightPx}"`);
    expect(svg).toContain(`viewBox="0 0 ${WIDTH} ${totalHeightPx}"`);
    expect(svg).toContain('<title>chr1:0-100</title>');
    expect(svg).toContain('translate(0 0)');
    expect(svg).toContain('translate(0 28)');
    expect(svg).toContain('translate(0 158)');
    expect(count(svg, '<clipPath')).toBe(0);
  });

  it('rounds a fractional natural height up', () => {
    const { totalHeightPx } = buildCompositeSvg({
      title: 't',
      trackPanels: [panel(50, 50.2)],
      rulerSvg: null,
      totalWidthPx: WIDTH,
    });
    expect(totalHeightPx).toBe(51);
  });

  it('clips each panel to its configured height when clip is on', () => {
    const { svg, totalHeightPx } = buildCompositeSvg({
      title: 't',
      trackPanels: [panel(80, 130), panel(60, 40)],
      rulerSvg: ruler(28),
      totalWidthPx: WIDTH,
      clip: true,
    });
    expect(totalHeightPx).toBe(28 + 80 + 60);
    expect(count(svg, '<clipPath')).toBe(2);
    expect(svg).toContain('clip-path="url(#panel-clip-0)"');
    expect(svg).toContain('clip-path="url(#panel-clip-1)"');
    expect(svg).toContain(`width="${WIDTH}" height="80"`);
    expect(svg).toContain(`width="${WIDTH}" height="60"`);
    // The ruler is never clipped.
    expect(count(svg, 'clip-path=')).toBe(2);
  });

  it('uses the configured height when a renderer left naturalHeight unset', () => {
    const { totalHeightPx } = buildCompositeSvg({
      title: 't',
      trackPanels: [panel(70, null)],
      rulerSvg: null,
      totalWidthPx: WIDTH,
    });
    expect(totalHeightPx).toBe(70);
  });

  it('skips panels with no track svg (collapsed tracks)', () => {
    const collapsed = document.createElement('div');
    const { svg, totalHeightPx } = buildCompositeSvg({
      title: 't',
      trackPanels: [collapsed, panel(40, 40, 3)],
      rulerSvg: null,
      totalWidthPx: WIDTH,
    });
    expect(totalHeightPx).toBe(40);
    expect(count(svg, '<rect')).toBe(3);
  });

  it('moves glyphs into the composite without touching the live panels', () => {
    const live = panel(40, 40, 5);
    buildCompositeSvg({
      title: 't',
      trackPanels: [live],
      rulerSvg: null,
      totalWidthPx: WIDTH,
    });
    expect(live.querySelectorAll('rect').length).toBe(5);
  });
});

describe('estimateGlyphCount', () => {
  it('counts drawable elements across panels and ignores groups', () => {
    const a = panel(40, 40, 4);
    const b = panel(40, 40, 2);
    const svg = b.querySelector('svg')!;
    svg.appendChild(svgEl('g'));
    svg.appendChild(svgEl('text'));
    svg.appendChild(svgEl('path'));
    expect(estimateGlyphCount([a, b, document.createElement('div')])).toBe(4 + 2 + 2);
  });
});
