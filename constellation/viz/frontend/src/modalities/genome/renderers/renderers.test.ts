// Renderer parity snapshots.
//
// Each case feeds a renderer the Arrow table its Python kernel really
// emits (fixtures under `__fixtures__/genome/`) and pins the SVG it
// draws. The snapshots are standalone `.svg` files — open one in a
// browser to see the track. A refactor that is meant to preserve drawn
// output must leave every file here byte-identical.

import { describe, expect, it } from 'vitest';
import type { Table } from 'apache-arrow';
import type { TrackMode } from '../../../engine/arrow_client';
import { xScale } from '../scales';
import { ensureSvg } from '../../../engine/svg_layer';
import { loadFixture, metadataFor } from '../__fixtures__/load';
import { getRenderer, registeredKinds } from '.';

interface Case {
  name: string;
  kind: string;
  fixture: string;
  mode?: TrackMode;
  domain: [number, number];
  widthPx?: number;
  heightPx?: number;
  showLabels?: boolean;
  style?: Record<string, unknown>;
  filter?: Record<string, unknown>;
  /** Draw from a zero-row slice of the fixture (empty-state path). */
  empty?: boolean;
}

function draw(c: Case): SVGSVGElement {
  const renderer = getRenderer(c.kind);
  if (!renderer) throw new Error(`no renderer for ${c.kind}`);
  const full: Table = loadFixture(c.fixture);
  const table = c.empty ? full.slice(0, 0) : full;
  const widthPx = c.widthPx ?? 1000;
  const heightPx = c.heightPx ?? 120;
  const host = document.createElement('div');
  const svg = ensureSvg(host, widthPx, heightPx);
  renderer.render(table, c.mode ?? 'vector', {
    svg,
    widthPx,
    heightPx,
    xScale: xScale(c.domain, widthPx),
    meta: metadataFor(c.kind),
    showLabels: c.showLabels,
    style: c.style,
    filter: c.filter,
  });
  return svg;
}

/** One element per line, so a snapshot diff points at the glyph that moved. */
function serialize(svg: SVGSVGElement): string {
  return `${svg.outerHTML.replace(/></g, '>\n<')}\n`;
}

const CASES: Case[] = [
  // --- reference_sequence -------------------------------------------
  { name: 'reference_sequence.letters', kind: 'reference_sequence',
    fixture: 'reference_sequence.letters', domain: [0, 48], widthPx: 800, heightPx: 24 },
  { name: 'reference_sequence.blocks', kind: 'reference_sequence',
    fixture: 'reference_sequence.letters', domain: [0, 48], widthPx: 200, heightPx: 24 },
  { name: 'reference_sequence.decimated', kind: 'reference_sequence',
    fixture: 'reference_sequence.decimated', domain: [0, 12_000], heightPx: 24 },
  { name: 'reference_sequence.styled', kind: 'reference_sequence',
    fixture: 'reference_sequence.letters', domain: [0, 48], widthPx: 800, heightPx: 24,
    style: {
      'palette.A': '#ff0000',
      palette: { N: '#00ff00' },
      letter_font_family: 'Courier New',
      letter_font_size_px: 14,
    } },
  { name: 'reference_sequence.threshold-raised', kind: 'reference_sequence',
    fixture: 'reference_sequence.letters', domain: [0, 48], widthPx: 800, heightPx: 24,
    style: { letter_threshold_px_per_bp: 30 } },
  { name: 'reference_sequence.empty', kind: 'reference_sequence',
    fixture: 'reference_sequence.letters', domain: [0, 48], heightPx: 24, empty: true },

  // --- gene_annotation ----------------------------------------------
  { name: 'gene_annotation.default', kind: 'gene_annotation',
    fixture: 'gene_annotation', domain: [0, 3000], heightPx: 60 },
  { name: 'gene_annotation.labels-off', kind: 'gene_annotation',
    fixture: 'gene_annotation', domain: [0, 3000], heightPx: 60, showLabels: false },
  { name: 'gene_annotation.styled', kind: 'gene_annotation',
    fixture: 'gene_annotation', domain: [0, 3000], heightPx: 60, showLabels: false,
    style: {
      row_height_px: 20,
      feature_opacity: 0.5,
      label_font_family: 'Georgia',
      label_font_size_px: 12,
      label_min_width_px: 0,
      strand_chevron_min_width_px: 40,
      show_labels: true,
      'palette.gene': '#123456',
      'palette.CDS': '#abcdef',
    } },
  { name: 'gene_annotation.chevrons-off', kind: 'gene_annotation',
    fixture: 'gene_annotation', domain: [0, 3000], heightPx: 60,
    style: { show_chevrons: false, show_labels: false } },
  { name: 'gene_annotation.filtered', kind: 'gene_annotation',
    fixture: 'gene_annotation', domain: [0, 3000], heightPx: 60,
    // `visible_sources` is matched against each feature's own `source`
    // column (the GFF source, "RefSeq" here), not the binding's
    // reference/derived tag.
    filter: {
      visible_types: ['gene', 'exon', 'pseudogene'],
      visible_strands: ['+'],
      visible_sources: ['RefSeq'],
      min_length_bp: 250,
    } },
  { name: 'gene_annotation.source-filter-miss', kind: 'gene_annotation',
    fixture: 'gene_annotation', domain: [0, 3000], heightPx: 60,
    filter: { visible_sources: ['reference'] } },
  { name: 'gene_annotation.empty', kind: 'gene_annotation',
    fixture: 'gene_annotation', domain: [0, 3000], heightPx: 60, empty: true },

  // --- coverage_histogram -------------------------------------------
  { name: 'coverage_histogram.default', kind: 'coverage_histogram',
    fixture: 'coverage_histogram', domain: [0, 600], widthPx: 600, heightPx: 80 },
  { name: 'coverage_histogram.styled', kind: 'coverage_histogram',
    fixture: 'coverage_histogram', domain: [0, 600], widthPx: 600, heightPx: 80,
    style: {
      y_scale: 'log',
      fill_opacity: 0.8,
      stroke_width_px: 2,
      show_sample_labels: false,
      show_max_depth: true,
      'palette.1': '#00aa00',
    } },
  { name: 'coverage_histogram.annotations-off', kind: 'coverage_histogram',
    fixture: 'coverage_histogram', domain: [0, 600], widthPx: 600, heightPx: 80,
    style: { show_max_depth: false, show_sample_labels: false } },
  { name: 'coverage_histogram.filtered', kind: 'coverage_histogram',
    fixture: 'coverage_histogram', domain: [0, 600], widthPx: 600, heightPx: 80,
    filter: { visible_samples: [1], min_depth: 5 } },
  { name: 'coverage_histogram.filtered-to-nothing', kind: 'coverage_histogram',
    fixture: 'coverage_histogram', domain: [0, 600], widthPx: 600, heightPx: 80,
    filter: { min_depth: 1000 } },
  { name: 'coverage_histogram.empty', kind: 'coverage_histogram',
    fixture: 'coverage_histogram', domain: [0, 600], widthPx: 600, heightPx: 80, empty: true },

  // --- read_pileup --------------------------------------------------
  { name: 'read_pileup.default', kind: 'read_pileup',
    fixture: 'read_pileup', domain: [0, 1000], heightPx: 240 },
  { name: 'read_pileup.styled', kind: 'read_pileup',
    fixture: 'read_pileup', domain: [0, 1000], heightPx: 240,
    style: {
      'palette.1': '#008080',
      'palette.+': '#101010',
      'palette.mismatch': '#ff00ff',
      'palette.intron': '#0000ff',
      intron_stroke_dasharray: '4,1',
      intron_stroke_width_px: 2,
      mismatch_glyph_size_px: 10,
      min_row_height_px: 6,
      max_row_height_px: 12,
      read_opacity: 0.6,
    } },
  { name: 'read_pileup.filtered', kind: 'read_pileup',
    fixture: 'read_pileup', domain: [0, 1000], heightPx: 240,
    filter: { visible_samples: [1], visible_strands: ['+'] } },
  { name: 'read_pileup.filtered-to-nothing', kind: 'read_pileup',
    fixture: 'read_pileup', domain: [0, 1000], heightPx: 240,
    filter: { visible_strands: [] } },
  { name: 'read_pileup.short-panel', kind: 'read_pileup',
    fixture: 'read_pileup', domain: [0, 1000], heightPx: 24 },
  { name: 'read_pileup.hybrid', kind: 'read_pileup',
    fixture: 'hybrid_frame', mode: 'hybrid', domain: [0, 1000], heightPx: 240 },
  { name: 'read_pileup.empty', kind: 'read_pileup',
    fixture: 'read_pileup', domain: [0, 1000], heightPx: 240, empty: true },

  // --- cluster_pileup -----------------------------------------------
  { name: 'cluster_pileup.clusters.default', kind: 'cluster_pileup',
    fixture: 'cluster_pileup.clusters', domain: [0, 1000], heightPx: 200 },
  { name: 'cluster_pileup.clusters.styled', kind: 'cluster_pileup',
    fixture: 'cluster_pileup.clusters', domain: [0, 1000], heightPx: 200,
    style: {
      'palette.em': '#aa00aa',
      min_row_height_px: 8,
      max_row_height_px: 16,
      opacity_min: 0.1,
      opacity_max: 0.9,
    } },
  { name: 'cluster_pileup.clusters.filtered', kind: 'cluster_pileup',
    fixture: 'cluster_pileup.clusters', domain: [0, 1000], heightPx: 200,
    filter: { visible_modes: ['genome', 'em'], visible_strands: ['+'], min_reads: 2 } },
  { name: 'cluster_pileup.members.default', kind: 'cluster_pileup',
    fixture: 'cluster_pileup.members', domain: [0, 1000], heightPx: 200 },
  { name: 'cluster_pileup.members.filtered', kind: 'cluster_pileup',
    fixture: 'cluster_pileup.members', domain: [0, 1000], heightPx: 200,
    filter: { visible_strands: ['+'], visible_clusters: [3] } },
  { name: 'cluster_pileup.hybrid', kind: 'cluster_pileup',
    fixture: 'hybrid_frame', mode: 'hybrid', domain: [0, 1000], heightPx: 200 },
  { name: 'cluster_pileup.empty', kind: 'cluster_pileup',
    fixture: 'cluster_pileup.clusters', domain: [0, 1000], heightPx: 200, empty: true },

  // --- splice_junctions ---------------------------------------------
  { name: 'splice_junctions.default', kind: 'splice_junctions',
    fixture: 'splice_junctions', domain: [0, 3000], heightPx: 80 },
  { name: 'splice_junctions.styled', kind: 'splice_junctions',
    fixture: 'splice_junctions', domain: [0, 3000], heightPx: 80,
    style: {
      arc_stroke_min_px: 2,
      arc_stroke_max_px: 3,
      arc_opacity: 0.3,
      'palette.GC-AG': '#00ffff',
      'palette.default': '#ffff00',
    } },
  { name: 'splice_junctions.filtered', kind: 'splice_junctions',
    fixture: 'splice_junctions', domain: [0, 3000], heightPx: 80,
    filter: { visible_motifs: ['GT-AG', 'GC-AG'], min_support: 5, annotated_only: false } },
  { name: 'splice_junctions.annotated-only', kind: 'splice_junctions',
    fixture: 'splice_junctions', domain: [0, 3000], heightPx: 80,
    filter: { annotated_only: true } },
  { name: 'splice_junctions.empty', kind: 'splice_junctions',
    fixture: 'splice_junctions', domain: [0, 3000], heightPx: 80, empty: true },
];

describe('track renderers', () => {
  it('covers every registered kind', () => {
    const covered = new Set(CASES.map((c) => c.kind));
    expect([...covered].sort()).toEqual(registeredKinds());
  });

  it.each(CASES)('$name', async (c) => {
    const svg = draw(c);
    // Every renderer stamps naturalHeight; SVG export sizes panels from it.
    expect(Number(svg.dataset.naturalHeight)).toBeGreaterThan(0);
    await expect(serialize(svg)).toMatchFileSnapshot(`./__snapshots__/${c.name}.svg`);
  });

  it('is deterministic across repeated renders into the same svg', () => {
    const c = CASES.find((x) => x.name === 'read_pileup.default')!;
    expect(serialize(draw(c))).toBe(serialize(draw(c)));
  });
});
