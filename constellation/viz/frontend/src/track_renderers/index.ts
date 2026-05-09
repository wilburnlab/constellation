// Renderer registry — populated at module-load by importing every
// track-renderer module. Mirrors the Python-side `register_track`
// pattern in `constellation/viz/tracks/__init__.py`.

import { TrackRenderer } from './base';
import coverage_histogram from './coverage_histogram';
import gene_annotation from './gene_annotation';
import reference_sequence from './reference_sequence';
import splice_junctions from './splice_junctions';
import read_pileup from './read_pileup';
import cluster_pileup from './cluster_pileup';

const renderers: Record<string, TrackRenderer> = {
  [coverage_histogram.kind]: coverage_histogram,
  [gene_annotation.kind]: gene_annotation,
  [reference_sequence.kind]: reference_sequence,
  [splice_junctions.kind]: splice_junctions,
  [read_pileup.kind]: read_pileup,
  [cluster_pileup.kind]: cluster_pileup,
};

export function getRenderer(kind: string): TrackRenderer | null {
  return renderers[kind] ?? null;
}

export function registeredKinds(): string[] {
  return Object.keys(renderers).sort();
}
