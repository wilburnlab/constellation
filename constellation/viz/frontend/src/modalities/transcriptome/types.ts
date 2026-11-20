// Wire shapes of the transcriptome modality's routes
// (viz/modalities/transcriptome/endpoints.py). Ids above 2^53 (content ids)
// arrive as strings in JSON; template ids, (round << 40) | row, stay below.

export interface RunInfo {
  key: string;
  label: string;
  path: string;
  finished: boolean;
  rounds: number[];
  n_clusters: number;
  n_reads: number;
  has_cluster_nodes: boolean;
  has_mstep_record: boolean;
  has_compact_evidence: boolean;
  has_round_params: boolean;
  /** Sample id → name; empty when the demultiplexing output was not found. */
  samples: Record<string, string>;
  demux_dir: string | null;
}

export interface SampleCount {
  sample_id: number | null;
  sample: string | null;
  count: number;
  tpm: number | null;
}

export interface ClusterDetail {
  cluster_id: number;
  n_reads: number;
  n_unique_sequences: number | null;
  orf_start: number | null;
  orf_end: number | null;
  length: number;
  protein_length: number;
  round: number | null;
  parent_template_id: number | null;
  haplotype_id: number | null;
  node_template_id: number | null;
  rebuilt: boolean | null;
  n_absorbed: number | null;
  consensus_content_id: string | number | null;
  protein_content_id: string | number | null;
  protein: string | null;
  samples: SampleCount[];
}

export interface NodeCount {
  haplotype_id: number;
  n_reads: number;
}

/** What the support view is of. */
export type SupportTarget =
  | { kind: 'cluster'; clusterId: number }
  | { kind: 'template'; round: number; templateId: number };

/** `cluster` (a cluster's own node), `pooled`, or a haplotype id. */
export type NodeChoice = 'cluster' | 'pooled' | number;
export type Space = 'consensus' | 'frame';

export interface SupportMeta {
  kind: 'node' | 'merged' | 'rebuilt' | 'template';
  round: number;
  verified: boolean;
  mismatch: string[];
  warnings: string[];
  cluster_id?: number;
  haplotype_id?: number;
  template_id: number | null;
  params_source?: string;
  status?: string | null;
  n_reads: number;
  n_sampled: number;
  nodes: NodeCount[];
  n_absorbed_reads: number;
  reference: string;
  space: Space;
  consensus: string;
  n_columns: number;
  rows: Record<string, number>;
}

export const SUPPORT_TABLES = ['reads', 'departures', 'insertions', 'columns', 'arcs', 'differences'] as const;
export type SupportTableName = (typeof SUPPORT_TABLES)[number];

/** Candidate-column route, a mask (em/columns.py: ROUTE_ALLELIC = 1,
 *  ROUTE_COVERAGE = 2). */
export function routeName(route: number): string {
  if (route === 3) return 'allelic + coverage';
  if (route === 2) return 'coverage';
  if (route === 1) return 'allelic';
  return '—';
}
