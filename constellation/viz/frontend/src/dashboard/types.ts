// Wire types mirroring the backend's
// constellation/viz/introspect/schema.py TypedDicts.

export type ArgumentType =
  | 'str'
  | 'int'
  | 'float'
  | 'flag'
  | 'enum'
  | 'path'
  | 'multi';

export interface ArgumentSchema {
  dest: string;
  option_strings: string[];
  metavar: string | null;
  help: string | null;
  type: ArgumentType;
  default: unknown;
  choices: unknown[] | null;
  required: boolean;
  nargs: string | number | null;
  is_positional: boolean;
}

export interface CommandSchema {
  name: string;
  path: string[];
  help: string | null;
  arguments: ArgumentSchema[];
  subcommands: CommandSchema[];
}

export interface CuratedEntry {
  path: string[];
  label: string;
  group?: string;
  hint?: string;
}

export interface CliSchema {
  prog: string;
  help: string | null;
  arguments: ArgumentSchema[];
  subcommands: CommandSchema[];
  curated: CuratedEntry[];
}

// /api/commands wire shapes

export interface CommandResponse {
  job_id: string;
  argv: string[];
  started_at: string;
  state: string;
}

export interface JobSnapshot {
  job_id: string;
  argv: string[];
  started_at: string;
  ended_at: string | null;
  exit_code: number | null;
  state: string;
}

export interface OutputFrame {
  stream: 'stdout' | 'stderr' | 'exit';
  line: string;
}

// ---------------------------------------------------------------------
// Reference cache + session entry endpoints
// ---------------------------------------------------------------------

export interface InstalledReference {
  handle: string;
  organism: string;
  release_slug: string;
  source: string;
  release: string;
  path: string;
  assembly_accession: string | null;
  assembly_name: string | null;
  annotation_release: string | null;
  fetched_at: string | null;
  size_bytes: number | null;
  scientific_name: string | null;
  is_default: boolean;
}

export interface SourceInspection {
  path: string;
  kind: 'align' | 'cluster';
  reference_handle: string | null;
  reference_path: string | null;
  assembly_accession: string | null;
  samples: string[];
}

export interface OpenSessionResult {
  session_id: string;
  label: string;
  reference_handle: string;
  reference_path: string;
  n_sources: number;
  stages_present: Record<string, boolean>;
  warnings: string[];
  saved_as: string | null;
}

export interface SavedSessionSummary {
  slug: string;
  label: string;
  reference_handle: string;
  n_sources: number;
  saved_at: string;
  last_viewed_locus: { contig: string; start: number; end: number } | null;
}

export interface SavedSessionPayload extends SavedSessionSummary {
  sources: Array<{ path: string; kind: string; label: string }>;
}
