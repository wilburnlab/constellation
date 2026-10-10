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

/** Directory-vs-file sub-kind for path-ish args (type 'path' or a
 *  path-ish 'multi'). Drives which mode the FilePicker opens in.
 *  'either' allows selecting a directory or a file. */
export type PathKind = 'dir' | 'file' | 'either';

/** Richer-widget hint decoupled from `type`. `reference` → an editable
 *  dropdown fed by GET /api/references. */
export type WidgetHint = 'reference';

export interface ArgumentSchema {
  dest: string;
  option_strings: string[];
  metavar: string | null;
  help: string | null;
  type: ArgumentType;
  path_kind?: PathKind | null;
  widget?: WidgetHint | null;
  glob?: string | null;
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
// Reference cache — GET /api/references. Read by the shell itself: any
// command form whose argument is a reference handle offers the installed
// ones. (A browser's own session shapes live with that browser, under
// modalities/<name>/.)
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
