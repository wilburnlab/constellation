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
