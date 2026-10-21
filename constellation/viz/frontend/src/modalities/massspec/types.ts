// Wire shapes of the mass-spec session routes (viz/modalities/massspec/endpoints.py).

/** One acquisition, as `GET /api/massspec/sessions/{sid}/acquisitions` describes it. */
export interface AcquisitionInfo {
  key: string;
  label: string;
  bundle: string;
  source_id: string;
  bundle_kind: string;
  source_file: string;
  source_sha256: string | null;
  acquisition_type: 'DIA' | 'DDA' | null;
  staggered: boolean;
  profile_mode: boolean;
  levels: number[];
  /** By level, as a string. */
  scans_per_level: Record<string, number>;
  analyzers: Record<string, string[]>;
  /** Seconds. */
  rt_range: [number, number];
  has_index: boolean;
  /** By level: `[value, "ppm" | "Da"]`. */
  default_tolerance: Record<string, [number, string]>;
  /** The experiments that searched it. */
  searches: SearchLink[];
}

/** One experiment's link to an acquisition. */
export interface SearchLink {
  experiment: string;
  experiment_source_id: string;
  experiment_label: string;
  acquisition_id: number;
  units: string[];
}

/** One DIA isolation window (`…/acquisitions/{key}/windows`). */
export interface IsolationWindow {
  level: number;
  window: number;
  isolation_lower: number;
  isolation_upper: number;
  center: number;
  n_scans: number;
}

/** What `POST /api/sessions/open` returns. */
export interface OpenedSession {
  session_id: string;
  label: string;
  modality: string;
  n_acquisitions: number;
  warnings: string[];
}
