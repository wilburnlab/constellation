// Wire shapes of the genome browser's session endpoints, as its entry
// form and the dashboard's registry entry for it read them. The Python
// side is constellation/viz/modalities/genome/ (`requests.py`,
// `session.py`) and the saved-session cache.

import { LayoutEntry } from '../../panels/layout';
import { BrowserOptions } from '../../panels/OptionsPopover';

/** POST /api/sessions/inspect-source */
export interface SourceInspection {
  path: string;
  kind: 'align' | 'cluster';
  reference_handle: string | null;
  reference_path: string | null;
  assembly_accession: string | null;
  samples: string[];
}

/** POST /api/sessions/open */
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

/** One row of GET /api/saved-sessions. */
export interface SavedSessionSummary {
  slug: string;
  /** Which browser the configuration is for. Absent on responses from a
   *  server that predates modalities, where every session is a genome one. */
  modality?: string;
  label: string;
  reference_handle: string;
  n_sources: number;
  saved_at: string;
  last_viewed_locus: { contig: string; start: number; end: number } | null;
}

/** GET /api/saved-sessions/{slug} */
export interface SavedSessionPayload extends SavedSessionSummary {
  sources: Array<{ path: string; kind: string; label: string }>;
  track_layout?: LayoutEntry[];
  options?: BrowserOptions;
}
