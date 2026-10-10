// TrackSettingsPanel — the gear popover for one genome track.
//
// A thin binding of the generic `SettingsPanel` to the genome track
// kinds: it looks up the schema the track's renderer declares. Sections:
//
//   General — the same few fields for every track.
//   Style   — the kind's palette and size knobs.
//   Filter  — the kind's dataset-slice controls (samples / modes /
//             motifs, strand toggles, minimum thresholds).
//
// Categorical lists (samples, modes, motifs) come from the track's
// metadata, which the browser already fetched when it mounted the track.

import { SettingsPanel } from '../../panels/SettingsPanel';
import { getRenderer } from './renderers';
import { TrackMetadata } from './renderers/base';
import { FALLBACK_SETTINGS } from './renderers/settings_common';

export interface TrackSettingsArgs {
  anchor: HTMLElement;
  kind: string;
  label: string;
  meta: TrackMetadata;
  style: Record<string, unknown>;
  filter: Record<string, unknown>;
  onStyleChange(style: Record<string, unknown>): void;
  onFilterChange(filter: Record<string, unknown>): void;
  onReset(): void;
  onClose(): void;
}

export class TrackSettingsPanel extends SettingsPanel {
  constructor(opts: TrackSettingsArgs) {
    super({
      ...opts,
      schema: getRenderer(opts.kind)?.settings ?? FALLBACK_SETTINGS,
    });
  }
}
