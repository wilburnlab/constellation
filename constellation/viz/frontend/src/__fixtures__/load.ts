// Test-only loaders for the kernel-generated fixtures under
// `__fixtures__/genome/` (see scripts/build-viz-frontend-fixtures.py).

import { readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { Table, tableFromIPC } from 'apache-arrow';
import type { TrackMetadata } from '../track_renderers/base';

// Resolved through the file path rather than `new URL(rel, import.meta.url)`:
// Vite rewrites that pattern as a bundled-asset reference.
const GENOME_DIR = join(dirname(fileURLToPath(import.meta.url)), 'genome');

/** Raw Arrow IPC stream bytes, as the server would put them on the wire. */
export function loadFixtureBytes(name: string): Uint8Array {
  return new Uint8Array(readFileSync(join(GENOME_DIR, `${name}.arrow`)));
}

export function loadFixture(name: string): Table {
  return tableFromIPC(loadFixtureBytes(name));
}

/** Kernel `metadata()` payloads, keyed `"<kind>/<binding_id>"`. */
export function loadMetadata(): Record<string, TrackMetadata> {
  return JSON.parse(
    readFileSync(join(GENOME_DIR, 'metadata.json'), 'utf-8'),
  ) as Record<string, TrackMetadata>;
}

/** The metadata for the first fixture binding of `kind`. */
export function metadataFor(kind: string): TrackMetadata {
  const all = loadMetadata();
  const key = Object.keys(all).find((k) => k.startsWith(`${kind}/`));
  if (!key) throw new Error(`no fixture metadata for kind ${kind}`);
  return all[key];
}
