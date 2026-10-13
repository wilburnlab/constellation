// Vite config — multi-entry build emitting to ../static/<entry>/.
//
// Each `index.<entry>.html` shell in this directory is one entry: a
// self-contained SPA (its own HTML + bundle) built to
// constellation/viz/static/<entry>/ and mounted by the FastAPI app at
// /static/<entry>/. Entries are discovered from the shell files — the
// same rule `build.py::known_entries()` uses — so adding one is adding
// its shell; nothing here names an entry.

import { defineConfig } from 'vite';
import { readdirSync } from 'node:fs';
import { resolve } from 'node:path';

const inputs: Record<string, string> = {};
for (const name of readdirSync(__dirname).sort()) {
  const match = /^index\.(.+)\.html$/.exec(name);
  if (match) inputs[match[1]] = resolve(__dirname, name);
}

const ENTRY = process.env.CONSTELLATION_VIZ_ENTRY ?? 'genome';
if (!(ENTRY in inputs)) {
  throw new Error(
    `unknown viz entry "${ENTRY}" — no index.${ENTRY}.html in ${__dirname} ` +
      `(found: ${Object.keys(inputs).join(', ') || 'none'})`,
  );
}

export default defineConfig({
  base: `/static/${ENTRY}/`,
  build: {
    outDir: resolve(__dirname, '..', 'static', ENTRY),
    emptyOutDir: true,
    sourcemap: true,
    rollupOptions: {
      input: inputs[ENTRY],
    },
  },
  server: {
    // Local dev server proxies /api to the FastAPI backend so
    // `pnpm dev` works against a running `constellation viz genome`.
    proxy: {
      '/api': 'http://127.0.0.1:8765',
    },
  },
});
