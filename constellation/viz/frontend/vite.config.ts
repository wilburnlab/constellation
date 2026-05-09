// Vite config — multi-entry build emitting to ../static/<entry>/.
//
// PR 1 ships the `genome` entry; PR 2 adds `dashboard`. Each entry is a
// self-contained SPA (its own HTML + bundle) under
// constellation/viz/static/<entry>/, mounted by the FastAPI app at
// /static/<entry>/.

import { defineConfig } from 'vite';
import { resolve } from 'node:path';

const ENTRY = process.env.CONSTELLATION_VIZ_ENTRY ?? 'genome';

const inputs: Record<string, string> = {
  genome: resolve(__dirname, 'index.genome.html'),
  // PR 2 will add: dashboard: resolve(__dirname, 'index.dashboard.html'),
};

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
