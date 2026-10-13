// Vitest config — unit tests for the viz frontend (`pnpm test`).
//
// Kept separate from vite.config.ts so the bundle build (base path,
// per-entry rollup input) is untouched by test settings. jsdom supplies
// the DOM/SVG APIs the renderers and widgets call; it has no layout
// engine, so anything that depends on measured sizes is stubbed in the
// individual test.

import { defineConfig } from 'vitest/config';

export default defineConfig({
  test: {
    environment: 'jsdom',
    include: ['src/**/*.test.ts'],
  },
});
