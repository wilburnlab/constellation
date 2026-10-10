// Layering of the frontend source tree.
//
//   engine/      rendering + transport primitives         ┐
//   panels/      modality-neutral panel chrome            ├─ never import modalities/
//   widgets/     reusable inputs (path picker, …)         ┘
//   modalities/<name>/   everything specific to one browser; may import
//                        the three above, never another modality
//   dashboard/   the shell; reaches a modality only through viz_registry.ts
//
// The genome browser used to be woven through engine/ and widgets/. A
// second browser can only reuse the generic layers if nothing in them
// quietly depends on genome code again, so the rule is checked here.

import { readdirSync, readFileSync } from 'node:fs';
import { dirname, join, relative, resolve, sep } from 'node:path';
import { fileURLToPath } from 'node:url';
import { describe, expect, it } from 'vitest';

const SRC = dirname(fileURLToPath(import.meta.url));
const GENERIC_LAYERS = ['engine', 'panels', 'widgets'];
/** The one dashboard file allowed to name a modality. */
const REGISTRY = join('dashboard', 'viz_registry.ts');

interface SourceFile {
  /** Path relative to src/, with native separators. */
  path: string;
  text: string;
}

function sourceFiles(dir = SRC): SourceFile[] {
  const out: SourceFile[] = [];
  for (const entry of readdirSync(dir, { withFileTypes: true })) {
    const full = join(dir, entry.name);
    if (entry.isDirectory()) {
      out.push(...sourceFiles(full));
    } else if (entry.name.endsWith('.ts')) {
      out.push({ path: relative(SRC, full), text: readFileSync(full, 'utf-8') });
    }
  }
  return out;
}

/** Relative specifiers of every static import, dynamic import and vi.mock. */
function relativeImports(text: string): string[] {
  const pattern = /(?:\bfrom|\bimport|\bmock)\s*\(?\s*['"](\.{1,2}(?:\/[^'"]*)?)['"]/g;
  return Array.from(text.matchAll(pattern), (m) => m[1]);
}

/** `['modalities', 'genome']` for a path under modalities/genome/, else null. */
function modalityOf(path: string): string | null {
  const parts = path.split(sep);
  return parts[0] === 'modalities' && parts.length > 2 ? parts[1] : null;
}

function violations(files: SourceFile[]): string[] {
  const problems: string[] = [];
  for (const file of files) {
    const layer = file.path.split(sep)[0];
    const own = modalityOf(file.path);
    for (const spec of relativeImports(file.text)) {
      const target = relative(SRC, resolve(SRC, dirname(file.path), spec));
      const imported = modalityOf(`${target}${sep}x`);
      if (imported === null) continue;
      if (GENERIC_LAYERS.includes(layer)) {
        problems.push(`${file.path}: ${layer}/ imports modality code (${spec})`);
      } else if (own !== null && own !== imported) {
        problems.push(`${file.path}: modality ${own} imports modality ${imported} (${spec})`);
      } else if (layer === 'dashboard' && file.path !== REGISTRY) {
        problems.push(`${file.path}: dashboard/ imports modality code outside viz_registry.ts (${spec})`);
      }
    }
  }
  return problems;
}

describe('frontend layering', () => {
  it('generic layers and the dashboard shell stay free of modality imports', () => {
    expect(violations(sourceFiles())).toEqual([]);
  });

  it('has something to check in every layer it guards', () => {
    const layers = new Set(sourceFiles().map((f) => f.path.split(sep)[0]));
    for (const layer of [...GENERIC_LAYERS, 'modalities', 'dashboard']) {
      expect(layers.has(layer), layer).toBe(true);
    }
  });

  it('reports each kind of violation', () => {
    const p = (...parts: string[]): string => parts.join(sep);
    const planted: SourceFile[] = [
      { path: p('engine', 'x.ts'), text: `import { a } from '../modalities/genome/scales';` },
      { path: p('panels', 'y.ts'), text: `const m = await import('../modalities/genome/GenomeBrowser');` },
      { path: p('widgets', 'z.test.ts'), text: `vi.mock('../modalities/genome/scales', () => ({}));` },
      { path: p('dashboard', 'Shell.ts'), text: `import { F } from '../modalities/genome/GenomeBrowserForm';` },
      { path: p('modalities', 'massspec', 'b.ts'), text: `import { L } from '../genome/viewport_bus';` },
      // Allowed: the registry, a modality using the generic layers and itself.
      { path: p('dashboard', 'viz_registry.ts'), text: `import { F } from '../modalities/genome/GenomeBrowserForm';` },
      { path: p('modalities', 'genome', 'a.ts'), text: `import { s } from '../../engine/svg_layer';\nimport { r } from './renderers';` },
    ];
    const found = violations(planted);
    expect(found).toHaveLength(5);
    expect(found.map((v) => v.split(':')[0])).toEqual([
      p('engine', 'x.ts'),
      p('panels', 'y.ts'),
      p('widgets', 'z.test.ts'),
      p('dashboard', 'Shell.ts'),
      p('modalities', 'massspec', 'b.ts'),
    ]);
  });
});
