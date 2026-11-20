import { describe, expect, it } from 'vitest';
import { isUnknownBrowser, taskInitForCommand } from './TaskPanel';
import type { ArgumentSchema, CommandSchema } from './types';

function cmd(path: string[], options: string[][]): CommandSchema {
  return {
    name: path.at(-1)!,
    path,
    help: null,
    subcommands: [],
    arguments: options.map((o) => ({ dest: o[0].replace(/^-+/, ''), option_strings: o }) as unknown as ArgumentSchema),
  };
}

describe('viz commands in the dashboard', () => {
  it('opens the known browsers in place, the transcriptome browser among them', () => {
    for (const name of ['genome', 'massspec', 'transcriptome']) {
      expect(taskInitForCommand(cmd(['viz', name], [['--no-browser']])).kind).toBe('viz');
    }
  });

  it('refuses a browser it has no descriptor for, rather than serving it in a terminal', () => {
    expect(isUnknownBrowser(cmd(['viz', 'newthing'], [['--port'], ['--no-browser']]))).toBe(true);
    // A viz command that serves nothing is an ordinary command.
    expect(isUnknownBrowser(cmd(['viz', 'install-frontend'], [['--from']]))).toBe(false);
    expect(isUnknownBrowser(cmd(['viz', 'transcriptome'], [['--no-browser']]))).toBe(false);
    expect(isUnknownBrowser(cmd(['massspec', 'search'], [['--no-browser']]))).toBe(false);
  });
});
