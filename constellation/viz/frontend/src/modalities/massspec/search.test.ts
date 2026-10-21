import { describe, expect, it } from 'vitest';
import { matchIdentifications, parseQuery } from './search';

describe('parseQuery', () => {
  it.each([
    ['scan 12345', { kind: 'scan', scan: 12345 }],
    ['#77', { kind: 'scan', scan: 77 }],
    ['rt 35.2', { kind: 'rt', minutes: 35.2 }],
    ['512.27', { kind: 'mz', mz: 512.27 }],
    ['mz 600', { kind: 'mz', mz: 600 }],
    ['PEPTIDEK', { kind: 'peptide', text: 'PEPTIDEK', charge: null }],
    ['PEPTIDEK/2', { kind: 'peptide', text: 'PEPTIDEK', charge: 2 }],
    ['EM[Oxidation]EVTSESK/2', { kind: 'peptide', text: 'EM[Oxidation]EVTSESK', charge: 2 }],
    ['[Acetyl]-MDFQHRPGGK/3', { kind: 'peptide', text: '[Acetyl]-MDFQHRPGGK', charge: 3 }],
    ['sp|P78371|TCPB_HUMAN', { kind: 'text', text: 'sp|P78371|TCPB_HUMAN' }],
    ['tubulin', { kind: 'text', text: 'tubulin' }],
    ['  ', { kind: 'empty' }],
  ])('%s', (input, expected) => {
    expect(parseQuery(input)).toEqual(expected);
  });
});

describe('matchIdentifications', () => {
  const rows = [
    { modified_sequence: 'PEPTIDEK', sequence: 'PEPTIDEK', proteins: 'sp|P1|A_HUMAN', charge: 2 },
    { modified_sequence: 'PEPTIDEK', sequence: 'PEPTIDEK', proteins: 'sp|P1|A_HUMAN', charge: 3 },
    { modified_sequence: 'PEPTIDEKR', sequence: 'PEPTIDEKR', proteins: 'sp|P2|B_HUMAN', charge: 2 },
    { modified_sequence: 'PEPTM[UNIMOD:35]IDEK', sequence: 'PEPTMIDEK', proteins: 'sp|P3|TUBB_HUMAN', charge: 2 },
  ];

  it('ranks exact peptides before prefixes and honours the charge', () => {
    const hits = matchIdentifications(rows, parseQuery('PEPTIDEK'));
    expect(hits.map((r) => r.modified_sequence)).toEqual(['PEPTIDEK', 'PEPTIDEK', 'PEPTIDEKR']);
    expect(matchIdentifications(rows, parseQuery('PEPTIDEK/3')).map((r) => r.charge)).toEqual([3]);
  });

  it('matches text in proteins and sequences, ignoring case', () => {
    expect(matchIdentifications(rows, parseQuery('tubb')).map((r) => r.sequence)).toEqual(['PEPTMIDEK']);
    expect(matchIdentifications(rows, parseQuery('scan 4'))).toEqual([]);
  });
});
