// Encoders for the genome kernels' server-side ("pushdown") filters.
//
// Each maps a stored filter value to the query parameter the kernel's
// query model declares (`ReadPileupQuery.min_mapq`,
// `ClusterPileupQuery.cluster_view`), or `undefined` to leave it out.

import { PushdownEncoder } from '../../../panels/kind';

/** MAPQ threshold: sent only when positive (0 admits every alignment,
 *  which is the server's default). Accepts a number or a numeric string. */
export const minMapq: PushdownEncoder = (value) => {
  let n = 0;
  if (typeof value === 'number' && Number.isFinite(value)) {
    n = value;
  } else if (typeof value === 'string') {
    const parsed = Number(value);
    if (Number.isFinite(parsed)) n = parsed;
  }
  return n > 0 ? String(n) : undefined;
};

/** Cluster pile-up view: sent only for one of the two known views. */
export const clusterView: PushdownEncoder = (value) =>
  value === 'clusters' || value === 'members' ? value : undefined;
