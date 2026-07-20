"""Tests for the de novo first-round assembly stage (Phase 3, Cut 1)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import pytest

from constellation.core.sequence.nucleic import reverse_complement
from constellation.sequencing.transcriptome.cluster.denovo.candidates import (
    generate_candidates,
)
from constellation.sequencing.transcriptome.cluster.denovo.cluster_graph import (
    connected_components,
)
from constellation.sequencing.transcriptome.cluster.denovo.consensus import (
    MemberSpec,
    centroid_consensus,
)
from constellation.sequencing.transcriptome.cluster.denovo.dereplicate import (
    dereplicate,
)
from constellation.sequencing.transcriptome.cluster.denovo.encode import (
    INVALID_HASH,
    encode_block,
)
from constellation.sequencing.transcriptome.cluster.denovo.minimizers import (
    extract_minimizers,
)
from constellation.sequencing.transcriptome.cluster.denovo.pipeline import (
    assemble_clusters,
    cluster_transcripts,
)
from constellation.sequencing.transcriptome.cluster.denovo.verify import (
    verify_candidates,
)


edlib = pytest.importorskip("edlib")


def _rand(rng, n):
    return "".join(rng.choice(list("ACGT"), n))


def _mutate(rng, s, err=0.008, trim5=0, trim3=0):
    out = []
    for ch in s:
        r = rng.random()
        if r < err * 0.6:
            out.append(rng.choice(list("ACGT")))
        elif r < err * 0.8:
            continue
        elif r < err:
            out.append(ch)
            out.append(rng.choice(list("ACGT")))
        else:
            out.append(ch)
    o = "".join(out)
    return o[trim5 : len(o) - trim3 if trim3 else len(o)]


# ── encode + minimizers ───────────────────────────────────────────────


def test_canonical_minimizers_revcomp_invariant():
    rng = np.random.default_rng(0)
    s = _rand(rng, 400)
    col = pa.array([s, reverse_complement(s)], type=pa.large_string())
    mi = extract_minimizers(col, k=15, w=10)
    h = mi.mini_hash.numpy()
    u = mi.uniq_id.numpy()
    assert sorted(h[u == 0].tolist()) == sorted(h[u == 1].tolist())
    assert h[u == 0].shape[0] > 0


def test_window_min_matches_bruteforce():
    rng = np.random.default_rng(1)
    s = _rand(rng, 300)
    kh = encode_block(s.encode(), np.array([0, len(s)], dtype=np.int64), k=15)
    ch = kh.canon_hash.numpy()
    valid = kh.valid.numpy()
    w = 10
    n_pos = len(ch)
    brute = set()
    for i in range(0, n_pos - w + 1):
        if not (valid[i] and valid[i + w - 1]):
            continue
        win = ch[i : i + w]
        j = int(np.argmin(win))
        brute.add((i + j, int(win.min())))
    from constellation.sequencing.transcriptome.cluster.denovo.minimizers import (
        _block_minimizers,
    )

    h, _u, p = _block_minimizers(
        kh.canon_hash, kh.valid, kh.seq_id, kh.pos_in_seq, w=w, uniq_offset=0
    )
    assert set(zip(p.numpy().tolist(), h.numpy().tolist())) == brute


def test_minimizer_chunk_invariant():
    rng = np.random.default_rng(2)
    col = pa.array(
        [_rand(rng, 300), "N" * 20 + _rand(rng, 280), _rand(rng, 50), _rand(rng, 8)],
        type=pa.large_string(),
    )

    def keyset(mi):
        return set(
            zip(
                mi.uniq_id.numpy().tolist(),
                mi.pos.numpy().tolist(),
                mi.mini_hash.numpy().tolist(),
            )
        )

    full = extract_minimizers(col, k=15, w=10, chunk_bases=10**9)
    chunked = extract_minimizers(col, k=15, w=10, chunk_bases=64)
    assert keyset(full) == keyset(chunked)


def test_short_sequence_no_minimizers():
    col = pa.array(["ACGT"], type=pa.large_string())
    assert extract_minimizers(col, k=15, w=10).mini_hash.shape[0] == 0
    assert INVALID_HASH > 0


# ── dereplicate ───────────────────────────────────────────────────────


def test_dereplicate_counts_and_map():
    reads = pa.table(
        {
            "read_id": ["a", "b", "c", "d"],
            "sequence": ["ACGT", "ACGT", "TTTT", "ACGT"],
            "sample_id": pa.array([0, 0, 1, 1], pa.int64()),
        }
    )
    uniq, rmap = dereplicate(reads)
    assert uniq.num_rows == 2
    abund = dict(
        zip(uniq.column("sequence").to_pylist(), uniq.column("abundance").to_pylist())
    )
    assert abund == {"ACGT": 3, "TTTT": 1}
    # read_map points each read at its unique sequence
    seq_of_uniq = uniq.column("sequence").to_pylist()
    for rid, uid, seq in zip(
        rmap.column("read_id").to_pylist(),
        rmap.column("uniq_id").to_pylist(),
        ["ACGT", "ACGT", "TTTT", "ACGT"],
    ):
        assert seq_of_uniq[uid] == seq


# ── verify: overhang normalization both orientations ──────────────────


def test_verify_overhang_both_orientations():
    rng = np.random.default_rng(3)
    base = _rand(rng, 300)
    # b is base with a 25 nt 5' extension (within 30) → should accept
    ext = _rand(rng, 25) + base
    seqs = [base, ext]
    cand = pa.table(
        {"uniq_a": [0], "uniq_b": [1], "n_shared": pa.array([5], pa.int32())}
    )
    acc = verify_candidates(cand, seqs, identity=0.98, max_5p=30, max_3p=30)
    assert acc.num_rows == 1
    # a 40 nt extension exceeds max_5p=30 → reject
    ext2 = _rand(rng, 40) + base
    acc2 = verify_candidates(
        pa.table({"uniq_a": [0], "uniq_b": [1], "n_shared": pa.array([5], pa.int32())}),
        [base, ext2],
        identity=0.98,
        max_5p=30,
        max_3p=30,
    )
    assert acc2.num_rows == 0


def test_verify_rejects_low_identity():
    rng = np.random.default_rng(4)
    a = _rand(rng, 400)
    b = _mutate(rng, a, err=0.10)  # ~10% divergent
    acc = verify_candidates(
        pa.table({"uniq_a": [0], "uniq_b": [1], "n_shared": pa.array([5], pa.int32())}),
        [a, b],
        identity=0.98,
        max_5p=30,
        max_3p=30,
    )
    assert acc.num_rows == 0


# ── consensus projection both orientations ────────────────────────────


def _consensus_one(centroid, member, w_c=5.0, w_m=1.0):
    short, long = (
        (centroid, member) if len(centroid) <= len(member) else (member, centroid)
    )
    a = edlib.align(short, long, mode="HW", task="path")
    spec = MemberSpec(
        member_seq=member,
        weight=w_m,
        cigar=a["cigar"],
        centroid_is_query=(centroid == short),
        ref_start=a["locations"][0][0],
    )
    return centroid_consensus(centroid, w_c, [spec]).consensus


def test_consensus_projection_orientations():
    rng = np.random.default_rng(5)
    c = _rand(rng, 300)
    m = list(c)
    m[50] = "A" if c[50] != "A" else "C"
    m = "".join(m)
    # centroid-dominant → unchanged; member-dominant → flips pos 50
    assert _consensus_one(c, m, 5.0, 1.0) == c
    flipped = _consensus_one(c, m, 1.0, 5.0)
    assert flipped[50] == m[50] and flipped[:50] == c[:50] and flipped[51:] == c[51:]
    # member longer (3' extension) and shorter (5' truncation) both project
    assert _consensus_one(c, c + _rand(rng, 20), 5.0, 1.0) == c
    assert _consensus_one(c, c[25:], 5.0, 1.0) == c


# ── connected components ──────────────────────────────────────────────


def test_connected_components_isolated_and_linked():
    abundance = np.array([10, 1, 1, 5], dtype=np.int64)
    seq_len = np.array([100, 100, 100, 100], dtype=np.int64)
    # edges: 0-1, 1-2 (one component {0,1,2}); 3 isolated
    comp = connected_components(
        4, abundance, seq_len, np.array([0, 1]), np.array([1, 2])
    )
    labels = comp.cluster_of
    assert labels[0] == labels[1] == labels[2]
    assert labels[3] != labels[0]
    # centroid of the {0,1,2} component is the highest-abundance member (0)
    assert comp.centroid_uniq[labels[0]] == 0


# ── end-to-end assemble_clusters on a synthetic panel ─────────────────


@pytest.fixture(scope="module")
def synthetic_panel():
    rng = np.random.default_rng(2024)
    n_tx = 120
    trues = [_rand(rng, int(rng.integers(500, 1100))) for _ in range(n_tx)]
    depths = rng.integers(1, 60, size=n_tx)
    reads, truth, rid = [], {}, 0
    for ti, (T, d) in enumerate(zip(trues, depths)):
        for _ in range(int(d)):
            w = _mutate(
                rng,
                T,
                err=0.008,
                trim5=int(rng.integers(0, 25)),
                trim3=int(rng.integers(0, 25)),
            )
            reads.append((f"r{rid}", w, int(rng.integers(0, 3))))
            truth[f"r{rid}"] = ti
            rid += 1
    table = pa.table(
        {
            "read_id": [r[0] for r in reads],
            "sequence": [r[1] for r in reads],
            "sample_id": pa.array([r[2] for r in reads], pa.int64()),
        }
    )
    return table, trues, depths, truth


def _ari(true_labels, pred_labels):
    from collections import Counter
    from math import comb

    n = len(true_labels)
    a = Counter(true_labels)
    b = Counter(pred_labels)
    ab = Counter(zip(true_labels, pred_labels))
    sa = sum(comb(v, 2) for v in a.values())
    sb = sum(comb(v, 2) for v in b.values())
    s = sum(comb(v, 2) for v in ab.values())
    exp = sa * sb / comb(n, 2)
    maxi = (sa + sb) / 2
    return (s - exp) / (maxi - exp) if maxi != exp else 1.0


def test_assemble_clusters_recovers_panel(synthetic_panel):
    table, trues, depths, truth = synthetic_panel
    res = assemble_clusters(table, identity=0.97, predict_orfs=True, min_aa_length=40)
    # one cluster per well-supported transcript
    n_expected = int((depths >= 5).sum())
    big = sum(1 for x in res.clusters.column("n_reads").to_pylist() if x >= 5)
    assert big >= int(0.95 * n_expected)
    # membership covers every input read exactly once
    assert res.membership.num_rows == table.num_rows
    # high agreement with ground truth
    r2c = dict(
        zip(
            res.membership.column("read_id").to_pylist(),
            res.membership.column("cluster_id").to_pylist(),
        )
    )
    common = [r for r in r2c if r in truth]
    assert _ari([truth[r] for r in common], [r2c[r] for r in common]) > 0.95
    # consensus accuracy on the deep clusters
    cons = res.clusters.column("consensus_sequence").to_pylist()
    nr = res.clusters.column("n_reads").to_pylist()
    errs = []
    for c, n in zip(cons, nr):
        if n >= 15:
            best = min(edlib.align(c, T, mode="NW")["editDistance"] for T in trues)
            errs.append(best / max(len(c), 1))
    assert np.median(errs) < 0.02


def test_assemble_clusters_schema_and_quant(synthetic_panel):
    table, *_ = synthetic_panel
    res = assemble_clusters(table, identity=0.97, predict_orfs=True, min_aa_length=40)
    # cluster table mode + nulled genome columns
    assert set(res.clusters.column("mode").to_pylist()) == {"de-novo"}
    assert res.clusters.column("contig_id").null_count == res.clusters.num_rows
    # feature_quant origin/engine + per-sample TPM sums ~1e6
    fq = res.feature_quant
    assert set(fq.column("feature_origin").to_pylist()) == {"cluster_id"}
    for s in set(fq.column("sample_id").to_pylist()):
        tot = sum(
            t
            for t, ss in zip(
                fq.column("tpm").to_pylist(), fq.column("sample_id").to_pylist()
            )
            if ss == s
        )
        assert abs(tot - 1e6) < 1.0
    # membership roles drawn from the allowed vocab
    roles = set(res.membership.column("role").to_pylist())
    assert roles <= {"representative", "member", "duplicate", "drift_filtered"}


def test_assemble_clusters_empty():
    empty = pa.table(
        {
            "read_id": pa.array([], pa.string()),
            "sequence": pa.array([], pa.large_string()),
            "sample_id": pa.array([], pa.int64()),
        }
    )
    res = assemble_clusters(empty)
    assert res.clusters.num_rows == 0
    assert res.membership.num_rows == 0


# ── full I/O path against a synthetic demux dir ───────────────────────


def _write_demux_dir(tmp_path: Path, rows: list[tuple]) -> Path:
    lead, trail = "GGGGGGGGGG", "CCCCCCCCCC"
    reads_tbl = pa.table(
        {
            "read_id": [r[0] for r in rows],
            "sequence": [lead + r[1] + trail for r in rows],
            "quality": ["I" * (len(r[1]) + 20) for r in rows],
        }
    )
    demux_tbl = pa.table(
        {
            "read_id": [r[0] for r in rows],
            "transcript_segment_index": [0] * len(rows),
            "sample_id": pa.array([r[2] for r in rows], pa.int64()),
            "transcript_start": pa.array([len(lead)] * len(rows), pa.int32()),
            "transcript_end": pa.array(
                [len(lead) + len(r[1]) for r in rows], pa.int32()
            ),
            "score": pa.array([1.0] * len(rows), pa.float32()),
            "is_chimera": [False] * len(rows),
            "status": ["Complete"] * len(rows),
            "is_fragment": [False] * len(rows),
            "artifact": ["none"] * len(rows),
        }
    )
    demux = tmp_path / "demux"
    (demux / "reads").mkdir(parents=True)
    (demux / "read_demux").mkdir(parents=True)
    pq.write_table(reads_tbl, demux / "reads" / "part-00000.parquet")
    pq.write_table(demux_tbl, demux / "read_demux" / "part-00000.parquet")
    return demux


def _member_spec(centroid, member, weight=1.0):
    short, long = (
        (centroid, member) if len(centroid) <= len(member) else (member, centroid)
    )
    a = edlib.align(short, long, mode="HW", task="path")
    return MemberSpec(
        member_seq=member,
        weight=weight,
        cigar=a["cigar"],
        centroid_is_query=(centroid == short),
        ref_start=a["locations"][0][0],
    )


def test_call_variants_real_vs_error_and_depth_gate():
    from constellation.sequencing.transcriptome.cluster.denovo.variants import (
        ErrorModel,
        call_variants,
    )

    rng = np.random.default_rng(9)
    c = _rand(rng, 400)
    alt = "A" if c[100] != "A" else "C"
    specs = []
    for i in range(50):
        m = list(c)
        if i < 20:  # 20/50 carry a real substitution at pos 100
            m[100] = alt
        j = int(rng.integers(len(m)))  # one scattered error each
        m[j] = rng.choice(list("ACGT"))
        specs.append(_member_spec(c, "".join(m)))
    cres = centroid_consensus(c, 1.0, specs)
    rows = call_variants(cres, model=ErrorModel(), n_min=5, a_min=2)
    at100 = [r for r in rows if r[0] == 100]
    assert at100 and at100[0][10] == "real"  # call == 'real'
    assert at100[0][2] == alt  # minor allele
    # depth gate: a shallow cluster with a minor allele < n_min reads → ambiguous
    shallow = [_member_spec(c, c), _member_spec(c, c)]
    mm = list(c)
    mm[100] = alt
    shallow.append(_member_spec(c, "".join(mm)))
    shallow.append(_member_spec(c, "".join(mm)))
    cres2 = centroid_consensus(c, 1.0, shallow)
    rows2 = call_variants(cres2, n_min=10, a_min=2)
    assert all(r[10] == "ambiguous" for r in rows2)


def test_homopolymer_indel_classified():
    from constellation.sequencing.transcriptome.cluster.denovo.variants import (
        call_variants,
    )

    rng = np.random.default_rng(12)
    # centroid with an internal A-homopolymer run of length 6
    c = _rand(rng, 200) + "AAAAAA" + _rand(rng, 200)
    hp_start = 200
    specs = []
    for i in range(40):
        m = list(c)
        if i < 12:  # a deletion in the homopolymer (drops one A)
            del m[hp_start]
        specs.append(_member_spec(c, "".join(m)))
    cres = centroid_consensus(c, 1.0, specs)
    rows = call_variants(cres, n_min=5, a_min=2)
    hp_rows = [r for r in rows if r[3] == "homopolymer_indel"]
    assert hp_rows  # at least one homopolymer-indel variant flagged
    assert all(r[4] is not None and r[4] >= 2 for r in hp_rows)  # run length set
    # homopolymer indels get a higher ε than the substitution prior
    assert all(r[9] > 0.003 for r in hp_rows)  # epsilon_class


def test_haplotype_phasing_and_map(tmp_path):
    rng = np.random.default_rng(31)
    base = _rand(rng, 600)
    p1, p2 = 100, 300
    a1 = "A" if base[p1] != "A" else "C"
    a2 = "G" if base[p2] != "G" else "T"

    def make(alt):
        s = list(base)
        if alt:
            s[p1], s[p2] = a1, a2
        j = int(rng.integers(len(s)))
        s[j] = rng.choice(list("ACGT"))
        return "".join(s)

    rows, rid = [], 0
    for alt in (True, False):
        for _ in range(30):
            rows.append((f"r{rid}", make(alt), 0))
            rid += 1
    table = pa.table(
        {
            "read_id": [r[0] for r in rows],
            "sequence": [r[1] for r in rows],
            "sample_id": pa.array([r[2] for r in rows], pa.int64()),
        }
    )
    res = assemble_clusters(table, identity=0.97, predict_orfs=False)
    # the two planted variants are called real and perfectly phased (r²≈1)
    reals = [r for r in res.variants.to_pylist() if r["call"] == "real"]
    planted = {r["consensus_pos"]: r for r in reals}
    assert p1 in planted and p2 in planted
    assert planted[p1]["max_linkage_r2"] > 0.9
    assert planted[p2]["max_linkage_r2"] > 0.9
    # the two true haplotypes dominate
    haps = sorted(res.haplotypes.to_pylist(), key=lambda r: -r["abundance"])
    assert haps[0]["abundance"] >= 20 and haps[1]["abundance"] >= 20
    assert haps[0]["is_complete"]
    # haplotype-map SVG renders under detail/
    import pyarrow.parquet as _pq

    d = tmp_path / "hapcheck"
    d.mkdir()
    _pq.write_table(res.clusters, d / "clusters.parquet")
    _pq.write_table(res.variants, d / "cluster_variants.parquet")
    _pq.write_table(res.haplotypes, d / "cluster_haplotypes.parquet")
    from constellation.sequencing.transcriptome.cluster.denovo.diagnostics import (
        emit_cluster_details,
    )

    assert emit_cluster_details(d, top_n=5) >= 1
    svgs = list(d.glob("detail/*/haplotype_map.svg"))
    assert svgs and svgs[0].read_text().startswith("<?xml")


def test_haplotype_abundance_sums_to_cluster_reads():
    # Every read is assigned to exactly one haplotype, so the per-cluster
    # haplotype abundances sum to the cluster's read count (quant-grade) —
    # not just the consensus-voting subset that fed the PWM.
    rng = np.random.default_rng(77)
    base = _rand(rng, 700)
    p = 350
    alt = "A" if base[p] != "A" else "C"
    rows, rid = [], 0
    for i in range(160):
        s = list(base)
        if i < 64:  # ~40% carry an interior SNP → a real, phaseable column
            s[p] = alt
        j = int(rng.integers(len(s)))  # one scattered error each
        s[j] = rng.choice(list("ACGT"))
        s = "".join(s)
        s = s[int(rng.integers(0, 25)) : len(s) - int(rng.integers(0, 25))]  # ragged
        rows.append((f"r{rid}", s, 0))
        rid += 1
    table = pa.table(
        {
            "read_id": [r[0] for r in rows],
            "sequence": [r[1] for r in rows],
            "sample_id": pa.array([r[2] for r in rows], pa.int64()),
        }
    )
    res = assemble_clusters(table, identity=0.95, predict_orfs=False)
    assert res.haplotypes.num_rows > 0
    n_reads = dict(
        zip(
            res.clusters.column("cluster_id").to_pylist(),
            res.clusters.column("n_reads").to_pylist(),
        )
    )
    hap_total: dict[int, int] = {}
    for cid, ab in zip(
        res.haplotypes.column("cluster_id").to_pylist(),
        res.haplotypes.column("abundance").to_pylist(),
    ):
        hap_total[cid] = hap_total.get(cid, 0) + ab
    # for every cluster that produced haplotypes, abundances sum to its reads
    for cid, total in hap_total.items():
        assert total == n_reads[cid]
    # and the dominant cluster captured (nearly) all 160 reads
    assert max(hap_total.values()) >= 150


def test_estimate_error_rates_recovers_ratios():
    from constellation.sequencing.transcriptome.cluster.denovo.variants import (
        estimate_error_rates,
    )

    # (class_code, run_len) -> (minor_sum, total_sum)
    stats = {
        (0, 0): (300.0, 100000.0),  # substitution 0.3%
        (2, 0): (50.0, 10000.0),  # non-hp indel 0.5%
        (1, 3): (100.0, 5000.0),  # hp run 3 → 2%
        (1, 6): (300.0, 5000.0),  # hp run 6 → 6%
    }
    m = estimate_error_rates(stats, min_total=100)
    assert abs(m.eps_sub - 0.003) < 1e-4
    assert abs(m.eps_indel - 0.005) < 1e-4
    # run-length scaling recovered by the weighted linear fit
    assert abs(m.epsilon_homopolymer(3) - 0.02) < 5e-3
    assert abs(m.epsilon_homopolymer(6) - 0.06) < 5e-3


def test_empirical_and_betabinom_more_conservative(synthetic_panel):
    table, *_ = synthetic_panel
    from collections import Counter

    default = assemble_clusters(table, identity=0.97, predict_orfs=False)
    empirical = assemble_clusters(
        table, identity=0.97, predict_orfs=False, fit_empirical=True
    )
    betabinom = assemble_clusters(
        table, identity=0.97, predict_orfs=False, overdispersion=0.1
    )
    n_real = lambda r: Counter(r.variants.column("call").to_pylist())["real"]  # noqa: E731
    # empirical + overdispersion both tighten the real-variant calls vs the
    # default prior (they fit / widen the error null), never loosen.
    assert n_real(empirical) <= n_real(default)
    assert n_real(betabinom) <= n_real(default)
    # the variant table still validates against the schema
    assert set(empirical.variants.column("call").to_pylist()) <= {
        "real",
        "ambiguous",
        "collapsed_error",
    }


def test_in_core_excludes_terminal_length_variation():
    rng = np.random.default_rng(3)
    base = _rand(rng, 700)
    p = 350
    alt = "A" if base[p] != "A" else "C"
    rows, rid = [], 0
    for i in range(200):
        s = list(base)
        if i < 80:
            s[p] = alt
        s = "".join(s)
        s = s[int(rng.integers(0, 25)) : len(s) - int(rng.integers(0, 25))]  # ragged
        rows.append((f"r{rid}", s, 0))
        rid += 1
    table = pa.table(
        {
            "read_id": [r[0] for r in rows],
            "sequence": [r[1] for r in rows],
            "sample_id": pa.array([r[2] for r in rows], pa.int64()),
        }
    )
    res = assemble_clusters(table, identity=0.95, predict_orfs=False)
    # the variant catalog carries the in_core flag
    assert "in_core" in res.variants.column_names
    clen = max(
        len(s or "") for s in res.clusters.column("consensus_sequence").to_pylist()
    )
    assert res.haplotypes.num_rows > 0
    vps = res.haplotypes.column("variant_positions").to_pylist()[0]
    # haplotype columns are interior (no terminal 5'/3' ramp positions) …
    assert all(15 <= q <= clen - 15 for q in vps)
    # … and base substitutions only (no gap/length columns)
    v = {r["consensus_pos"]: r for r in res.variants.to_pylist()}
    assert all(v[q]["minor_allele"] in "ACGT" for q in vps)
    # the planted ~40% interior SNP is captured as a real, phaseable column
    assert any(
        v[q]["minor_allele"] in "ACGT" and 0.3 <= v[q]["minor_fraction"] <= 0.5
        for q in vps
    )
    # terminal gap "variants" are catalogued but kept out of the haplotype columns
    term_gaps = [
        cp
        for cp, r in v.items()
        if r["minor_allele"] == "-" and (cp < 8 or cp > clen - 8)
    ]
    assert all(cp not in vps for cp in term_gaps)


def test_emit_alignments(synthetic_panel):
    from constellation.sequencing.transcriptome.cluster.denovo._cigar import parse_cigar

    table, *_ = synthetic_panel
    res = assemble_clusters(
        table, identity=0.96, predict_orfs=False, emit_alignments=True
    )
    aln = res.alignments
    assert aln.num_rows > 0
    assert set(aln.column("role").to_pylist()) <= {"representative", "member"}
    rep = aln.filter(pc.equal(aln.column("role"), "representative"))
    assert set(rep.column("cluster_id").to_pylist()) <= set(
        res.clusters.column("cluster_id").to_pylist()
    )
    # centroid frame length per cluster = the representative row's n_match
    # (the centroid aligned to itself as "<len>=").
    centroid_len = dict(
        zip(rep.column("cluster_id").to_pylist(), rep.column("n_match").to_pylist())
    )
    for r in aln.to_pylist():
        ref_consumed = sum(
            n for n, op in parse_cigar(r["cigar"]) if op in ("=", "X", "D", "M")
        )
        # CIGAR ref span matches the match/mismatch/delete counts and the
        # member→centroid alignment stays inside the centroid frame.
        assert ref_consumed == r["n_match"] + r["n_mismatch"] + r["n_delete"]
        assert r["ref_start"] >= 0
        assert r["ref_start"] + ref_consumed <= centroid_len[r["cluster_id"]]
        assert 0.0 <= r["identity"] <= 1.0
    # default run does not emit alignments
    res2 = assemble_clusters(table, identity=0.96, predict_orfs=False)
    assert res2.alignments.num_rows == 0


def test_cluster_transcripts_io(tmp_path):
    rng = np.random.default_rng(7)
    trues = [_rand(rng, int(rng.integers(500, 800))) for _ in range(20)]
    rows, rid = [], 0
    for T in trues:
        for _ in range(15):
            rows.append(
                (f"r{rid}", _mutate(rng, T, err=0.006), int(rng.integers(0, 2)))
            )
            rid += 1
    demux = _write_demux_dir(tmp_path, rows)
    out = tmp_path / "out"
    paths = cluster_transcripts(
        demux, output_dir=out, identity=0.97, predict_orfs=True, min_aa_length=40
    )
    assert (out / "_SUCCESS").exists()
    clusters = pq.read_table(paths["clusters"])
    assert 18 <= clusters.num_rows <= 24  # ~20 transcripts
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["kind"] == "cluster"
    assert manifest["align_dir"] == ""
    assert manifest["parameters"]["mode"] == "de-novo"
    # FASTA + protein + counts + variants + report emitted
    assert (out / "cluster.fa").read_text().count(">") == clusters.num_rows
    assert (out / "cluster_counts.tsv").exists()
    assert (out / "cluster_variants.parquet").exists()
    variants = pq.read_table(out / "cluster_variants.parquet")
    assert set(variants.column("call").to_pylist()) <= {
        "real",
        "ambiguous",
        "collapsed_error",
    }
    assert (out / "cluster_haplotypes.parquet").exists()
    assert (out / "diagnostics" / "report.md").exists()
    report = (out / "diagnostics" / "report.md").read_text()
    assert "Variant composition" in report and "Haplotypes" in report


def test_haplotype_columns_gated_on_real_call():
    # Only FDR-supported ('real') in-core substitutions define haplotypes.
    # A low-frequency in-core substitution that the caller classifies as
    # 'ambiguous'/'collapsed_error' (minor mass consistent with error) must be
    # catalogued in cluster_variants but kept OUT of the haplotype columns —
    # otherwise a handful of error reads fragment a clean cluster.
    rng = np.random.default_rng(11)
    base = _rand(rng, 600)
    p_real, p_noise = 300, 150
    a_real = next(b for b in "ACGT" if b != base[p_real])
    a_noise = next(b for b in "ACGT" if b != base[p_noise])
    rows = []
    for i in range(300):
        s = list(base)
        if i < 120:  # 40% -> a real, phaseable column
            s[p_real] = a_real
        if i < 4:  # ~1.3% in-core substitution -> not 'real'
            s[p_noise] = a_noise
        rows.append((f"r{i}", "".join(s), 0))
    table = pa.table(
        {
            "read_id": [r[0] for r in rows],
            "sequence": [r[1] for r in rows],
            "sample_id": pa.array([r[2] for r in rows], pa.int64()),
        }
    )
    res = assemble_clusters(table, identity=0.97, predict_orfs=False)
    v = {r["consensus_pos"]: r for r in res.variants.to_pylist()}
    # both positions are tested and in-core; only the 40% one is 'real'
    assert v[p_real]["call"] == "real"
    assert p_noise in v and v[p_noise]["in_core"] and v[p_noise]["call"] != "real"
    hap_positions = set()
    for vps in res.haplotypes.column("variant_positions").to_pylist():
        hap_positions.update(vps)
    assert p_real in hap_positions  # real column kept
    assert p_noise not in hap_positions  # non-real in-core substitution dropped
    # with only the single real column, the cluster resolves to 2 haplotypes,
    # not the 3+ that the noise column would have manufactured
    assert res.haplotypes.num_rows == 2
