"""I/O for the de novo cluster stage — window loading + output writing.

Kept separate from :mod:`.pipeline` so the pure ``assemble_clusters``
core has no filesystem or demux-format dependency (and is unit-testable
on an in-memory reads table).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from constellation.sequencing.align.map import (
    _iter_demux_read_batches,
    transcript_window_buffers,
)


_READS_SCHEMA = pa.schema(
    [
        pa.field("read_id", pa.string(), nullable=False),
        pa.field("sequence", pa.large_string(), nullable=False),
        pa.field("sample_id", pa.int64(), nullable=True),
        # The `qs:f` Dorado tag, null on demux dirs that predate it. Used by
        # the EM seeder (quality selects read accuracy ~4x; length does not)
        # and by the round-1 assignment tie-break.
        pa.field("dorado_quality", pa.float32(), nullable=True),
    ]
)


def _trim_batch(batch: pa.RecordBatch) -> pa.Table:
    """Extract the transcript window of each read, orientation-aware.

    Delegates the gather to :func:`transcript_window_buffers` so this
    stage cannot drift from the align / FASTQ consumers: the window
    offsets index the chosen-orientation frame, so a ``'-'`` read's
    window is the reverse-complement of an interval of the stored bytes.
    Slicing the stored bytes directly used to put reverse-oriented reads
    in clusters of their own instead of joining their forward twins.
    """
    buf, new_off = transcript_window_buffers(batch)
    n = batch.num_rows
    if new_off[-1] == 0:
        window = pa.array([""] * n, type=pa.large_string())
    else:
        window = pa.LargeStringArray.from_buffers(
            length=n,
            value_offsets=pa.py_buffer(new_off.tobytes()),
            data=pa.py_buffer(buf.tobytes()),
            null_bitmap=None,
            null_count=0,
        )
    names = set(batch.schema.names)
    sample = (
        batch.column("sample_id") if "sample_id" in names else pa.nulls(n, pa.int64())
    )
    quality = (
        batch.column("dorado_quality")
        if "dorado_quality" in names
        else pa.nulls(n, pa.float32())
    )
    return pa.table(
        {
            "read_id": batch.column("read_id"),
            "sequence": window,
            "sample_id": pa.array(sample).cast(pa.int64()),
            "dorado_quality": pa.array(quality).cast(pa.float32()),
        },
        schema=_READS_SCHEMA,
    )


def load_demux_windows(
    demux_dir: Path, *, max_window_length: int | None = None
) -> tuple[pa.Table, dict[str, int]]:
    """Stream + trim Complete, non-fragment transcript windows from a demux dir.

    Returns ``(reads, stats)``.

    ``max_window_length`` drops windows longer than it. This is **the** place
    to do it, not the seeding stage: three consumers read this corpus — the
    ORF seeder (a table), the reads FASTA the E-step aligns (a path), and the
    per-read sequence map the M-step projects into its PWM. Filtering at
    seeding removes an oversized read from the first only. It still wins a
    banded hit, still lands on a template, and still contributes its unaligned
    flank as a terminal extension event, and ``min_insertion_support`` guards
    only against a *singleton* — concatemers come in classes, so a pair of
    40 kb reads is enough to reserve a 40 kb block on someone else's template.

    Filtering here also runs before ``dereplicate``, so unique abundances are
    right without a remap.

    Oversized windows are **dropped, never trimmed**: a 361,908 nt "cDNA" is a
    concatemer, and truncating it to 15 kb manufactures a read that was never
    sequenced.

    ``None`` (the library default) disables the filter, so the returned table
    is byte-identical to the pre-filter behaviour.
    """
    parts: list[pa.Table] = []
    n_input = 0
    n_dropped_long = 0
    max_seen = 0
    for batch in _iter_demux_read_batches(demux_dir, only_complete=True):
        if batch.num_rows == 0:
            continue
        trimmed = _trim_batch(batch)
        n_input += trimmed.num_rows
        lengths = pc.utf8_length(trimmed.column("sequence"))
        if trimmed.num_rows:
            max_seen = max(max_seen, int(pc.max(lengths).as_py() or 0))
        if max_window_length is not None and max_window_length > 0:
            keep = pc.less_equal(lengths, max_window_length)
            n_dropped_long += trimmed.num_rows - int(pc.sum(keep).as_py() or 0)
            trimmed = trimmed.filter(keep)
        parts.append(trimmed)
    stats = {
        "n_input": n_input,
        "n_dropped_long": n_dropped_long,
        "max_input_length": max_seen,
    }
    if not parts:
        return _READS_SCHEMA.empty_table(), stats
    return pa.concat_tables(parts), stats


def _write_fasta(path: Path, ids: list[str], seqs: list[str]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for name, seq in zip(ids, seqs):
            if not seq:
                continue
            fh.write(f">{name}\n{seq}\n")


def _sample_name_map(samples: Any | None) -> dict[int, str]:
    if samples is None:
        return {}
    try:
        tbl = samples.samples
        return {
            int(sid): str(name)
            for sid, name in zip(
                tbl.column("sample_id").to_pylist(),
                tbl.column("sample_name").to_pylist(),
            )
        }
    except Exception:
        return {}


def _write_counts_tsv(
    path: Path, feature_quant: pa.Table, name_map: dict[int, str]
) -> None:
    """Wide cluster × sample count matrix."""
    if feature_quant.num_rows == 0:
        # Still emit the registry's columns. Returning a bare header
        # dropped every registered sample whenever all clusters were
        # filtered out, so an empty result had a DIFFERENT schema from a
        # populated one — the same schema-depends-on-content problem the
        # populated path had.
        headers = ["cluster_id"] + [
            name_map.get(s, f"sample_{s}") for s in sorted(name_map)
        ]
        path.write_text("\t".join(headers) + "\n", encoding="utf-8")
        return
    cid = feature_quant.column("feature_id").to_numpy(zero_copy_only=False)
    sid = feature_quant.column("sample_id").to_numpy(zero_copy_only=False)
    cnt = feature_quant.column("count").to_numpy(zero_copy_only=False)
    clusters = sorted(set(int(c) for c in cid))
    # Union with the persisted sample registry, not just what was
    # observed. A sample whose reads all failed to survive into a
    # cluster has no feature_quant row, and deriving the columns from
    # that table alone silently dropped it — making the matrix schema
    # depend on the counts it contains. An all-zero column is the
    # correct answer and keeps downstream comparisons aligned.
    samples = sorted(set(int(s) for s in sid) | {int(s) for s in name_map})
    row_of = {c: i for i, c in enumerate(clusters)}
    col_of = {s: i for i, s in enumerate(samples)}
    mat = np.zeros((len(clusters), len(samples)), dtype=np.int64)
    for c, s, v in zip(cid, sid, cnt):
        mat[row_of[int(c)], col_of[int(s)]] = int(v)
    headers = ["cluster_id"] + [name_map.get(s, f"sample_{s}") for s in samples]
    with path.open("w", encoding="utf-8") as fh:
        fh.write("\t".join(headers) + "\n")
        for i, c in enumerate(clusters):
            fh.write("\t".join([str(c)] + [str(int(x)) for x in mat[i]]) + "\n")


def write_outputs(
    result: Any,
    *,
    output_dir: Path,
    demux_dir: Path,
    samples: Any | None,
    write_fasta: bool,
    predict_orfs: bool,
    parameters: dict[str, Any],
    emit_cluster_detail: bool = False,
    detail_top_n: int = 50,
) -> dict[str, Path]:
    from constellation.sequencing.transcriptome.manifest import write_cluster_manifest

    output_dir = Path(output_dir)
    outputs: dict[str, str] = {}
    paths: dict[str, Path] = {}

    clusters_path = output_dir / "clusters.parquet"
    pq.write_table(result.clusters, clusters_path)
    outputs["clusters"] = "clusters.parquet"
    paths["clusters"] = clusters_path

    membership_path = output_dir / "cluster_membership.parquet"
    pq.write_table(result.membership, membership_path)
    outputs["cluster_membership"] = "cluster_membership.parquet"
    paths["membership"] = membership_path

    fq_path = output_dir / "feature_quant.parquet"
    pq.write_table(result.feature_quant, fq_path)
    outputs["feature_quant"] = "feature_quant.parquet"
    paths["feature_quant"] = fq_path

    variants_path = output_dir / "cluster_variants.parquet"
    pq.write_table(result.variants, variants_path)
    outputs["cluster_variants"] = "cluster_variants.parquet"
    paths["variants"] = variants_path

    haplotypes = getattr(result, "haplotypes", None)
    if haplotypes is not None:
        hap_path = output_dir / "cluster_haplotypes.parquet"
        pq.write_table(haplotypes, hap_path)
        outputs["cluster_haplotypes"] = "cluster_haplotypes.parquet"
        paths["haplotypes"] = hap_path

    alignments = getattr(result, "alignments", None)
    if alignments is not None and alignments.num_rows:
        aln_path = output_dir / "cluster_alignments.parquet"
        pq.write_table(alignments, aln_path)
        outputs["cluster_alignments"] = "cluster_alignments.parquet"
        paths["alignments"] = aln_path

    name_map = _sample_name_map(samples)
    counts_path = output_dir / "cluster_counts.tsv"
    _write_counts_tsv(counts_path, result.feature_quant, name_map)
    outputs["cluster_counts"] = "cluster_counts.tsv"
    paths["cluster_counts"] = counts_path

    if write_fasta and result.clusters.num_rows:
        cids = result.clusters.column("cluster_id").to_pylist()
        cons = result.clusters.column("consensus_sequence").to_pylist()
        fa_path = output_dir / "cluster.fa"
        _write_fasta(fa_path, [f"cluster_{c}" for c in cids], [s or "" for s in cons])
        outputs["cluster_fa"] = "cluster.fa"
        paths["fasta"] = fa_path
        if predict_orfs:
            prot = result.clusters.column("predicted_protein").to_pylist()
            prot_ids = [f"cluster_{c}" for c, p in zip(cids, prot) if p]
            prot_seqs = [p for p in prot if p]
            prot_path = output_dir / "proteins.fasta"
            _write_fasta(prot_path, prot_ids, prot_seqs)
            outputs["proteins_fasta"] = "proteins.fasta"
            paths["proteins"] = prot_path

    if emit_cluster_detail:
        try:
            from constellation.sequencing.transcriptome.cluster.denovo.diagnostics import (  # noqa: E501
                emit_cluster_details,
            )

            emit_cluster_details(output_dir, top_n=detail_top_n)
            outputs["detail"] = "detail/"
        except Exception:  # noqa: BLE001 — detail never breaks a successful run
            pass

    sample_names = None
    if samples is not None:
        try:
            sample_names = sorted(
                set(samples.samples.column("sample_name").to_pylist())
            )
        except Exception:
            sample_names = None

    write_cluster_manifest(
        output_dir / "manifest.json",
        reference_handle=None,
        reference_path=None,
        assembly_accession=None,
        align_dir="",
        demux_dir=str(demux_dir),
        parameters=parameters,
        stages={
            "n_input_reads": int(result.n_input_reads),
            "n_unique_sequences": int(result.n_unique),
            "n_clusters": int(result.clusters.num_rows),
            "n_membership_rows": int(result.membership.num_rows),
        },
        outputs=outputs,
        samples=sample_names,
    )
    (output_dir / "_SUCCESS").write_bytes(b"")
    paths["manifest"] = output_dir / "manifest.json"
    return paths


__all__ = ["load_demux_windows", "write_outputs"]
