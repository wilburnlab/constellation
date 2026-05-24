"""Tier A tests for `constellation.massspec.search.novel`.

Synthetic inputs (no real .elib / FASTA fixtures required). One test per
classification class plus a couple of end-to-end coverage tests that
exercise the full `classify_novel_peptides` workflow on a hand-built
2-protein corpus.

The classifier logic is a direct port of cartographer's CIGAR-walking
algorithm — see `cartographer/data/nanopore.py` lines 685-850 for the
reference implementation.
"""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pytest

from constellation.core.io.schemas import (
    ALIGNMENT_HIT_TABLE,
    read_mmseqs_tab,
)
from constellation.massspec.search.novel import (
    _CLASSIFICATION_PRIORITY,
    classify_novel_peptides,
    classify_single_peptide,
    read_fasta_proteins,
)


# ──────────────────────────────────────────────────────────────────────
# classify_single_peptide — one synthetic case per class
# ──────────────────────────────────────────────────────────────────────


def _hit(query="N1", target="R1", qstart=1, qend=10, tstart=1, tend=10,
         cigar="10M", tier="reference"):
    """Build a hit dict shaped like one row of ALIGNMENT_HIT_TABLE."""
    return {
        "query": query, "target": target, "evalue": 1e-50,
        "qstart": qstart, "qend": qend, "tstart": tstart, "tend": tend,
        "cigar": cigar, "alignment_tier": tier,
    }


def test_classify_snp() -> None:
    """Single mismatch in the peptide span → snp."""
    # Novel protein has K at position 5 (1-indexed); reference has R.
    # Peptide spans positions 3-7 (0-indexed 2-7) which is inside the
    # 10-residue alignment.
    novel    = "MAGCKLPPGR"  # peptide G[CKL]P → "GCKLP"
    ref      = "MAGCRLPPGR"  # same positions but with R instead of K
    hit = _hit(qstart=1, qend=10, tstart=1, tend=10, cigar="10M")
    refs = {"R1": ref}
    cls, ref_seq = classify_single_peptide("GCKLP", novel, hit, refs)
    assert cls == "snp"
    assert ref_seq == "GCRLP"  # reference at the peptide span


def test_classify_insertion() -> None:
    """Inserted residue in query within peptide span → insertion."""
    # 3-residue insertion in the middle of the alignment.
    # Novel: MAGCK AAA LPPGR (positions 1-13)
    # Ref:   MAGCK     LPPGR (positions 1-10)
    # CIGAR (query-centric): 5M 3I 5M
    novel    = "MAGCKAAALPPGR"
    ref      = "MAGCKLPPGR"
    hit = _hit(qstart=1, qend=13, tstart=1, tend=10, cigar="5M3I5M")
    refs = {"R1": ref}
    # Peptide spans the inserted region: "CKAAAL"
    cls, _ = classify_single_peptide("CKAAAL", novel, hit, refs)
    assert cls == "insertion"


def test_classify_deletion() -> None:
    """Deleted residue in query (gap in novel) within peptide span → deletion."""
    # Novel:        MAG CK     LPPGR  (query positions 1-10)
    # Ref:          MAG CK AAA LPPGR  (target positions 1-13)
    # CIGAR (query-centric): 5M 3D 5M
    novel    = "MAGCKLPPGR"
    ref      = "MAGCKAAALPPGR"
    hit = _hit(qstart=1, qend=10, tstart=1, tend=13, cigar="5M3D5M")
    refs = {"R1": ref}
    cls, _ = classify_single_peptide("CKLPP", novel, hit, refs)
    assert cls == "deletion"


def test_classify_complex() -> None:
    """Two or more event types within the peptide span → complex."""
    # Mismatch + insertion within span
    # Novel:  M A G C K A A A L P P G R   (13 residues, K mismatch at pos 5)
    # Ref:    M A G C R           L P P G R  (10 residues, R at pos 5)
    # CIGAR: 5M 3I 5M (the 5 at the start contains the mismatch)
    novel    = "MAGCKAAALPPGR"
    ref      = "MAGCRLPPGR"
    hit = _hit(qstart=1, qend=13, tstart=1, tend=10, cigar="5M3I5M")
    refs = {"R1": ref}
    cls, _ = classify_single_peptide("CKAAAL", novel, hit, refs)
    assert cls == "complex"


def test_classify_n_terminal_truncation() -> None:
    """Peptide flush at novel N-terminus; reference extends further upstream."""
    # Novel:  MKPR             (4 residues; peptide MKPR matches full novel)
    # Ref:    ...PRELUDEMKPR   (peptide MKPR is FOUND in ref but at offset > 0)
    # CIGAR all-match, no deviations
    novel = "MKPR"
    ref = "PRELUDEMKPR"
    hit = _hit(qstart=1, qend=4, tstart=8, tend=11, cigar="4M")
    refs = {"R1": ref}
    cls, _ = classify_single_peptide("MKPR", novel, hit, refs)
    assert cls == "n_terminal_truncation"


def test_classify_met_clipped_n_terminal_truncation() -> None:
    """A Met-clipped N-terminal peptide (initiator Met excised) that the
    reference extends upstream of must classify as n_terminal_truncation,
    not unknown/cutsite. The peptide starts at protein position 1 (the M
    sits at 0), so the effective-N-terminus rule must recognise it."""
    # Novel:  M[AAAK]GGGGR  → Met-clipped N-term peptide "AAAK" at pos 1
    # Ref:    PRELUDEAAAKGGGGR  → "AAAK" found at offset 7 (> pep_start 1)
    novel = "MAAAKGGGGR"
    ref = "PRELUDEAAAKGGGGR"
    # Alignment maps novel AAAK (pos 1-4) → ref AAAK (pos 7-10), all-match.
    hit = _hit(qstart=2, qend=5, tstart=8, tend=11, cigar="4M")
    refs = {"R1": ref}
    cls, _ = classify_single_peptide("AAAK", novel, hit, refs)
    assert cls == "n_terminal_truncation"


def test_classify_met_clipped_no_spurious_cutsite() -> None:
    """A Met-clipped N-terminal peptide whose left 'flank' is the excised
    Met must NOT fire trypsin_cutsite_mutation off that M — the left
    boundary is the protein start, not a tryptic cut. Only a right-flank
    difference is a real cutsite mutation."""
    # Novel:  M[AAAK]GGGGR   peptide "AAAK" at pos 1; right flank novel = 'G'
    # Ref:    Q[AAAK]GGGGR   "AAAK" at ref pos 1; right flank ref = 'G' (same)
    # Left flanks differ (novel 'M' vs ref 'Q') but that's the N-terminus,
    # so it must NOT be called a cutsite mutation. Peptide is at the
    # reference's own N-terminus (pep_start_ref == pep_start == 1) so it's
    # not a truncation either → falls through to unknown (no real event).
    novel = "MAAAKGGGGR"
    ref = "QAAAKGGGGR"
    hit = _hit(qstart=2, qend=5, tstart=2, tend=5, cigar="4M")
    refs = {"R1": ref}
    cls, _ = classify_single_peptide("AAAK", novel, hit, refs)
    # The left-flank M/Q difference is the N-terminus, not a cut site,
    # so cutsite must NOT fire; right flanks match → no event → unknown.
    assert cls != "trypsin_cutsite_mutation"


def test_classify_c_terminal_truncation() -> None:
    """Peptide flush at novel C-terminus; reference extends further downstream."""
    novel = "MKPR"
    ref = "MKPRPOSTLUDE"
    hit = _hit(qstart=1, qend=4, tstart=1, tend=4, cigar="4M")
    refs = {"R1": ref}
    cls, _ = classify_single_peptide("MKPR", novel, hit, refs)
    assert cls == "c_terminal_truncation"


def test_classify_trypsin_cutsite_mutation() -> None:
    """Peptide is identical in novel and reference but a flanking AA
    differs — i.e. a SNP outside the peptide span created (or removed)
    a tryptic cut site. The cartographer rule: novel_left != ref_left
    OR novel_right != ref_right (peptide_span all-match)."""
    # Novel:  AAA K LIGHTPEPT R BBB    (position 3 = 'K' — cut site)
    # Ref:    AAA P LIGHTPEPT R BBB    (position 3 = 'P' — NO cut site)
    # Both proteins are 17 residues; CIGAR all-match across them.
    # Position 3 is OUTSIDE the peptide span (4..13) so has_mismatch
    # stays False inside the peptide → falls through to all-match
    # branch → flank check detects the K vs P difference at pep_start-1.
    novel = "AAAKLIGHTPEPTRBBB"
    ref   = "AAAPLIGHTPEPTRBBB"
    hit = _hit(qstart=1, qend=17, tstart=1, tend=17, cigar="17M")
    refs = {"R1": ref}
    cls, _ = classify_single_peptide("LIGHTPEPT", novel, hit, refs)
    assert cls == "trypsin_cutsite_mutation"


def test_classify_n_term_deviation() -> None:
    """Peptide starts before the alignment range → n_term_deviation."""
    # 20-residue novel; alignment covers positions 11-20 (qstart=11)
    # Peptide is from positions 1-9 — entirely before the alignment.
    novel = "MAGCKLAPLRDGGGQRSTAV"  # 20 residues — valid AAs only
    ref = "DGGGQRSTAV"               # 10-residue reference, aligns to novel[10:20]
    hit = _hit(qstart=11, qend=20, tstart=1, tend=10, cigar="10M")
    refs = {"R1": ref}
    cls, _ = classify_single_peptide("MAGCKLAPL", novel, hit, refs)
    assert cls == "n_term_deviation"


def test_classify_c_term_deviation() -> None:
    """Peptide ends after the alignment range → c_term_deviation."""
    novel = "MAGCKLAPLRDGGGQRSTAV"   # 20 residues — valid AAs only
    ref = "MAGCKLAPLR"                # 10-residue ref, aligns to novel[0:10]
    hit = _hit(qstart=1, qend=10, tstart=1, tend=10, cigar="10M")
    refs = {"R1": ref}
    cls, _ = classify_single_peptide("GGGQRSTAV", novel, hit, refs)
    assert cls == "c_term_deviation"


def test_classify_unknown_no_hit() -> None:
    """No alignment hit for the protein → unknown via the batch path."""
    # NOVEL_NO_HIT tryptic digest produces "LIGHTPEPTIDK" as the
    # post-MAGCR cut peptide. Reference has no equivalent so the
    # peptide is in theoretical_novel; with no hit in alignments,
    # the batch path emits "unknown".
    detected = pa.table({"peptide_sequence": pa.array(["LIGHTPEPTIDK"])})
    novel = pa.table(
        {
            "protein_id": pa.array(["NOVEL_NO_HIT"]),
            "sequence": pa.array(["MAGCRLIGHTPEPTIDKRAWN"]),
        }
    )
    reference = pa.table(
        {
            "protein_id": pa.array(["REF1"]),
            "sequence": pa.array(["MAGYRDIFFERENTREFPRTEIN"]),
        }
    )
    alignments = ALIGNMENT_HIT_TABLE.empty_table()
    result = classify_novel_peptides(
        detected_peptides=detected,
        alignments=alignments,
        reference_proteins=reference,
        novel_proteins=novel,
    )
    classifications = result.column("classification").to_pylist()
    assert classifications == ["unknown"]


def test_classify_non_reference() -> None:
    """Hit target NOT in reference_proteins → non_reference."""
    novel = "MAGCKLIGHTPEPTKRAW"
    # Pass an alignment_tier explicitly set to anything other than
    # "reference" — classify_single_peptide short-circuits to
    # non_reference.
    hit = _hit(target="SWISSPROT_HIT", qstart=1, qend=18, tstart=1,
               tend=18, cigar="18M", tier="swissprot")
    refs = {"SWISSPROT_HIT": novel}
    cls, ref_seq = classify_single_peptide("CKLIGHTPEPTKR", novel, hit, refs)
    assert cls == "non_reference"
    assert ref_seq == ""


# ──────────────────────────────────────────────────────────────────────
# read_mmseqs_tab — schema + parsing
# ──────────────────────────────────────────────────────────────────────


def test_read_mmseqs_tab_basic(tmp_path: Path) -> None:
    """Parse a 3-line synthetic mmseqs2 .tab into ALIGNMENT_HIT_TABLE."""
    tab = tmp_path / "alignments.tab"
    tab.write_text(
        "NOVEL1\tREF1\t1.2e-30\t1\t100\t1\t100\t100M\n"
        "NOVEL2\tREF2\t3.4e-25\t1\t80\t10\t89\t40M5I40M\n"
        "NOVEL3\tSWISS_X\t9.9e-18\t5\t50\t1\t46\t46M\n"
    )
    table = read_mmseqs_tab(tab, alignment_tier="reference")
    assert table.num_rows == 3
    assert table.schema.equals(ALIGNMENT_HIT_TABLE)
    assert table.column("query").to_pylist() == ["NOVEL1", "NOVEL2", "NOVEL3"]
    assert table.column("target").to_pylist() == ["REF1", "REF2", "SWISS_X"]
    assert table.column("cigar").to_pylist() == ["100M", "40M5I40M", "46M"]
    assert table.column("alignment_tier").to_pylist() == ["reference"] * 3
    # Coordinates are int64 (mmseqs2 1-indexed convention preserved)
    assert table.column("qstart").to_pylist() == [1, 1, 5]


def test_read_mmseqs_tab_no_tier(tmp_path: Path) -> None:
    """alignment_tier left null when caller doesn't specify."""
    tab = tmp_path / "alignments.tab"
    tab.write_text("Q\tT\t1e-50\t1\t10\t1\t10\t10M\n")
    table = read_mmseqs_tab(tab)
    assert table.column("alignment_tier").to_pylist() == [None]


def test_read_mmseqs_tab_empty(tmp_path: Path) -> None:
    tab = tmp_path / "empty.tab"
    tab.touch()
    table = read_mmseqs_tab(tab)
    assert table.num_rows == 0
    assert table.schema.equals(ALIGNMENT_HIT_TABLE)


def test_read_mmseqs_tab_missing_file(tmp_path: Path) -> None:
    table = read_mmseqs_tab(tmp_path / "does_not_exist.tab")
    assert table.num_rows == 0
    assert table.schema.equals(ALIGNMENT_HIT_TABLE)


def test_read_mmseqs_tab_with_header(tmp_path: Path) -> None:
    """A header row is detected (evalue column non-numeric) + skipped."""
    tab = tmp_path / "alignments.tab"
    tab.write_text(
        "query\ttarget\tevalue\tqstart\tqend\ttstart\ttend\tcigar\n"
        "NOVEL1\tREF1\t1.2e-30\t1\t100\t1\t100\t100M\n"
        "NOVEL2\tREF2\t3.4e-25\t1\t80\t10\t89\t80M\n"
    )
    table = read_mmseqs_tab(tab)
    assert table.num_rows == 2
    assert table.column("query").to_pylist() == ["NOVEL1", "NOVEL2"]
    assert table.column("evalue").to_pylist()[0] == pytest.approx(1.2e-30)


def test_read_mmseqs_tab_ninth_tier_column(tmp_path: Path) -> None:
    """A 9th alignment_tier column (pre-tiered .tab) is read directly,
    and the file value wins over the kwarg default."""
    tab = tmp_path / "alignments.tab"
    tab.write_text(
        "NOVEL1\tREF1\t1.2e-30\t1\t100\t1\t100\t100M\treference\n"
        "NOVEL2\tSWISS_X\t9.9e-18\t5\t50\t1\t46\t46M\tnon_reference\n"
    )
    table = read_mmseqs_tab(tab)
    assert table.num_rows == 2
    assert table.column("alignment_tier").to_pylist() == [
        "reference", "non_reference",
    ]


def test_read_mmseqs_tab_header_and_tier(tmp_path: Path) -> None:
    """Header + 9th alignment_tier column together — the exact shape the
    lab's pre-tiered .tab files have (regression for the OSC failure)."""
    tab = tmp_path / "alignments.tab"
    tab.write_text(
        "query\ttarget\tevalue\tqstart\tqend\ttstart\ttend\tcigar\talignment_tier\n"
        "NOVEL1\tREF1\t1.2e-30\t1\t100\t1\t100\t100M\treference\n"
        "NOVEL2\tSWISS_X\t9.9e-18\t5\t50\t1\t46\t46M\tnon_reference\n"
    )
    table = read_mmseqs_tab(tab)
    assert table.num_rows == 2
    assert table.schema.equals(ALIGNMENT_HIT_TABLE)
    assert table.column("query").to_pylist() == ["NOVEL1", "NOVEL2"]
    assert table.column("alignment_tier").to_pylist() == [
        "reference", "non_reference",
    ]


def test_read_mmseqs_tab_quoted_header_and_fields(tmp_path: Path) -> None:
    """Header + data quoted (pandas QUOTE_ALL style) still parses — the
    evalue-numeric sniff strips quotes before the float check."""
    tab = tmp_path / "alignments.tab"
    tab.write_text(
        '"query"\t"target"\t"evalue"\t"qstart"\t"qend"\t"tstart"\t"tend"\t"cigar"\t"alignment_tier"\n'
        '"NOVEL1"\t"REF1"\t"1.2e-30"\t"1"\t"100"\t"1"\t"100"\t"100M"\t"reference"\n'
    )
    table = read_mmseqs_tab(tab)
    assert table.num_rows == 1
    assert table.column("query").to_pylist() == ["NOVEL1"]
    assert table.column("alignment_tier").to_pylist() == ["reference"]


# ──────────────────────────────────────────────────────────────────────
# Tier inference (target ∈ reference_proteins → "reference")
# ──────────────────────────────────────────────────────────────────────


def test_tier_inference_marks_reference_targets_as_reference() -> None:
    """When alignment_tier is null on input, the classifier infers
    tier from target membership in reference_proteins. Tests that the
    OUTPUT row carries forward the inferred tier."""
    # snp within the peptide span: novel has L at position 9, ref has A.
    # Both proteins yield a unique novel peptide post-K cut.
    novel = pa.table(
        {
            "protein_id": pa.array(["NOVEL1"]),
            "sequence": pa.array(["MAGCKLLLLLPPGR"]),  # 14 residues
        }
    )
    reference = pa.table(
        {
            "protein_id": pa.array(["REF1"]),
            # K→K kept (still cuts), L at pos 9 → A so the post-K peptide
            # differs by 1 AA between novel and reference.
            "sequence": pa.array(["MAGCKLLLLAPPGR"]),
        }
    )
    alignments = pa.table(
        {
            "query": ["NOVEL1"],
            "target": ["REF1"],
            "evalue": [1e-50],
            "qstart": [1],
            "qend": [14],
            "tstart": [1],
            "tend": [14],
            "cigar": ["14M"],
            "alignment_tier": pa.array([None], type=pa.string()),
        }
    )
    detected = pa.table(
        {"peptide_sequence": pa.array(["LLLLLPPGR"])}
    )
    result = classify_novel_peptides(
        detected, alignments, reference, novel,
    )
    # Should have one row classified as snp (single mismatch in peptide span)
    assert result.num_rows == 1
    assert result.column("classification").to_pylist() == ["snp"]
    # alignment_tier should be the inferred "reference" (target REF1 ∈ ref proteome)
    tier_vals = result.column("alignment_tier").to_pylist()
    assert tier_vals[0] == "reference"


def test_tier_inference_marks_non_reference_targets() -> None:
    """When target ∉ reference_proteins, tier infers to 'non_reference'
    and short-circuits to that classification."""
    novel = pa.table(
        {
            "protein_id": pa.array(["NOVEL1"]),
            "sequence": pa.array(["MAGCKWAVLPPGR"]),
        }
    )
    reference = pa.table(
        {
            "protein_id": pa.array(["REF_TOTALLY_DIFFERENT"]),
            "sequence": pa.array(["NREATEDSEQENCKLYDIF"]),  # canonical AAs only
        }
    )
    alignments = pa.table(
        {
            "query": ["NOVEL1"],
            "target": ["SWISSPROTHITNOTINREF"],
            "evalue": [1e-30],
            "qstart": [1],
            "qend": [13],
            "tstart": [1],
            "tend": [13],
            "cigar": ["13M"],
            "alignment_tier": pa.array([None], type=pa.string()),
        }
    )
    # WAVLPPGR is the post-K novel peptide (8 residues); not in reference's
    # tryptic digest.
    detected = pa.table(
        {"peptide_sequence": pa.array(["WAVLPPGR"])}
    )
    result = classify_novel_peptides(
        detected, alignments, reference, novel,
    )
    assert result.column("classification").to_pylist() == ["non_reference"]


def test_foreign_tier_vocabulary_recomputed_from_membership() -> None:
    """Regression for the OSC all-non_reference bug: the input .tab
    carries cartographer's ``refseq`` / ``swissprot`` tier vocabulary
    (not ``reference`` / ``non_reference``). The classifier must
    recompute the tier from target membership — a refseq hit whose
    target IS in the reference proteome must still CIGAR-walk to snp,
    not short-circuit to non_reference just because the literal string
    isn't ``"reference"``."""
    novel = pa.table(
        {
            "protein_id": pa.array(["NOVEL_R", "NOVEL_S"]),
            "sequence": pa.array(["MAGCKLLLLLPPGR", "MAGCKWAVLPPGR"]),
        }
    )
    reference = pa.table(
        {
            # REF1 is in the reference proteome (refseq target);
            # the swissprot target P00552 is NOT.
            "protein_id": pa.array(["REF1"]),
            "sequence": pa.array(["MAGCKLLLLAPPGR"]),
        }
    )
    alignments = pa.table(
        {
            "query": ["NOVEL_R", "NOVEL_S"],
            "target": ["REF1", "P00552"],   # refseq-in-ref, swissprot-not-in-ref
            "evalue": [1e-50, 1e-30],
            "qstart": [1, 1],
            "qend": [14, 13],
            "tstart": [1, 1],
            "tend": [14, 13],
            "cigar": ["14M", "13M"],
            # The foreign vocabulary that broke the real run:
            "alignment_tier": pa.array(["refseq", "swissprot"], type=pa.string()),
        }
    )
    detected = pa.table(
        {"peptide_sequence": pa.array(["LLLLLPPGR", "WAVLPPGR"])}
    )
    result = classify_novel_peptides(detected, alignments, reference, novel)
    by_pep = {
        r["peptide_sequence"]: r["classification"]
        for r in result.to_pylist()
    }
    # refseq hit (target ∈ reference) → CIGAR-walked → snp.
    assert by_pep["LLLLLPPGR"] == "snp"
    # swissprot hit (target ∉ reference) → non_reference.
    assert by_pep["WAVLPPGR"] == "non_reference"
    # And the OUTPUT tier carries the canonical recomputed vocabulary.
    tiers = {
        r["peptide_sequence"]: r["alignment_tier"] for r in result.to_pylist()
    }
    assert tiers["LLLLLPPGR"] == "reference"
    assert tiers["WAVLPPGR"] == "non_reference"


# ──────────────────────────────────────────────────────────────────────
# Deduplication by classification priority
# ──────────────────────────────────────────────────────────────────────


def test_deduplication_keeps_most_specific_class() -> None:
    """If the same peptide appears in two novel proteins with different
    classifications, the more-specific (lower priority value) class
    survives. Priority: snp (0) < insertion (1).

    Both novel proteins have an identical sequence — the same tryptic
    peptide ``LIGHTPEPTR`` reaches the digest of both. The alignments
    differ:
      NOVEL_A → REF_A: same-length protein with a snp inside the
                       peptide span (T6→A6) → "snp"
      NOVEL_B → REF_B: ref is shorter by one residue (no I at pos 3)
                       so query has an insertion at pos 3 within the
                       peptide span → "insertion"
    Same peptide, different classifications. snp wins per priority.
    """
    novel = pa.table(
        {
            "protein_id": pa.array(["NOVEL_A", "NOVEL_B"]),
            "sequence": pa.array(
                [
                    "MKLIGHTPEPTRAAA",  # 15 residues
                    "MKLIGHTPEPTRAAA",  # same sequence — both yield the same tryptic peptides
                ]
            ),
        }
    )
    reference = pa.table(
        {
            "protein_id": pa.array(["REF_A", "REF_B"]),
            "sequence": pa.array(
                [
                    # T at pos 6 of novel → A at pos 6 of REF_A (snp in peptide span)
                    "MKLIGHAPEPTRAAA",   # 15 residues
                    # REF_B missing the I at pos 3 — alignment shows
                    # NOVEL_B has an insertion within the peptide span
                    "MKLGHTPEPTRAAA",    # 14 residues
                ]
            ),
        }
    )
    alignments = pa.table(
        {
            "query": ["NOVEL_A", "NOVEL_B"],
            "target": ["REF_A", "REF_B"],
            "evalue": [1e-50, 1e-50],
            "qstart": [1, 1],
            "qend": [15, 15],
            "tstart": [1, 1],
            "tend": [15, 14],
            "cigar": ["15M", "3M1I11M"],
            "alignment_tier": pa.array([None, None], type=pa.string()),
        }
    )
    # NOVEL tryptic digest produces "LIGHTPEPTR" as the post-K peptide.
    detected = pa.table({"peptide_sequence": pa.array(["LIGHTPEPTR"])})
    result = classify_novel_peptides(
        detected, alignments, reference, novel,
        min_peptide_length=5,
    )
    # One row per unique peptide_sequence (no modforms recorded → single
    # row via the (None, None) fallback)
    assert result.num_rows == 1
    # snp (0) < insertion (1) → snp wins
    assert result.column("classification").to_pylist() == ["snp"]


# ──────────────────────────────────────────────────────────────────────
# Modform-level output (one row per detected modified_sequence)
# ──────────────────────────────────────────────────────────────────────


def test_output_keyed_at_modform_level() -> None:
    """A bare sequence detected as multiple modforms (e.g. N-terminal
    acetylated + bare) yields one row per modform, all sharing the bare
    sequence's classification — proteoforms stay distinct for per-
    precursor quant instead of collapsing to the bare-sequence level."""
    novel = pa.table(
        {
            "protein_id": pa.array(["NOVEL1"]),
            "sequence": pa.array(["MAGCKLLLLLPPGR"]),
        }
    )
    reference = pa.table(
        {
            "protein_id": pa.array(["REF1"]),
            "sequence": pa.array(["MAGCKLLLLAPPGR"]),  # snp in the peptide span
        }
    )
    alignments = pa.table(
        {
            "query": ["NOVEL1"],
            "target": ["REF1"],
            "evalue": [1e-50],
            "qstart": [1], "qend": [14],
            "tstart": [1], "tend": [14],
            "cigar": ["14M"],
            "alignment_tier": pa.array([None], type=pa.string()),
        }
    )
    # The same bare peptide detected as two proteoforms: N-term acetyl
    # and bare. Both must survive as separate output rows.
    detected = pa.table(
        {
            "peptide_sequence": pa.array(["LLLLLPPGR", "LLLLLPPGR"]),
            "modified_sequence": pa.array(
                ["[Acetyl]-LLLLLPPGR", "LLLLLPPGR"]
            ),
        }
    )
    result = classify_novel_peptides(detected, alignments, reference, novel)
    assert result.num_rows == 2
    rows = result.to_pylist()
    # Both rows share the bare sequence + classification...
    assert {r["peptide_sequence"] for r in rows} == {"LLLLLPPGR"}
    assert {r["classification"] for r in rows} == {"snp"}
    # ...but carry the two distinct modforms.
    assert {r["modified_sequence"] for r in rows} == {
        "[Acetyl]-LLLLLPPGR", "LLLLLPPGR",
    }


def test_reference_selenoprotein_not_falsely_novel() -> None:
    """A reference protein carrying a non-canonical residue (selenocysteine
    U) must still contribute ALL its canonical tryptic peptides to the
    reference digest. Otherwise a peptide upstream of the U is never
    subtracted and looks 'novel' despite being byte-identical to the
    reference. Regression for the OSC unknown-that-isn't-novel case."""
    # Reference protein: a clean peptide (LIGHTPEPTK) then a C-terminal
    # selenocysteine. Novel protein: the same but N-terminally truncated
    # (missing the reference's leading MGGGR), and WITHOUT the U so it
    # digests cleanly on its own.
    shared = "LIGHTPEPTK"
    reference = pa.table(
        {
            "protein_id": pa.array(["REF1"]),
            "sequence": pa.array(["MGGGR" + shared + "AAAKU"]),  # U at the end
        }
    )
    novel = pa.table(
        {
            "protein_id": pa.array(["NOVEL1"]),
            "sequence": pa.array([shared + "AAAK"]),  # truncated, canonical
        }
    )
    alignments = pa.table(
        {
            "query": ["NOVEL1"],
            "target": ["REF1"],
            "evalue": [1e-50],
            "qstart": [1], "qend": [14],
            "tstart": [6], "tend": [19],
            "cigar": ["14M"],
            "alignment_tier": pa.array([None], type=pa.string()),
        }
    )
    detected = pa.table({"peptide_sequence": pa.array([shared])})
    result = classify_novel_peptides(detected, alignments, reference, novel)
    # LIGHTPEPTK is a clean tryptic peptide of the reference (MGGGR | ...);
    # it must be subtracted as reference, NOT classified as novel.
    assert result.num_rows == 0


def test_output_single_row_when_no_modforms() -> None:
    """No modified_sequence column → one row per bare sequence (the
    pre-change behaviour), with a null modified_sequence."""
    novel = pa.table(
        {"protein_id": pa.array(["NOVEL1"]), "sequence": pa.array(["MAGCKLLLLLPPGR"])}
    )
    reference = pa.table(
        {"protein_id": pa.array(["REF1"]), "sequence": pa.array(["MAGCKLLLLAPPGR"])}
    )
    alignments = pa.table(
        {
            "query": ["NOVEL1"], "target": ["REF1"], "evalue": [1e-50],
            "qstart": [1], "qend": [14], "tstart": [1], "tend": [14],
            "cigar": ["14M"], "alignment_tier": pa.array([None], type=pa.string()),
        }
    )
    detected = pa.table({"peptide_sequence": pa.array(["LLLLLPPGR"])})
    result = classify_novel_peptides(detected, alignments, reference, novel)
    assert result.num_rows == 1
    assert result.column("modified_sequence").to_pylist() == [None]


# ──────────────────────────────────────────────────────────────────────
# read_fasta_proteins
# ──────────────────────────────────────────────────────────────────────


def test_read_fasta_proteins_basic(tmp_path: Path) -> None:
    fasta = tmp_path / "tiny.fasta"
    fasta.write_text(
        ">protA description here\n"
        "MAGCKL\nVERPDF\n"           # multi-line sequence joined
        ">protB gene=BRCA1\n"
        "SEMA\n"
    )
    table = read_fasta_proteins(fasta)
    assert table.column("protein_id").to_pylist() == ["protA", "protB"]
    assert table.column("sequence").to_pylist() == ["MAGCKLVERPDF", "SEMA"]


def test_read_fasta_proteins_empty(tmp_path: Path) -> None:
    fasta = tmp_path / "empty.fasta"
    fasta.touch()
    table = read_fasta_proteins(fasta)
    assert table.num_rows == 0


# ──────────────────────────────────────────────────────────────────────
# Schema sanity
# ──────────────────────────────────────────────────────────────────────


def test_classification_priority_taxonomy_is_complete() -> None:
    """The 11 classes plus a fallback are in the priority table."""
    assert set(_CLASSIFICATION_PRIORITY) == {
        "snp", "insertion", "deletion", "complex",
        "n_terminal_truncation", "c_terminal_truncation",
        "trypsin_cutsite_mutation",
        "n_term_deviation", "c_term_deviation",
        "unknown", "non_reference",
    }
    # snp wins over everything else; non_reference is the lowest priority
    assert _CLASSIFICATION_PRIORITY["snp"] == 0
    assert _CLASSIFICATION_PRIORITY["non_reference"] == 10


def test_empty_input_returns_empty_schema() -> None:
    """Empty detected_peptides → empty NOVEL_PEPTIDE_TABLE."""
    from constellation.massspec.search.schemas import NOVEL_PEPTIDE_TABLE

    detected = pa.table({"peptide_sequence": pa.array([], type=pa.string())})
    alignments = ALIGNMENT_HIT_TABLE.empty_table()
    reference = pa.table(
        {
            "protein_id": pa.array(["R1"]),
            "sequence": pa.array(["MAGCKL"]),
        }
    )
    novel = pa.table(
        {
            "protein_id": pa.array(["N1"]),
            "sequence": pa.array(["MAGCKL"]),
        }
    )
    result = classify_novel_peptides(detected, alignments, reference, novel)
    assert result.num_rows == 0
    assert result.schema.equals(NOVEL_PEPTIDE_TABLE)


def test_build_gene_map_bracketed_and_gn(tmp_path: Path) -> None:
    """Parser handles bare `gene=`, bracketed `[gene=]`, and SwissProt `GN=`."""
    from constellation.massspec.search.novel import build_gene_map_from_fasta_headers

    fasta = tmp_path / "mixed.fasta"
    fasta.write_text(
        ">P1 gene=BARE accession=A1 source=transcriptome\nMAGCKL\n"
        ">P2 [gene=BRACK] [accession=A2] source=transcriptome\nMAGCKL\n"
        ">sp|P3|PROT_HUMAN desc OS=Human GN=SWISS PE=1 SV=1\nMAGCKL\n"
        ">P4 [other=X] source=transcriptome\nMAGCKL\n"
    )
    out = build_gene_map_from_fasta_headers([fasta])
    assert out["P1"] == "BARE"
    assert out["P2"] == "BRACK"
    # SwissProt-style accession is "sp|P3|PROT_HUMAN" — first whitespace token
    assert out["sp|P3|PROT_HUMAN"] == "SWISS"
    # P4 has no gene tag → not in map
    assert "P4" not in out


def test_classify_picks_most_abundant_transcript() -> None:
    """When multiple novel proteins produce the same peptide with the same
    winning classification, the row chosen for `protein_id` should be the
    one with the highest abundance in the supplied lookup."""
    from constellation.massspec.search.novel import classify_novel_peptides

    # Same peptide produced by two novel proteins (both whole-protein ORFs
    # tryptic-digesting to one peptide of length >= 7). The novel has an
    # insertion relative to the reference; higher-abundance protein should
    # win the row.
    pep = "PEPTIDEK"
    detected = pa.table({"peptide_sequence": pa.array([pep])})
    novel = pa.table(
        {
            "protein_id": pa.array(["N_HIGH", "N_LOW"]),
            "sequence": pa.array([pep, pep]),
        }
    )
    reference = pa.table(
        {
            "protein_id": pa.array(["R1"]),
            "sequence": pa.array(["PEPTIDK"]),  # 7 aa, missing the 'E' at position 7
        }
    )
    alignments = pa.table(
        {
            "query":  pa.array(["N_HIGH", "N_LOW"]),
            "target": pa.array(["R1", "R1"]),
            "evalue": pa.array([1e-30, 1e-30]),
            "qstart": pa.array([1, 1]),
            "qend":   pa.array([8, 8]),
            "tstart": pa.array([1, 1]),
            "tend":   pa.array([7, 7]),
            "cigar":  pa.array(["6M1I1M", "6M1I1M"]),
            "alignment_tier": pa.array(["reference", "reference"]),
        }
    )

    # Abundance map — N_HIGH is 10x more abundant.
    abundance = {"N_HIGH": 10.0, "N_LOW": 1.0}
    result = classify_novel_peptides(
        detected, alignments, reference, novel,
        protein_abundance=abundance,
    )
    rows = result.to_pylist()
    insertion_rows = [r for r in rows if r["classification"] == "insertion"]
    assert len(insertion_rows) >= 1
    # The chosen protein_id should be the high-abundance one
    assert insertion_rows[0]["protein_id"] == "N_HIGH"

    # Sanity: when no abundance map is supplied, falls back to the legacy
    # first-encountered behaviour and still returns a valid row.
    result_legacy = classify_novel_peptides(detected, alignments, reference, novel)
    assert result_legacy.num_rows >= 1
