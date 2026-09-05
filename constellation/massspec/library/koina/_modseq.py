"""ProForma 2.0 <-> Koina modified-sequence dialect.

Koina accepts a restricted ProForma subset: residue-attached UNIMOD
accession brackets only, as in ``YIC[UNIMOD:4]DNQDTISSK``. This is far
simpler than the EncyclopeDIA translation in
``massspec.io.encyclopedia._modseq``, which has to resolve ``[+57.021]``
mass deltas through ``UNIMOD.find_by_mass`` plus specificity rules —
Koina carries explicit accessions, so there is no lookup and no
ambiguity.

What it does *not* accept, measured live against
``Prosit_2020_intensity_HCD`` on 2026-08-03:

* **N-terminal mods in any form.** ``[UNIMOD:1]-PEPTIDEK``,
  ``[UNIMOD:1]PEPTIDEK`` and ``(UNIMOD:1)PEPTIDEK`` were all rejected.
  This is why ``ProteinNTermAcetyl`` and ``PyroGluQ`` — two of
  Constellation's default variable mods — cannot be predicted here yet.
* Mods outside the model's training set. Phospho and Acetyl-K both fail
  server-side with an opaque "at least one request failed", which is the
  reason ``supported`` pre-flighting exists: to name the offending
  accession before the request is sent.
"""

from __future__ import annotations

from collections.abc import Collection

from constellation.core.sequence.proforma import (
    MultiPeptidoform,
    Peptidoform,
    parse_proforma,
)


class KoinaModSeqError(ValueError):
    """A peptidoform cannot be expressed in Koina's dialect."""


def _accession(tagged) -> str:
    """``TaggedMod`` → ``"UNIMOD:N"``, or raise with the reason."""
    mod = tagged.mod
    if mod is None:
        raise KoinaModSeqError(
            "bare localization-group label has no modification payload; "
            "Koina cannot represent ambiguity groups"
        )
    if mod.cv is None or mod.accession is None:
        detail = (
            f"mass delta {mod.mass_delta}"
            if mod.mass_delta is not None
            else f"name {mod.name!r}"
            if mod.name is not None
            else repr(mod)
        )
        raise KoinaModSeqError(
            f"modification given as {detail} has no CV accession; Koina "
            f"requires explicit UNIMOD accessions (e.g. [UNIMOD:35])"
        )
    return f"{mod.cv}:{mod.accession}"


def format_koina_modseq(
    peptidoform: Peptidoform,
    *,
    supported: Collection[str] | None = None,
    allow_n_term_mods: bool = False,
) -> str:
    """``Peptidoform`` → a Koina modified-sequence string.

    ``supported``, when given, is the set of UNIMOD accessions the target
    model was trained on; anything outside it raises rather than being
    sent off to fail opaquely server-side.

    ``allow_n_term_mods`` opts into the N-terminal dialect for models
    that accept it — see :attr:`Ms2Model.supports_n_term_mods`. The
    2020-series models reject a terminal mod in every bracket form, but
    the PTM-aware series accepts the ProForma ``[UNIMOD:N]-PEPTIDE``
    form. Measured against koina.wilhelmlab.org:

    =========================== ====================== ==================
    model                       ``[UNIMOD:1]-PEPTIDEK`` ``[UNIMOD:1]PEPTIDEK``
    =========================== ====================== ==================
    Prosit_2025_intensity_22PTM accepted               rejected
    Prosit_2020_intensity_HCD   rejected               rejected
    =========================== ====================== ==================

    C-terminal mods stay rejected regardless: the overlay flag speaks
    only for the N-terminus, which is what was measured. Sending a mod
    the model silently ignores would return intensities that look valid
    and are not.

    Raises ``KoinaModSeqError`` for unsupported terminal mods,
    mass-delta-only mods, and every ProForma feature above compliance
    level 2 (ranges, cross-links, branches, labile, unknown-position,
    global isotopes, global fixed mods) — none of which Koina can
    express.
    """
    if isinstance(peptidoform, MultiPeptidoform):
        raise KoinaModSeqError(
            "multi-chain (cross-linked) peptidoforms are not predictable "
            "via Koina"
        )

    if peptidoform.c_term_mods:
        raise KoinaModSeqError(
            "C-terminal modifications are not accepted by the Prosit models "
            "on Koina; drop the terminal mod or use the EncyclopeDIA backend"
        )
    if peptidoform.n_term_mods and not allow_n_term_mods:
        raise KoinaModSeqError(
            "N-terminal modifications are not accepted by this model "
            "(verified for [UNIMOD:N]-, [UNIMOD:N] and (UNIMOD:N) forms); "
            "drop the terminal mod, pick a PTM-aware model, or use the "
            "EncyclopeDIA backend"
        )
    for attr, label in (
        ("labile_mods", "labile modifications"),
        ("unknown_pos_mods", "unknown-position modifications"),
        ("ranges", "position ranges"),
        ("global_isotopes", "global isotope labels"),
        ("fixed_mods", "global fixed-mod declarations"),
    ):
        if getattr(peptidoform, attr):
            raise KoinaModSeqError(f"{label} cannot be expressed in Koina's dialect")

    seq = peptidoform.sequence
    out: list[str] = []
    for tagged in peptidoform.n_term_mods or ():
        acc = _accession(tagged)
        if supported is not None and acc not in supported:
            raise KoinaModSeqError(
                f"N-terminal modification {acc} is outside the model's "
                f"supported set {sorted(supported)}"
            )
        out.append(f"[{acc}]-")
    for i, residue in enumerate(seq):
        out.append(residue)
        for tagged in peptidoform.residue_mods.get(i, ()):
            acc = _accession(tagged)
            if supported is not None and acc not in supported:
                raise KoinaModSeqError(
                    f"modification {acc} at {residue}{i + 1} is outside the "
                    f"model's supported set {sorted(supported)}; predictions "
                    f"for it would be meaningless even if the server accepted "
                    f"the request"
                )
            out.append(f"[{acc}]")
    return "".join(out)


def parse_koina_modseq(modseq: str) -> Peptidoform:
    """Koina modified-sequence string → ``Peptidoform``.

    Koina's dialect is a strict ProForma subset, so this delegates to the
    canonical parser rather than implementing a second grammar — keeping
    a single source of truth for modseq semantics.
    """
    parsed = parse_proforma(modseq)
    if isinstance(parsed, MultiPeptidoform):
        raise KoinaModSeqError(f"not a single-chain peptidoform: {modseq!r}")
    return parsed


__all__ = ["KoinaModSeqError", "format_koina_modseq", "parse_koina_modseq"]
