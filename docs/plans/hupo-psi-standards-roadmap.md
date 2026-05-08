# HUPO-PSI Standards Integration Plan

## Context

Two prior sessions integrated the **HUPO-PSI ProForma 2.0** spec into Constellation (`core.sequence.proforma`). That work was triggered by an EncyclopeDIA reader-port discovery: EncyclopeDIA collapses `[Mod]-X` (N-terminal α-amine modification) and `X[Mod]` (residue side-chain modification) into the same `X[Mod]` notation, but they are **chemically distinct molecules** (e.g. K[Acetyl]VERPD vs [Acetyl]-KVERPD — lysine ε-amine vs N-terminal α-amine acetylation, different RT/fragmentation/localization scoring).

ProForma 2.0 distinguishes them — but ProForma is only one of ~17 standards HUPO-PSI promotes, and Constellation has been adopting them ad-hoc. Before resuming the EncyclopeDIA reader port, this plan:

1. Surveys all PSI standards and partitions in-scope vs deferred vs out-of-scope.
2. Designs the integration shape for the highest-priority in-scope standards: **mzPAF v1.0** (peak annotations), **mzSpecLib v1.0** (library exchange), **USI** (spectrum identifiers), and **PSI-MS CV** (metadata-key vocabulary).
3. Drafts a binding **CLAUDE.md commandment** that elevates PSI compliance from convention to architectural invariant — so future modules default to PSI shapes without per-feature deliberation.
4. Resolves the EncyclopeDIA N-terminal-mod ambiguity at the reader boundary via UNIMOD position-constraint lookup, un-blocking the dlib/elib/.dia port.

The user's intended outcome is twofold: ship a PSI-compliant in-memory model (without inverting our relational Arrow schemas), and embed PSI-first thinking into CLAUDE.md so future contributions adopt standards by default.

---

## PSI Standards Survey

| Standard | Status (as of 2026-04) | Disposition | Module home |
|---|---|---|---|
| **ProForma 2.0** | Ratified | **Shipped** | [core.sequence.proforma](constellation/core/sequence/proforma.py) |
| **mzPAF v1.0** | Ratified Aug 2025 | **Phase 1** | `massspec.annotation.mzpaf` (new) |
| **mzSpecLib v1.0** | Ratified 2025 | **Phase 3** | `massspec.library.readers.mzspeclib` (new) |
| **USI** | Ratified Jul 2021 | **Phase 1** | `massspec.annotation.usi` (new) |
| **PSI-MS CV** | Stable, versioned | **Phase 2** | `core.ontology.PSI_MS` (new module) |
| **PEFF** | Ratified 2019 | Deferred | `core.sequence.protein.peff` (when consumer arrives) |
| **mzML** | Stable | Deferred | Lands with mzML reader port |
| **mzIdentML** | Stable | Deferred | Lands with `massspec.search` |
| **mzTab / mzTab-M** | mzTab v1.0 stable, mzTab-M v2.1 draft | Deferred | Lands with results-export tooling |
| **SDRF-Proteomics** | Stable | Deferred | Lands with cross-modality experiment design |
| **mzPeak** | Draft | Deferred | Track upstream; our parquet form already similar |
| **mzQC** | Stable | Deferred | Lands with QC tooling |
| **proBAM / proBed** | Stable | Deferred | Lands with genome→transcriptome→proteome bridge work; proteogenomic alignments + features fit naturally once `sequencing` (POD5→protein) and a lab-genome layer are wired together |
| **PSI-MI XML** | Stable | **Out of scope** | Molecular-interactions domain |
| **TraML** | Stable | **Out of scope** | Legacy SRM, superseded by mzML |
| **FeatureTab** | Draft | **Out of scope** | Niche |

---

## Architectural Decisions (settled)

1. **PSI-bridge, not PSI-first.** Relational Arrow schemas remain canonical in-memory; PSI formats are reader/writer adapters. Rationale: invariant 2 (Arrow in / Arrow out) and invariant 3 (torch numerics, vmap-compatible hot paths) both want flat columnar shapes; mzSpecLib's hierarchical CV-keyed attribute bag is a *document* model, fine for serialization but wrong for in-memory math. The shipped Reader/Writer Protocol pattern (ParquetDir / dlib / elib) already encodes the bridge — mzSpecLib slots in as one more peer.

2. **mzPAF = derive-on-write.** `LIBRARY_FRAGMENT_TABLE.ion_type/position/charge/loss_id/mz_theoretical` remain source-of-truth; mzPAF strings are emitted at write time via a vectorized projection (`fragment_table_to_mzpaf(table) -> pa.Array`). The `annotation` column stays nullable and is only populated when a reader supplied a non-derivable mzPAF (e.g. multi-analyte alternates from external libraries).

3. **Per-spectrum metadata = parallel `PRECURSOR_METADATA` table.** mzSpecLib carries arbitrary CV-keyed per-spectrum attributes that don't map to `PRECURSOR_TABLE` columns. New 6th Library schema: `(precursor_id int64 FK, accession string, value string)` long-form. Star-schema fact-and-attribute split: queryable, lossless, scales to any CV. Library now has six tables: proteins, peptides, precursors, edges, fragments, precursor_metadata.

4. **EncyclopeDIA modseq normalization at reader boundary.** When parsing `[+N.NNN]` mass deltas at residue 0, look up via `UNIMOD.find_by_mass(delta, tol=0.001)`; if exactly one UNIMOD match has N-term-only position constraint, rewrite to ProForma terminal form `[UNIMOD:N]-X`. Unresolvable deltas remain as ProForma mass-delta level. The reader is the right boundary: the K[Acetyl] vs [Acetyl]- bug already declared (in CLAUDE.md conventions) that ProForma must distinguish them.

5. **PSI-MS CV accessions are the preferred metadata-key form.** `MS:NNNNNNN` replaces `x.ms.<key>` everywhere a PSI accession exists; `x.<domain>.<key>` remains the fallback for genuinely Constellation-specific keys (`x.hplc.wavelength_nm`, `x.constellation.fragment_ladder_version`).

6. **`core.ontology` is a new leaf module**, not a generalization of `core.chem.modifications`. UNIMOD/MOD/RESID/XLMOD/GNO are chemistry ontologies (carry `delta_composition`); PSI-MS is taxonomy (no chemistry). `core.ontology` sits at the bottom of the DAG alongside `core.io`, consumed by both `core.chem.modifications` (cross-ontology mod resolution) and domain modules (mzSpecLib/mzPAF accession validation). **Naming note:** the namespace is *ontology*, not *cv* — "CV" is reserved for collective variable in future MD / structure code; PSI-MS / UNIMOD / MOD are OBO-Foundry ontologies in everything but name (cf. GO, ChEBI).

7. **mzPAF parser belongs at `massspec.annotation`** (new package, sibling to `peptide/`). mzPAF references multiple peptidoforms (multi-analyte chimeric annotations) and non-peptide ion types (reporter, formula, SMILES, named) — broader than `peptide/`. Independent grammar implementation per the same rule that bans pyteomics from `core.sequence.proforma`; the `mzpaf` PyPI package earns a slot only in cross-validation tests.

8. **USI lives in `massspec.annotation.usi`** as a sibling to mzpaf — both are MS-domain annotation/identifier types, both depend on `core.sequence.proforma`.

---

## Phased Delivery

### Phase 1 — Un-block EncyclopeDIA reader (highest priority; pre-blocks the user's next session)

**Schema enrichment:**
- [constellation/data/unimod.json](constellation/data/unimod.json) — regenerate to include UNIMOD `<specificity>` `position` and `site` constraints. Currently dropped at build time.
- [scripts/build-unimod-json.py](scripts/build-unimod-json.py) — extract `position` (Anywhere / N-term / C-term / Protein N-term / Protein C-term) and `site` (residue) from UNIMOD XML.
- [constellation/core/chem/modifications.py](constellation/core/chem/modifications.py) — extend `Modification` dataclass with `specificities: tuple[Specificity, ...]` where `Specificity = (position: str, site: str)`. Multi-position because mods like Acetyl carry multiple constraints. Add `Modification.has_n_term_specificity` / `.has_c_term_specificity` predicate properties.

**New `massspec.annotation` package:**
- [constellation/massspec/annotation/__init__.py](constellation/massspec/annotation/__init__.py) — public API surface.
- [constellation/massspec/annotation/mzpaf.py](constellation/massspec/annotation/mzpaf.py) — `Annotation`, `PeakAnnotation`, `parse_mzpaf`, `format_mzpaf`, `MzPAFSyntaxError`. Plus projection helpers `fragment_row_to_mzpaf(row) -> str` and `fragment_table_to_mzpaf(table) -> pa.Array` (vectorized via Arrow compute). Loss-token round-trip uses `core.chem.modifications.UNIMOD.find_by_mass` for resolved-form (`-[Phospho]`) and `LOSS_REGISTRY` reverse lookup for formula-form (`-H2O`).
- [constellation/massspec/annotation/_grammar.py](constellation/massspec/annotation/_grammar.py) — token-level helpers; mirror `core/sequence/proforma.py` Lark-grammar pattern.
- [constellation/massspec/annotation/usi.py](constellation/massspec/annotation/usi.py) — frozen-slot `USI` dataclass with `parse(s)`, `format()`, `from_spectrum(...)` classmethod.

**EncyclopeDIA modseq normalization (un-blocks dlib/elib reader):**
- [constellation/thirdparty/encyclopedia.py](constellation/thirdparty/encyclopedia.py) — extend (or create) with `normalize_encyclopedia_modseq(modseq, *, vocab=UNIMOD, tolerance_da=0.001) -> str`. Implements rule from decision (4) above. Adapter-tier per the existing CLAUDE.md convention ("EncyclopeDIA-style `[+mass_delta]` translation goes through `UNIMOD.find_by_mass(...)` in a thirdparty adapter").

**Tests:**
- [tests/test_mzpaf.py](tests/test_mzpaf.py) — grammar coverage (peptide ion series, immonium, internal, precursor, reporter, named, formula, SMILES, unknown; charge/loss/isotope/adduct/mass-error/confidence decorations; alternates and multi-analyte parsing).
- [tests/test_mzpaf_pypi_parity.py](tests/test_mzpaf_pypi_parity.py) — cross-validate against `mzpaf` PyPI on a fixture set (drift detection only).
- [tests/test_usi.py](tests/test_usi.py) — round-trip on the spec examples (basic + ProForma interpretation).
- [tests/test_encyclopedia_modseq.py](tests/test_encyclopedia_modseq.py) — N-terminal placement rules: K[+42.011]VERPD at residue 0 → [UNIMOD:1]-KVERPD; K[+42.011]VERPD at residue ≥1 → KVERPD with K[UNIMOD:1] at the right position; ambiguous mass with no N-term match stays as bare mass delta.

**CLAUDE.md update (Phase 1 ships the commandment too — see §"CLAUDE.md commandment" below):**
- [CLAUDE.md](CLAUDE.md) — insert new architecture invariant #2 (PSI compliance), update metadata-key convention bullet (PSI accessions preferred, `x.<domain>.<key>` fallback), update module index with new `core.ontology` and `massspec.annotation` rows.

**Phase 1 deliverable:** EncyclopeDIA reader port can call `normalize_encyclopedia_modseq` to produce ProForma 2.0 with terminal correctness. mzPAF and USI round-trip. No Library schema changes yet.

### Phase 2 — `core.ontology` + PSI-MS resolver

- [constellation/core/ontology/__init__.py](constellation/core/ontology/__init__.py) — public API.
- [constellation/core/ontology/vocabulary.py](constellation/core/ontology/vocabulary.py) — `OntologyTerm(accession, name, definition, parents, units)`, `Ontology` / `ControlledVocabulary` (get/has/find_by_name/iter), `OntologyResolver` (multi-ontology dispatcher). Public singleton `PSI_MS`.
- [constellation/core/ontology/loaders.py](constellation/core/ontology/loaders.py) — OBO + JSON loaders.
- [constellation/data/ontology/psi-ms.json](constellation/data/ontology/psi-ms.json) — generated.
- [scripts/build-psi-ms-ontology-json.py](scripts/build-psi-ms-ontology-json.py) — pin upstream version, OBO → JSON.
- [tests/test_ontology.py](tests/test_ontology.py) — accession lookup, parent-chain traversal, units validation.

DAG impact: `core.ontology` is leaf-tier alongside `core.io`. `core.chem.modifications` may *optionally* import `core.ontology` to register UNIMOD as an `Ontology` instance (forward-compat, not phase-blocking).

### Phase 3 — mzSpecLib reader/writer + `PRECURSOR_METADATA` schema

**Library readers refactor (precondition):**
- Create [constellation/massspec/library/readers/__init__.py](constellation/massspec/library/readers/__init__.py).
- Move `ParquetDirReader`/`Writer` from `library/io.py` → `library/readers/parquet_dir.py`.
- Move `DlibReader`/`Writer` stubs from `library/io.py` → `library/readers/dlib.py`.
- `library/io.py` retains Protocols, registry, `save_library`/`load_library`, multi-suffix `_resolve_*` (handle `.mzspeclib.txt` and `.mzspeclib.json`).

**Schema addition:**
- [constellation/massspec/library/schemas.py](constellation/massspec/library/schemas.py) — add `PRECURSOR_METADATA` Arrow schema: `precursor_id int64 (FK), accession string, value string`. Self-register with `core.io.schemas.register_schema`.
- [constellation/massspec/library/library.py](constellation/massspec/library/library.py) — `Library` constructor accepts the new table; PK uniqueness on `(precursor_id, accession)`; FK closure validation extends to include precursor_metadata → precursors.
- Update `Library.subset_proteins(ids)` cascade to include `precursor_metadata`.

**mzSpecLib reader/writer:**
- [constellation/massspec/library/readers/mzspeclib.py](constellation/massspec/library/readers/mzspeclib.py) — `MzSpecLibTextReader` and `MzSpecLibTextWriter`. Streaming line parser for section markers `<mzSpecLib>`, `<Spectrum=...>`, `<Analyte=...>`, `<Interpretation=...>`, `<Peaks>`. Attribute mapping (CV accession → schema column or `PRECURSOR_METADATA` row):
  - `MS:1003049` (ProForma sequence) → `PEPTIDE_TABLE.modified_sequence`
  - `MS:1000041` (charge state) → `PRECURSOR_TABLE.charge`
  - `MS:1000744` (selected ion m/z) → `PRECURSOR_TABLE.precursor_mz`
  - `MS:1000894` (RT) → `PRECURSOR_TABLE.rt_predicted` (Library context)
  - `MS:1002954` (CCS) → `PRECURSOR_TABLE.ccs_predicted`
  - All other per-spectrum attributes → `PRECURSOR_METADATA` rows
  - All library-level attributes → `Library.metadata_extras["x.psi.<accession>"]`
  - Peaks (`m/z intensity mzPAF`) → `LIBRARY_FRAGMENT_TABLE` rows; mzPAF parses to `ion_type/position/charge/loss_id` via `mzpaf_to_fragment_row`; raw mzPAF kept in `annotation` only when alternates / multi-analyte present.

**Tests:**
- [tests/test_mzspeclib_text.py](tests/test_mzspeclib_text.py) — round-trip on canonical examples.
- [tests/data/mzspeclib/*.mzspeclib.txt](tests/data/mzspeclib/) — fixtures (NIST-derived, EncyclopeDIA-derived).

### Phase 4 — JSON / MSP dialects + `Quant` mzSpecLib alignment

- [constellation/massspec/library/readers/mzspeclib_json.py](constellation/massspec/library/readers/mzspeclib_json.py)
- [constellation/massspec/library/readers/msp.py](constellation/massspec/library/readers/msp.py) — NIST MSP via `mzspeclib` PyPI bridge (extras_require dep `pip install constellation[psi]`).
- Cross-validation tests against `mzspeclib` PyPI.

### Phase 5 — Apply commandment retroactively

- Audit existing modules for legacy `x.ms.*` metadata keys → replace with PSI accessions.
- Update tests asserting on `x.ms.*` literals.
- Document the migration in CHANGELOG-style commit body.

---

## CLAUDE.md commandment (insert as new architectural invariant #2)

Insert into the "Architecture invariants (load-bearing — design principles from the PI)" numbered list, between current #1 (physically grounded statistical models) and current #2 (PyArrow tables in memory) — pushing the rest down by one. Phrasing:

> **2. HUPO-PSI standards compliance is the default for proteomics-domain serialization, identifiers, and metadata keys.** Where a HUPO-PSI standard exists for a concept Constellation handles, the PSI form is the canonical *external* representation; relational Arrow tables remain the canonical *in-memory* shape (PSI-bridge, not PSI-first). In-scope standards: **ProForma 2.0** (modseq strings — `core.sequence.proforma`), **mzPAF v1.0** (peak annotations — `massspec.annotation.mzpaf`), **mzSpecLib v1.0** (spectral library exchange — `massspec.library.readers.mzspeclib`), **USI** (spectrum identifiers — `massspec.annotation.usi`), **PSI-MS CV** (metadata-key vocabulary — `core.ontology.PSI_MS`). PSI-MS CV accessions (`MS:NNNNNNN`) are the preferred metadata-key form anywhere a CV term exists; the `x.<domain>.<key>` namespace remains the fallback for Constellation-specific keys with no PSI equivalent. Deferred until corresponding readers/writers drive them: **mzML** (raw spectra), **mzIdentML** (identifications), **mzTab** (results), **SDRF-Proteomics** (sample metadata), **PEFF** (extended FASTA), **mzPeak** (when PSI ratifies), **mzQC** (QC tooling), **proBAM / proBed** (proteogenomic alignments and features — lands with the genome→transcriptome→proteome bridge work). Out of scope: **PSI-MI XML** (molecular interactions, different domain), **TraML** (legacy SRM, superseded by mzML), **FeatureTab** (niche). New PSI ratifications enter the in-scope set unless a written rationale rejects them.

Also update the existing "Code-level invariants" bullet on metadata convention from:

> "Modality-specific information rides as **namespaced schema metadata** (`x.<domain>.<key>`), never as columns."

to:

> "Modality-specific information rides as **schema metadata, never as columns**. Preferred key form: PSI-MS CV accession (`MS:1000744`) when one exists for the attribute; fallback to namespaced `x.<domain>.<key>` (`x.hplc.wavelength_nm`, `x.pod5.read_id`) for Constellation-specific keys with no PSI equivalent."

---

## Critical files (Phase 1, in dependency order)

1. [constellation/data/unimod.json](constellation/data/unimod.json) (regenerated)
2. [scripts/build-unimod-json.py](scripts/build-unimod-json.py)
3. [constellation/core/chem/modifications.py](constellation/core/chem/modifications.py)
4. [constellation/massspec/annotation/__init__.py](constellation/massspec/annotation/__init__.py) (new)
5. [constellation/massspec/annotation/_grammar.py](constellation/massspec/annotation/_grammar.py) (new)
6. [constellation/massspec/annotation/mzpaf.py](constellation/massspec/annotation/mzpaf.py) (new)
7. [constellation/massspec/annotation/usi.py](constellation/massspec/annotation/usi.py) (new)
8. [constellation/thirdparty/encyclopedia.py](constellation/thirdparty/encyclopedia.py)
9. [CLAUDE.md](CLAUDE.md) — invariant #2 + metadata-convention bullet update
10. [tests/test_mzpaf.py](tests/test_mzpaf.py), [tests/test_usi.py](tests/test_usi.py), [tests/test_encyclopedia_modseq.py](tests/test_encyclopedia_modseq.py), [tests/test_mzpaf_pypi_parity.py](tests/test_mzpaf_pypi_parity.py)

---

## Verification

After Phase 1:

```bash
# Smoke imports — new package registers cleanly
pytest tests/test_imports.py

# mzPAF grammar coverage
pytest tests/test_mzpaf.py -v

# USI round-trip
pytest tests/test_usi.py -v

# EncyclopeDIA modseq rewriting
pytest tests/test_encyclopedia_modseq.py -v

# Cross-validation drift detection vs mzpaf PyPI
pip install mzpaf
pytest tests/test_mzpaf_pypi_parity.py -v

# Ad-hoc check: parse a PRIDE USI and round-trip it
python -c "
from constellation.massspec.annotation import USI
u = USI.parse('mzspec:PXD000561:Adult_Frontalcortex_bRP_Elite_85_f09:scan:17555:VLHPLEGAVVIIFK/2')
assert u.format() == 'mzspec:PXD000561:Adult_Frontalcortex_bRP_Elite_85_f09:scan:17555:VLHPLEGAVVIIFK/2'
print(u.interpretation.sequence)  # 'VLHPLEGAVVIIFK'
"

# Ad-hoc check: EncyclopeDIA N-terminal acetyl rewriting
python -c "
from constellation.thirdparty.encyclopedia import normalize_encyclopedia_modseq
assert normalize_encyclopedia_modseq('K[+42.011]VERPD') == '[UNIMOD:1]-KVERPD'  # N-term Acetyl
"

# Lint
ruff check .
```

After Phase 3:

```bash
# mzSpecLib text round-trip
pytest tests/test_mzspeclib_text.py -v

# Read a NIST-derived mzSpecLib file, write it back, diff
python -c "
from constellation.massspec.library import load_library, save_library
lib = load_library('tests/data/mzspeclib/nist_human_yeast.mzspeclib.txt')
save_library(lib, '/tmp/roundtrip.mzspeclib.txt')
"
diff tests/data/mzspeclib/nist_human_yeast.mzspeclib.txt /tmp/roundtrip.mzspeclib.txt
```

---

## Deferred / open

- **PEFF placement** (`core.sequence.protein.peff` vs `massspec.peff`) — defer until first concrete consumer.
- **mzPeak draft** — track upstream; revisit when ratified.
- **`x.ms.*` migration** — eager (Phase 1) or lazy (Phase 5)? Lean lazy: avoids touching unrelated tests in Phase 1; the PSI commandment in CLAUDE.md gates new code into the right convention immediately.
- **`mzspeclib` PyPI as `extras_require[psi]`** — needed only for Phase 4 niche-format reads (NIST MSP / SPTXT / DIA-NN / Spectronaut). Not a runtime dep for Phases 1–3.
