# Constellation

An integrative bioinformatics platform for analysis of diverse omic and
bioanalytical datasets.

Constellation stitches together several lab-internal tools (Cartographer
for mass spectrometry proteomics, NanoporeAnalysis for long-read cDNA,
CoLLAGE for codon optimization, Contour for structure preparation) into
a single package with a shared core of physically grounded primitives:
compositions, distributions, optimizers, graphs, and I/O schemas. Each
modality is a domain module that consumes the core — never another
domain module — so statistical and structural similarities across
experiments become explicit rather than siloed.

**Status: early scaffold.** The architecture is fixed (see [CLAUDE.md](CLAUDE.md));
the core primitives and domain modules are being rewritten from the
existing packages. `constellation doctor` works today; most subcommands
are placeholders.

## Quickstart

```bash
# Create the conda env (Python 3.12; torch/koinapy/pythonnet via pip).
conda env create -f environment.yml
conda activate constellation

# Confirm the package is importable and the doctor CLI runs.
constellation doctor
```

The `doctor` command prints a table of registered third-party tools and
whether each one is installed. Missing tools show the install-script
path to run (or the environment variable to set).

## Optional third-party tools

Constellation wraps a growing set of open-source tools (and a few
proprietary ones like Thermo CommonCore DLLs). Each is optional and
discovered at runtime:

- Set `$CONSTELLATION_<TOOL>_HOME` to a pre-installed copy, or
- Run the matching `scripts/install-<tool>.sh` to fetch a hash-pinned
  release into `third_party/<tool>/<version>/`.

Available installers: see [`scripts/README.md`](scripts/README.md).

## Development

```bash
pip install -e .[dev]
pytest
ruff check .
```

## License

Apache-2.0.
