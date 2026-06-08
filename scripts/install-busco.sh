#!/usr/bin/env bash
# Install BUSCO into Constellation's third_party/ layout.
#
# Primary install path is conda (bioconda — in environment.yml), which
# also pulls BUSCO's external dependencies (metaeuk / hmmer / sepp ...).
# This script is the no-conda FALLBACK: a venv pip-install of BUSCO. Note
# that pip does NOT install BUSCO's external aligners — those must already
# be on PATH for a real run. Lineage datasets download separately.
#
# Layout (matches the thirdparty.registry contract — artifact 'busco'):
#   third_party/busco/<version>/busco -> venv/bin/busco
#   third_party/busco/current -> <version>/   (symlink)
#
# Lineage data: set $BUSCO_DOWNLOADS_PATH (the runner passes it through as
# `--download_path --offline`), or let BUSCO auto-download online. Override
# the tool location with $CONSTELLATION_BUSCO_HOME.
#
# Usage:
#   bash scripts/install-busco.sh
#   bash scripts/install-busco.sh --force
#   bash scripts/install-busco.sh --version 5.7.1
#   bash scripts/install-busco.sh --lineage eukaryota_odb10   # pre-stage a dataset
#
# BUSCO is MIT-licensed.
set -euo pipefail

VERSION="5.7.1"
LINEAGE=""
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

force=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --force) force=1; shift ;;
        --version) VERSION="$2"; shift 2 ;;
        --lineage) LINEAGE="$2"; shift 2 ;;
        -h|--help)
            cat <<'USAGE'
install-busco.sh — pip-install BUSCO into a dedicated venv (no-conda fallback).

Primary install path is conda (bioconda — in environment.yml), which also
installs BUSCO's external aligners. pip does NOT — they must be on PATH.

Writes to:
  third_party/busco/<version>/busco  (-> venv/bin/busco)
  third_party/busco/current -> <version>/   (symlink)

Lineage data: set $BUSCO_DOWNLOADS_PATH, or pre-stage with --lineage NAME.
Override the tool path via $CONSTELLATION_BUSCO_HOME.

BUSCO is MIT-licensed.
USAGE
            exit 0
            ;;
        *) echo "unknown arg: $1" >&2; exit 2 ;;
    esac
done

TARGET_DIR="${REPO_ROOT}/third_party/busco/${VERSION}"
CURRENT_LINK="${REPO_ROOT}/third_party/busco/current"
ENTRY="${TARGET_DIR}/busco"

if [[ -e "${ENTRY}" && ${force} -eq 0 ]]; then
    echo "busco ${VERSION} already installed at ${ENTRY}"
else
    if ! command -v python3 >/dev/null 2>&1; then
        echo "error: python3 not on PATH" >&2
        exit 1
    fi
    rm -rf "${TARGET_DIR}"
    mkdir -p "${TARGET_DIR}"
    echo "creating venv + installing BUSCO ${VERSION}..."
    python3 -m venv "${TARGET_DIR}/venv"
    "${TARGET_DIR}/venv/bin/pip" install --quiet --upgrade pip
    "${TARGET_DIR}/venv/bin/pip" install --quiet "busco==${VERSION}"
    if [[ ! -x "${TARGET_DIR}/venv/bin/busco" ]]; then
        echo "error: busco entry point not found in the venv after install" >&2
        exit 1
    fi
    ln -sfn "venv/bin/busco" "${ENTRY}"
    echo "note: pip did not install BUSCO's external aligners (metaeuk/hmmer/...)." >&2
    echo "      ensure they are on PATH, or use the conda install instead." >&2
fi

tmp_link="${CURRENT_LINK}.tmp.$$"
ln -sfn "${VERSION}" "${tmp_link}"
mv -Tf "${tmp_link}" "${CURRENT_LINK}"

if [[ -n "${LINEAGE}" ]]; then
    dl="${BUSCO_DOWNLOADS_PATH:-${HOME}/.constellation/busco_downloads}"
    echo "pre-staging lineage ${LINEAGE} into ${dl}..."
    "${ENTRY}" --download_path "${dl}" --download "${LINEAGE}" || \
        echo "warning: lineage download failed; fetch manually with \`busco --download ${LINEAGE}\`" >&2
fi

echo "installed: ${ENTRY}"
echo "current:   ${CURRENT_LINK} -> ${VERSION}/"
echo
echo "run \`constellation doctor\` to confirm registry discovery."
