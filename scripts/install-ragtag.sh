#!/usr/bin/env bash
# Install RagTag into Constellation's third_party/ layout.
#
# Primary install path is conda (bioconda — in environment.yml). This
# script is the FALLBACK for nodes / containers without conda: it creates
# a dedicated venv, pip-installs RagTag into it, and exposes the
# `ragtag.py` entry point where the registry expects it.
#
# Layout (matches the thirdparty.registry contract — artifact 'ragtag.py'):
#   third_party/ragtag/<version>/ragtag.py -> venv/bin/ragtag.py
#   third_party/ragtag/current -> <version>/   (symlink)
#
# RagTag needs minimap2 on PATH (already in environment.yml). Override
# with $CONSTELLATION_RAGTAG_HOME to use a shared install.
#
# Usage:
#   bash scripts/install-ragtag.sh
#   bash scripts/install-ragtag.sh --force
#   bash scripts/install-ragtag.sh --version 2.1.0
#
# MIT (matches upstream RagTag license — installs freely).
set -euo pipefail

# Build the venv in isolation: an exported PYTHONPATH (common on HPC login
# shells) leaks the parent environment into the venv, so pip would see its
# packages as "already satisfied" and skip installing RagTag's own deps —
# leaving a venv that breaks when run without that leak.
unset PYTHONPATH

VERSION="2.1.0"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

force=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --force) force=1; shift ;;
        --version) VERSION="$2"; shift 2 ;;
        -h|--help)
            cat <<'USAGE'
install-ragtag.sh — pip-install RagTag into a dedicated venv.

Primary install path is conda (bioconda — in environment.yml). This is
the no-conda fallback.

Writes to:
  third_party/ragtag/<version>/ragtag.py  (-> venv/bin/ragtag.py)
  third_party/ragtag/current -> <version>/   (symlink)

Needs python3 + minimap2 on PATH. Override via $CONSTELLATION_RAGTAG_HOME.

MIT (matches upstream RagTag license).
USAGE
            exit 0
            ;;
        *) echo "unknown arg: $1" >&2; exit 2 ;;
    esac
done

TARGET_DIR="${REPO_ROOT}/third_party/ragtag/${VERSION}"
CURRENT_LINK="${REPO_ROOT}/third_party/ragtag/current"
ENTRY="${TARGET_DIR}/ragtag.py"

if [[ -e "${ENTRY}" && ${force} -eq 0 ]]; then
    echo "ragtag ${VERSION} already installed at ${ENTRY}"
else
    if ! command -v python3 >/dev/null 2>&1; then
        echo "error: python3 not on PATH" >&2
        exit 1
    fi
    if ! command -v minimap2 >/dev/null 2>&1; then
        echo "warning: minimap2 not on PATH — RagTag needs it at runtime." >&2
    fi
    rm -rf "${TARGET_DIR}"
    mkdir -p "${TARGET_DIR}"
    echo "creating venv + installing RagTag ${VERSION}..."
    python3 -m venv "${TARGET_DIR}/venv"
    "${TARGET_DIR}/venv/bin/pip" install --quiet --upgrade pip
    "${TARGET_DIR}/venv/bin/pip" install --quiet "RagTag==${VERSION}"
    if [[ ! -x "${TARGET_DIR}/venv/bin/ragtag.py" ]]; then
        echo "error: ragtag.py not found in the venv after install" >&2
        exit 1
    fi
    ln -sfn "venv/bin/ragtag.py" "${ENTRY}"
fi

if [[ ! -e "${ENTRY}" ]]; then
    echo "error: post-install ${ENTRY} missing" >&2
    exit 1
fi

tmp_link="${CURRENT_LINK}.tmp.$$"
ln -sfn "${VERSION}" "${tmp_link}"
mv -Tf "${tmp_link}" "${CURRENT_LINK}"

echo "installed: ${ENTRY}"
echo "current:   ${CURRENT_LINK} -> ${VERSION}/"
echo
echo "run \`constellation doctor\` to confirm registry discovery."
