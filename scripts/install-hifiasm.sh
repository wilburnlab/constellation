#!/usr/bin/env bash
# Build hifiasm from source into Constellation's third_party/ layout.
#
# Primary install path is conda (bioconda — in environment.yml). This
# script is the FALLBACK for nodes / containers without conda. hifiasm
# ships source only (no prebuilt release binary), so this builds with
# `make` (needs a C++ compiler + zlib headers).
#
# Layout (matches the thirdparty.registry contract):
#   third_party/hifiasm/<version>/hifiasm
#   third_party/hifiasm/current -> <version>/   (symlink)
#
# Override with $CONSTELLATION_HIFIASM_HOME to use a shared install
# elsewhere; in that case skip this script — the registry resolves via
# the env var.
#
# Usage:
#   bash scripts/install-hifiasm.sh
#   bash scripts/install-hifiasm.sh --force
#   bash scripts/install-hifiasm.sh --version 0.25.0
#   bash scripts/install-hifiasm.sh --checksum <SHA256>   # verify source tarball
#
# MIT (matches upstream hifiasm license — bundles / installs freely).
set -euo pipefail

VERSION="0.25.0"
# SHA256 of the GitHub source tarball for the pinned version. Left as a
# placeholder — pass --checksum <SHA> (verified from the release page) to
# enable verification. Without it the script warns and proceeds.
EXPECTED_SHA="REPLACE_WITH_VERIFIED_SHA256"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

force=0
override_sha=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --force) force=1; shift ;;
        --version) VERSION="$2"; shift 2 ;;
        --checksum) override_sha="$2"; shift 2 ;;
        -h|--help)
            cat <<'USAGE'
install-hifiasm.sh — build the pinned hifiasm from source.

Primary install path is conda (bioconda — in environment.yml). This
script is the fallback for nodes / containers without conda.

Writes to:
  third_party/hifiasm/<version>/hifiasm
  third_party/hifiasm/current -> <version>/   (symlink)

Usage:
  bash scripts/install-hifiasm.sh
  bash scripts/install-hifiasm.sh --force
  bash scripts/install-hifiasm.sh --version 0.25.0
  bash scripts/install-hifiasm.sh --checksum <SHA256>

Requires a C++ compiler (g++) + GNU make + zlib headers.
Override via $CONSTELLATION_HIFIASM_HOME to use a shared install.

MIT (matches upstream hifiasm license).
USAGE
            exit 0
            ;;
        *) echo "unknown arg: $1" >&2; exit 2 ;;
    esac
done

TARGET_DIR="${REPO_ROOT}/third_party/hifiasm/${VERSION}"
CURRENT_LINK="${REPO_ROOT}/third_party/hifiasm/current"
BIN_PATH="${TARGET_DIR}/hifiasm"

if [[ -x "${BIN_PATH}" && ${force} -eq 0 ]]; then
    echo "hifiasm ${VERSION} already built at ${BIN_PATH}"
else
    for tool in make g++; do
        if ! command -v "${tool}" >/dev/null 2>&1; then
            echo "error: '${tool}' not on PATH — hifiasm builds from source" >&2
            echo "  install build tools (e.g. apt-get install build-essential zlib1g-dev)" >&2
            echo "  or just use bioconda: conda install -c bioconda hifiasm" >&2
            exit 1
        fi
    done

    mkdir -p "${TARGET_DIR}"
    TARBALL="hifiasm-${VERSION}.tar.gz"
    URL="https://github.com/chhylp123/hifiasm/archive/refs/tags/${VERSION}.tar.gz"
    work_tar="${TARGET_DIR}/${TARBALL}.part"
    echo "downloading hifiasm ${VERSION} source from ${URL}"
    if command -v curl >/dev/null 2>&1; then
        curl -fL --retry 3 -o "${work_tar}" "${URL}"
    elif command -v wget >/dev/null 2>&1; then
        wget -O "${work_tar}" "${URL}"
    else
        echo "error: neither curl nor wget is available on PATH" >&2
        exit 1
    fi

    EXPECTED_SHA="${override_sha:-${EXPECTED_SHA}}"
    if [[ "${EXPECTED_SHA}" == REPLACE_WITH_VERIFIED_SHA256 ]]; then
        echo "warning: no verified SHA256 pinned for ${VERSION}; skipping checksum." >&2
        echo "  pass --checksum <SHA> (from the release page) to enable verification." >&2
    else
        echo "verifying sha256..."
        actual_sha="$(sha256sum "${work_tar}" | awk '{print $1}')"
        if [[ "${actual_sha}" != "${EXPECTED_SHA}" ]]; then
            echo "error: sha256 mismatch for ${work_tar}" >&2
            echo "  expected: ${EXPECTED_SHA}" >&2
            echo "  actual:   ${actual_sha}" >&2
            rm -f "${work_tar}"
            exit 1
        fi
    fi

    build_dir="${TARGET_DIR}/src"
    rm -rf "${build_dir}"
    mkdir -p "${build_dir}"
    tar -xzf "${work_tar}" -C "${build_dir}" --strip-components=1
    rm -f "${work_tar}"

    echo "building hifiasm (make)..."
    make -C "${build_dir}" -j"$(nproc 2>/dev/null || echo 4)"
    cp "${build_dir}/hifiasm" "${BIN_PATH}"
    rm -rf "${build_dir}"
fi

if [[ ! -x "${BIN_PATH}" ]]; then
    echo "error: post-build ${BIN_PATH} not executable" >&2
    exit 1
fi

tmp_link="${CURRENT_LINK}.tmp.$$"
ln -sfn "${VERSION}" "${tmp_link}"
mv -Tf "${tmp_link}" "${CURRENT_LINK}"

echo "installed: ${BIN_PATH}"
echo "current:   ${CURRENT_LINK} -> ${VERSION}/"
echo
echo "run \`constellation doctor\` to confirm registry discovery."
