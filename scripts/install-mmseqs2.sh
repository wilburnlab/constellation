#!/usr/bin/env bash
# Install mmseqs2 static binary into Constellation's third_party/ layout.
#
# Primary install path is conda (bioconda — already in environment.yml).
# This script is the FALLBACK for nodes/containers without conda.
#
# Layout (matches the thirdparty.registry contract):
#   third_party/mmseqs2/<version>/bin/mmseqs
#   third_party/mmseqs2/current -> <version>/   (symlink)
#
# Override with $CONSTELLATION_MMSEQS2_HOME if you keep a shared
# install elsewhere; in that case skip this script — the registry
# will find it via the env var.
#
# Usage:
#   bash scripts/install-mmseqs2.sh
#   bash scripts/install-mmseqs2.sh --force
#   bash scripts/install-mmseqs2.sh --variant sse2     # AVX2-incapable CPUs
#   bash scripts/install-mmseqs2.sh --variant arm64    # ARM64 hosts
#
# GPLv3 (matches upstream mmseqs2 license).
set -euo pipefail

VERSION="15-6f452"
VARIANT="avx2"   # default; override via --variant {avx2|sse41|sse2|arm64}

# SHA256s for each upstream tarball at this version. avx2 verified
# 2026-05-19 against the actual GitHub release download
# (https://github.com/soedinglab/MMseqs2/releases/tag/15-6f452).
# The other variants' hashes are placeholders — pass --checksum <SHA>
# (verified from the release page) until they're filled in. If upstream
# rebuilds the tag, these need re-verification.
declare -A SHA256_BY_VARIANT
SHA256_BY_VARIANT[avx2]="8dc61321ebe00cfdce2773b63bce9d6a226bc2a9520ca6fee30957915eadd0a6"
SHA256_BY_VARIANT[sse41]="REPLACE_WITH_VERIFIED_SHA256_FOR_SSE41"
SHA256_BY_VARIANT[sse2]="REPLACE_WITH_VERIFIED_SHA256_FOR_SSE2"
SHA256_BY_VARIANT[arm64]="REPLACE_WITH_VERIFIED_SHA256_FOR_ARM64"

# Repo root = parent of this script's dir.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

force=0
override_sha=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --force) force=1; shift ;;
        --variant)
            VARIANT="$2"; shift 2 ;;
        --checksum)
            override_sha="$2"; shift 2 ;;
        -h|--help)
            cat <<'USAGE'
install-mmseqs2.sh — fetch the pinned mmseqs2 static binary.

Primary install path is conda (bioconda — already in environment.yml).
This script is the fallback for nodes / containers without conda.

Writes to:
  third_party/mmseqs2/<version>/bin/mmseqs
  third_party/mmseqs2/current -> <version>/   (symlink)

Usage:
  bash scripts/install-mmseqs2.sh
  bash scripts/install-mmseqs2.sh --force
  bash scripts/install-mmseqs2.sh --variant {avx2|sse41|sse2|arm64}
  bash scripts/install-mmseqs2.sh --checksum <SHA256>   # override pinned hash

Override via $CONSTELLATION_MMSEQS2_HOME to use a shared install
elsewhere; in that case skip this script — the registry resolves
via the env var.

GPLv3 (matches upstream mmseqs2 license).
USAGE
            exit 0
            ;;
        *) echo "unknown arg: $1" >&2; exit 2 ;;
    esac
done

if [[ -z "${SHA256_BY_VARIANT[${VARIANT}]:-}" ]]; then
    echo "error: unknown --variant ${VARIANT}" >&2
    echo "  supported: avx2 sse41 sse2 arm64" >&2
    exit 2
fi
EXPECTED_SHA="${override_sha:-${SHA256_BY_VARIANT[${VARIANT}]}}"

if [[ "${EXPECTED_SHA}" == REPLACE_WITH_VERIFIED_SHA256* ]]; then
    echo "error: SHA256 placeholder for --variant ${VARIANT} — verify upstream and re-run with --checksum" >&2
    echo "  upstream releases: https://github.com/soedinglab/MMseqs2/releases/tag/${VERSION}" >&2
    exit 1
fi

TARBALL="mmseqs-linux-${VARIANT}.tar.gz"
URL="https://github.com/soedinglab/MMseqs2/releases/download/${VERSION}/${TARBALL}"

TARGET_DIR="${REPO_ROOT}/third_party/mmseqs2/${VERSION}"
CURRENT_LINK="${REPO_ROOT}/third_party/mmseqs2/current"
BIN_PATH="${TARGET_DIR}/bin/mmseqs"

mkdir -p "${TARGET_DIR}"

if [[ -f "${BIN_PATH}" && ${force} -eq 0 ]]; then
    echo "mmseqs2 ${VERSION} already installed at ${BIN_PATH}"
else
    work_tar="${TARGET_DIR}/${TARBALL}.part"
    echo "downloading mmseqs2 ${VERSION} (${VARIANT}) from ${URL}"
    if command -v curl >/dev/null 2>&1; then
        curl -fL --retry 3 -o "${work_tar}" "${URL}"
    elif command -v wget >/dev/null 2>&1; then
        wget -O "${work_tar}" "${URL}"
    else
        echo "error: neither curl nor wget is available on PATH" >&2
        exit 1
    fi

    echo "verifying sha256..."
    actual_sha="$(sha256sum "${work_tar}" | awk '{print $1}')"
    if [[ "${actual_sha}" != "${EXPECTED_SHA}" ]]; then
        echo "error: sha256 mismatch for ${work_tar}" >&2
        echo "  expected: ${EXPECTED_SHA}" >&2
        echo "  actual:   ${actual_sha}" >&2
        rm -f "${work_tar}"
        exit 1
    fi

    # Extract; the upstream tarball lays out as ./mmseqs/{bin,examples,...}.
    # Strip that top-level so we land at <version>/bin/mmseqs directly.
    tar -xzf "${work_tar}" -C "${TARGET_DIR}" --strip-components=1
    rm -f "${work_tar}"
fi

if [[ ! -x "${BIN_PATH}" ]]; then
    echo "error: post-install ${BIN_PATH} not executable" >&2
    exit 1
fi

# Atomic `current` symlink retarget so concurrent `constellation doctor`
# runs see a consistent state.
tmp_link="${CURRENT_LINK}.tmp.$$"
ln -sfn "${VERSION}" "${tmp_link}"
mv -Tf "${tmp_link}" "${CURRENT_LINK}"

echo "installed: ${BIN_PATH}"
echo "current:   ${CURRENT_LINK} -> ${VERSION}/"
echo
echo "run \`constellation doctor\` to confirm registry discovery."
