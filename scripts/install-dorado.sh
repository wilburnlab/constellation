#!/usr/bin/env bash
# Install the Dorado basecaller into Constellation's third_party/ layout.
#
# Dorado is NOT redistributed by Constellation. It is downloaded from
# Oxford Nanopore's CDN at install time and is governed by the Oxford
# Nanopore Technologies PLC. Public License Version 1.0 — RESEARCH
# PURPOSES ONLY. Invoking Dorado (directly or via the Constellation
# wrapper) means accepting ONT's terms. This script never bundles the
# binary; it only fetches it for you.
#
# Layout (matches the thirdparty.registry contract — artifact 'bin/dorado'):
#   third_party/dorado/<version>/bin/dorado
#   third_party/dorado/current -> <version>/   (symlink)
#
# Override with $CONSTELLATION_DORADO_HOME to use a shared install.
#
# Usage:
#   bash scripts/install-dorado.sh
#   bash scripts/install-dorado.sh --force
#   bash scripts/install-dorado.sh --version 0.8.3
#   bash scripts/install-dorado.sh --checksum <SHA256>
#   CONSTELLATION_ACCEPT_DORADO_LICENSE=1 bash scripts/install-dorado.sh  # non-interactive
#
# Wrapper code is Apache-2.0 (Constellation); Dorado itself is ONT's.
set -euo pipefail

VERSION="0.8.3"
PINNED_VERSION="0.8.3"
# Per-platform SHA256 of the ONT CDN tarballs for PINNED_VERSION. The ONT CDN
# artifacts are byte-stable, so these are reliable integrity pins. linux-x64
# verified 2026-06-09; arm64 / osx-arm64 are placeholders — pass --checksum
# to verify those (or fill them in once verified). For any other --version,
# pass --checksum.
declare -A PINNED_SHA
PINNED_SHA[linux-x64]="8b679ed7faa61299af2df591322b2737d61106a53f3175cc2d4efe0a31242ec2"
PINNED_SHA[linux-arm64]="REPLACE_WITH_VERIFIED_SHA256"
PINNED_SHA[osx-arm64]="REPLACE_WITH_VERIFIED_SHA256"

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
install-dorado.sh — fetch the Oxford Nanopore Dorado basecaller.

Dorado is downloaded from ONT's CDN (never bundled) and is licensed for
RESEARCH PURPOSES ONLY (Oxford Nanopore PLC Public License v1.0). You
must acknowledge the license once (interactive prompt, or set
CONSTELLATION_ACCEPT_DORADO_LICENSE=1).

Writes to:
  third_party/dorado/<version>/bin/dorado
  third_party/dorado/current -> <version>/   (symlink)

Usage:
  bash scripts/install-dorado.sh [--force] [--version X] [--checksum SHA]

Override via $CONSTELLATION_DORADO_HOME to use a shared install.
USAGE
            exit 0
            ;;
        *) echo "unknown arg: $1" >&2; exit 2 ;;
    esac
done

# Platform → ONT CDN artifact suffix.
os="$(uname -s)"
arch="$(uname -m)"
case "${os}-${arch}" in
    Linux-x86_64)  PLATFORM="linux-x64" ;;
    Linux-aarch64) PLATFORM="linux-arm64" ;;
    Darwin-arm64)  PLATFORM="osx-arm64" ;;
    *)
        echo "error: unsupported platform ${os}-${arch}" >&2
        echo "  Dorado ships linux-x64 / linux-arm64 / osx-arm64 — see" >&2
        echo "  https://github.com/nanoporetech/dorado#installation" >&2
        exit 1
        ;;
esac

TARGET_DIR="${REPO_ROOT}/third_party/dorado/${VERSION}"
CURRENT_LINK="${REPO_ROOT}/third_party/dorado/current"
BIN_PATH="${TARGET_DIR}/bin/dorado"

# ── License acknowledgement (once per install) ───────────────────────
ACK_FILE="${TARGET_DIR}/.dorado_license_accepted"
if [[ ! -f "${ACK_FILE}" ]]; then
    cat <<'NOTICE'

  Dorado is distributed by Oxford Nanopore Technologies under the
  Oxford Nanopore Technologies PLC. Public License Version 1.0 —
  RESEARCH PURPOSES ONLY. Constellation does not redistribute Dorado;
  this script downloads it from ONT's CDN. By proceeding you acknowledge
  that your use of Dorado is governed by ONT's license, not Constellation's.
  Full text: https://github.com/nanoporetech/dorado/blob/master/LICENCE.txt

NOTICE
    if [[ "${CONSTELLATION_ACCEPT_DORADO_LICENSE:-}" != "1" ]]; then
        if [[ ! -t 0 ]]; then
            echo "error: no TTY and CONSTELLATION_ACCEPT_DORADO_LICENSE != 1; aborting." >&2
            exit 1
        fi
        read -r -p "Type 'accept' to proceed, anything else to abort: " resp
        if [[ "${resp}" != "accept" ]]; then
            echo "aborted — license not accepted." >&2
            exit 1
        fi
    fi
fi

if [[ -x "${BIN_PATH}" && ${force} -eq 0 ]]; then
    echo "dorado ${VERSION} already installed at ${BIN_PATH}"
else
    mkdir -p "${TARGET_DIR}"
    TARBALL="dorado-${VERSION}-${PLATFORM}.tar.gz"
    URL="https://cdn.oxfordnanoportal.com/software/analysis/${TARBALL}"
    work_tar="${TARGET_DIR}/${TARBALL}.part"
    echo "downloading dorado ${VERSION} (${PLATFORM}) from ${URL}"
    if command -v curl >/dev/null 2>&1; then
        curl -fL --retry 3 -o "${work_tar}" "${URL}"
    elif command -v wget >/dev/null 2>&1; then
        wget -O "${work_tar}" "${URL}"
    else
        echo "error: neither curl nor wget is available on PATH" >&2
        exit 1
    fi

    if [[ -n "${override_sha}" ]]; then
        EXPECTED_SHA="${override_sha}"
    elif [[ "${VERSION}" == "${PINNED_VERSION}" ]]; then
        EXPECTED_SHA="${PINNED_SHA[${PLATFORM}]:-}"
    else
        EXPECTED_SHA=""
    fi
    if [[ -z "${EXPECTED_SHA}" || "${EXPECTED_SHA}" == REPLACE_WITH_VERIFIED_SHA256 ]]; then
        echo "warning: no verified SHA256 pinned for dorado ${VERSION}/${PLATFORM}; skipping checksum." >&2
        echo "  pass --checksum <SHA> (from ONT) to enable verification." >&2
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

    # Tarball lays out as ./dorado-<ver>-<platform>/{bin,lib}. Strip the
    # top level so we land at <version>/bin/dorado directly.
    tar -xzf "${work_tar}" -C "${TARGET_DIR}" --strip-components=1
    rm -f "${work_tar}"
fi

if [[ ! -x "${BIN_PATH}" ]]; then
    echo "error: post-install ${BIN_PATH} not executable" >&2
    exit 1
fi

# Record acknowledgement now that the install dir exists.
date -u +"%Y-%m-%dT%H:%M:%SZ" > "${ACK_FILE}" 2>/dev/null || true

tmp_link="${CURRENT_LINK}.tmp.$$"
ln -sfn "${VERSION}" "${tmp_link}"
mv -Tf "${tmp_link}" "${CURRENT_LINK}"

echo "installed: ${BIN_PATH}"
echo "current:   ${CURRENT_LINK} -> ${VERSION}/"
echo
echo "run \`constellation doctor\` to confirm registry discovery."
