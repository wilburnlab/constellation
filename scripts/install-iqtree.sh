#!/usr/bin/env bash
# Install IQ-TREE 3 static binary into Constellation's third_party/ layout.
#
# IQ-TREE is a self-contained C++ binary (no Python / numpy deps), so we
# fetch the pinned upstream release tarball directly — no conda env needed.
# Used by the phylogenomics / cross-validation stages of the genome
# pipeline (Phase 4).
#
# Layout (matches the thirdparty.registry contract):
#   third_party/iqtree/<version>/bin/iqtree3
#   third_party/iqtree/current -> <version>/   (symlink)
#
# Override with $CONSTELLATION_IQTREE_HOME if you keep a shared install
# elsewhere; in that case skip this script — the registry resolves it via
# the env var.
#
# Usage:
#   bash scripts/install-iqtree.sh
#   bash scripts/install-iqtree.sh --force
#   bash scripts/install-iqtree.sh --variant arm        # ARM64 hosts
#   bash scripts/install-iqtree.sh --variant generic    # older / non-AVX CPUs
#   bash scripts/install-iqtree.sh --checksum <SHA256>  # override pinned hash
#
# IQ-TREE is GPLv2-licensed.
set -euo pipefail

VERSION="3.1.2"

# Variant -> release-asset suffix. 'intel' = x86_64, 'arm' = aarch64,
# 'generic' = the no-suffix Linux build (broadest CPU compatibility, e.g.
# pre-AVX hosts).
declare -A SUFFIX_BY_VARIANT
SUFFIX_BY_VARIANT[intel]="-intel"
SUFFIX_BY_VARIANT[arm]="-arm"
SUFFIX_BY_VARIANT[generic]=""

# SHA256s for each upstream tarball at this version. 'intel' verified
# 2026-06-15 against the actual GitHub release download
# (https://github.com/iqtree/iqtree3/releases/tag/v3.1.2). The other
# variants' hashes are placeholders — pass --checksum <SHA> (verified from
# the release page) until they're filled in. If upstream rebuilds the tag,
# these need re-verification.
declare -A SHA256_BY_VARIANT
SHA256_BY_VARIANT[intel]="8616116b55850f297319a613ea54ea29b49cb5eafcc1398983498a19320eb6bd"
SHA256_BY_VARIANT[arm]="REPLACE_WITH_VERIFIED_SHA256_FOR_ARM"
SHA256_BY_VARIANT[generic]="REPLACE_WITH_VERIFIED_SHA256_FOR_GENERIC"

# Default variant from host arch; overridable via --variant.
case "$(uname -m)" in
    aarch64|arm64) VARIANT="arm" ;;
    *)             VARIANT="intel" ;;
esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

force=0
override_sha=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --force) force=1; shift ;;
        --version) VERSION="$2"; shift 2 ;;
        --variant) VARIANT="$2"; shift 2 ;;
        --checksum) override_sha="$2"; shift 2 ;;
        -h|--help)
            cat <<'USAGE'
install-iqtree.sh — fetch the pinned IQ-TREE 3 static binary.

IQ-TREE is a standalone C++ binary (no Python deps), fetched from the
upstream GitHub release — not conda.

Writes to:
  third_party/iqtree/<version>/bin/iqtree3
  third_party/iqtree/current -> <version>/   (symlink)

Usage:
  bash scripts/install-iqtree.sh
  bash scripts/install-iqtree.sh --force
  bash scripts/install-iqtree.sh --variant {intel|arm|generic}
  bash scripts/install-iqtree.sh --checksum <SHA256>   # override pinned hash

Override via $CONSTELLATION_IQTREE_HOME to use a shared install elsewhere.

IQ-TREE is GPLv2-licensed.
USAGE
            exit 0
            ;;
        *) echo "unknown arg: $1" >&2; exit 2 ;;
    esac
done

if [[ -z "${SUFFIX_BY_VARIANT[${VARIANT}]+x}" ]]; then
    echo "error: unknown --variant ${VARIANT}" >&2
    echo "  supported: intel arm generic" >&2
    exit 2
fi
EXPECTED_SHA="${override_sha:-${SHA256_BY_VARIANT[${VARIANT}]}}"

if [[ "${EXPECTED_SHA}" == REPLACE_WITH_VERIFIED_SHA256* ]]; then
    echo "error: SHA256 placeholder for --variant ${VARIANT} — verify upstream and re-run with --checksum" >&2
    echo "  upstream releases: https://github.com/iqtree/iqtree3/releases/tag/v${VERSION}" >&2
    exit 1
fi

TARBALL="iqtree-${VERSION}-Linux${SUFFIX_BY_VARIANT[${VARIANT}]}.tar.gz"
URL="https://github.com/iqtree/iqtree3/releases/download/v${VERSION}/${TARBALL}"

TARGET_DIR="${REPO_ROOT}/third_party/iqtree/${VERSION}"
CURRENT_LINK="${REPO_ROOT}/third_party/iqtree/current"
BIN_PATH="${TARGET_DIR}/bin/iqtree3"

mkdir -p "${TARGET_DIR}"

if [[ -f "${BIN_PATH}" && ${force} -eq 0 ]]; then
    echo "iqtree ${VERSION} already installed at ${BIN_PATH}"
else
    work_tar="${TARGET_DIR}/${TARBALL}.part"
    echo "downloading IQ-TREE ${VERSION} (${VARIANT}) from ${URL}"
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

    # Upstream tarball lays out as ./iqtree-<version>-Linux<suffix>/{bin,...}.
    # Strip that top-level so we land at <version>/bin/iqtree3 directly.
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
