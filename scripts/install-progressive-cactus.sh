#!/usr/bin/env bash
# Install Progressive Cactus into Constellation's third_party/ layout.
#
# Cactus (whole-genome aligner; Cactus-graph based) ships a precompiled
# binary release: a tree of native binaries (bin/) + shared libs (lib/) +
# a Python package that installs the `cactus` driver into a self-contained
# virtualenv. We follow upstream BIN-INSTALL.md exactly, then emit a
# wrapper that bakes the PATH / PYTHONPATH / LD_LIBRARY_PATH the native
# binaries need. Used by the de novo / comparative stages of the genome
# pipeline (Phase 3/4); also the optional HAL synteny backend for Ragout.
#
# Layout (matches the thirdparty.registry contract — artifact 'cactus'):
#   third_party/cactus/<version>/dist/             (extracted bin/ + lib/ + venv)
#   third_party/cactus/<version>/dist/venv-cactus/ (the cactus venv)
#   third_party/cactus/<version>/cactus            (wrapper)
#   third_party/cactus/current -> <version>/       (symlink)
#
# Requires python3 (>= 3.10) + a C toolchain (toil's deps may build wheels)
# + curl/wget. Override with $CONSTELLATION_CACTUS_HOME for a shared install.
#
# Usage:
#   bash scripts/install-progressive-cactus.sh
#   bash scripts/install-progressive-cactus.sh --force
#   bash scripts/install-progressive-cactus.sh --legacy        # older CPUs
#   bash scripts/install-progressive-cactus.sh --checksum <SHA256>
#
# Cactus is MIT-licensed.
set -euo pipefail

# Build the venv in isolation: an exported PYTHONPATH (common on HPC login
# shells) leaks the parent environment into the venv at install time. The
# runtime wrapper re-sets PYTHONPATH to the cactus lib/ dir itself.
unset PYTHONPATH

VERSION="3.2.1"
asset="main"   # 'main' = cactus-bin; 'legacy' = cactus-bin-legacy (older CPUs)

# SHA256s for each upstream tarball at this version. 'main' verified
# 2026-06-15 against the actual GitHub release download
# (https://github.com/ComparativeGenomicsToolkit/cactus/releases/tag/v3.2.1).
# 'legacy' is a placeholder — pass --checksum <SHA> (verified from the
# release page) when using --legacy until it's filled in. If upstream
# rebuilds the tag, these need re-verification.
declare -A SHA256_BY_ASSET
SHA256_BY_ASSET[main]="abfb625ee20caacc499d071a55bdfca947ed2fa049f59d4936370c43bcbb2828"
SHA256_BY_ASSET[legacy]="REPLACE_WITH_VERIFIED_SHA256_FOR_LEGACY"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

force=0
override_sha=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --force) force=1; shift ;;
        --version) VERSION="$2"; shift 2 ;;
        --legacy) asset="legacy"; shift ;;
        --checksum) override_sha="$2"; shift 2 ;;
        -h|--help)
            cat <<'USAGE'
install-progressive-cactus.sh — install the Progressive Cactus binary release.

Downloads the upstream cactus-bin tarball and builds its self-contained
virtualenv per BIN-INSTALL.md, then emits a wrapper exposing the
PATH/PYTHONPATH/LD_LIBRARY_PATH the native binaries require.

Writes to:
  third_party/cactus/<version>/dist/        (bin/ + lib/ + venv-cactus/)
  third_party/cactus/<version>/cactus       (wrapper)
  third_party/cactus/current -> <version>/   (symlink)

Requires python3 (>= 3.10) + curl/wget. Use --legacy on older CPUs.
Override via $CONSTELLATION_CACTUS_HOME for a shared install.

Cactus is MIT-licensed.
USAGE
            exit 0
            ;;
        *) echo "unknown arg: $1" >&2; exit 2 ;;
    esac
done

EXPECTED_SHA="${override_sha:-${SHA256_BY_ASSET[${asset}]}}"
if [[ "${EXPECTED_SHA}" == REPLACE_WITH_VERIFIED_SHA256* ]]; then
    echo "error: SHA256 placeholder for the ${asset} asset — verify upstream and re-run with --checksum" >&2
    echo "  upstream releases: https://github.com/ComparativeGenomicsToolkit/cactus/releases/tag/v${VERSION}" >&2
    exit 1
fi

if [[ "${asset}" == "legacy" ]]; then
    TARBALL="cactus-bin-legacy-v${VERSION}.tar.gz"
else
    TARBALL="cactus-bin-v${VERSION}.tar.gz"
fi
URL="https://github.com/ComparativeGenomicsToolkit/cactus/releases/download/v${VERSION}/${TARBALL}"

TARGET_DIR="${REPO_ROOT}/third_party/cactus/${VERSION}"
DIST="${TARGET_DIR}/dist"
VENV="${DIST}/venv-cactus"
CURRENT_LINK="${REPO_ROOT}/third_party/cactus/current"
ENTRY="${TARGET_DIR}/cactus"

if [[ -e "${ENTRY}" && ${force} -eq 0 ]]; then
    echo "cactus ${VERSION} already installed at ${ENTRY}"
else
    if ! command -v python3 >/dev/null 2>&1; then
        echo "error: python3 not on PATH (Cactus needs python3 >= 3.10)" >&2
        exit 1
    fi
    rm -rf "${TARGET_DIR}"
    mkdir -p "${DIST}"

    work_tar="${TARGET_DIR}/${TARBALL}.part"
    echo "downloading Cactus ${VERSION} (${asset}) from ${URL}"
    echo "  (this is a large download, ~300+ MB)"
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

    # Upstream tarball lays out as ./cactus-bin-v<ver>/{bin,lib,setup.py,...}.
    # Strip that top-level so we land at <dist>/{bin,lib,...} directly.
    tar -xzf "${work_tar}" -C "${DIST}" --strip-components=1
    rm -f "${work_tar}"

    echo "building the cactus virtualenv..."
    python3 -m venv "${VENV}"
    "${VENV}/bin/pip" install --quiet --upgrade setuptools pip wheel
    # Upstream order: install cactus (`.`) first, then re-pin the validated
    # toil from toil-requirement.txt. Both run with CWD = dist so the local
    # source tree and the relative requirement file resolve.
    ( cd "${DIST}" && "${VENV}/bin/pip" install --quiet -U . )
    ( cd "${DIST}" && "${VENV}/bin/pip" install --quiet -U -r ./toil-requirement.txt )

    if [[ ! -x "${VENV}/bin/cactus" ]]; then
        echo "error: cactus console script not found in the venv after install (${VENV}/bin/cactus)" >&2
        exit 1
    fi

    # Wrapper: the native binaries in dist/bin link against dist/lib, and
    # the cactus python package imports from dist/lib. Bake absolute paths
    # (the `current` symlink swap stays transparent). The wrapper sets these
    # itself, so it is robust even if a caller strips PYTHONPATH.
    cat > "${ENTRY}" <<EOF
#!/usr/bin/env bash
# Auto-generated by scripts/install-progressive-cactus.sh — runs the
# cactus driver with the PATH/PYTHONPATH/LD_LIBRARY_PATH its native
# binaries + python package require.
export PATH="${DIST}/bin:${VENV}/bin:\$PATH"
export PYTHONPATH="${DIST}/lib\${PYTHONPATH:+:\$PYTHONPATH}"
export LD_LIBRARY_PATH="${DIST}/lib\${LD_LIBRARY_PATH:+:\$LD_LIBRARY_PATH}"
exec "${VENV}/bin/cactus" "\$@"
EOF
    chmod +x "${ENTRY}"
fi

tmp_link="${CURRENT_LINK}.tmp.$$"
ln -sfn "${VERSION}" "${tmp_link}"
mv -Tf "${tmp_link}" "${CURRENT_LINK}"

echo "installed: ${ENTRY}"
echo "current:   ${CURRENT_LINK} -> ${VERSION}/"
echo
echo "run \`constellation doctor\` to confirm registry discovery."
