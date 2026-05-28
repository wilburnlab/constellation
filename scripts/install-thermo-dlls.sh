#!/usr/bin/env bash
# Install Thermo CommonCore DLLs into Constellation's third_party/ layout.
#
# Layout (matches the thirdparty.registry contract):
#   third_party/thermo/<dll_pack_version>/{ThermoFisher.CommonCore.Data.dll,
#                                          ThermoFisher.CommonCore.RawFileReader.dll,
#                                          OpenMcdf.dll,
#                                          THERMO_LICENSE,
#                                          .thermo_license_accepted}
#   third_party/thermo/current  -> <dll_pack_version>/   (symlink)
#
# DLLs come from the ThermoRawFileParser GitHub release artifact
# (Apache-2.0, compomics/ThermoRawFileParser) which is the canonical
# public redistribution channel for the Thermo RawFileReader SDK.
# Constellation (Apache-2.0) never redistributes them itself.
#
# Override with $CONSTELLATION_THERMO_HOME if you keep a shared install
# elsewhere; in that case, don't run this script — the registry will
# find it via the env var.
#
# Usage:
#   bash scripts/install-thermo-dlls.sh
#   bash scripts/install-thermo-dlls.sh --force            # overwrite existing
#   bash scripts/install-thermo-dlls.sh --skip-hash-check  # don't fail on sha mismatch
#   CONSTELLATION_ACCEPT_THERMO_LICENSE=1 bash scripts/install-thermo-dlls.sh
set -euo pipefail

# Upstream ThermoRawFileParser release that bundles the CommonCore SDK.
# Bump in lockock with _EXPECTED_*_SHA256 below and with
# THERMO_DLL_PACK_VERSION in constellation/thirdparty/thermo.py.
VERSION="1.4.5"
ARCHIVE_URL="https://github.com/compomics/ThermoRawFileParser/releases/download/v${VERSION}/ThermoRawFileParser${VERSION}.zip"

# Pins — sha256 of the upstream artifacts. Recomputed when the version
# constant changes.
ARCHIVE_SHA256="be1f8fd6f85b20750d5f59c324f031882f911035ff4e93c4e98c9b2d73944b37"
DATA_DLL_SHA256="8b10b14f885c00619a90d0a8434d08098f3bdffcc260d59feb00ea8b7efbc139"
RAWREADER_DLL_SHA256="a9b2fc190367881c6541291caf57e114f7d305a10ba933ccbb0040d3a8a33888"
OPENMCDF_DLL_SHA256="a38c2cce6561862aacd6afc500234ec011ba3d4297df6271d79ccf3f79c15f7e"

REQUIRED_MEMBERS=(
    "ThermoFisher.CommonCore.Data.dll"
    "ThermoFisher.CommonCore.RawFileReader.dll"
    "OpenMcdf.dll"
    "THERMO_LICENSE"
)

# Repo root = directory containing pyproject.toml (this script lives in scripts/).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

TARGET_DIR="${REPO_ROOT}/third_party/thermo/${VERSION}"
CURRENT_LINK="${REPO_ROOT}/third_party/thermo/current"

force=0
skip_hash_check=0
for arg in "$@"; do
    case "${arg}" in
        --force) force=1 ;;
        --skip-hash-check) skip_hash_check=1 ;;
        -h|--help)
            cat <<'USAGE'
install-thermo-dlls.sh — fetch the pinned Thermo CommonCore DLLs.

Writes to:
  third_party/thermo/<version>/{ThermoFisher.CommonCore.Data.dll,
                                ThermoFisher.CommonCore.RawFileReader.dll,
                                OpenMcdf.dll,
                                THERMO_LICENSE}
  third_party/thermo/current  -> <version>/   (symlink)

Usage:
  bash scripts/install-thermo-dlls.sh
  bash scripts/install-thermo-dlls.sh --force            # overwrite existing install
  bash scripts/install-thermo-dlls.sh --skip-hash-check  # don't fail on sha mismatch
  CONSTELLATION_ACCEPT_THERMO_LICENSE=1 bash scripts/install-thermo-dlls.sh

Override via $CONSTELLATION_THERMO_HOME to use a shared install elsewhere;
in that case skip this script — the registry will find it via the env var.

The Thermo RawFileReader SDK license permits redistribution subject to
attribution requirements. You must acknowledge the license once per
install (interactive prompt, or set CONSTELLATION_ACCEPT_THERMO_LICENSE=1).
USAGE
            exit 0
            ;;
        *) echo "unknown arg: ${arg}" >&2; exit 2 ;;
    esac
done

# Pre-flight: .NET 8 runtime needs to be on PATH for pythonnet to load
# the DLLs later. We don't *need* dotnet to install — but if it's
# absent, the install can't actually be used.
if ! command -v dotnet >/dev/null 2>&1; then
    cat >&2 <<'EOF'
warning: `dotnet` not found on PATH.

Constellation needs the .NET 8 runtime to load Thermo CommonCore DLLs.
After installing the DLLs, install the runtime into the active conda
env:

    conda env update -f environment.yml --prune

(environment.yml pulls in `dotnet-runtime=8.0.*` + `pythonnet`.)
Continuing with DLL install — but `constellation massspec convert` will
fail until the runtime is present.
EOF
fi

# Sanity tools
for tool in sha256sum unzip; do
    if ! command -v "${tool}" >/dev/null 2>&1; then
        echo "error: \`${tool}\` is required but not on PATH" >&2
        exit 1
    fi
done

if [[ -d "${TARGET_DIR}" && ${force} -eq 0 ]]; then
    have_all=1
    for f in "${REQUIRED_MEMBERS[@]}"; do
        if [[ ! -f "${TARGET_DIR}/${f}" ]]; then
            have_all=0
            break
        fi
    done
    if [[ ${have_all} -eq 1 ]]; then
        echo "thermo DLL pack ${VERSION} already installed at ${TARGET_DIR}"
        # Re-point `current` defensively (idempotent).
        tmp_link="${CURRENT_LINK}.tmp.$$"
        ln -sfn "${VERSION}" "${tmp_link}"
        mv -Tf "${tmp_link}" "${CURRENT_LINK}"
        echo "current:   ${CURRENT_LINK} -> ${VERSION}/"
        exit 0
    fi
fi

# License acknowledgement. Honoured non-interactively when
# CONSTELLATION_ACCEPT_THERMO_LICENSE=1 is set.
ACK_FILE="${TARGET_DIR}/.thermo_license_accepted"
if [[ ! -f "${ACK_FILE}" ]]; then
    cat <<'EOF'
Thermo Fisher Scientific CommonCore / RawFileReader SDK — Licensing Notice
==========================================================================

The DLLs about to be downloaded are Thermo Fisher Scientific's
RawFileReader SDK (CommonCore assemblies). Their license permits free
redistribution and use for reading Thermo RAW files, subject to:

  1. Reproducing Thermo's copyright notice and license text with any
     redistribution of the binaries.
  2. No reverse engineering, decompilation, or derivation of the source.
  3. No warranty expressed or implied.

Constellation (Apache-2.0) does NOT redistribute these DLLs. They are
downloaded on your machine on demand from the ThermoRawFileParser
GitHub release artifact (Apache-2.0, compomics/ThermoRawFileParser),
the canonical public redistribution channel for the SDK. Your use of
the DLLs is governed by Thermo's license, not Constellation's.

The full Thermo RawFileReader license text (THERMO_LICENSE) is copied
alongside the installed DLLs.

By proceeding, you acknowledge these terms.
EOF
    if [[ "${CONSTELLATION_ACCEPT_THERMO_LICENSE:-}" != "1" ]]; then
        if [[ ! -t 0 ]]; then
            echo "error: no TTY and CONSTELLATION_ACCEPT_THERMO_LICENSE != 1; aborting." >&2
            exit 1
        fi
        read -r -p "Type 'accept' to proceed, anything else to abort: " resp
        if [[ "${resp,,}" != "accept" ]]; then
            echo "Thermo license not accepted; aborting install." >&2
            exit 1
        fi
    fi
fi

mkdir -p "${TARGET_DIR}"

# Download to a tempfile so a partial fetch doesn't leave a stale archive
# behind.
TMPDIR_RUN="$(mktemp -d)"
trap 'rm -rf "${TMPDIR_RUN}"' EXIT
ARCHIVE_PATH="${TMPDIR_RUN}/ThermoRawFileParser${VERSION}.zip"

echo "downloading Thermo CommonCore (TRFP ${VERSION}) from ${ARCHIVE_URL}"
if command -v curl >/dev/null 2>&1; then
    curl -fL --retry 3 -o "${ARCHIVE_PATH}" "${ARCHIVE_URL}"
elif command -v wget >/dev/null 2>&1; then
    wget -O "${ARCHIVE_PATH}" "${ARCHIVE_URL}"
else
    echo "error: neither curl nor wget is available on PATH" >&2
    exit 1
fi

echo "verifying archive sha256..."
actual_archive_sha="$(sha256sum "${ARCHIVE_PATH}" | awk '{print $1}')"
if [[ "${actual_archive_sha}" != "${ARCHIVE_SHA256}" ]]; then
    msg="archive sha256 mismatch: expected ${ARCHIVE_SHA256}, got ${actual_archive_sha}"
    if [[ ${skip_hash_check} -eq 0 ]]; then
        echo "error: ${msg}" >&2
        exit 1
    fi
    echo "warning: ${msg} (continuing because --skip-hash-check)" >&2
fi

# Extract only the members we need. unzip with explicit names errors out
# if any member is missing — which is what we want.
echo "extracting required members → ${TARGET_DIR}"
unzip -j -o "${ARCHIVE_PATH}" "${REQUIRED_MEMBERS[@]}" -d "${TARGET_DIR}" >/dev/null

verify_one() {
    local fname="$1" expected="$2"
    local actual
    actual="$(sha256sum "${TARGET_DIR}/${fname}" | awk '{print $1}')"
    if [[ "${actual}" != "${expected}" ]]; then
        msg="sha256 mismatch for ${fname}: expected ${expected}, got ${actual}"
        if [[ ${skip_hash_check} -eq 0 ]]; then
            echo "error: ${msg}" >&2
            return 1
        fi
        echo "warning: ${msg} (continuing because --skip-hash-check)" >&2
    else
        echo "  ${fname}: sha256 verified"
    fi
}

verify_one "ThermoFisher.CommonCore.Data.dll" "${DATA_DLL_SHA256}"
verify_one "ThermoFisher.CommonCore.RawFileReader.dll" "${RAWREADER_DLL_SHA256}"
verify_one "OpenMcdf.dll" "${OPENMCDF_DLL_SHA256}"

# Record license acceptance alongside the DLLs (durable until --force or
# manual removal). Survives re-runs as a "previously accepted" marker.
touch "${ACK_FILE}"

# Point `current` at this version. Recreate atomically so concurrent
# `constellation doctor` runs see a consistent state.
tmp_link="${CURRENT_LINK}.tmp.$$"
ln -sfn "${VERSION}" "${tmp_link}"
mv -Tf "${tmp_link}" "${CURRENT_LINK}"

echo "installed: ${TARGET_DIR}/"
for f in "${REQUIRED_MEMBERS[@]}"; do
    echo "  ${f}"
done
echo "current:   ${CURRENT_LINK} -> ${VERSION}/"
echo
echo "run \`constellation doctor\` to confirm registry discovery."
