#!/usr/bin/env bash
# Install Thermo CommonCore DLLs into Constellation's third_party/ layout.
#
# Layout (matches the thirdparty.registry contract):
#   third_party/thermo/<version>/{ThermoFisher.CommonCore.Data.dll,
#                                 ThermoFisher.CommonCore.RawFileReader.dll,
#                                 OpenMcdf.dll, ...,
#                                 the licenses,
#                                 .thermo_license_accepted}
#   third_party/thermo/current  -> <version>/   (symlink)
#
# Default: Thermo's RawFileReader 8.0.42 for .NET 8, the NuGet packages
# Thermo publishes in its own repository (github.com/thermofisherlsms/
# RawFileReader, Libs/NetCore/Net8), plus the two packages they depend on
# from nuget.org (OpenMcdf 3.1.4, CommandLineParser 2.9.1). Thermo replaces
# the packages in that repository with each release and keeps no older
# ones, so they are fetched from a fixed commit, and every file is
# SHA-256 pinned. 8.x reads the instrument method embedded in a .raw on
# Linux, which the CommonCore 5.0.0.93 of ThermoRawFileParser 1.4.5 does
# not (it returns None).
#
# --legacy-1.4.5 installs that older pack instead, from the
# ThermoRawFileParser 1.4.5 release artifact (Apache-2.0,
# compomics/ThermoRawFileParser). Both versions give identical peaks and
# scan values; 8.x formats filter strings about 4x slower per scan.
#
# Constellation (Apache-2.0) never redistributes these DLLs: they are
# downloaded on your machine, and your use of them is governed by
# Thermo's license.
#
# Override with $CONSTELLATION_THERMO_HOME if you keep a shared install
# elsewhere; in that case, don't run this script — the registry will
# find it via the env var.
#
# Usage:
#   bash scripts/install-thermo-dlls.sh
#   bash scripts/install-thermo-dlls.sh --legacy-1.4.5     # CommonCore 5.0.0.93
#   bash scripts/install-thermo-dlls.sh --keep-current     # install, leave `current` alone
#   bash scripts/install-thermo-dlls.sh --force            # overwrite existing
#   bash scripts/install-thermo-dlls.sh --skip-hash-check  # don't fail on sha mismatch
#   CONSTELLATION_ACCEPT_THERMO_LICENSE=1 bash scripts/install-thermo-dlls.sh
set -euo pipefail

# Keep in step with THERMO_DLL_PACK_VERSION in constellation/thirdparty/thermo.py.
VERSION="8.0.42"
THERMO_COMMIT="cd0a429a9894c13cc8bc0a5a10d54f5b7eb991bb"
THERMO_BASE="https://raw.githubusercontent.com/thermofisherlsms/RawFileReader/${THERMO_COMMIT}"
NUGET_BASE="https://api.nuget.org/v3-flatcontainer"

# One line per file to fetch: name|url|sha256|members to extract (space-
# separated, "zip-path=installed-name"); an empty member list installs the
# file itself under its name.
PACKAGES_8=(
    "thermofisher.commoncore.data.8.0.42.nupkg|${THERMO_BASE}/Libs/NetCore/Net8/thermofisher.commoncore.data.8.0.42.nupkg|0b1d2fff2f81532e0c56471b22bc0e2c0c6685f37aa43b0b134ad82ae75ae444|lib/net8.0/ThermoFisher.CommonCore.Data.dll=ThermoFisher.CommonCore.Data.dll"
    "thermofisher.commoncore.rawfilereader.8.0.42.nupkg|${THERMO_BASE}/Libs/NetCore/Net8/thermofisher.commoncore.rawfilereader.8.0.42.nupkg|af259d466e24227ba6706872330a0736d6fc2f552ad9d46a78a4697f0f62e1a6|lib/net8.0/ThermoFisher.CommonCore.RawFileReader.dll=ThermoFisher.CommonCore.RawFileReader.dll"
    "thermofisher.commoncore.randomaccessreaderplugin.8.0.42.nupkg|${THERMO_BASE}/Libs/NetCore/Net8/thermofisher.commoncore.randomaccessreaderplugin.8.0.42.nupkg|98e1831a10e7f9126802c85353a4ccca0d6169c451d414332f8983b78ad42cda|lib/net8.0/ThermoFisher.CommonCore.RandomAccessReaderPlugin.dll=ThermoFisher.CommonCore.RandomAccessReaderPlugin.dll"
    "openmcdf.3.1.4.nupkg|${NUGET_BASE}/openmcdf/3.1.4/openmcdf.3.1.4.nupkg|fa9290d5491361b7c69c5fbe7ce1c6be5a08f7d6d2ad4e8d2940f80224b73cc4|lib/net8.0/OpenMcdf.dll=OpenMcdf.dll"
    "commandlineparser.2.9.1.nupkg|${NUGET_BASE}/commandlineparser/2.9.1/commandlineparser.2.9.1.nupkg|02953dcb5c97eb475a4a33e4dca60305e27b5d9072296f21b1ef4d45919c8dea|lib/netstandard2.0/CommandLine.dll=CommandLine.dll License.md=COMMANDLINEPARSER_LICENSE.md"
    "THERMO_LICENSE.doc|${THERMO_BASE}/License.doc|4f3cce3f9f436df2c2e31523f49ede19fc06cf0b943a36d121b2c8ecb950224b|"
)
REQUIRED_8=(
    "ThermoFisher.CommonCore.Data.dll"
    "ThermoFisher.CommonCore.RawFileReader.dll"
    "ThermoFisher.CommonCore.RandomAccessReaderPlugin.dll"
    "OpenMcdf.dll"
    "CommandLine.dll"
    "THERMO_LICENSE.doc"
)

LEGACY_VERSION="1.4.5"
PACKAGES_LEGACY=(
    "ThermoRawFileParser1.4.5.zip|https://github.com/compomics/ThermoRawFileParser/releases/download/v1.4.5/ThermoRawFileParser1.4.5.zip|be1f8fd6f85b20750d5f59c324f031882f911035ff4e93c4e98c9b2d73944b37|ThermoFisher.CommonCore.Data.dll=ThermoFisher.CommonCore.Data.dll ThermoFisher.CommonCore.RawFileReader.dll=ThermoFisher.CommonCore.RawFileReader.dll OpenMcdf.dll=OpenMcdf.dll THERMO_LICENSE=THERMO_LICENSE"
)
REQUIRED_LEGACY=(
    "ThermoFisher.CommonCore.Data.dll"
    "ThermoFisher.CommonCore.RawFileReader.dll"
    "OpenMcdf.dll"
    "THERMO_LICENSE"
)
# The DLLs inside the legacy archive, pinned too.
declare -A LEGACY_DLL_SHA256=(
    ["ThermoFisher.CommonCore.Data.dll"]="8b10b14f885c00619a90d0a8434d08098f3bdffcc260d59feb00ea8b7efbc139"
    ["ThermoFisher.CommonCore.RawFileReader.dll"]="a9b2fc190367881c6541291caf57e114f7d305a10ba933ccbb0040d3a8a33888"
    ["OpenMcdf.dll"]="a38c2cce6561862aacd6afc500234ec011ba3d4297df6271d79ccf3f79c15f7e"
)

# Repo root = directory containing pyproject.toml (this script lives in scripts/).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

force=0
skip_hash_check=0
keep_current=0
legacy=0
for arg in "$@"; do
    case "${arg}" in
        --force) force=1 ;;
        --skip-hash-check) skip_hash_check=1 ;;
        --keep-current) keep_current=1 ;;
        --legacy-1.4.5) legacy=1 ;;
        -h|--help)
            cat <<'USAGE'
install-thermo-dlls.sh — fetch the pinned Thermo CommonCore DLLs.

Writes to:
  third_party/thermo/<version>/   the DLLs and their licenses
  third_party/thermo/current  -> <version>/   (symlink)

Usage:
  bash scripts/install-thermo-dlls.sh                    # RawFileReader 8.0.42
  bash scripts/install-thermo-dlls.sh --legacy-1.4.5     # CommonCore 5.0.0.93 (TRFP 1.4.5)
  bash scripts/install-thermo-dlls.sh --keep-current     # leave `current` where it is
  bash scripts/install-thermo-dlls.sh --force            # overwrite existing install
  bash scripts/install-thermo-dlls.sh --skip-hash-check  # don't fail on sha mismatch
  CONSTELLATION_ACCEPT_THERMO_LICENSE=1 bash scripts/install-thermo-dlls.sh

Override via $CONSTELLATION_THERMO_HOME to use a shared install elsewhere;
in that case skip this script — the registry will find it via the env var.

You must acknowledge Thermo's RawFileReader license once per install
(interactive prompt, or set CONSTELLATION_ACCEPT_THERMO_LICENSE=1).
USAGE
            exit 0
            ;;
        *) echo "unknown arg: ${arg}" >&2; exit 2 ;;
    esac
done

if [[ ${legacy} -eq 1 ]]; then
    VERSION="${LEGACY_VERSION}"
    PACKAGES=("${PACKAGES_LEGACY[@]}")
    REQUIRED=("${REQUIRED_LEGACY[@]}")
    SOURCE="the ThermoRawFileParser ${LEGACY_VERSION} release artifact (Apache-2.0, compomics/ThermoRawFileParser)"
    LICENSE_FILE="THERMO_LICENSE"
else
    PACKAGES=("${PACKAGES_8[@]}")
    REQUIRED=("${REQUIRED_8[@]}")
    SOURCE="Thermo's own repository (github.com/thermofisherlsms/RawFileReader at commit ${THERMO_COMMIT:0:8}) and nuget.org (OpenMcdf, MPL-2.0; CommandLineParser, MIT)"
    LICENSE_FILE="THERMO_LICENSE.doc"
fi
TARGET_DIR="${REPO_ROOT}/third_party/thermo/${VERSION}"
CURRENT_LINK="${REPO_ROOT}/third_party/thermo/current"

point_current() {
    if [[ ${keep_current} -eq 1 ]]; then
        echo "current:   left as it was (--keep-current)"
        return
    fi
    # Recreate atomically so concurrent `constellation doctor` runs see a
    # consistent state.
    local tmp_link="${CURRENT_LINK}.tmp.$$"
    ln -sfn "${VERSION}" "${tmp_link}"
    mv -Tf "${tmp_link}" "${CURRENT_LINK}"
    echo "current:   ${CURRENT_LINK} -> ${VERSION}/"
}

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

for tool in sha256sum unzip; do
    if ! command -v "${tool}" >/dev/null 2>&1; then
        echo "error: \`${tool}\` is required but not on PATH" >&2
        exit 1
    fi
done
if command -v curl >/dev/null 2>&1; then
    fetch() { curl -fL --retry 3 -o "$2" "$1"; }
elif command -v wget >/dev/null 2>&1; then
    fetch() { wget -O "$2" "$1"; }
else
    echo "error: neither curl nor wget is available on PATH" >&2
    exit 1
fi

if [[ -d "${TARGET_DIR}" && ${force} -eq 0 ]]; then
    have_all=1
    for f in "${REQUIRED[@]}"; do
        if [[ ! -f "${TARGET_DIR}/${f}" ]]; then
            have_all=0
            break
        fi
    done
    if [[ ${have_all} -eq 1 ]]; then
        echo "thermo DLL pack ${VERSION} already installed at ${TARGET_DIR}"
        point_current
        exit 0
    fi
fi

# License acknowledgement. Honoured non-interactively when
# CONSTELLATION_ACCEPT_THERMO_LICENSE=1 is set.
ACK_FILE="${TARGET_DIR}/.thermo_license_accepted"
if [[ ! -f "${ACK_FILE}" ]]; then
    cat <<EOF
Thermo Fisher Scientific CommonCore / RawFileReader SDK — Licensing Notice
==========================================================================

The DLLs about to be downloaded are Thermo Fisher Scientific's
RawFileReader SDK (CommonCore assemblies). Their license permits use
for reading Thermo RAW files, subject to its terms, among them:

  1. Reproducing Thermo's copyright notice and license text with any
     redistribution of the binaries.
  2. No reverse engineering, decompilation, or derivation of the source.
  3. No warranty expressed or implied.

Constellation (Apache-2.0) does NOT redistribute these DLLs. They are
downloaded on your machine on demand from
${SOURCE}.
Your use of the DLLs is governed by Thermo's license, not Constellation's.

Thermo's full license (${LICENSE_FILE}) is copied alongside the
installed DLLs.

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

# Downloads go to a temporary directory so a partial fetch leaves nothing
# behind.
TMPDIR_RUN="$(mktemp -d)"
trap 'rm -rf "${TMPDIR_RUN}"' EXIT

check_sha() {
    local path="$1" expected="$2" what="$3"
    local actual
    actual="$(sha256sum "${path}" | awk '{print $1}')"
    if [[ "${actual}" != "${expected}" ]]; then
        local msg="sha256 mismatch for ${what}: expected ${expected}, got ${actual}"
        if [[ ${skip_hash_check} -eq 0 ]]; then
            echo "error: ${msg}" >&2
            exit 1
        fi
        echo "warning: ${msg} (continuing because --skip-hash-check)" >&2
    else
        echo "  ${what}: sha256 verified"
    fi
}

for entry in "${PACKAGES[@]}"; do
    IFS='|' read -r name url sha members <<<"${entry}"
    path="${TMPDIR_RUN}/${name}"
    echo "downloading ${name}"
    fetch "${url}" "${path}"
    check_sha "${path}" "${sha}" "${name}"
    if [[ -z "${members}" ]]; then
        cp "${path}" "${TARGET_DIR}/${name}"
        continue
    fi
    for member in ${members}; do
        src="${member%%=*}"
        dest="${member#*=}"
        # unzip with an explicit member errors out if it is missing.
        unzip -p "${path}" "${src}" > "${TARGET_DIR}/${dest}"
        if [[ ! -s "${TARGET_DIR}/${dest}" ]]; then
            echo "error: ${name} has no ${src}" >&2
            exit 1
        fi
    done
done

if [[ ${legacy} -eq 1 ]]; then
    for f in "${!LEGACY_DLL_SHA256[@]}"; do
        check_sha "${TARGET_DIR}/${f}" "${LEGACY_DLL_SHA256[${f}]}" "${f}"
    done
fi

# Record license acceptance alongside the DLLs (durable until --force or
# manual removal). Survives re-runs as a "previously accepted" marker.
touch "${ACK_FILE}"

echo "installed: ${TARGET_DIR}/"
for f in "${REQUIRED[@]}"; do
    echo "  ${f}"
done
point_current
echo
echo "run \`constellation doctor\` to confirm registry discovery."
