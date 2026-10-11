#!/usr/bin/env bash
# Install EncyclopeDIA (>= 6.9.23) into Constellation's per-user cache.
#
# Constellation pins EncyclopeDIA to >= 6.9.23, the build its checked-in
# command snapshot (constellation/data/_raw/encyclopedia/) was captured
# from. This script never installs anything older.
#
# Install location (NOT third_party/ — the install4j build is a heavy app
# dir with a bundled JRE, ~100 library jars and native MSRawJava libs, so it
# lives in the per-user home cache alongside
# ~/.constellation/{references,catalogs,...}):
#
#   ~/.constellation/encyclopedia/<version>/encyclopedia-<version>.jar
#   ~/.constellation/encyclopedia/<version>/lib/...   (the jar's Class-Path)
#   ~/.constellation/encyclopedia/<version>/jre/...   (bundled by install4j)
#   ~/.constellation/encyclopedia/current  -> <version>/   (symlink)
#
# The jar is NOT standalone: its manifest Class-Path points at lib/ beside
# it, so the whole app dir has to be installed, never the jar on its own.
#
# `constellation.thirdparty.registry.find("encyclopedia")` auto-discovers
# this location (env var -> ~/.constellation/encyclopedia/current ->
# third_party). Override with $CONSTELLATION_ENCYCLOPEDIA_HOME to point at
# an install elsewhere (e.g. a shared cluster install); in that case you
# don't need this script.
#
# Three modes:
#   --installer <path>   run a local install4j self-extracting installer
#                        (the lab's build). WORKS TODAY.
#   --from-dir <path>    copy an already-extracted install directory (the
#                        one holding encyclopedia-<version>.jar, lib/ and
#                        jre/). WORKS TODAY. Use this to move an install off
#                        a slow or foreign filesystem (a Windows drive under
#                        WSL, a network mount).
#   (default)            download from the public release URL. The public
#                        >= 6.9.23 release is NOT YET PUBLISHED, so the
#                        default mode currently errors and points you at
#                        the two modes above.
#
# Usage:
#   bash scripts/install-encyclopedia.sh --installer /path/to/EncyclopeDIA_unix_6_9_23.sh
#   bash scripts/install-encyclopedia.sh --from-dir /path/to/EncyclopeDIA
#   bash scripts/install-encyclopedia.sh --installer ... --sha256 <hex>   # verify the installer blob
#   bash scripts/install-encyclopedia.sh --installer ... --force          # overwrite an existing install
#   bash scripts/install-encyclopedia.sh                                  # default URL mode (pending public release)
#
# Apache-2.0; no license-acknowledgement prompt required.
set -euo pipefail

MINIMUM_VERSION="6.9.23"

# Public >= 6.9.23 release artifact — NOT YET PUBLISHED. Leave these as the
# sentinel until a public release exists; default mode errors helpfully
# while RELEASE_URL is unset. When a release ships, set RELEASE_URL to the
# download (the install4j `.sh`) and RELEASE_SHA256 to its
# verified digest. It must be the install4j `.sh`: a bare jar cannot run.
RELEASE_URL="__UNSET__"
RELEASE_SHA256="__UNSET__"

INSTALL_ROOT="${HOME}/.constellation/encyclopedia"

# ── Arg parsing ─────────────────────────────────────────────────────────
installer_path=""
from_dir=""
sha256_override=""
force=0

print_usage() {
    # Print the leading comment block (after the shebang) as help text,
    # stopping at the first non-comment line — robust to line-number drift.
    awk 'NR==1 {next} /^#/ {sub(/^# ?/, ""); print; next} {exit}' "${BASH_SOURCE[0]}"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --installer) installer_path="${2:?--installer requires a path}"; shift 2 ;;
        --from-dir)  from_dir="${2:?--from-dir requires a path}"; shift 2 ;;
        --sha256)    sha256_override="${2:?--sha256 requires a hex digest}"; shift 2 ;;
        --force)     force=1; shift ;;
        -h|--help)   print_usage; exit 0 ;;
        *) echo "unknown arg: $1" >&2; exit 2 ;;
    esac
done

# ── Preflight ───────────────────────────────────────────────────────────
# `sort -V` (GNU coreutils version sort) drives the version comparison.
if ! printf '1\n2\n' | sort -V >/dev/null 2>&1; then
    echo "error: 'sort -V' (GNU coreutils version sort) is required but unavailable." >&2
    echo "       Constellation targets Linux/WSL/conda where coreutils provides it." >&2
    exit 1
fi

# EncyclopeDIA is a JVM tool, but the install4j build bundles the JRE it
# was built against (jre/bin/java beside the jar) and Constellation's runner
# prefers that one, so no system `java` is needed. The jar itself warns
# outside Java 11-21.

# ── Helpers ─────────────────────────────────────────────────────────────
is_sentinel() { [[ "$1" == "__UNSET__" ]]; }

# version_ge HAVE MIN -> exit 0 iff HAVE >= MIN (numeric, dotted).
version_ge() {
    [[ "$1" == "$2" ]] && return 0
    local smaller
    smaller="$(printf '%s\n%s\n' "$1" "$2" | sort -V | head -n1)"
    [[ "$smaller" == "$2" ]]
}

# Extract X.Y.Z from an encyclopedia jar filename (mirrors the Python
# adapter's _VERSION_RE). Prints nothing on no match.
_jar_version() {
    basename "$1" | sed -nE 's/^encyclopedia-([0-9]+\.[0-9]+\.[0-9]+)(-executable)?\.jar$/\1/p'
}

_check_version() {  # ver jar_basename ; exits nonzero on failure
    local ver="$1" name="$2"
    [[ -n "${ver}" ]] || { echo "error: cannot parse a version from ${name}" >&2; exit 1; }
    if ! version_ge "${ver}" "${MINIMUM_VERSION}"; then
        echo "error: ${name} is EncyclopeDIA ${ver}, but Constellation requires >= ${MINIMUM_VERSION}." >&2
        echo "       Provide a >= ${MINIMUM_VERSION} build via --installer or --from-dir." >&2
        exit 1
    fi
}

fetch() {  # url dest
    if command -v curl >/dev/null 2>&1; then
        curl -fL --retry 3 -o "$2" "$1"
    elif command -v wget >/dev/null 2>&1; then
        wget -O "$2" "$1"
    else
        echo "error: neither curl nor wget is available on PATH" >&2
        exit 1
    fi
}

verify_sha256() {  # file expected_hex
    echo "verifying sha256 of $(basename "$1")..."
    local actual
    actual="$(sha256sum "$1" | awk '{print $1}')"
    if [[ "${actual}" != "$2" ]]; then
        echo "error: sha256 mismatch for $1" >&2
        echo "  expected: $2" >&2
        echo "  actual:   ${actual}" >&2
        exit 1
    fi
}

# Move a prepared app dir (jar + bundled jre + ...) into the versioned
# install location and repoint `current`. `src` must be on the same
# filesystem as INSTALL_ROOT so the moves are atomic renames.
finalize_install() {  # app_root ver
    local src="$1" ver="$2"
    local target="${INSTALL_ROOT}/${ver}"
    if [[ -e "${target}" && ${force} -eq 0 ]]; then
        echo "encyclopedia ${ver} already installed at ${target} (use --force to overwrite); repointing current."
    else
        rm -rf "${target}.tmp.$$"
        mv "${src}" "${target}.tmp.$$"
        rm -rf "${target}"
        mv -Tf "${target}.tmp.$$" "${target}"
        echo "installed: ${target}"
    fi
    # Atomic `current` repoint — recreate via tmp link + rename so a
    # concurrent doctor run never sees a half-written symlink.
    local current="${INSTALL_ROOT}/current"
    ln -sfn "${ver}" "${current}.tmp.$$"
    mv -Tf "${current}.tmp.$$" "${current}"
    echo "current:   ${current} -> ${ver}/"
}

# Run a local install4j self-extracting installer into a fresh subdir of
# the staging area, locate the produced jar, version-check it, and install.
install_from_installer() {  # installer_sh staging
    local installer="$1" staging="$2"
    [[ -f "${installer}" ]] || { echo "error: installer not found: ${installer}" >&2; exit 1; }
    if [[ -n "${sha256_override}" ]]; then
        verify_sha256 "${installer}" "${sha256_override}"
    fi
    chmod +x "${installer}" 2>/dev/null || true
    local dest="${staging}/install"   # fresh, non-existent -> install4j creates it
    echo "running EncyclopeDIA install4j installer (unattended)..."
    "${installer}" -q -dir "${dest}"
    local jar
    jar="$(find "${dest}" -maxdepth 3 -name 'encyclopedia-*.jar' -print -quit)"
    [[ -n "${jar}" ]] || { echo "error: no encyclopedia-*.jar produced by the installer under ${dest}" >&2; exit 1; }
    local ver
    ver="$(_jar_version "${jar}")"
    _check_version "${ver}" "$(basename "${jar}")"
    # The jar's parent dir is the app root (jre/ sits beside the jar) — move
    # THAT so the jar lands at the top of the version dir and the registry's
    # non-recursive glob + jvm.py's <jar>/../jre/bin/java fallback both work.
    finalize_install "$(dirname "${jar}")" "${ver}"
}

# Copy an already-extracted install directory (jar + lib/ + jre/) into the
# staging area, version-check it, and install. The source is left untouched.
install_from_dir() {  # src_dir staging
    local src="$1" staging="$2"
    [[ -d "${src}" ]] || { echo "error: --from-dir is not a directory: ${src}" >&2; exit 1; }
    local jar
    jar="$(find "${src}" -maxdepth 1 -name 'encyclopedia-*.jar' -print | sort -V | tail -n1)"
    [[ -n "${jar}" ]] || { echo "error: no encyclopedia-*.jar at the top of ${src}" >&2; exit 1; }
    local ver
    ver="$(_jar_version "${jar}")"
    _check_version "${ver}" "$(basename "${jar}")"
    if [[ ! -d "${src}/lib" ]]; then
        echo "error: ${src} has no lib/ directory. The jar is not standalone (its" >&2
        echo "       Class-Path points at lib/); copy the whole install directory." >&2
        exit 1
    fi
    [[ -x "${src}/jre/bin/java" ]] || echo "warning: ${src} has no bundled jre/bin/java; a system Java 11-21 will be needed." >&2
    echo "copying EncyclopeDIA ${ver} from ${src} ..."
    local dest="${staging}/app"
    cp -a "${src}" "${dest}"
    finalize_install "${dest}" "${ver}"
}

# Download the public release and install it (artifact-type-agnostic).
install_default() {  # staging
    local staging="$1"
    if is_sentinel "${RELEASE_URL}"; then
        cat >&2 <<EOF
error: no public EncyclopeDIA >= ${MINIMUM_VERSION} release is published yet,
and RELEASE_URL in this script is still a placeholder.

Install the lab's build now, from its install4j installer or from an
already-extracted install directory:
    bash scripts/install-encyclopedia.sh --installer /path/to/EncyclopeDIA_unix_<version>.sh
    bash scripts/install-encyclopedia.sh --from-dir  /path/to/EncyclopeDIA

When a public >= ${MINIMUM_VERSION} release ships, set RELEASE_URL (+ RELEASE_SHA256)
in this script and re-run with no --installer flag. Or point
\$CONSTELLATION_ENCYCLOPEDIA_HOME at an existing install dir.
EOF
        exit 1
    fi
    local download
    download="${staging}/$(basename "${RELEASE_URL}")"
    echo "downloading EncyclopeDIA from ${RELEASE_URL}"
    fetch "${RELEASE_URL}" "${download}"
    if ! is_sentinel "${RELEASE_SHA256}"; then
        verify_sha256 "${download}" "${RELEASE_SHA256}"
    fi
    case "${download}" in
        *.sh)
            # Public release ships as an install4j installer.
            install_from_installer "${download}" "${staging}" ;;
        *.jar)
            # Since 6.x the jar's Class-Path points at lib/ beside it, so a
            # jar on its own cannot start.
            echo "error: ${download} is a bare jar, which cannot run without the" >&2
            echo "       install directory's lib/. Point RELEASE_URL at the install4j .sh." >&2
            exit 1 ;;
        *)
            echo "error: unrecognized download artifact: ${download} (expected .sh or .jar)" >&2
            exit 1 ;;
    esac
}

# ── Run ─────────────────────────────────────────────────────────────────
mkdir -p "${INSTALL_ROOT}"
STAGING="$(mktemp -d "${INSTALL_ROOT}/.staging.XXXXXX")"
trap 'rm -rf "${STAGING}"' EXIT

if [[ -n "${installer_path}" && -n "${from_dir}" ]]; then
    echo "error: pass --installer or --from-dir, not both" >&2
    exit 2
fi

if [[ -n "${installer_path}" ]]; then
    install_from_installer "${installer_path}" "${STAGING}"
elif [[ -n "${from_dir}" ]]; then
    install_from_dir "${from_dir}" "${STAGING}"
else
    install_default "${STAGING}"
fi

echo
echo "run \`constellation doctor\` to confirm registry discovery."
