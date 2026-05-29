#!/usr/bin/env bash
# Install EncyclopeDIA (>= 6.5.15) into Constellation's per-user cache.
#
# Constellation pins EncyclopeDIA to >= 6.5.15 — the version available at
# public release and the floor the transcriptome->proteome pipeline needs
# (older builds lack the `-convert -fastaToJChronologerLibrary`
# predict-library utility). This script never installs anything older.
#
# Install location (NOT third_party/ — the install4j build is a heavy app
# dir with a bundled JRE + native MSRawJava libs, so it lives in the
# per-user home cache alongside ~/.constellation/{references,catalogs,...}):
#
#   ~/.constellation/encyclopedia/<version>/encyclopedia-<version>.jar
#   ~/.constellation/encyclopedia/<version>/jre/...   (bundled by install4j)
#   ~/.constellation/encyclopedia/current  -> <version>/   (symlink)
#
# `constellation.thirdparty.registry.find("encyclopedia")` auto-discovers
# this location (env var -> ~/.constellation/encyclopedia/current ->
# third_party -> $PATH). Override with $CONSTELLATION_ENCYCLOPEDIA_HOME to
# point at an install elsewhere (e.g. a shared cluster install or an
# ad-hoc dev build); in that case you don't need this script.
#
# Two modes:
#   --installer <path>   run a local install4j self-extracting installer
#                        (the lab's 6.5.15+ build). WORKS TODAY.
#   (default)            download from the public release URL. The public
#                        >= 6.5.15 release is NOT YET PUBLISHED, so the
#                        default mode currently errors and points you at
#                        --installer.
#
# Usage:
#   bash scripts/install-encyclopedia.sh --installer /path/to/encyclopedia-6.5.15.sh
#   bash scripts/install-encyclopedia.sh --installer ... --sha256 <hex>   # verify the installer blob
#   bash scripts/install-encyclopedia.sh --installer ... --force          # overwrite an existing install
#   bash scripts/install-encyclopedia.sh                                  # default URL mode (pending public release)
#
# Apache-2.0; no license-acknowledgement prompt required.
set -euo pipefail

MINIMUM_VERSION="6.5.15"

# Public >= 6.5.15 release artifact — NOT YET PUBLISHED. Leave these as the
# sentinel until a public release exists; default mode errors helpfully
# while RELEASE_URL is unset. When a release ships, set RELEASE_URL to the
# download (a `.jar` or an install4j `.sh`) and RELEASE_SHA256 to its
# verified digest. The download artifact type is auto-detected.
RELEASE_URL="__UNSET__"
RELEASE_SHA256="__UNSET__"

INSTALL_ROOT="${HOME}/.constellation/encyclopedia"

# ── Arg parsing ─────────────────────────────────────────────────────────
installer_path=""
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

# EncyclopeDIA is a JVM tool. The install4j build bundles its own JRE, so a
# system `java` is not strictly required to RUN it — but a bare-jar install
# (default-mode `.jar`) would need one. Warn rather than block.
if ! command -v java >/dev/null 2>&1; then
    cat >&2 <<'EOF'
warning: no `java` found on PATH.

  The install4j build bundles its own JRE (jre/bin/java beside the jar),
  so it will still run. A bare-jar install would need Java — get it via:
      conda env update -f environment.yml --prune    # pulls in openjdk
EOF
fi

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
        echo "       Provide a >= ${MINIMUM_VERSION} build (the lab's install4j installer) via --installer." >&2
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

# Download the public release and install it (artifact-type-agnostic).
install_default() {  # staging
    local staging="$1"
    if is_sentinel "${RELEASE_URL}"; then
        cat >&2 <<EOF
error: no public EncyclopeDIA >= ${MINIMUM_VERSION} release is published yet,
and RELEASE_URL in this script is still a placeholder.

Install the lab's install4j build now:
    bash scripts/install-encyclopedia.sh --installer /path/to/encyclopedia-${MINIMUM_VERSION}.sh

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
            local jar_base ver app_root
            jar_base="$(basename "${download}")"
            ver="$(_jar_version "${download}")"
            _check_version "${ver}" "${jar_base}"
            app_root="${staging}/app"
            mkdir -p "${app_root}"
            mv "${download}" "${app_root}/${jar_base}"
            finalize_install "${app_root}" "${ver}" ;;
        *)
            echo "error: unrecognized download artifact: ${download} (expected .sh or .jar)" >&2
            exit 1 ;;
    esac
}

# ── Run ─────────────────────────────────────────────────────────────────
mkdir -p "${INSTALL_ROOT}"
STAGING="$(mktemp -d "${INSTALL_ROOT}/.staging.XXXXXX")"
trap 'rm -rf "${STAGING}"' EXIT

if [[ -n "${installer_path}" ]]; then
    install_from_installer "${installer_path}" "${STAGING}"
else
    install_default "${STAGING}"
fi

echo
echo "run \`constellation doctor\` to confirm registry discovery."
