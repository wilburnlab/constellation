#!/usr/bin/env bash
# Install the EncyclopeDIA jar into Constellation's third_party/ layout.
#
# Layout (matches the thirdparty.registry contract):
#   third_party/encyclopedia/<version>/encyclopedia-<version>-executable.jar
#   third_party/encyclopedia/current  -> <version>/   (symlink)
#
# Override with $CONSTELLATION_ENCYCLOPEDIA_HOME if you keep a shared
# install elsewhere (e.g. /opt/bioinfo/encyclopedia/3.0.4/); in that
# case, don't run this script — the registry will find it via the env var.
#
# Usage:
#   bash scripts/install-encyclopedia.sh
#   bash scripts/install-encyclopedia.sh --force   # overwrite existing install
#
# Apache-2.0; no license-acknowledgement prompt required.
set -euo pipefail

VERSION="3.0.4"
JAR_FILENAME="encyclopedia-${VERSION}-executable.jar"
JAR_URL="https://bitbucket.org/searleb/encyclopedia/downloads/${JAR_FILENAME}"
JAR_SHA256="a21ef9248aa05c0e531d950af1201d6deefa5c2a2ef200f6a13102041d825071"

# Repo root = directory containing pyproject.toml (this script lives in scripts/).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

TARGET_DIR="${REPO_ROOT}/third_party/encyclopedia/${VERSION}"
CURRENT_LINK="${REPO_ROOT}/third_party/encyclopedia/current"
JAR_PATH="${TARGET_DIR}/${JAR_FILENAME}"

force=0
for arg in "$@"; do
    case "${arg}" in
        --force) force=1 ;;
        -h|--help)
            cat <<'USAGE'
install-encyclopedia.sh — fetch the pinned EncyclopeDIA jar.

Writes to:
  third_party/encyclopedia/<version>/encyclopedia-<version>-executable.jar
  third_party/encyclopedia/current  -> <version>/   (symlink)

Usage:
  bash scripts/install-encyclopedia.sh
  bash scripts/install-encyclopedia.sh --force   # overwrite existing install

Override via $CONSTELLATION_ENCYCLOPEDIA_HOME to use a shared install
elsewhere (e.g. /opt/bioinfo/encyclopedia/3.0.4/); in that case skip
this script — the registry will find it via the env var.

Apache-2.0; no license-acknowledgement prompt required.
USAGE
            exit 0
            ;;
        *) echo "unknown arg: ${arg}" >&2; exit 2 ;;
    esac
done

# Pre-flight: EncyclopeDIA is a JVM tool. Catch the missing-java case early.
if ! command -v java >/dev/null 2>&1; then
    cat >&2 <<'EOF'
error: `java` not found on PATH.

EncyclopeDIA is a JVM tool. Install openjdk into the active conda env:

    conda env update -f environment.yml --prune

(environment.yml pulls in `openjdk`.) Then re-run this script.
EOF
    exit 1
fi

mkdir -p "${TARGET_DIR}"

if [[ -f "${JAR_PATH}" && ${force} -eq 0 ]]; then
    echo "encyclopedia ${VERSION} already installed at ${JAR_PATH}"
else
    echo "downloading EncyclopeDIA ${VERSION} from ${JAR_URL}"
    # Prefer curl; fall back to wget. Both are ubiquitous.
    if command -v curl >/dev/null 2>&1; then
        curl -fL --retry 3 -o "${JAR_PATH}.part" "${JAR_URL}"
    elif command -v wget >/dev/null 2>&1; then
        wget -O "${JAR_PATH}.part" "${JAR_URL}"
    else
        echo "error: neither curl nor wget is available on PATH" >&2
        exit 1
    fi
    mv "${JAR_PATH}.part" "${JAR_PATH}"
fi

echo "verifying sha256..."
actual_sha="$(sha256sum "${JAR_PATH}" | awk '{print $1}')"
if [[ "${actual_sha}" != "${JAR_SHA256}" ]]; then
    echo "error: sha256 mismatch for ${JAR_PATH}" >&2
    echo "  expected: ${JAR_SHA256}" >&2
    echo "  actual:   ${actual_sha}" >&2
    rm -f "${JAR_PATH}"
    exit 1
fi

# Point `current` at this version. Recreate atomically so concurrent
# `constellation doctor` runs see a consistent state.
tmp_link="${CURRENT_LINK}.tmp.$$"
ln -sfn "${VERSION}" "${tmp_link}"
mv -Tf "${tmp_link}" "${CURRENT_LINK}"

echo "installed: ${JAR_PATH}"
echo "current:   ${CURRENT_LINK} -> ${VERSION}/"
echo
echo "run \`constellation doctor\` to confirm registry discovery."
