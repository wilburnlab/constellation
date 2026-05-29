# scripts/

Idempotent, hash-pinned installers for optional third-party tools.

Each script follows the same contract:

- writes into `third_party/<tool>/<version>/...`
- (re)points `third_party/<tool>/current` at the installed version
- verifies a sha256 pin, exits nonzero on mismatch
- accepts `--force` to overwrite an existing install
- pre-flight checks for any required runtime (`java`, `dotnet`, …)

> **Exception — heavy app-dir tools.** `install-encyclopedia.sh` installs
> to the per-user home cache `~/.constellation/encyclopedia/<version>/`
> (alongside `~/.constellation/{references,catalogs,taxonomy,sessions}`),
> not `third_party/`. The EncyclopeDIA install4j build is a full app dir
> (jar + bundled JRE + native MSRawJava libs, hundreds of MB); the home
> cache keeps it out of the repo tree and shared across checkouts.

`constellation.thirdparty.registry.find(tool)` resolves:

1. `$CONSTELLATION_<TOOL>_HOME` (user override — e.g. a shared cluster install)
2. `~/.constellation/<tool>/current/` (home cache — opt-in per tool via `user_cache_dir`)
3. `third_party/<tool>/current/` (what most of these scripts create)
4. `$PATH` via `shutil.which` (for conda-installed binaries like `mmseqs`)
5. raises `ToolNotFoundError` with the script name in the message

Run `constellation doctor` to see what's installed and what's missing.

## Available installers

| Script | Tool | License | Notes |
|---|---|---|---|
| `install-encyclopedia.sh` | EncyclopeDIA ≥ 6.5.15 | Apache-2.0 | Installs to `~/.constellation/encyclopedia/`. `--installer <install4j.sh>` works today; default URL mode pending the public ≥ 6.5.15 release. |

More installers (Thermo DLLs, Dorado, IQ-TREE, …) land here as the
corresponding domain modules are wired.
