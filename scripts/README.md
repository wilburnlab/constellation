# scripts/

Idempotent, hash-pinned installers for optional third-party tools.

Each script follows the same contract:

- writes into `third_party/<tool>/<version>/...`
- (re)points `third_party/<tool>/current` at the installed version
- verifies a sha256 pin, exits nonzero on mismatch
- accepts `--force` to overwrite an existing install
- pre-flight checks for any required runtime (`java`, `dotnet`, …)

`constellation.thirdparty.registry.find(tool)` resolves:

1. `$CONSTELLATION_<TOOL>_HOME` (user override — e.g. a shared cluster install)
2. `third_party/<tool>/current/` (what these scripts create)
3. `$PATH` via `shutil.which` (for conda-installed binaries like `mmseqs`)
4. raises `ToolNotFoundError` with the script name in the message

Run `constellation doctor` to see what's installed and what's missing.

## Available installers

| Script | Tool | License | Notes |
|---|---|---|---|
| `install-encyclopedia.sh` | EncyclopeDIA 2.12.30 | Apache-2.0 | Bitbucket release; no license prompt |

More installers (Thermo DLLs, Dorado, IQ-TREE, …) land here as the
corresponding domain modules are wired.
