# third_party/

On-disk install target for optional external binaries, jars, and DLLs
fetched by `scripts/install-*.sh`.

## Layout

```
third_party/
├── <tool>/
│   ├── <version>/           # concrete install tree for this release
│   │   └── <artifact>
│   └── current -> <version>/ # symlink that the registry resolves to
```

Two versions of the same tool can coexist here (one shared by daily
work, another being validated) — pointing `current` swaps the default.

## Discovery

`constellation.thirdparty.registry.find(tool)` resolves in this order:

1. `$CONSTELLATION_<TOOL>_HOME` (user override)
2. `third_party/<tool>/current/<artifact>` (these installs)
3. `shutil.which(<bin>)` (conda/system installs)
4. `ToolNotFoundError`

## Git hygiene

Binaries under `third_party/` are ignored by `.gitignore`. The directory
itself is tracked so `constellation doctor` has a home to look in on a
fresh checkout — the per-tool subdirectories are created by the install
scripts on demand.
