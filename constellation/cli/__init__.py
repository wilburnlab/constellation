"""CLI entry points — `constellation <subcommand>` dispatcher.

Primary binary is `constellation`; thin per-domain shims (`mzpeak`,
`koina-library`, ...) in [project.scripts] forward to the same
subcommand dispatchers for muscle-memory compatibility with Cartographer.

See `__main__.py` for the dispatcher.
"""
