"""Modalities the viz server can host.

A *modality* is one kind of browser: the genome browser today, a
mass-spec data browser next. Each is a subpackage here
(``viz/modalities/<name>/``) owning everything specific to it — session
classes, track kernels, extra endpoints — while the server core
(``viz/server/``, ``viz/tracks/base.py``) stays modality-neutral.
"""
