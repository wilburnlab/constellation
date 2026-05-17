"""Shared stdlib-HTTP helpers for catalog fetchers.

Matches the conventions established in
``constellation.sequencing.reference.fetch`` — ``urllib.request``,
no third-party deps, transparent gzip decoding via ``.gz`` suffix
sniffing.
"""

from __future__ import annotations

import gzip
import urllib.request

from constellation import __version__ as _CONSTELLATION_VERSION


_USER_AGENT = f"constellation/{_CONSTELLATION_VERSION}"


def http_get_bytes(url: str, *, timeout: int = 120) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = resp.read()
    if url.endswith(".gz") or url.endswith(".gzip"):
        return gzip.decompress(body)
    return body


def http_get_text(url: str, *, timeout: int = 120, encoding: str = "utf-8") -> str:
    return http_get_bytes(url, timeout=timeout).decode(encoding, errors="replace")


__all__ = ["http_get_bytes", "http_get_text"]
