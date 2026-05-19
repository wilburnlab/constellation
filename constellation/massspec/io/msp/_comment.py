"""Quote-aware tokenizer for NIST .msp Comment lines.

The Comment line in an MSP entry is a free-form sequence of
``key=value`` pairs separated by whitespace:

    Parent=778.4131 Mods=0 Modstring=AAAAAAAAAAAAAAAASAGGK///2 iRT=85.41

Values may be double-quoted to embed whitespace::

    Inst="HCD" Spec="Consensus" Collision_energy=27.0

A bareword token (no ``=``) is treated as a boolean flag with value
``True``. Values are returned as raw strings — numeric coercion is the
caller's responsibility.

No regex — a small character scanner handles the corner cases (quoted
values containing ``=`` or ``/``; trailing whitespace; embedded
backslash-escaped quotes are NOT supported because NIST/MaxQuant/MSFragger
exports don't emit them in practice).
"""

from __future__ import annotations


def tokenize_comment(line: str) -> dict[str, str | bool]:
    """Parse an MSP Comment line into a ``{key: value}`` dict.

    Strips a leading ``Comment:`` header if present, then walks the
    line character-by-character collecting whitespace-separated
    tokens. Tokens of the form ``key=value`` populate the dict;
    barewords with no ``=`` populate the dict with value ``True``.

    Returns an empty dict for an empty / whitespace-only line.
    """
    s = line.rstrip("\r\n")
    if s.lower().startswith("comment:"):
        s = s[len("comment:") :]
    s = s.strip()
    if not s:
        return {}

    out: dict[str, str | bool] = {}
    i = 0
    n = len(s)
    while i < n:
        # Skip leading whitespace
        while i < n and s[i].isspace():
            i += 1
        if i >= n:
            break

        # Read key up to '=' or whitespace
        key_start = i
        while i < n and s[i] != "=" and not s[i].isspace():
            i += 1
        key = s[key_start:i]
        if not key:
            break

        if i >= n or s[i].isspace():
            # Bareword: no '=' present
            out[key] = True
            continue

        # s[i] == '='; consume it
        i += 1

        # Read value — may be quoted
        if i < n and s[i] == '"':
            i += 1  # consume opening quote
            val_start = i
            while i < n and s[i] != '"':
                i += 1
            value = s[val_start:i]
            if i < n:
                i += 1  # consume closing quote
        else:
            val_start = i
            while i < n and not s[i].isspace():
                i += 1
            value = s[val_start:i]

        out[key] = value
    return out


__all__ = ["tokenize_comment"]
