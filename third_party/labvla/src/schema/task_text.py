"""Task-text normalization heuristics — single source of truth.

Without this module, the NUL-padded-ASCII decode heuristic and the tokenized-
task normalization would exist as two hand-synced copies:

  * ``src/adapters/lerobot_v30.py`` (training adapter; env-driven coercion
    escape hatch),
  * ``data_process/meta_reader/v30.py`` (read-only correctness probe;
    deliberately NO escape hatch).

Hand-syncing a 40-line heuristic is exactly how the two stacks drift; both
callers now bind THIS implementation with their policy made explicit via
``allow_tokenized_coercion`` (no hidden env read in the shared code — the
adapter passes its env flag, the meta reader passes a hard ``False``).
"""
from __future__ import annotations

from typing import Optional

import numpy as np


def decode_ascii_byte_array(seq) -> Optional[str]:
    """Return the decoded string if ``seq`` looks like a NUL-padded UTF-8 byte
    array (TFDS-style fixed-length string encoding used by OXE
    ``language_table_*`` sub-repos), else None.

    Heuristic tightened to avoid accepting low-id tokenizer output as text:
      (a) every element in [0, 127] (ASCII range);
      (b) NUL padding (trailing zeros) present, OR the array is fully
          printable;
      (c) after stripping NULs, >= 90% printable ASCII (0x20..0x7E) or
          whitespace (\\t \\n \\r);
      (d) result is at least 2 chars long (single chars are too ambiguous —
          could be any low-id token).

    Returns None on any failure — callers fall through to their
    tokenized-task policy (fail-loud, or str() coercion when explicitly
    allowed).
    """
    if seq is None or len(seq) == 0:
        return None
    try:
        ints = [int(x) for x in seq]
    except (TypeError, ValueError):
        return None
    if any(not (0 <= b <= 127) for b in ints):
        return None
    # (b) require NUL-padding evidence (proves TFDS fixed-length encoding)
    # OR a fully printable array (unambiguously a string).
    n_trailing_zeros = 0
    for b in reversed(ints):
        if b == 0:
            n_trailing_zeros += 1
        else:
            break
    def printable_or_ws(c: int) -> bool:
        return (0x20 <= c <= 0x7E) or c in (0x09, 0x0A, 0x0D)

    non_pad = ints[: len(ints) - n_trailing_zeros] if n_trailing_zeros > 0 else ints
    if not non_pad:
        return None
    all_printable = all(printable_or_ws(c) for c in non_pad)
    if n_trailing_zeros == 0 and not all_printable:
        # No padding AND not fully printable: likely tokenizer output.
        return None
    # (c) >=90% printable within the unpadded prefix.
    n_printable = sum(1 for c in non_pad if printable_or_ws(c))
    if n_printable < 0.9 * len(non_pad):
        return None
    decoded = bytes(non_pad).decode("ascii", errors="replace").rstrip()
    # (d) single-char results are too ambiguous (could be any low-id token).
    if len(decoded) < 2:
        return None
    return decoded


def normalize_tasks(
    val,
    *,
    allow_tokenized_coercion: bool,
    undecodable_msg: str,
):
    """Normalize one ``tasks`` cell so it is always VLM-tokenizable.

    ``list<string>`` / scalar values pass through unchanged. A tokenized
    (``list<int>``) element is first run through
    :func:`decode_ascii_byte_array` (OXE ``language_table_*`` stores strings
    as NUL-padded ASCII byte arrays — text, not tokenizer output). When the
    heuristic rejects it:

      * ``allow_tokenized_coercion=True``  → legacy ``str(list(...))``
        coercion (inspection-only runs; training on digit strings is wrong);
      * ``allow_tokenized_coercion=False`` → ``ValueError(undecodable_msg)``
        — the caller supplies its own actionable message.
    """
    if val is None:
        return None
    if isinstance(val, (list, np.ndarray)):
        out = []
        for item in val:
            if isinstance(item, (list, np.ndarray)):
                item_list = list(item) if isinstance(item, np.ndarray) else item
                decoded = decode_ascii_byte_array(item_list)
                if decoded is not None:
                    out.append(decoded)
                elif allow_tokenized_coercion:
                    out.append(str(list(item_list)))
                else:
                    raise ValueError(undecodable_msg)
            else:
                out.append(str(item) if item is not None else "")
        return out
    return val


__all__ = ["decode_ascii_byte_array", "normalize_tasks"]
