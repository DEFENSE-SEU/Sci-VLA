"""Shared helpers for v2 and v3 stats readers.

v2 and v3 both need to:
  * Read `observation.state`/`action` columns from a parquet view.
  * Stack those columns per schema key with target dim.
  * Tolerate shards/episodes lacking required columns.
  * Tolerate legacy column-name aliases (`state` ↔ `observation.state`,
    `actions` ↔ `action`) so the stats sample set matches what the training
    adapter produces (see ``LeRobotAdapterBase.CANONICAL_ALT_KEYS``).

Neither reader owns the iteration shape (v2 iterates per-episode parquets,
v3 iterates per-shard with slicing), so a full base class does not simplify
either side. Instead we hoist only the genuinely shared primitives so future
changes can keep the two files in lock-step.

Keep this module free of I/O orchestration (ThreadPoolExecutor, progress
prints) — those concerns live in the concrete readers and differ by format.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Iterable

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


# Mirror of ``src/adapters/lerobot_base.py::LeRobotAdapterBase.CANONICAL_ALT_KEYS``.
# Duplicated (not imported) to keep ``data_process`` independent of
# ``src/adapters`` so the stats CLI runs without the adapter import path. The
# two MUST stay in sync; ``tests/integration/test_stats_running.py`` checks both.
#
# The map is BIDIRECTIONAL: a legacy schema requesting ``state``/``actions``
# against a parquet using modern ``observation.state``/``action`` names would
# otherwise hit "missing column" and silently drop the shard, shrinking the
# sample set. Listing both directions lets either name resolve to the file's.
CANONICAL_ALT_KEYS: dict[str, str] = {
    "observation.state": "state",
    "state": "observation.state",
    "action": "actions",
    "actions": "action",
}


# Schema → per-key dim targets (centralizes the zip pattern).
def build_target_dims(
    schema,
) -> tuple[dict[str, int], dict[str, int], list[str], tuple[str, ...], tuple[str, ...]]:
    """Return physical read keys and target dims for a DatasetSchema.

    ``need_cols`` is the de-duplicated list of columns both readers request
    from pyarrow (order preserved).
    """
    read_state_keys = (
        tuple(getattr(schema, "source_state_keys", ()) or ())
        or tuple(getattr(schema, "state_keys", ()) or ())
    )
    read_state_dims = (
        tuple(getattr(schema, "source_state_dims", ()) or ())
        or tuple(getattr(schema, "state_dims", ()) or ())
    )
    read_action_keys = (
        tuple(getattr(schema, "source_action_keys", ()) or ())
        or tuple(getattr(schema, "action_keys", ()) or ())
    )
    read_action_dims = (
        tuple(getattr(schema, "source_action_dims", ()) or ())
        or tuple(getattr(schema, "action_dims", ()) or ())
    )
    # Virtual state keys ("virtual.<name>") are materialized by the training
    # adapter as a SAME-FRAME copy of a physical column, so for
    # stats purposes the virtual key's per-frame value IS the physical
    # column's value. Resolve to the physical name here — both readers then
    # request and stack the real on-disk column with the virtual key's
    # declared dim, and the resulting state array matches what the adapter
    # feeds the transform chain. (validate_schema forbids virtual keys
    # alongside source_state_keys and in action_keys, so only the
    # state-keys branch can carry them.)
    vss = dict(getattr(schema, "virtual_state_sources", None) or {})
    if vss:
        read_state_keys = tuple(vss.get(k, k) for k in read_state_keys)

    # Next-frame synthetic action keys (the adapter materializes key[t] from
    # SOURCE[t+1]; e.g. robointer's
    # other_information.action_gripper_position) have NO physical column —
    # the old behavior read a nonexistent column, dropped every parquet and
    # aborted on the coverage gate. Resolve them to their source column here;
    # the per-episode readers then shift those columns by +1 and drop the
    # terminal row across the whole action array, so stats see EXACTLY the
    # labels training supervises (terminal frames are padded out of the loss).
    from src.adapters.lerobot_base import LeRobotAdapterBase as _AB  # lazy: torch
    _nf = {k: _AB.NEXT_FRAME_ACTION_SOURCES[k]
           for k in read_action_keys if k in _AB.NEXT_FRAME_ACTION_SOURCES}
    next_frame_cols = frozenset(_nf.values())
    if _nf:
        read_action_keys = tuple(_nf.get(k, k) for k in read_action_keys)
        read_action_dims = tuple(read_action_dims)

    def _targets(keys, dims, label: str) -> dict[str, int]:
        out: dict[str, int] = {}
        for k, d in zip(keys, dims):
            if k in out and out[k] != int(d):
                raise ValueError(
                    f"build_target_dims: {label} column {k!r} declared with "
                    f"conflicting dims {out[k]} vs {int(d)} (after virtual-"
                    f"state resolution) — ambiguous stacking target."
                )
            out[k] = int(d)
        return out

    state_targets = _targets(read_state_keys, read_state_dims, "state")
    action_targets = _targets(read_action_keys, read_action_dims, "action")
    need_cols = list(
        dict.fromkeys(list(read_state_keys) + list(read_action_keys))
    )
    return (state_targets, action_targets, need_cols, read_state_keys,
            read_action_keys, next_frame_cols)


# Per-key concat helper, hoisted so changes to the stacking strategy land once.
def stack_keys_to_array(
    table_view: pa.Table,
    keys: Iterable[str],
    targets: dict[str, int],
    stack_fn,
    next_frame_shift: frozenset = frozenset(),
) -> np.ndarray:
    """Concatenate a per-schema-key stack along the last dim.

    ``stack_fn`` is `_stack_column_pa` from stats.v2.reader, passed in to avoid
    a circular import and keep the dependency arrow one-directional.

    ``next_frame_shift``: resolved source columns whose schema key is a
    next-frame synthetic action. The table_view MUST then span exactly ONE
    episode: shifted keys take rows [1:], all other keys take rows [:-1], so
    every emitted row t pairs same-frame columns with src[t+1] for the
    shifted keys — exactly the training labels (terminal frame excluded, as
    training pads it out of the loss).
    """
    if not next_frame_shift:
        return np.concatenate(
            [stack_fn(table_view, k, target_dim=targets[k]) for k in keys],
            axis=-1,
        )
    arrs = []
    for k in keys:
        a = stack_fn(table_view, k, target_dim=targets[k])
        arrs.append(a[1:] if k in next_frame_shift else a[:-1])
    return np.concatenate(arrs, axis=-1)


# Missing-column error detection: both readers key off the same arrow message
# to decide whether a parquet should be dropped vs. re-raised.
_FIELDREF_MISS_MARKER = "No match for FieldRef"


def is_missing_column_error(exc: BaseException) -> bool:
    """True iff ``exc`` is an arrow "column not found" error.

    Both v2 and v3 readers drop parquets that raise this — any other pyarrow
    exception propagates (corrupt file, permission error, unreadable schema).
    """
    return _FIELDREF_MISS_MARKER in str(exc)


# Alias-aware column resolution: match ``LeRobotAdapterBase`` so the stats
# sample set never diverges from the training sample set.
@lru_cache(maxsize=16)
def _resolve_alias_cols_cached(
    available: frozenset[str], need_cols: tuple[str, ...]
) -> tuple[tuple[str, ...], tuple[tuple[str, str], ...]]:
    """Cached core: hashable inputs → hashable outputs. See public wrapper below."""
    read_cols: list[str] = []
    renames: list[tuple[str, str]] = []
    missing: list[str] = []
    for c in need_cols:
        if c in available:
            read_cols.append(c)
        elif c in CANONICAL_ALT_KEYS and CANONICAL_ALT_KEYS[c] in available:
            alias = CANONICAL_ALT_KEYS[c]
            read_cols.append(alias)
            renames.append((alias, c))
        else:
            missing.append(c)
    if missing:
        # Raise arrow-style so existing ``is_missing_column_error`` drop-catch
        # in v2/v3 readers fires uniformly. ``ParquetFile.read`` silently
        # filters unknown columns (unlike ``pq.read_table(path, columns=…)``),
        # so without this we'd surface the miss only at downstream
        # ``table.column(key)`` as ``KeyError`` — past the drop boundary.
        raise pa.ArrowInvalid(
            f"No match for FieldRef.Name({missing[0]!r}) in {sorted(available)}"
        )
    return tuple(read_cols), tuple(renames)


def open_and_read(
    path: Path | str, need_cols: Iterable[str]
) -> pa.Table:
    """Open ``path``, read ``need_cols`` honouring alias fallback, return table.

    Files in a homogeneous dataset share one column set, so the alias resolution
    is cached on ``(frozenset(available), tuple(need_cols))`` — for the typical
    100k-episode v2.1 run this collapses ~100k Python loops to one per dataset.

    Raises ``pyarrow.lib.ArrowInvalid`` when a requested column has no
    canonical-or-alias match; the caller recognises this via
    :func:`is_missing_column_error` and drops the file.
    """
    pf = pq.ParquetFile(str(path))
    available = frozenset(pf.schema_arrow.names)
    read_cols, renames = _resolve_alias_cols_cached(available, tuple(need_cols))
    table = pf.read(columns=list(read_cols))
    if renames:
        rename_map = dict(renames)
        table = table.rename_columns(
            [rename_map.get(n, n) for n in table.column_names]
        )
    return table
