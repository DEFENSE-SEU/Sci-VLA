"""Disk-persistent scan-cache infrastructure shared by the v2.1 and v3.0
adapters.

``lerobot_v21`` re-exports the names defined here for any out-of-tree caller.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_SCAN_CACHE_FILE = ".labvla_scan_cache.json"


def _shard_fingerprint(data_root: Optional[Path]) -> dict:
    """Summarize the data/ shard parquet files by count + total size + max mtime.

    A cheap content fingerprint so the scan cache invalidates on in-place shard
    repair/replacement that keeps the episode list unchanged (keying only on
    episodes.jsonl mtime missed those). O(num_shards) ``stat()`` calls, no
    parquet open. Returns zeros when ``data_root`` is absent (degrades to the
    previous episode-list-only behavior).
    """
    if data_root is None:
        return {"shard_count": 0, "shard_total_size": 0, "shard_max_mtime": 0.0}
    count = 0
    total_size = 0
    max_mtime = 0.0
    if data_root.is_dir():
        for p in data_root.rglob("*.parquet"):
            try:
                st = p.stat()
            except OSError:
                continue
            count += 1
            total_size += int(st.st_size)
            if st.st_mtime > max_mtime:
                max_mtime = st.st_mtime
    return {
        "shard_count": count,
        "shard_total_size": total_size,
        "shard_max_mtime": max_mtime,
    }


def _scan_cache_key(
    meta_root: Path,
    schema_id: str,
    chunks_size: int,
    data_root: Optional[Path] = None,
) -> dict:
    """Fingerprint used to invalidate the cached scan.

    Invalidated when any of these change from the stored key:
      - schema_id: a different schema filters different episodes
      - chunks_size: repartitioning changes chunk_ok semantics
      - episodes_jsonl_mtime: the canonical "episode list changed" signal
      - shard_*: count/total-size/max-mtime of data/**/*.parquet, so the cache
        also invalidates when shard CONTENTS change even though the episode list
        is identical (v3 has no episodes.jsonl and relies entirely on this).
    """
    ep_jsonl = meta_root / "episodes.jsonl"
    from src.utils import env_flags as _env_flags
    return {
        "schema_id": str(schema_id),
        "chunks_size": int(chunks_size),
        # These env gates change which episodes/shards the integrity filter
        # accepts; flipping them must invalidate the cache. Read via the
        # env_flags registry (not raw os.environ) so an unset flag and an
        # explicit "=0" resolve to the same default rather than colliding with
        # opposite scan behavior.
        "allow_truncate": _env_flags.get("LABVLA_ALLOW_TRUNCATE") == "1",
        "v21_validate_per_file": str(_env_flags.get("LABVLA_V21_VALIDATE_PER_FILE")),
        "episodes_jsonl_mtime": (
            ep_jsonl.stat().st_mtime if ep_jsonl.exists() else 0.0
        ),
        **_shard_fingerprint(data_root),
    }


def _load_scan_cache(meta_root: Path, expected_key: dict) -> Optional[tuple[dict, set]]:
    """Return (chunk_ok_map, existing_episodes) if cache is valid, else None.

    Cache file is `meta/.labvla_scan_cache.json`; hidden to avoid cluttering.
    Bypass by setting env `LABVLA_SCAN_CACHE=0`.
    """
    from src.utils import env_flags as _env_flags
    if _env_flags.get("LABVLA_SCAN_CACHE") == "0":
        return None
    cache_path = meta_root / _SCAN_CACHE_FILE
    if not cache_path.exists():
        return None
    try:
        cached = json.loads(cache_path.read_text())
    except Exception:
        return None
    # MA2: expected_key may contain list[tuple] (declared_dims); the cached key
    # round-tripped through JSON is list[list]. Compare canonical JSON so tuples
    # and lists match instead of always reporting a (false) cache miss.
    if json.dumps(cached.get("key"), sort_keys=True) != json.dumps(expected_key, sort_keys=True):
        return None
    # json keys are strings → restore int dict
    chunk_ok = {int(k): bool(v) for k, v in cached.get("chunk_ok", {}).items()}
    existing = set(int(x) for x in cached.get("existing_episodes", []))
    return chunk_ok, existing


def _save_scan_cache(
    meta_root: Path,
    key: dict,
    chunk_ok: dict[int, bool],
    existing_episodes: set[int],
) -> None:
    """Atomically write scan result. Failure is non-fatal (just log)."""
    cache_path = meta_root / _SCAN_CACHE_FILE
    tmp_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
    payload = {
        "key": key,
        "chunk_ok": {str(k): bool(v) for k, v in chunk_ok.items()},
        "existing_episodes": sorted(int(x) for x in existing_episodes),
    }
    try:
        tmp_path.write_text(json.dumps(payload))
        import os
        os.replace(tmp_path, cache_path)
    except Exception as e:
        logger.warning("[scan-cache] failed to write scan cache at %s: %s", cache_path, e)
