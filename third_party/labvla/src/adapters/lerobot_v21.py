"""LeRobot v2.1 dataset adapter.

v2.1 layout:
  <root>/meta/info.json        (codebase_version="v2.1")
  <root>/meta/episodes.jsonl   (one episode per line: {episode_index, length, tasks})
  <root>/meta/tasks.jsonl
  <root>/meta/stats.json       (absolute-value per-feature stats)
  <root>/data/chunk-XXX/episode_NNNNNN.parquet   (one parquet per episode)
  <root>/videos/chunk-XXX/<cam_key>/episode_NNNNNN.mp4

Interface satisfies `BaseAdapter` — train.py's `TransformedAdapterDataset`
only needs `adapter[idx]`, `len(adapter)`, and `adapter.meta`. No
HuggingFace lerobot dependency; this adapter reads parquet/mp4 directly.
"""
from __future__ import annotations

import json
import logging
import os
import re
from fractions import Fraction
from functools import lru_cache
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch

from .base import DatasetMeta
from .lerobot_base import LeRobotAdapterBase, _get_shared_video_cache
from src.utils.storage_retry import (
    read_parquet_with_storage_retry,
    run_with_storage_retry,
    storage_path_exists,
)

logger = logging.getLogger(__name__)


def _read_episodes_jsonl(meta_root: Path) -> list[dict]:
    out = []
    with open(meta_root / "episodes.jsonl") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    out.sort(key=lambda e: e["episode_index"])
    return out


# Scan-cache infrastructure lives in adapters/_scan_cache.py and _ep_starts_lens
# in lerobot_base (shared indexing math); re-exported here for out-of-tree callers.
from ._scan_cache import (  # noqa: F401
    _SCAN_CACHE_FILE,
    _load_scan_cache,
    _save_scan_cache,
    _scan_cache_key,
    _shard_fingerprint,
)
from .lerobot_base import _ep_starts_lens  # noqa: F401

# Episode index extracted from a v2.1 per-episode filename
# ("episode_000123.parquet" → 123). Any digit width accepted.
_EP_NUM_RE = re.compile(r"episode_(\d+)")


def _parquet_ok_v21(
    p: Path,
    *,
    needed: list,
    alt: dict,
    declared_dim: dict,
    ep_len_by_idx: dict,
    repo_id: str,
    warn_key_root: str,
    allow_truncate: bool = False,
) -> bool:
    """Schema-satisfies check for a sample parquet.

    (a) every schema key must be present (or its v2.1 alt variant);
    (b) any key with a per-episode list dim must not EXCEED the
        schema's declared dim. Source-robot episodes whose native
        state/action exceeds the declared envelope (e.g. a 49-dim
        observation.state when schema says 30) are skipped rather
        than truncated — truncation would lose actual joint info.
        Shorter is OK; Phase-1 zero-pads up to declared dim.
    (c) the parquet's row count must equal the episodes.jsonl `length`
        for this episode — a longer meta length means an IndexError
        mid-training; a shorter one means silently unsampled tail
        frames. Either way the episode is dropped with a warning.
    """
    import pyarrow.parquet as _pq
    try:
        table = _pq.read_table(str(p))
    except Exception:
        return False
    _m = _EP_NUM_RE.search(p.name)
    if _m is not None:
        expected_len = ep_len_by_idx.get(int(_m.group(1)))
        if expected_len is not None and expected_len != len(table):
            from src.utils.logging_utils import warn_once
            warn_once(
                logger,
                ("v21_row_len_mismatch", warn_key_root),
                "[v21-adapter] %s: episode_%06d parquet has %d "
                "rows but episodes.jsonl declares length=%d — "
                "dropping episode (meta/data inconsistency; "
                "further mismatches in this repo are dropped "
                "with the same policy, warned once).",
                repo_id, int(_m.group(1)),
                len(table), expected_len,
            )
            return False
    cols = set(table.schema.names)
    for k in needed:
        key_in_parquet = k if k in cols else alt.get(k)
        if key_in_parquet is None or key_in_parquet not in cols:
            return False
        # Overflow check only for keys with a declared dim.
        decl = declared_dim.get(k)
        if decl is None or len(table) == 0:
            continue
        raw = table[key_in_parquet][0].as_py()
        if isinstance(raw, (list, tuple)) and len(raw) > decl and not allow_truncate:
            # v30 honors LABVLA_ALLOW_TRUNCATE here (keep the shard, truncate at
            # read time); on v2.1 the over-wide episode is dropped instead.
            return False
    return True



class LeRobotV21Adapter(LeRobotAdapterBase):
    """Adapter for LeRobot v2.1 datasets (one parquet + one mp4 per episode)."""

    def __init__(
        self,
        repo_id: str,
        root: str | Path,
        delta_timestamps: dict | None = None,
        image_transforms=None,
        external_stats: dict | None = None,
        override_schema=None,
        video_backend: str = "pyav",
        episode_filter: list[int] | tuple[int, ...] | None = None,
    ):
        """
        Args:
            episode_filter: optional set of episode_index values to keep; all
                others are excluded. Used by the task-uniform sampler to build
                per-task views of one repo — the adapter then reports
                num_episodes/num_frames/sample() as if it held only those.

        __init__ is a thin orchestrator over the four phase methods below.
        """
        self._resolve_root(
            repo_id, root, delta_timestamps, image_transforms, video_backend
        )
        stats, schema = self._load_meta_and_stats(external_stats, override_schema)

        # Filter episodes whose parquet lacks schema-required state/action keys
        # (e.g. robocoin chunks missing gripper_open_scale_* would crash
        # downstream). Stats compute applies the same filter to stay aligned.
        if schema is not None:
            self._filter_episodes_by_schema(schema)

        # Task-uniform support: restrict to a user-supplied episode subset,
        # applied AFTER schema/parquet filters (view = filter ∩ schema-valid).
        if episode_filter is not None:
            allowed = {int(i) for i in episode_filter}
            before = len(self._episodes)
            self._episodes = [
                ep for ep in self._episodes
                if int(ep.get("episode_index", -1)) in allowed
            ]
            self._ep_starts, self._ep_lens = _ep_starts_lens(self._episodes)
            logger.info(
                "[v21-adapter] %s: episode_filter kept %d/%d episodes",
                repo_id, len(self._episodes), before,
            )

        self._drop_terminal_samples_for_next_frame_actions(schema)

        self._finalize_meta_and_caches(stats, schema)

    # ---- __init__ phase methods --------------------------------------------

    def _resolve_root(
        self,
        repo_id: str,
        root: str | Path,
        delta_timestamps: dict | None,
        image_transforms,
        video_backend: str,
    ) -> None:
        """Phase 1: root resolution, info.json version gate, episodes index."""
        self.repo_id = repo_id
        # Caller can pass either <parent>/<repo_id> or <root> directly.
        root_p = Path(root)
        if root_p.name != Path(repo_id).name and (root_p / repo_id).exists():
            root_p = root_p / repo_id
        self.root = root_p
        self.meta_root = self.root / "meta"
        self.data_root = self.root / "data"
        self.video_root = self.root / "videos"
        self.delta_timestamps = delta_timestamps or {}
        self.image_transforms = image_transforms
        self.video_backend = video_backend

        with open(self.meta_root / "info.json") as _f:
            info = json.load(_f)
        v = info.get("codebase_version", "")
        if not v.startswith("v2"):
            raise ValueError(
                f"LeRobotV21Adapter: {self.root} is codebase_version={v!r}, "
                f"expected v2.x. Use the *_old suffix dataset if this is v3."
            )
        self._info = info
        self._chunks_size = int(info.get("chunks_size", 1000))
        self._episodes = _read_episodes_jsonl(self.meta_root)
        self._ep_starts, self._ep_lens = _ep_starts_lens(self._episodes)

    def _load_meta_and_stats(self, external_stats: dict | None, override_schema):
        """Phase 2: tasks index, stats (+ external override), schema
        discovery, next-frame stats patch. Returns ``(stats, schema)``."""
        info = self._info

        # Load tasks.jsonl -> task_index (int) -> task (str). v2.1 parquet stores
        # integer `task_index` per frame; downstream transforms expect the string
        # `task` key (language instruction). Resolve on adapter load.
        self._tasks_by_idx: dict[int, str] = {}
        tasks_jsonl = self.meta_root / "tasks.jsonl"
        if tasks_jsonl.exists():
            with open(tasks_jsonl) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    ti = rec.get("task_index")
                    if ti is not None:
                        self._tasks_by_idx[int(ti)] = rec.get("task", "")

        # Stats: absolute-value per-feature dict. external_stats overrides.
        stats_path = self.meta_root / "stats.json"
        if stats_path.exists():
            from src.dataset.utils import cast_stats_to_numpy
            with open(stats_path) as _f:
                stats: dict = cast_stats_to_numpy(json.load(_f))
        else:
            stats = {}
        if external_stats:
            stats = {**stats, **external_stats}

        # Schema discovery (Tier 0 override > Tier 1 manifest > Tier 2 infer).
        from src.schema import discover_schema, SchemaDiscoveryError
        try:
            schema = discover_schema(
                self.root,
                robot_type=info.get("robot_type"),
                override=override_schema,
            )
        except SchemaDiscoveryError as e:
            if override_schema is not None:
                # The override came from --dataset_schema; discover_schema also
                # VALIDATES it against the on-disk meta (camera mapping,
                # annotation columns, ...). Swallowing the failure here would
                # train on the unvalidated override with only a warning.
                raise
            logger.warning("[v21-adapter] schema discovery failed: %s", e)
            schema = override_schema

        stats = self.patch_stats_for_next_frame_actions(stats, schema)
        return stats, schema

    def _filter_episodes_by_schema(self, schema) -> None:
        """Phase 3: drop episodes whose parquet lacks schema-required columns
        or whose row count contradicts episodes.jsonl (scan-cached)."""
        repo_id = self.repo_id
        read_state_keys = (
            tuple(getattr(schema, "source_state_keys", ()) or ())
            or tuple(getattr(schema, "state_keys", ()) or ())
        )
        read_action_keys = (
            tuple(getattr(schema, "source_action_keys", ()) or ())
            or tuple(getattr(schema, "action_keys", ()) or ())
        )
        # _resolve_physical_column also maps virtual state keys to their
        # physical source — a virtual key must never appear in `needed` or
        # every episode would be dropped for "missing" it.
        needed = list(dict.fromkeys(
            self._resolve_physical_column(schema, k)
            for k in (list(read_state_keys) + list(read_action_keys))
        ))
        # For canonical single-key schemas ("observation.state", "action")
        # also accept the v2.1 legacy variants ("state", "actions").
        alt = {"observation.state": "state", "action": "actions"}

        # Per-key declared dim from schema (for the overflow check below).
        # Also register SOURCE keys/dims: `needed` holds source column names
        # when the schema declares them, so a canonical-only map would skip
        # the overflow check for source-layout schemas.
        declared_dim: dict[str, int] = {}
        for k, d in zip(getattr(schema, "state_keys", ()),
                        getattr(schema, "state_dims", ())):
            declared_dim[k] = int(d)
        for k, d in zip(getattr(schema, "action_keys", ()),
                        getattr(schema, "action_dims", ())):
            declared_dim[k] = int(d)
        for k, d in zip(getattr(schema, "source_state_keys", ()) or (),
                        getattr(schema, "source_state_dims", ()) or ()):
            declared_dim[k] = int(d)
        for k, d in zip(getattr(schema, "source_action_keys", ()) or (),
                        getattr(schema, "source_action_dims", ()) or ()):
            declared_dim[k] = int(d)
        # `needed` holds the PHYSICAL source column for virtual state keys;
        # attach the virtual key's declared dim to that physical name so the
        # overflow check still probes it. setdefault: a column with its own
        # schema-declared dim (e.g. it doubles as an action key) keeps that dim.
        for vk, vsrc in (
            getattr(schema, "virtual_state_sources", None) or {}
        ).items():
            if vk in declared_dim:
                declared_dim.setdefault(str(vsrc), declared_dim[vk])

        from src.utils import env_flags as _env_flags_t
        _allow_truncate = _env_flags_t.get("LABVLA_ALLOW_TRUNCATE") == "1"

        # Expected per-episode frame count from episodes.jsonl, cross-checked
        # against the parquet's actual row count below.
        _ep_len_by_idx: dict[int, int] = {
            int(ep["episode_index"]): int(ep.get("length", -1))
            for ep in self._episodes
        }

        def _parquet_ok(p: Path) -> bool:
            # Shim binding the phase-local context to module-level _parquet_ok_v21.
            return _parquet_ok_v21(
                p,
                needed=needed,
                alt=alt,
                declared_dim=declared_dim,
                ep_len_by_idx=_ep_len_by_idx,
                repo_id=repo_id,
                warn_key_root=str(self.root),
                allow_truncate=_allow_truncate,
            )

        # Disk-persistent scan cache: for N-task adapters sharing a
        # source_root, only the first pays the scan cost (~10s); the rest
        # read the cached result (~10ms).
        cache_key = _scan_cache_key(
            self.meta_root,
            schema_id=getattr(schema, "schema_id", "unknown"),
            chunks_size=self._chunks_size,
            data_root=self.data_root,
        )
        cache_key["required_columns"] = list(needed)
        # Scan semantics include the per-episode row-count check, source-key
        # overflow dims, declared dims and env gates; bump to invalidate caches
        # written by an older scan version.
        cache_key["integrity_checks"] = 3
        cache_key["declared_dims"] = sorted(
            (str(k), int(v)) for k, v in (declared_dim or {}).items()
        )
        cached = _load_scan_cache(self.meta_root, cache_key)

        if cached is not None:
            chunk_ok, existing_ep_indices = cached
        else:
            # Per-file validation is the safe DEFAULT. The alternative
            # (LABVLA_V21_VALIDATE_PER_FILE=0) trusts one sampled parquet
            # per chunk — fast on huge known-clean OXE datasets, but it can
            # keep bad episodes from chunks whose sample happened to be fine.
            from src.utils import env_flags as _env_flags
            _validate_per_file = (
                _env_flags.get("LABVLA_V21_VALIDATE_PER_FILE") != "0"
            )
            chunk_ok: dict[int, bool] = {}
            file_ok: dict[Path, bool] = {}
            for ep in self._episodes:
                chunk = int(ep.get("episode_index", 0)) // self._chunks_size
                if chunk in chunk_ok:
                    continue
                chunk_dir = self.data_root / f"chunk-{chunk:03d}"
                if _validate_per_file:
                    # Validate every parquet; chunk is OK if at least one
                    # passes. Cache per-file results to drop individual bad
                    # episodes below.
                    any_ok = False
                    if chunk_dir.is_dir():
                        for child in chunk_dir.iterdir():
                            if child.suffix != ".parquet":
                                continue
                            ok = _parquet_ok(child)
                            file_ok[child] = ok
                            any_ok = any_ok or ok
                    chunk_ok[chunk] = any_ok
                else:
                    sample_path = None
                    if chunk_dir.is_dir():
                        for child in chunk_dir.iterdir():
                            if child.suffix == ".parquet":
                                sample_path = child
                                break
                    chunk_ok[chunk] = bool(sample_path and _parquet_ok(sample_path))

            # Also filter per-episode by parquet existence ("cleaned"
            # datasets sometimes keep the episodes.jsonl entry but delete
            # the parquet) and, under per-file validation, by whether that
            # specific parquet passed — otherwise file_ok would go unused
            # and a bad file in an OK chunk would slip through.
            import re as _re
            _ep_re = _re.compile(r"episode_(\d+)")
            existing_ep_indices: set[int] = set()
            for _chunk, _ok in chunk_ok.items():
                if not _ok:
                    continue
                _chunk_dir = self.data_root / f"chunk-{_chunk:03d}"
                if not _chunk_dir.is_dir():
                    continue
                for _child in _chunk_dir.iterdir():
                    if _child.suffix != ".parquet":
                        continue
                    if _validate_per_file and not file_ok.get(_child, False):
                        continue
                    _m = _ep_re.match(_child.stem)
                    if _m:
                        existing_ep_indices.add(int(_m.group(1)))

            # Persist to disk for subsequent adapter inits on the same repo.
            _save_scan_cache(
                self.meta_root, cache_key, chunk_ok, existing_ep_indices,
            )

        keep_eps: list[dict] = []
        dropped = 0
        for ep in self._episodes:
            chunk = int(ep.get("episode_index", 0)) // self._chunks_size
            if chunk_ok.get(chunk, False):
                keep_eps.append(ep)
            else:
                dropped += 1

        before_existence = len(keep_eps)
        keep_eps = [
            ep for ep in keep_eps
            if int(ep.get("episode_index", -1)) in existing_ep_indices
        ]
        missing_parquets = before_existence - len(keep_eps)
        if missing_parquets:
            logger.warning(
                "[v21-adapter] %s: dropped %d additional episodes whose "
                "parquet file is missing on disk (e.g. 'episode_%06d.parquet' "
                "absent from data/chunk-*/)",
                repo_id,
                missing_parquets,
                next(
                    (int(ep.get("episode_index", 0)) for ep in self._episodes
                     if int(ep.get("episode_index", -1)) not in existing_ep_indices
                     and int(ep.get("episode_index", 0)) // self._chunks_size in chunk_ok
                     and chunk_ok[int(ep.get("episode_index", 0)) // self._chunks_size]),
                    0,
                ),
            )
            dropped += missing_parquets

        if dropped:
            # Key on (repo_id, tuple(needed)) so a second repo with the
            # same missing columns still emits once.
            from src.utils.logging_utils import warn_once
            warn_once(
                logger,
                ("repo_col_drop", repo_id, tuple(needed)),
                "[v21-adapter] %s: dropped %d/%d episodes lacking "
                "schema-required columns (%s)",
                repo_id, dropped, len(self._episodes), needed,
            )
            self._episodes = keep_eps
            self._ep_starts, self._ep_lens = _ep_starts_lens(self._episodes)

    def _finalize_meta_and_caches(self, stats, schema) -> None:
        """Phase 4: feature/camera key extraction, fps gate, DatasetMeta,
        video/parquet caches, shared counters."""
        info = self._info
        repo_id = self.repo_id

        feats = info.get("features", {}) or {}
        video_keys = [
            k for k, val in feats.items()
            if isinstance(val, dict) and val.get("dtype") == "video"
        ]
        # dtype=image (PNG-in-parquet) cameras: datasets like
        # LabUtopia/Level3_open_old ship images embedded in parquet, not as
        # mp4. The base __getitem__ decodes them inline via self._image_keys.
        # v21 reads full parquet (no projection), so image columns are already
        # included.
        image_keys = [
            k for k, val in feats.items()
            if isinstance(val, dict) and val.get("dtype") == "image"
        ]
        self._image_keys = tuple(image_keys)

        # Frame-index boundaries per episode — consumed by MultiLeRobotDataset
        # to build cross-repo combined indexing.
        _ep_ends = self._ep_starts + self._ep_lens
        # Fail loudly if 'fps' is missing/non-numeric: a silent default of 10
        # produced wrong delta_timestamps on 30fps videos (~3x drift per chunk).
        _raw_fps = info.get("fps")
        if _raw_fps is None:
            raise ValueError(
                f"info.json at {repo_id} missing required 'fps' field. "
                f"Set it explicitly in meta/info.json."
            )
        if not isinstance(_raw_fps, (int, float)) or isinstance(_raw_fps, bool):
            raise TypeError(
                f"info.json fps must be int or float, got {type(_raw_fps).__name__}: {_raw_fps!r}"
            )
        self.meta = DatasetMeta(
            robot_type=info.get("robot_type"),
            stats=stats if stats else None,
            fps=float(_raw_fps),
            total_episodes=len(self._episodes),
            total_frames=int(self._ep_lens.sum()),
            features=feats,
            video_keys=video_keys,
            # camera_keys = full set (mp4 video + PNG-in-parquet image);
            # video_keys alone is empty for image-only datasets.
            camera_keys=list(video_keys) + list(image_keys),
            schema=schema,
            episodes={
                "dataset_from_index": self._ep_starts.astype(int).tolist(),
                "dataset_to_index":   _ep_ends.astype(int).tolist(),
            },
        )
        logger.info(
            "[v21-adapter] %s: %d episodes, %d frames, %d video_cameras, "
            "%d image_cameras, schema=%s",
            repo_id, self.meta.total_episodes, self.meta.total_frames,
            len(video_keys), len(image_keys),
            getattr(schema, "schema_id", None),
        )

        # Warn once per (repo, horizon) if the longest offset clips past most
        # of the shortest episode. Cheap; does not change clip behavior.
        self._validate_delta_timestamps_frame_aligned()
        self._validate_delta_timestamps_vs_episode_lens()

        # Hook into the process-wide PyAV container LRU (shared with v30):
        # consecutive frames in one episode reuse the cached container instead
        # of paying a fresh av.open+seek+close per frame.
        self._video_cache = _get_shared_video_cache()
        self._video_cache_owner_id = id(self)

        # Fork-shared zero-frame fallback counter (must exist before DataLoader
        # workers fork so worker increments reach the main process).
        self._init_zero_frame_shared()

        # Per-INSTANCE parquet LRU. A class-level @lru_cache on the method keyed
        # (self, ep_idx) would make all adapters in a mixture share ONE 8-slot
        # cache (near-zero hit rate at 60+ adapters) and retain strong `self`
        # references. Binding lru_cache to the instance gives each adapter its
        # own 8-slot window. Fork-only DataLoader workers inherit the bound
        # wrapper without pickling.
        self._cached_ep_parquet = lru_cache(maxsize=8)(self._load_ep_parquet_uncached)

    def _close_video_containers(self) -> None:
        """Best-effort flush of cached PyAV containers (used at adapter teardown).

        Drops only entries owned by this adapter from the shared cache;
        other adapters in the same worker keep their entries.
        """
        try:
            self._video_cache.drop_owner(self._video_cache_owner_id)
        except Exception:
            pass

    def __del__(self):
        try:
            self._close_video_containers()
        except Exception:
            pass

    # ---- format-specific I/O hooks (overrides LeRobotAdapterBase) ----

    # maxsize=8 (bound per-instance in __init__): each adapter holds an
    # independent cache, so larger values multiply parquet residency across
    # adapters×ranks×workers. 8 covers the typical consecutive-episode window.
    def _load_ep_parquet_uncached(self, ep_idx: int) -> pd.DataFrame:
        chunk = ep_idx // self._chunks_size
        p = self.data_root / f"chunk-{chunk:03d}" / f"episode_{ep_idx:06d}.parquet"
        return read_parquet_with_storage_retry(p)

    def _load_ep_parquet(self, ep_idx: int) -> pd.DataFrame:
        return self._cached_ep_parquet(ep_idx).copy()

    def _read_video_frame(self, ep_idx: int, vkey: str, frame: int) -> torch.Tensor:
        """Read a single mp4 frame using keyframe seek + short forward decode.

        Decoding from frame 0 is O(frame_index) and saturates CPU/IO at scale.
        Instead, seek to the nearest keyframe <= target and forward-decode a
        handful of frames: O(keyframe_interval), ~3-6x faster, pixel-identical.
        The opened container is kept in the process-wide cache so consecutive
        frames in one chunk reuse it (50x fewer av.open calls).
        """
        chunk = ep_idx // self._chunks_size
        p = self.video_root / f"chunk-{chunk:03d}" / vkey / f"episode_{ep_idx:06d}.mp4"
        if not storage_path_exists(p):
            # Missing video is common in multi-robot merges (differing camera
            # layouts); return a black frame for the model to treat as padding.
            return self._zero_frame(reason="missing_file")
        import av

        full_key = (self._video_cache_owner_id, vkey, ep_idx)
        video_cache = self._video_cache

        def _drop_cached_container(_exc, _attempt) -> None:
            cached_container = video_cache.pop(full_key)
            if cached_container is not None:
                try:
                    cached_container[0].close()
                except Exception:
                    pass

        def _decode_once():
            cached = video_cache.get(full_key)
            if cached is None:
                container = av.open(str(p))
                stream = container.streams.video[0]
                # thread_type "AUTO" hits a pyav/libav internal futex deadlock
                # at 128-process scale (each with its own thread pool). Episode
                # mp4s are short, so single-threaded decode is fine and safe.
                stream.thread_type = "NONE"
                stream.thread_count = 1
                # LRU eviction (closes evicted container) handled inside .put().
                video_cache.put(full_key, (container, stream))
            else:
                container, stream = cached

            # Exact Fraction frame arithmetic (mirrors the v30 adapter). A float
            # `round(pts*tb*fps)` mapping can skip the target frame on
            # non-integer average_rate / VFR streams and silently fall back to a
            # zero frame (reason=index_overshoot).
            fps_frac = Fraction(stream.average_rate) if stream.average_rate else Fraction(30)
            fps = float(fps_frac)
            tb = stream.time_base  # e.g. 1/15360 for 30 fps

            # Seek to the nearest keyframe <= target; on failure (malformed
            # mp4) fall through to linear decode.
            target_pts = int(frame / fps / tb) if tb else 0
            try:
                container.seek(target_pts, stream=stream)
            except av.AVError as e:
                # Log (once per path) so genuine corruption surfaces instead of
                # silently degrading to O(frame) decode.
                logger.warning(
                    "[v21-adapter] seek failed for ep=%s key=%s frame=%s pts=%s: %s; "
                    "falling back to linear decode",
                    ep_idx, vkey, frame, target_pts, e,
                )

            # Forward-decode from the keyframe to the exact frame.
            for f in container.decode(stream):
                if f.pts is None:
                    continue
                cur = round(f.pts * tb * fps_frac)  # exact Fraction math
                if cur < frame:
                    continue  # still catching up from keyframe
                if cur > frame:
                    break  # overshoot (shouldn't happen with backward seek)
                img = f.to_ndarray(format="rgb24")
                t = torch.from_numpy(img).permute(2, 0, 1).contiguous().float() / 255.0
                if self.image_transforms is not None:
                    apply_seeded = getattr(self.image_transforms, "apply_with_seed", None)
                    if callable(apply_seeded):
                        t = apply_seeded(t, seed_parts=(self.repo_id, ep_idx, vkey, frame))
                    else:
                        t = self.image_transforms(t)
                return t
            return None

        try:
            decoded = run_with_storage_retry(
                _decode_once,
                path=p,
                description="v21 video decode",
                on_retry=_drop_cached_container,
            )
            if decoded is not None:
                return decoded
        except Exception as e:
            # Drop the cached container on error; it may be in a bad state.
            cached = video_cache.pop(full_key)
            if cached is not None:
                try:
                    cached[0].close()
                except Exception:
                    pass
            logger.warning("[v21-adapter] video decode failed for %s (%s); "
                           "returning zero frame.", p, e)
            return self._zero_frame(reason="decode_error")
        # Frame past end or seek mis-aligned: drop the cached container (its
        # cursor is past `frame`, so reopen next call) and pad with zero.
        cached = video_cache.pop(full_key)
        if cached is not None:
            try:
                cached[0].close()
            except Exception:
                pass
        return self._zero_frame(reason="index_overshoot")
