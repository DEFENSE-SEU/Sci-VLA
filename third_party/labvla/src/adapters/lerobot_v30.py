"""LeRobot v3.0 dataset adapter.

v3.0 layout (vs v2.1):
  <root>/meta/info.json        (codebase_version="v3.0")
  <root>/meta/episodes/chunk-XXX/file-NNN.parquet   (per-episode indexing + stats)
  <root>/meta/tasks.parquet    (task_index → task string)
  <root>/meta/stats.json       (global dataset stats — same format as v2.1)
  <root>/data/chunk-XXX/file-NNN.parquet   (shard: many episodes per file)
  <root>/videos/<cam>/chunk-XXX/file-NNN.mp4   (shard: many episodes per file)

Key difference: v3 packs many episodes into each shard. Per-episode row
range is in the episodes meta parquet as `dataset_from_index / dataset_to_index`
— the adapter opens the shard, slices to those rows, and hands a standard
per-episode `pd.DataFrame` to the shared transformation logic.

Shares `LeRobotAdapterBase`'s per-frame transformation pipeline (padding,
task resolution, delta-timestamp expansion) via inheritance; only the
three format-specific methods are defined locally:
  1. __init__ — reads episodes.parquet + tasks.parquet
  2. _load_ep_parquet — slices a shard
  3. _read_video_frame — opens a packed mp4 with from_timestamp offset
"""
from __future__ import annotations

import json
import logging
from collections import OrderedDict
from fractions import Fraction
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from .base import DatasetMeta
from .lerobot_base import (
    LeRobotAdapterBase,
    _ep_starts_lens,
    _get_shared_video_cache,
)
# Task-text heuristics are unified in schema/task_text.
from src.schema.task_text import normalize_tasks as _normalize_tasks_unified
from src.utils.storage_retry import (
    read_parquet_with_storage_retry,
    run_with_storage_retry,
    storage_path_exists,
)

logger = logging.getLogger(__name__)


def _is_missing_scalar(value) -> bool:
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _read_episodes_parquet_v3(meta_root: Path) -> list[dict]:
    """Concat meta/episodes/chunk-*/file-*.parquet → list of per-episode dicts.

    Kept columns (everything we need for slicing + video seek):
      episode_index, length, tasks,
      data/chunk_index, data/file_index, dataset_from_index, dataset_to_index,
      videos/<cam>/chunk_index, videos/<cam>/file_index,
      videos/<cam>/from_timestamp, videos/<cam>/to_timestamp
    """
    ep_root = meta_root / "episodes"
    if not ep_root.is_dir():
        raise FileNotFoundError(f"v3 meta/episodes dir missing at {ep_root}")
    paths = sorted(ep_root.rglob("*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No episode parquets under {ep_root}")

    # Go through pandas for heterogeneous-schema-safe concat: merged
    # oxe-auge_clean has differing camera-key-sets and two `tasks` column types
    # (list<string> vs list<list<int>>). pa.concat_tables can't union these;
    # pd.concat(sort=False) can.
    dfs_with_paths = [
        (
            p,
            run_with_storage_retry(
                lambda _p=p: pq.read_table(_p).to_pandas(),
                path=p,
                description="read v3 episodes parquet",
            ),
        )
        for p in paths
    ]
    dfs = [df for _, df in dfs_with_paths]

    def _normalize_tasks(val):
        # Normalize the `tasks` column so it's always VLM-tokenizable.
        # list<string> / scalar pass through unchanged. Tokenized
        # (list<list<int>>) tasks fail-loud by default (coercing to repr would
        # train the VLM on digit strings); LABVLA_ALLOW_TOKENIZED_TASK_COERCION=1
        # restores the legacy coercion for inspection only.
        #
        # Exception: OXE `language_table_*` stores strings as NUL-padded ASCII
        # byte arrays (TFDS idiom), not tokenizer output — the unified
        # heuristic (schema/task_text) decodes those BEFORE the raise. The env
        # flag is read per call; the shared code takes the policy as an
        # explicit parameter.
        from src.utils import env_flags as _env_flags
        return _normalize_tasks_unified(
            val,
            allow_tokenized_coercion=(
                _env_flags.get("LABVLA_ALLOW_TOKENIZED_TASK_COERCION") == "1"
            ),
            undecodable_msg=(
                "v30 adapter: encountered a tokenized (list<int>) "
                "task element. Coercing it to str() would train the "
                "VLM on digit strings rather than language. Either "
                "decode the tokens back to text in the upstream "
                "dataset, exclude tokenized-task shards from "
                "training, or set "
                "LABVLA_ALLOW_TOKENIZED_TASK_COERCION=1 for a "
                "best-effort inspection-only run."
            ),
        )

    # Warn loud when heterogeneous `tasks` column types are detected, naming the
    # shards with tokenized (list<list<int>>) tasks so the operator can
    # spot-check before training.
    tokenized_paths: list[Path] = []
    for p, df in dfs_with_paths:
        if "tasks" not in df.columns:
            continue
        sample = next((v for v in df["tasks"].values if v is not None), None)
        if isinstance(sample, (list, np.ndarray)) and len(sample) > 0:
            first = sample[0]
            if isinstance(first, (list, np.ndarray)):
                tokenized_paths.append(p)
    if tokenized_paths:
        from src.utils.logging_utils import warn_once
        # Heads-up that _normalize_tasks will raise at first read unless
        # LABVLA_ALLOW_TOKENIZED_TASK_COERCION=1.
        warn_once(
            logger,
            ("v30_tasks_heterogeneous", tuple(str(p) for p in tokenized_paths)),
            "[v30-adapter] Heterogeneous tasks column type detected — %d "
            "shard(s) contain tokenized (list<list<int>>) tasks while others "
            "contain list<string>. These will FAIL LOUD on first read; set "
            "LABVLA_ALLOW_TOKENIZED_TASK_COERCION=1 to fall back to legacy "
            "str() coercion (inspection-only — VLM should not be trained on "
            "the resulting digit strings). Affected paths: %s",
            len(tokenized_paths),
            ", ".join(str(p) for p in tokenized_paths[:5])
            + (" ..." if len(tokenized_paths) > 5 else ""),
        )

    for df in dfs:
        if "tasks" in df.columns:
            df["tasks"] = df["tasks"].map(_normalize_tasks)

    df = pd.concat(dfs, axis=0, ignore_index=True, sort=False)
    table = pa.Table.from_pandas(df, preserve_index=False)

    # Keep `videos/<cam>/…` columns verbatim for _read_video_frame; drop
    # `stats/<feature>/*` (per-episode stats the adapter doesn't use).
    keep_prefix = (
        "episode_index", "length", "tasks",
        "data/", "dataset_from_index", "dataset_to_index",
        "videos/",
    )
    keep_cols = [c for c in table.schema.names if any(
        c == p or c.startswith(p) for p in keep_prefix
    )]
    sub = table.select(keep_cols).to_pandas()
    out = [row.to_dict() for _, row in sub.iterrows()]
    # tasks is list<str> (np.ndarray from pandas); take the first (per-ep task).
    for e in out:
        tk = e.get("tasks")
        if isinstance(tk, (list, np.ndarray)) and len(tk) > 0:
            e["task"] = str(tk[0])
        else:
            e["task"] = ""
    out.sort(key=lambda e: int(e["episode_index"]))
    return out


def _read_tasks_parquet_v3(tasks_path: Path) -> dict[int, str]:
    """Parse meta/tasks.parquet → {task_index: task_string}.

    Observed conversion variants:
      (a) columns = {"task_index": int64, "task": string}
      (b) task string as the DataFrame index (`__index_level_0__` in pyarrow),
          `task_index: int64` as a column.
    Handle both.
    """
    if not storage_path_exists(tasks_path):
        return {}
    table = run_with_storage_retry(
        lambda: pq.read_table(tasks_path),
        path=tasks_path,
        description="read v3 tasks parquet",
    )
    df = table.to_pandas()
    if "task" in df.columns and "task_index" in df.columns:
        return {int(r["task_index"]): str(r["task"]) for _, r in df.iterrows()}
    # Variant (b): task string is df.index, task_index is a column.
    if "task_index" in df.columns:
        return {int(r["task_index"]): str(idx) for idx, r in df.iterrows()}
    logger.warning("[v30-adapter] unexpected tasks.parquet schema: cols=%s", list(df.columns))
    return {}




def _shard_ok_v30(
    ci: int,
    fi: int,
    *,
    data_root: Path,
    needed: list,
    alt: dict,
    shard_span: dict,
    declared_dim: dict,
    check_overwide: bool,
    repo_id: str,
) -> bool:
    """Schema-satisfies check for one v3 data shard.

    Checks: required columns (with v2.1 alt fallback), row-span sufficiency,
    and an overwide first-row probe.
    """
    p = data_root / f"chunk-{ci:03d}" / f"file-{fi:03d}.parquet"
    if not p.exists():
        return False
    try:
        pf = pq.ParquetFile(str(p))
        cols = set(pf.schema_arrow.names)
    except Exception:
        return False
    resolved: dict[str, str] = {}
    for k in needed:
        if k in cols:
            resolved[k] = k
            continue
        a = alt.get(k, None)
        if a in cols:
            resolved[k] = a
            continue
        return False
    # The shard must hold enough rows for every episode slice it
    # serves; otherwise _load_ep_parquet's iloc would return short
    # frames → IndexError mid-training.
    span = shard_span.get((ci, fi))
    if span is not None:
        try:
            n_rows = int(pf.metadata.num_rows)
        except Exception:
            return False
        need_rows = span[1] - span[0]
        if n_rows < need_rows:
            logger.warning(
                "[v30-adapter] %s: shard %s has %d rows but "
                "meta/episodes requires %d (span [%d, %d)) — "
                "dropping shard (meta/data inconsistency).",
                repo_id, p.name, n_rows, need_rows, span[0], span[1],
            )
            return False
    # Overwide-column probe on the first row.
    if check_overwide and declared_dim:
        try:
            probe_cols = [
                resolved[k] for k in declared_dim if k in resolved
            ]
            if probe_cols and pf.metadata.num_row_groups > 0:
                head = pf.read_row_group(0, columns=probe_cols)
                if len(head) > 0:
                    for k in declared_dim:
                        if k not in resolved:
                            continue
                        raw0 = head.column(resolved[k])[0].as_py()
                        if (isinstance(raw0, (list, tuple))
                                and len(raw0) > declared_dim[k]):
                            logger.warning(
                                "[v30-adapter] %s: shard %s column "
                                "%r is %d-wide > declared %d — "
                                "dropping shard (set "
                                "LABVLA_ALLOW_TRUNCATE=1 to keep "
                                "and truncate).",
                                repo_id, p.name, resolved[k],
                                len(raw0), declared_dim[k],
                            )
                            return False
        except Exception:
            return False
    return True


class LeRobotV30Adapter(LeRobotAdapterBase):
    """Adapter for LeRobot v3.0 shard-packed datasets.

    Inherits the shared per-frame transformation pipeline
    (``__getitem__``, row-padding, task resolution, delta-timestamp
    expansion, CRIT-05 ``_is_pad`` postcondition, zero-frame fallback)
    from ``LeRobotAdapterBase`` and overrides only the format-specific
    I/O hooks: ``_load_ep_parquet`` (slices a shard) and
    ``_read_video_frame`` (reads a packed mp4 with ``from_timestamp``
    offset).
    """

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
        """See LeRobotV21Adapter for episode_filter semantics.

        __init__ is a thin orchestrator over the four phase methods below.
        """
        self._resolve_root(
            repo_id, root, delta_timestamps, image_transforms, video_backend
        )
        stats, schema = self._load_meta_and_stats(external_stats, override_schema)

        # Filter episodes whose data shard lacks schema-required columns. In v3
        # a shard shares one schema, so check each unique (chunk, file) once and
        # drop episodes pointing at a failed shard.
        if schema is not None:
            self._filter_episodes_by_schema(schema)

        # Task-uniform support: restrict to a user-supplied episode subset.
        if episode_filter is not None:
            allowed = {int(i) for i in episode_filter}
            before = len(self._episodes)
            self._episodes = [
                ep for ep in self._episodes
                if int(ep.get("episode_index", -1)) in allowed
            ]
            self._ep_starts, self._ep_lens = _ep_starts_lens(self._episodes)
            logger.info(
                "[v30-adapter] %s: episode_filter kept %d/%d episodes",
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
        """Phase 1: root resolution, info.json version gate, episodes meta +
        length invariant."""
        self.repo_id = repo_id
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
        if not v.startswith("v3"):
            raise ValueError(
                f"LeRobotV30Adapter: {self.root} is codebase_version={v!r}, "
                f"expected v3.x."
            )
        self._info = info
        # Merged products re-index the global tasks.parquet; per-frame
        # task_index values still carry per-source indices and MUST NOT resolve
        # language through the merged table.
        self._task_index_unreliable = bool(info.get("task_index_remapped", False))
        self._chunks_size = int(info.get("chunks_size", 1000))
        self._episodes = _read_episodes_parquet_v3(self.meta_root)

        # Enforce `length == dataset_to_index - dataset_from_index`
        # for every episode. This is pure meta arithmetic (zero I/O) and is the
        # invariant the whole indexing path rests on: `_flat_to_ep` samples
        # frame_in_ep ∈ [0, length) while `_load_ep_parquet` slices [from, to).
        # length > (to-from) → IndexError mid-training; length < (to-from) →
        # tail frames silently never sampled. Fail loud at init instead.
        _bad_len = [
            (int(e["episode_index"]), int(e.get("length", -1)),
             int(e["dataset_to_index"]) - int(e["dataset_from_index"]))
            for e in self._episodes
            if int(e.get("length", -1))
            != int(e["dataset_to_index"]) - int(e["dataset_from_index"])
        ]
        if _bad_len:
            _sample = ", ".join(
                f"ep{ei}: length={ln} vs to-from={rng}" for ei, ln, rng in _bad_len[:5]
            )
            raise ValueError(
                f"[v30-adapter] {repo_id}: {len(_bad_len)} episode(s) violate "
                f"length == dataset_to_index - dataset_from_index ({_sample}"
                f"{' ...' if len(_bad_len) > 5 else ''}). meta/episodes is "
                f"internally inconsistent — fix the dataset before training "
                f"(mismatch means either mid-training IndexError or silently "
                f"unsampled tail frames)."
            )

        # Preserve unfiltered episodes for shard offsets: the offset is the
        # minimum dataset_from_index of ANY episode in the shard, since parquet
        # rows are numbered 0..N across all packed episodes (not just filtered).
        self._unfiltered_episodes = list(self._episodes)
        self._ep_starts, self._ep_lens = _ep_starts_lens(self._episodes)

    def _load_meta_and_stats(self, external_stats: dict | None, override_schema):
        """Phase 2: tasks index, stats (+ external override), schema
        discovery, next-frame stats patch. Returns ``(stats, schema)``."""
        info = self._info

        # tasks.parquet → {int: str} (v21 uses tasks.jsonl; same lookup path).
        self._tasks_by_idx = _read_tasks_parquet_v3(self.meta_root / "tasks.parquet")

        # stats.json — same format across v2.1 and v3.
        stats_path = self.meta_root / "stats.json"
        if stats_path.exists():
            from src.dataset.utils import cast_stats_to_numpy
            with open(stats_path) as _f:
                stats: dict = cast_stats_to_numpy(json.load(_f))
        else:
            stats = {}
        if external_stats:
            stats = {**stats, **external_stats}

        # Schema discovery (same chain as v21).
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
            logger.warning("[v30-adapter] schema discovery failed: %s", e)
            schema = override_schema

        stats = self.patch_stats_for_next_frame_actions(stats, schema)
        return stats, schema

    def _filter_episodes_by_schema(self, schema) -> None:
        """Phase 3: drop episodes whose shard lacks schema-required columns,
        rows, or width envelope (scan-cached; one check per unique shard)."""
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
        # every shard would be dropped for "missing" it.
        needed = list(dict.fromkeys(
            self._resolve_physical_column(schema, k)
            for k in (list(read_state_keys) + list(read_action_keys))
        ))
        alt = {"observation.state": "state", "action": "actions"}

        # Per-shard expected row span (min from / max to over ALL episodes in
        # the shard) so _shard_ok can verify the parquet really holds enough
        # rows for every episode slice.
        _shard_span: dict[tuple[int, int], tuple[int, int]] = {}
        for _e in self._episodes:
            _key = (int(_e["data/chunk_index"]), int(_e["data/file_index"]))
            _a = int(_e["dataset_from_index"])
            _b = int(_e["dataset_to_index"])
            _cur = _shard_span.get(_key)
            _shard_span[_key] = (
                (_a, _b) if _cur is None
                else (min(_cur[0], _a), max(_cur[1], _b))
            )

        # Per-read-key declared dims (SOURCE space when declared, mirroring
        # what the projection actually reads) for the overwide check, matching
        # v21's _parquet_ok. Without it an overwide shard crashes later in
        # _pad_row mid-training instead of being dropped at init. Honors
        # LABVLA_ALLOW_TRUNCATE.
        _rs_dims = (
            tuple(getattr(schema, "source_state_dims", ()) or ())
            or tuple(getattr(schema, "state_dims", ()) or ())
        )
        _ra_dims = (
            tuple(getattr(schema, "source_action_dims", ()) or ())
            or tuple(getattr(schema, "action_dims", ()) or ())
        )
        _declared_dim: dict[str, int] = {}
        for _k, _d in zip(read_state_keys, _rs_dims):
            _declared_dim[_k] = int(_d)
        for _k, _d in zip(read_action_keys, _ra_dims):
            _declared_dim[_k] = int(_d)
        # The overwide probe below resolves `needed` (physical names) — attach
        # each virtual state key's declared dim to its physical source column
        # so the probe still covers it. setdefault: a column with its own
        # schema-declared dim keeps that dim.
        for _vk, _vsrc in (
            getattr(schema, "virtual_state_sources", None) or {}
        ).items():
            if _vk in _declared_dim:
                _declared_dim.setdefault(str(_vsrc), _declared_dim[_vk])
        from src.utils import env_flags as _env_flags
        _check_overwide = _env_flags.get("LABVLA_ALLOW_TRUNCATE") != "1"

        def _shard_ok(ci: int, fi: int) -> bool:
            # Shim binding the phase-local context to module-level _shard_ok_v30.
            return _shard_ok_v30(
                ci, fi,
                data_root=self.data_root,
                needed=needed,
                alt=alt,
                shard_span=_shard_span,
                declared_dim=_declared_dim,
                check_overwide=_check_overwide,
                repo_id=repo_id,
            )

        # Disk-persistent scan cache (v3 variant — keyed by ok_shards). Shared
        # infrastructure in adapters/_scan_cache.py; v3 has no chunks_size, so
        # -1 sentinel.
        from ._scan_cache import (
            _scan_cache_key, _load_scan_cache, _save_scan_cache,
        )
        cache_key = _scan_cache_key(
            self.meta_root,
            schema_id=getattr(schema, "schema_id", "unknown"),
            chunks_size=-1,   # v3 sentinel
            data_root=self.data_root,
        )
        cache_key["required_columns"] = list(needed)
        # Scan semantics include the row-span check, overwide check, declared
        # dims and env gates; bump to invalidate caches written by an older
        # scan version.
        cache_key["integrity_checks"] = 3
        cache_key["declared_dims"] = sorted(
            (str(k), int(v)) for k, v in (_declared_dim or {}).items()
        )
        cached = _load_scan_cache(self.meta_root, cache_key)

        unique_shards = {
            (int(e["data/chunk_index"]), int(e["data/file_index"]))
            for e in self._episodes
        }

        # The cache packs (ci, fi) as ``ci*_PACK_BASE + fi`` — unambiguous
        # only while fi < _PACK_BASE. Assert loudly so >1M-shards-per-chunk
        # datasets fail at launch instead of mis-filtering silently.
        _PACK_BASE = 1_000_000
        max_fi = max((fi for (_, fi) in unique_shards), default=0)
        assert max_fi < _PACK_BASE, (
            f"[v30-adapter] shard fi={max_fi} ≥ cache pack base {_PACK_BASE}; "
            f"cache collision risk — update encoding before reusing."
        )

        if cached is not None:
            # Reuse the v21 cache's `existing_episodes` slot as our
            # ok_shards set (packed ints).
            _packed_ok = cached[1]
            ok_shards = {(p // _PACK_BASE, p % _PACK_BASE) for p in _packed_ok}
            # Intersect with the current episode set, which can drift
            # without a schema change (which would invalidate the cache).
            ok_shards = ok_shards & unique_shards
        else:
            ok_shards = {s for s in unique_shards if _shard_ok(*s)}
            _packed_ok = {ci * _PACK_BASE + fi for (ci, fi) in ok_shards}
            _save_scan_cache(
                self.meta_root, cache_key, chunk_ok={}, existing_episodes=_packed_ok,
            )

        if ok_shards != unique_shards:
            keep = [e for e in self._episodes
                    if (int(e["data/chunk_index"]), int(e["data/file_index"])) in ok_shards]
            dropped = len(self._episodes) - len(keep)
            logger.warning(
                "[v30-adapter] %s: dropped %d/%d episodes in %d shards "
                "lacking schema-required columns (%s)",
                repo_id, dropped, len(self._episodes),
                len(unique_shards) - len(ok_shards), needed,
            )
            self._episodes = keep
            self._ep_starts, self._ep_lens = _ep_starts_lens(self._episodes)

    def _finalize_meta_and_caches(self, stats, schema) -> None:
        """Phase 4: feature/camera keys, fps gate, DatasetMeta, shard
        offsets + column projection, video/shard caches, shared counters."""
        info = self._info
        repo_id = self.repo_id

        feats = info.get("features", {}) or {}
        video_keys = [
            k for k, val in feats.items()
            if isinstance(val, dict) and val.get("dtype") == "video"
        ]
        # dtype=image columns are PNG-in-parquet (HF Image feature), NOT mp4.
        # Datasets like LabUtopia/Level3_open ship images this way with no
        # videos/ dir; the base __getitem__ decodes them inline via
        # self._image_keys (else the video loop trains on all-black frames).
        image_keys = [
            k for k, val in feats.items()
            if isinstance(val, dict) and val.get("dtype") == "image"
        ]
        self._image_keys = tuple(image_keys)

        _ep_ends = self._ep_starts + self._ep_lens
        if "fps" not in info:
            raise ValueError(
                f"{self.meta_root}/info.json is missing 'fps'. v3.0 adapter "
                "requires an explicit fps to build delta_timestamps; a silent "
                "default of 10 caused action-vs-video drift when the real fps "
                "differs (e.g. 30)."
            )
        self.meta = DatasetMeta(
            robot_type=info.get("robot_type"),
            stats=stats if stats else None,
            fps=float(info["fps"]),
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

        # episode_index → meta row lookup, used by _load_ep_parquet /
        # _read_video_frame to recover shard + timestamp info O(1).
        self._ep_by_idx = {int(e["episode_index"]): e for e in self._episodes}

        # Per-shard global→local offset. `dataset_from_index` is cumulative
        # over the WHOLE dataset while each shard's rows are locally numbered;
        # offset = min(dataset_from_index) over ALL (unfiltered) episodes in
        # the shard — see _load_ep_parquet.
        self._shard_offsets: dict[tuple[int, int], int] = {}
        for _e in self._unfiltered_episodes:
            _key = (int(_e["data/chunk_index"]), int(_e["data/file_index"]))
            _a = int(_e["dataset_from_index"])
            _cur = self._shard_offsets.get(_key)
            if _cur is None or _a < _cur:
                self._shard_offsets[_key] = _a

        # Minimal column set for `_load_shard`: oxe-auge shards have ~49 columns
        # but we only need schema state/action + task-resolution fields.
        # Projecting cuts I/O ~10x and keeps LRU entries compact.
        needed: set[str] = set()
        if schema is not None:
            read_state_keys = (
                tuple(getattr(schema, "source_state_keys", ()) or ())
                or tuple(getattr(schema, "state_keys", ()) or ())
            )
            read_action_keys = (
                tuple(getattr(schema, "source_action_keys", ()) or ())
                or tuple(getattr(schema, "action_keys", ()) or ())
            )
            # Project the PHYSICAL source column for virtual state keys (a
            # virtual name never exists in the shard; projecting it would
            # silently drop the real source from the read).
            for k in read_state_keys:
                needed.add(self._resolve_physical_column(schema, k))
            for k in read_action_keys:
                needed.add(self._resolve_physical_column(schema, k))
            # Canonical siblings so the pluralization normalization can reach
            # them — ONLY when the schema actually reads a canonical-named
            # column. For fully non-canonical read sets (e.g. oxe-auge reads
            # only observation.joints), an unconditional update would drag the
            # heterogeneous raw observation.state through every sample dict
            # until ComposeFieldsTransform overwrote it — wasted shard I/O and
            # a trap for any future transform that iterates all dict keys.
            _canonical_siblings = {"observation.state", "state", "action", "actions"}
            if needed & _canonical_siblings:
                needed.update(_canonical_siblings)
        # Task resolution needs task_index (→ tasks.parquet) or
        # natural_language_instruction; include both.
        needed.update({"task_index", "natural_language_instruction", "task"})
        if schema is not None:
            for spec in getattr(schema, "annotation_losses", ()) or ():
                field = getattr(spec, "field", None)
                if field:
                    needed.add(str(field))
            # "annotation.unified" is SYNTHESIZED at transform time by
            # BuildUnifiedAnnotationTransformFn from the raw per-field
            # annotation columns — projecting only the synthesized name would
            # drop the raw columns from the shard read, and the builder would
            # silently emit an empty string (zero annotation CE). Project the
            # builder's source fields too; absent columns are filtered against
            # the shard's actual schema below, so heterogeneous shards stay fine.
            if any(
                getattr(spec, "field", None) == "annotation.unified"
                for spec in getattr(schema, "annotation_losses", ()) or ()
            ):
                # The source-field list is schema-layer data (shared by the
                # transform that synthesizes annotation.unified and this
                # projection) — import it from schema, not from transforms.
                from src.schema.annotation_loss import (
                    UNIFIED_ANNOTATION_FIELDS as _UNIFIED_RAW_FIELDS,
                )
                needed.update(str(f) for f in _UNIFIED_RAW_FIELDS)
        # Minimal metadata for any downstream transform that inspects index.
        needed.update({"timestamp", "frame_index", "episode_index", "index"})
        # Project dtype=image PNG columns too, else _load_shard drops them and
        # __getitem__ falls through to all-black _zero_frame tensors.
        needed.update(image_keys)
        self._shard_load_columns: tuple[str, ...] | None = (
            tuple(needed) if needed else None
        )

        logger.info(
            "[v30-adapter] %s: %d episodes, %d frames, %d video_cameras, "
            "%d image_cameras, schema=%s, shard_load_columns=%d",
            repo_id, self.meta.total_episodes, self.meta.total_frames,
            len(video_keys), len(image_keys),
            getattr(schema, "schema_id", None),
            len(self._shard_load_columns) if self._shard_load_columns else -1,
        )

        # Warn once per (repo, horizon) if the longest offset clips past most
        # of the shortest episode. See base class.
        self._validate_delta_timestamps_frame_aligned()  # M15
        self._validate_delta_timestamps_vs_episode_lens()

        # Process-wide PyAV container LRU (shared with v21): frames within a
        # sample share a packed mp4, so caching avoids a fresh av.open+seek per
        # frame. Keys namespaced by adapter id to avoid (vkey, ci, fi) collisions.
        self._video_cache = _get_shared_video_cache()
        self._video_cache_owner_id = id(self)

        # Fork-shared zero-frame fallback counter (must exist before DataLoader
        # workers fork so worker increments reach the main process).
        self._init_zero_frame_shared()

        # Per-INSTANCE shard LRU (see lerobot_v21 for rationale — a class-level
        # @lru_cache would share 64 slots across every adapter in a mixture and
        # pin `self` references in a class-level cache).
        self._load_shard = lru_cache(maxsize=64)(self._load_shard_uncached)

    def _close_video_containers(self) -> None:
        """Best-effort flush of cached PyAV containers (used at adapter teardown).

        Now drops only entries owned by this adapter from the shared cache;
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

    # ---- shard-sliced parquet loader (overrides v21) ----

    # Bound per-instance in __init__ via lru_cache(maxsize=64).
    def _load_shard_uncached(self, ci: int, fi: int) -> pd.DataFrame:
        """Whole-shard DataFrame, LRU-cached at 64 shards (a BS=64 batch spans
        many short oxe-auge episodes).

        Projects onto `self._shard_load_columns` (schema state/action + task
        cols): a full oxe-auge shard is ~50MB / 49 cols; projection drops it to
        ~5-10MB (~10x I/O reduction).
        """
        p = self.data_root / f"chunk-{ci:03d}" / f"file-{fi:03d}.parquet"
        cols = self._shard_load_columns
        if cols is None:
            return read_parquet_with_storage_retry(p)
        # Sub-repos differ in columns (e.g. natural_language_instruction only
        # in some OXE sources); read only the subset on disk to avoid ArrowInvalid.
        try:
            pf = run_with_storage_retry(
                lambda: pq.ParquetFile(str(p)),
                path=p,
                description="open parquet metadata",
            )
            avail = set(pf.schema_arrow.names)
            proj = [c for c in cols if c in avail]
            return run_with_storage_retry(
                lambda: pf.read(columns=proj).to_pandas(),
                path=p,
                description="read parquet projection",
            )
        except Exception:
            # Column-metadata read failed: full-column read (slow but functional).
            return read_parquet_with_storage_retry(p)

    def _load_ep_parquet(self, ep_idx: int) -> pd.DataFrame:
        ep = self._ep_by_idx[ep_idx]
        ci = int(ep["data/chunk_index"])
        fi = int(ep["data/file_index"])
        a = int(ep["dataset_from_index"])
        b = int(ep["dataset_to_index"])
        # Global cumulative frame indices → shard-local rows: subtract the
        # smallest dataset_from_index of any episode in this shard.
        offset = self._shard_offsets.get((ci, fi), 0)
        return (
            self._load_shard(ci, fi)
            .iloc[a - offset:b - offset]
            .copy()
            .reset_index(drop=True)
        )

    # ---- packed mp4 video loader (overrides v21) ----

    def _read_video_frame(self, ep_idx: int, vkey: str, frame: int) -> torch.Tensor:
        """Read frame N of episode `ep_idx` from a packed mp4 shard.

        v3 packs many episodes into a single mp4. The episode's sub-window
        is given by `videos/<cam>/from_timestamp` and `to_timestamp` in
        the episodes meta. Target pts = from_ts + frame/fps.
        """
        ep = self._ep_by_idx.get(ep_idx)
        if ep is None:
            return self._zero_frame(reason="missing_episode")
        vci_key = f"videos/{vkey}/chunk_index"
        vfi_key = f"videos/{vkey}/file_index"
        vfrom_key = f"videos/{vkey}/from_timestamp"
        if vci_key not in ep or vfi_key not in ep or vfrom_key not in ep:
            # Schema asked for a camera v3 info doesn't declare.
            return self._zero_frame(reason="schema_camera_missing")
        if any(_is_missing_scalar(ep.get(k)) for k in (vci_key, vfi_key, vfrom_key)):
            return self._zero_frame(reason="video_metadata_missing")
        ci = int(ep[vci_key])
        fi = int(ep[vfi_key])
        from_ts = float(ep[vfrom_key])
        p = self.video_root / vkey / f"chunk-{ci:03d}" / f"file-{fi:03d}.mp4"
        if not storage_path_exists(p):
            return self._zero_frame(reason="missing_file")

        import av
        full_key = (self._video_cache_owner_id, vkey, ci, fi)
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
                # "AUTO" hits a libav internal futex deadlock at 128-process
                # scale; "NONE" + thread_count=1 is the only safe production default.
                stream.thread_type = "NONE"
                stream.thread_count = 1
                # LRU eviction (closes evicted container) handled inside .put().
                video_cache.put(full_key, (container, stream))
            else:
                container, stream = cached

            fps_frac = Fraction(stream.average_rate) if stream.average_rate else Fraction(30)
            fps = float(fps_frac)
            tb = stream.time_base

            # Target = episode window start + local frame offset.
            target_sec = from_ts + frame / fps
            target_pts = int(target_sec / tb) if tb else 0
            try:
                container.seek(target_pts, stream=stream)
            except av.AVError:
                pass

            _start_pts = round(Fraction(from_ts) / tb) if tb else 0
            target_global_frame = round(_start_pts * tb * fps_frac) + frame
            for f in container.decode(stream):
                if f.pts is None:
                    continue
                cur = round(f.pts * tb * fps_frac)
                if cur < target_global_frame:
                    continue
                if cur > target_global_frame:
                    break
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
                description="v30 video decode",
                on_retry=_drop_cached_container,
            )
            if decoded is not None:
                return decoded
            cached = video_cache.pop(full_key)
            if cached is not None:
                try:
                    cached[0].close()
                except Exception:
                    pass
            return self._zero_frame(reason="frame_overshoot")
        except Exception as e:
            # Drop the cached container on error; it may be in a bad state.
            cached = video_cache.pop(full_key)
            if cached is not None:
                try:
                    cached[0].close()
                except Exception:
                    pass
            logger.warning("[v30-adapter] video decode failed for %s (%s); zero frame.", p, e)
            return self._zero_frame(reason="decode_error")
