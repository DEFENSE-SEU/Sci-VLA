"""MetaReader for LeRobot v3.0 (meta/episodes/chunk-*/file-*.parquet)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from .base import EpisodeMeta


def _normalize_tasks_col(val):
    """Merged datasets have heterogeneous `tasks` column types (list<string>
    for image datasets vs list<list<int>> for tokenized-task datasets).
    Normalize to list<string> before concat.

    The heuristic + normalization live in ``src.schema.task_text`` (shared
    verbatim with the training adapter, so the two stacks cannot drift). This
    read-only probe deliberately passes ``allow_tokenized_coercion=False`` so it
    never reports a digit-string task as normal while training would fail on the
    same shard. Lazy import: the CLI entrypoints run from the repo root, where
    ``src.`` imports resolve.
    """
    from src.schema.task_text import normalize_tasks

    return normalize_tasks(
        val,
        allow_tokenized_coercion=False,  # correctness probe: strict by design
        undecodable_msg=(
            "v30 meta_reader: encountered a tokenized (list<int>) "
            "task element that is not a decodable ASCII byte array. "
            "Coercing it to str() would misrepresent the task as "
            "natural language. Decode the tokens back to text in the "
            "upstream dataset or exclude tokenized-task shards. "
            "(This matches src/adapters/lerobot_v30.py fail-loud "
            "semantics.)"
        ),
    )


class V30MetaReader:
    """Reads concatenated meta/episodes/chunk-*/file-*.parquet shards.

    v3 episodes carry extra shard-pointer columns (data/chunk_index,
    dataset_from_index, videos/<cam>/from_timestamp, ...). Those are
    adapter-internal; for task discovery we only need episode_index,
    length, tasks. Shard pointers are preserved in EpisodeMeta.extra
    so that callers who build v3 adapters can forward them if needed
    (though our current adapter re-reads them itself).
    """

    def read(self, meta_root: Path) -> list[EpisodeMeta]:
        ep_root = Path(meta_root) / "episodes"
        if not ep_root.is_dir():
            raise FileNotFoundError(f"v3 meta/episodes dir missing at {ep_root}")
        paths = sorted(ep_root.rglob("*.parquet"))
        if not paths:
            raise FileNotFoundError(f"no episode parquets under {ep_root}")

        # Heterogeneous-schema-safe concat (see lerobot_v30 adapter for same
        # treatment): go through pandas so camera column unions and tasks
        # type mismatch (list<string> vs list<list<int>>) don't crash concat.
        dfs: list[pd.DataFrame] = []
        for p in paths:
            t = pq.read_table(p)
            # Drop per-episode stats columns early — they're large and unused.
            keep_cols = [c for c in t.schema.names if not c.startswith("stats/")]
            dfs.append(t.select(keep_cols).to_pandas())
        for df in dfs:
            if "tasks" in df.columns:
                df["tasks"] = df["tasks"].map(_normalize_tasks_col)
        df = pd.concat(dfs, axis=0, ignore_index=True, sort=False)

        out: list[EpisodeMeta] = []
        for _, row in df.iterrows():
            raw = row.to_dict()
            tasks_raw = raw.get("tasks")
            if isinstance(tasks_raw, (list, np.ndarray)):
                tasks = tuple(str(t) for t in tasks_raw if t)
            elif tasks_raw:
                tasks = (str(tasks_raw),)
            else:
                tasks = ()
            out.append(EpisodeMeta(
                episode_index=int(raw["episode_index"]),
                length=int(raw["length"]),
                tasks=tasks,
                source=raw.get("source"),   # v3 generally lacks this field
                extra={k: v for k, v in raw.items()
                       if k not in ("episode_index", "length", "tasks", "source")},
            ))
        out.sort(key=lambda e: e.episode_index)
        return out
