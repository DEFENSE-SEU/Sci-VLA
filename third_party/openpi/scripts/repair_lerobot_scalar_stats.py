#!/usr/bin/env python3
"""Repair scalar statistics in a LeRobot v2.1 dataset.

LeRobot 0.3.x requires every min/max/mean/std value in
``meta/episodes_stats.jsonl`` to have at least one dimension. Some dataset
exporters serialize scalar features as JSON numbers instead of singleton
lists, which makes ``LeRobotDatasetMetadata`` fail before the data pipeline is
created.
"""

import argparse
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any

STAT_NAMES = frozenset({"min", "max", "mean", "std"})


def _wrap_scalar_stats(stats: dict[str, Any]) -> tuple[dict[str, Any], int]:
    repaired = 0
    for feature_stats in stats.values():
        if not isinstance(feature_stats, dict):
            continue
        for stat_name, value in feature_stats.items():
            if stat_name in STAT_NAMES and not isinstance(value, list):
                feature_stats[stat_name] = [value]
                repaired += 1
    return stats, repaired


def _atomic_write(path: Path, text: str) -> None:
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as file:
        file.write(text)
        temporary_path = Path(file.name)
    try:
        os.replace(temporary_path, path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def _backup(path: Path) -> None:
    backup_path = path.with_name(f"{path.name}.bak")
    if not backup_path.exists():
        shutil.copy2(path, backup_path)


def repair_dataset_stats(dataset_root: Path, *, create_backup: bool = True) -> dict[str, int]:
    meta_dir = dataset_root.expanduser().resolve() / "meta"
    episodes_stats_path = meta_dir / "episodes_stats.jsonl"
    stats_path = meta_dir / "stats.json"
    for path in (episodes_stats_path, stats_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required LeRobot metadata file not found: {path}")

    repaired_episode_values = 0
    episode_lines: list[str] = []
    with episodes_stats_path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record.get("stats"), dict):
                raise ValueError(f"Missing stats object at {episodes_stats_path}:{line_number}")
            record["stats"], repaired = _wrap_scalar_stats(record["stats"])
            repaired_episode_values += repaired
            episode_lines.append(json.dumps(record, separators=(",", ":"), ensure_ascii=False))

    global_stats = json.loads(stats_path.read_text(encoding="utf-8"))
    if not isinstance(global_stats, dict):
        raise ValueError(f"Expected a JSON object in {stats_path}")
    global_stats, repaired_global_values = _wrap_scalar_stats(global_stats)

    if repaired_episode_values:
        if create_backup:
            _backup(episodes_stats_path)
        _atomic_write(episodes_stats_path, "\n".join(episode_lines) + "\n")
    if repaired_global_values:
        if create_backup:
            _backup(stats_path)
        _atomic_write(stats_path, json.dumps(global_stats, indent=2, ensure_ascii=False) + "\n")

    return {
        "episode_values": repaired_episode_values,
        "global_values": repaired_global_values,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_root", type=Path, help="LeRobot dataset root containing meta/")
    parser.add_argument("--no-backup", action="store_true", help="Do not create one-time .bak copies")
    args = parser.parse_args()

    result = repair_dataset_stats(args.dataset_root, create_backup=not args.no_backup)
    print(
        "Repaired "
        f"{result['episode_values']} episode-stat values and "
        f"{result['global_values']} global-stat values."
    )


if __name__ == "__main__":
    main()
