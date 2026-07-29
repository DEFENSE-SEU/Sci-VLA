#!/usr/bin/env python3
"""Remove depth assets and depth metadata from a LeRobot v2.1 dataset."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path


def atomic_write_json(path: Path, value: object) -> None:
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as f:
        json.dump(value, f, indent=2)
        f.write("\n")
        temporary_path = Path(f.name)
    os.replace(temporary_path, path)


def atomic_write_jsonl(path: Path, records: list[dict]) -> None:
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as f:
        for record in records:
            f.write(json.dumps(record, separators=(",", ":")) + "\n")
        temporary_path = Path(f.name)
    os.replace(temporary_path, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path, nargs="?", default=Path("data/mani_real"))
    args = parser.parse_args()
    root = args.dataset.resolve()
    meta = root / "meta"
    info_path, episodes_path = meta / "info.json", meta / "episodes.jsonl"
    info = json.loads(info_path.read_text())
    episodes = [json.loads(line) for line in episodes_path.read_text().splitlines() if line]
    depth_dir = root / "depth"
    if not depth_dir.is_dir():
        raise FileNotFoundError(f"Depth directory not found: {depth_dir}")

    for path in (info_path, episodes_path):
        shutil.copy2(path, path.with_name(path.name + ".pre_depth_removal"))
    info["features"].pop("depth", None)
    info["features"].pop("wrist_depth", None)
    info.pop("depth_path", None)
    info["total_depths"] = 0
    for episode in episodes:
        episode.pop("depth", None)
    atomic_write_json(info_path, info)
    atomic_write_jsonl(episodes_path, episodes)
    shutil.rmtree(depth_dir)
    print(f"Removed depth data and metadata from {root}")


if __name__ == "__main__":
    main()
