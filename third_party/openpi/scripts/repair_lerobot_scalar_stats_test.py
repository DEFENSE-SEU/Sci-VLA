import importlib.util
import json
from pathlib import Path
import sys

SCRIPT_PATH = Path(__file__).with_name("repair_lerobot_scalar_stats.py")
SPEC = importlib.util.spec_from_file_location("repair_lerobot_scalar_stats", SCRIPT_PATH)
repair = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = repair
SPEC.loader.exec_module(repair)


def _write_fixture(root: Path) -> None:
    meta = root / "meta"
    meta.mkdir()
    records = [
        {
            "episode_index": 0,
            "stats": {
                "gripper": {"min": 0.0, "max": 1.0, "mean": 0.5, "std": 0.5, "count": [2]},
                "actions": {
                    "min": [0.0, 0.0],
                    "max": [1.0, 1.0],
                    "mean": [0.5, 0.5],
                    "std": [0.5, 0.5],
                    "count": [2],
                },
            },
        },
        {
            "episode_index": 1,
            "stats": {
                "gripper": {"min": 0.2, "max": 0.8, "mean": 0.5, "std": 0.3, "count": [2]},
            },
        },
    ]
    (meta / "episodes_stats.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
    (meta / "stats.json").write_text(
        json.dumps({"gripper": {"min": 0.0, "max": 1.0, "mean": 0.5, "std": 0.4, "count": [4]}}),
        encoding="utf-8",
    )


def test_repairs_scalar_stats_and_is_idempotent(tmp_path):
    _write_fixture(tmp_path)

    result = repair.repair_dataset_stats(tmp_path)

    assert result == {"episode_values": 8, "global_values": 4}
    records = [json.loads(line) for line in (tmp_path / "meta" / "episodes_stats.jsonl").read_text().splitlines()]
    assert records[0]["stats"]["gripper"]["min"] == [0.0]
    assert records[1]["stats"]["gripper"]["std"] == [0.3]
    assert records[0]["stats"]["actions"]["mean"] == [0.5, 0.5]
    assert (tmp_path / "meta" / "episodes_stats.jsonl.bak").is_file()
    assert (tmp_path / "meta" / "stats.json.bak").is_file()

    assert repair.repair_dataset_stats(tmp_path) == {"episode_values": 0, "global_values": 0}
